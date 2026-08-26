"""One-shot JSONL RPC client for isolated Pi benchmark generations."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PiRequest:
    provider: str
    model: str
    thinking: str
    prompt: str
    timeout_seconds: int


@dataclass(frozen=True)
class PiResponse:
    text: str
    provider: str
    model: str
    stop_reason: str
    usage: Dict[str, object]
    session_tokens: Dict[str, int]
    elapsed_seconds: float
    stderr: str


class PiRpcError(RuntimeError):
    """Raised when a Pi benchmark request cannot produce a valid response."""


class _ExecutionFailure(Exception):
    pass


class PiRpcClient:
    """Run each benchmark prompt in a fresh, context-free Pi process."""

    _MAX_STDERR_LINES = 2000
    _MAX_STDERR_LINE_LENGTH = 4096
    _SHUTDOWN_TIMEOUT_SECONDS = 2.0

    def __init__(
        self,
        executable: str = "pi",
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.executable = str(executable)
        self._environment = env

    def build_command(self, request: PiRequest) -> List[str]:
        return [
            self.executable,
            "--mode",
            "rpc",
            "--provider",
            request.provider,
            "--model",
            request.model,
            "--thinking",
            request.thinking,
            "--no-session",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--system-prompt",
            "",
        ]

    def run(self, request: PiRequest) -> PiResponse:
        started = time.monotonic()
        process: Optional[subprocess.Popen[str]] = None
        stdout_thread: Optional[threading.Thread] = None
        stderr_thread: Optional[threading.Thread] = None
        stderr_lines: Deque[str] = deque(maxlen=self._MAX_STDERR_LINES)
        failure: Optional[str] = None
        response: Optional[PiResponse] = None

        try:
            environment = dict(os.environ)
            if self._environment is not None:
                environment.update(self._environment)
            environment["PI_SKIP_VERSION_CHECK"] = "1"
            environment["PI_TELEMETRY"] = "0"
            process = subprocess.Popen(
                self.build_command(request),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            output_queue: "queue.Queue[Optional[str]]" = queue.Queue()
            stdout_thread = self._start_stdout_reader(
                process.stdout, output_queue
            )
            stderr_thread = self._start_stderr_reader(
                process.stderr, stderr_lines
            )
            deadline = started + request.timeout_seconds

            self._send(
                process,
                {
                    "id": "benchmark-prompt",
                    "type": "prompt",
                    "message": request.prompt,
                },
            )
            message: Optional[Mapping[str, Any]] = None
            settled = False
            while not settled:
                item = self._next_output(
                    process, output_queue, deadline, request
                )
                if item is None:
                    raise _ExecutionFailure(
                        "process exited before agent_settled"
                    )
                event_name = self._event_name(item)
                if event_name == "message_end":
                    candidate = item.get("message")
                    if isinstance(candidate, Mapping) and (
                        candidate.get("role") == "assistant"
                    ):
                        message = candidate
                elif event_name == "agent_settled":
                    settled = True
                elif item.get("type") == "response":
                    self._check_response(item, "prompt")

            if message is None:
                raise _ExecutionFailure("no completed assistant message")

            last_text = self._request_response(
                process,
                output_queue,
                deadline,
                request,
                "benchmark-last-text",
                "get_last_assistant_text",
            )
            stats = self._request_response(
                process,
                output_queue,
                deadline,
                request,
                "benchmark-session-stats",
                "get_session_stats",
            )

            stop_reason = self._string_value(
                message, "stopReason", "stop_reason"
            )
            if stop_reason in {"error", "aborted", "length"}:
                raise _ExecutionFailure(
                    "assistant stopped with disallowed stop reason "
                    f"{stop_reason!r}"
                )
            text = self._text_from_response(last_text)
            if not text.strip():
                raise _ExecutionFailure("empty assistant text")

            provider = (
                self._string_value(message, "provider") or request.provider
            )
            model = self._string_value(message, "model") or request.model
            usage = self._mapping_value(message.get("usage"))
            session_tokens = self._session_tokens(stats)
            response = PiResponse(
                text=text,
                provider=provider,
                model=model,
                stop_reason=stop_reason or "",
                usage=dict(usage),
                session_tokens=session_tokens,
                elapsed_seconds=time.monotonic() - started,
                stderr="",
            )
        except _ExecutionFailure as error:
            failure = str(error)
        except (OSError, ValueError, TypeError, BrokenPipeError) as error:
            failure = str(error)
        finally:
            if process is not None:
                self._shutdown(process)
            for thread in (stdout_thread, stderr_thread):
                if thread is not None:
                    thread.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)

        stderr = "".join(stderr_lines)
        if failure is not None:
            raise PiRpcError(self._message(request, failure, stderr))
        assert response is not None
        return PiResponse(
            text=response.text,
            provider=response.provider,
            model=response.model,
            stop_reason=response.stop_reason,
            usage=response.usage,
            session_tokens=response.session_tokens,
            elapsed_seconds=response.elapsed_seconds,
            stderr=stderr,
        )

    @staticmethod
    def _start_stdout_reader(
        stream: Any, output_queue: "queue.Queue[Optional[str]]"
    ) -> threading.Thread:
        def read_stdout() -> None:
            try:
                for line in stream:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        thread = threading.Thread(target=read_stdout, daemon=True)
        thread.start()
        return thread

    def _start_stderr_reader(
        self, stream: Any, stderr_lines: Deque[str]
    ) -> threading.Thread:
        def read_stderr() -> None:
            for line in stream:
                stderr_lines.append(line[: self._MAX_STDERR_LINE_LENGTH])

        thread = threading.Thread(target=read_stderr, daemon=True)
        thread.start()
        return thread

    def _next_output(
        self,
        process: subprocess.Popen[str],
        output_queue: "queue.Queue[Optional[str]]",
        deadline: float,
        request: PiRequest,
    ) -> Optional[Dict[str, Any]]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ExecutionFailure("timed out waiting for RPC response")
        try:
            line = output_queue.get(timeout=remaining)
        except queue.Empty:
            raise _ExecutionFailure("timed out waiting for RPC response")
        if line is None:
            if process.poll() is None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is not None:
                raise _ExecutionFailure("process exited before RPC completed")
            raise _ExecutionFailure("stdout reader stopped unexpectedly")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise _ExecutionFailure(f"invalid JSONL from Pi: {error}")
        if not isinstance(value, dict):
            raise _ExecutionFailure("invalid JSONL from Pi: expected object")
        return value

    def _request_response(
        self,
        process: subprocess.Popen[str],
        output_queue: "queue.Queue[Optional[str]]",
        deadline: float,
        request: PiRequest,
        command_id: str,
        command: str,
    ) -> Mapping[str, Any]:
        self._send(process, {"id": command_id, "type": command})
        while True:
            item = self._next_output(process, output_queue, deadline, request)
            if item is None:
                raise _ExecutionFailure(
                    f"process exited before {command} response"
                )
            if item.get("type") != "response":
                continue
            if item.get("command") != command:
                continue
            self._check_response(item, command)
            data = item.get("data")
            if isinstance(data, Mapping):
                return data
            return {"value": data}

    @staticmethod
    def _send(
        process: subprocess.Popen[str], command: Mapping[str, Any]
    ) -> None:
        if process.stdin is None:
            raise _ExecutionFailure("Pi process stdin is unavailable")
        try:
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise _ExecutionFailure(f"could not send RPC command: {error}")

    @staticmethod
    def _event_name(item: Mapping[str, Any]) -> Optional[str]:
        event = item.get("event")
        if isinstance(event, str):
            return event
        item_type = item.get("type")
        return item_type if isinstance(item_type, str) else None

    @staticmethod
    def _check_response(item: Mapping[str, Any], command: str) -> None:
        if item.get("success") is False:
            error = item.get("error") or "unknown error"
            raise _ExecutionFailure(f"Pi {command} command failed: {error}")

    @staticmethod
    def _mapping_value(value: Any) -> Mapping[str, object]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _string_value(mapping: Mapping[str, Any], *keys: str) -> str:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str):
                return value
        return ""

    @classmethod
    def _text_from_response(cls, response: Mapping[str, Any]) -> str:
        value = response.get("text")
        if isinstance(value, str):
            return value
        value = response.get("value")
        return value if isinstance(value, str) else ""

    @classmethod
    def _session_tokens(cls, stats: Mapping[str, Any]) -> Dict[str, int]:
        value = stats.get("tokens")
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(item, int) and not isinstance(item, bool)
        }

    def _shutdown(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdin is not None:
            process.stdin.close()

    @staticmethod
    def _message(request: PiRequest, detail: str, stderr: str) -> str:
        message = (
            f"Pi RPC failed for {request.provider}/{request.model}: {detail}"
        )
        if stderr:
            message += f"; stderr: {stderr.strip()}"
        return message
