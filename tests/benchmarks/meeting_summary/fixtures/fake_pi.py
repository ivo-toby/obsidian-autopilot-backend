#!/usr/bin/env python3
"""Deterministic fake Pi RPC executable used by the benchmark tests."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ARGS = sys.argv[1:]
MODE = os.environ.get("FAKE_PI_MODE", "normal")


def emit(value: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


def record_invocation() -> None:
    capture = os.environ.get("FAKE_PI_CAPTURE")
    if capture:
        Path(capture).write_text(
            json.dumps(
                {
                    "args": ARGS,
                    "environment": {
                        key: os.environ[key]
                        for key in ("PI_SKIP_VERSION_CHECK", "PI_TELEMETRY")
                        if key in os.environ
                    },
                }
            ),
            encoding="utf-8",
        )


def main() -> None:
    record_invocation()
    line = sys.stdin.readline()
    if not line:
        return
    command = json.loads(line)
    if command.get("type") != "prompt":
        return

    if MODE == "invalid_json":
        sys.stdout.write("not json\n")
        sys.stdout.flush()
        return
    if MODE == "exit_early":
        return
    if MODE == "stderr_flood":
        sys.stderr.write("diagnostic stderr\n" * 100000)
        sys.stderr.flush()

    emit({"type": "response", "id": command.get("id"), "command": "prompt"})
    emit({"type": "event", "event": "agent_start"})
    if MODE == "agent_end_first":
        emit({"type": "event", "event": "agent_end"})

    stop_reason = "stop"
    text = "# Summary"
    if MODE == "valid_judgment":
        if "winner" in command.get("message", ""):
            text = json.dumps(
                {
                    "winner": "tie",
                    "reason": "Both summaries are equivalent.",
                    "critical_difference": "none",
                    "confidence": 3,
                }
            )
        else:
            text = json.dumps(
                {
                    "scores": {
                        "factual_accuracy": 4,
                        "decisions_and_actions": 4,
                        "technical_detail_and_blockers": 4,
                        "structure_and_compliance": 4,
                        "concision_and_usefulness": 4,
                    },
                    "critical_errors": [],
                    "missed_items": [],
                    "failure_tags": [],
                    "prompt_recommendations": [],
                    "verdict": "acceptable",
                }
            )
    if MODE == "error_stop":
        stop_reason = "error"
    elif MODE == "empty_text":
        text = ""
    elif MODE == "length_stop":
        stop_reason = "length"
    elif MODE == "aborted_stop":
        stop_reason = "aborted"

    emit(
        {
            "type": "event",
            "event": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "provider": "homelab",
                "model": "titan/ollama/gemma4:12b",
                "stopReason": stop_reason,
                "usage": {"input": 100, "output": 20},
            },
        }
    )
    if MODE == "agent_end_first":
        time.sleep(0.05)
    if MODE == "timeout":
        time.sleep(60)
        return

    emit({"type": "event", "event": "agent_settled"})
    for line in sys.stdin:
        request = json.loads(line)
        request_type = request.get("type")
        if request_type == "get_last_assistant_text":
            emit(
                {
                    "type": "response",
                    "id": request.get("id"),
                    "command": request_type,
                    "success": True,
                    "data": {"text": text},
                }
            )
        elif request_type == "get_session_stats":
            emit(
                {
                    "type": "response",
                    "id": request.get("id"),
                    "command": request_type,
                    "success": True,
                    "data": {
                        "tokens": {"input": 100, "output": 20, "total": 120}
                    },
                }
            )
        else:
            emit(
                {"type": "response", "command": request_type, "success": True}
            )


if __name__ == "__main__":
    main()
