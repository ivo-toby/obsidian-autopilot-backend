import json
import stat
import time
from pathlib import Path

import pytest

from benchmarks.meeting_summary.pi_rpc import (
    PiRequest,
    PiRpcClient,
    PiRpcError,
)


@pytest.fixture
def fake_pi_path() -> Path:
    path = Path(__file__).parent / "fixtures" / "fake_pi.py"
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def request(timeout_seconds: int = 10) -> PiRequest:
    return PiRequest(
        provider="homelab",
        model="titan/ollama/gemma4:12b",
        thinking="off",
        prompt="summarize this",
        timeout_seconds=timeout_seconds,
    )


def run(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = "normal"
):
    monkeypatch.setenv("FAKE_PI_MODE", mode)
    return PiRpcClient(executable=str(fake_pi_path)).run(request())


def test_rpc_client_returns_text_usage_and_elapsed(fake_pi_path: Path):
    client = PiRpcClient(executable=str(fake_pi_path))
    response = client.run(
        PiRequest(
            provider="homelab",
            model="titan/ollama/gemma4:12b",
            thinking="off",
            prompt="summarize this",
            timeout_seconds=10,
        )
    )

    assert response.text == "# Summary"
    assert response.provider == "homelab"
    assert response.model == "titan/ollama/gemma4:12b"
    assert response.usage["input"] == 100
    assert response.usage["output"] == 20
    assert response.elapsed_seconds >= 0


def test_rpc_command_disables_all_ambient_context(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("FAKE_PI_CAPTURE", str(capture))
    response = PiRpcClient(executable=str(fake_pi_path)).run(request())

    assert response.text == "# Summary"
    invocation = json.loads(capture.read_text(encoding="utf-8"))
    assert invocation["args"] == [
        "--mode",
        "rpc",
        "--provider",
        "homelab",
        "--model",
        "titan/ollama/gemma4:12b",
        "--thinking",
        "off",
        "--no-session",
        "--no-tools",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "--system-prompt",
        "",
    ]
    assert invocation["environment"]["PI_SKIP_VERSION_CHECK"] == "1"
    assert invocation["environment"]["PI_TELEMETRY"] == "0"


def test_rpc_client_waits_for_agent_settled_not_agent_end(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    response = run(fake_pi_path, monkeypatch, "agent_end_first")
    assert response.text == "# Summary"


def test_rpc_client_raises_on_error_stop_reason(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(
        PiRpcError, match="homelab.*titan/ollama/gemma4:12b.*error"
    ):
        run(fake_pi_path, monkeypatch, "error_stop")


def test_rpc_client_raises_on_empty_assistant_text(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(
        PiRpcError, match="homelab.*titan/ollama/gemma4:12b.*empty"
    ):
        run(fake_pi_path, monkeypatch, "empty_text")


def test_rpc_client_times_out_and_terminates_process(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    started = time.monotonic()
    with pytest.raises(
        PiRpcError, match="homelab.*titan/ollama/gemma4:12b.*timed out"
    ):
        monkeypatch.setenv("FAKE_PI_MODE", "timeout")
        PiRpcClient(executable=str(fake_pi_path)).run(request(1))
    assert time.monotonic() - started < 4


def test_rpc_client_surfaces_invalid_jsonl(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(
        PiRpcError, match="homelab.*titan/ollama/gemma4:12b.*JSON"
    ):
        run(fake_pi_path, monkeypatch, "invalid_json")


def test_rpc_client_surfaces_early_process_exit(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(
        PiRpcError, match="homelab.*titan/ollama/gemma4:12b.*exited"
    ):
        run(fake_pi_path, monkeypatch, "exit_early")


def test_rpc_client_captures_stderr_without_deadlock(
    fake_pi_path: Path, monkeypatch: pytest.MonkeyPatch
):
    response = run(fake_pi_path, monkeypatch, "stderr_flood")
    assert response.stderr.startswith("diagnostic stderr")
