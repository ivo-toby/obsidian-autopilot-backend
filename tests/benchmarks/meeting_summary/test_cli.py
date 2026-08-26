"""CLI contract tests for the meeting-summary benchmark."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.meeting_summary.cli as cli_module
from benchmarks.meeting_summary.cli import main


ROOT = Path(__file__).resolve().parents[3]
FAKE_PI = Path(__file__).with_name("fixtures") / "fake_pi.py"


def _config(tmp_path, models=("candidate-a", "candidate-b", "luna-control")):
    prompt = tmp_path / "prompt.md"
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    prompt.write_text("prompt", encoding="utf-8")
    transcript.write_text("private transcript", encoding="utf-8")
    golden.write_text("golden summary", encoding="utf-8")
    model_lines = []
    for model in models:
        kind = "baseline" if model == "luna-control" else "candidate"
        model_lines.append(
            "  - id: %s\n    provider: homelab\n"
            "    model: titan/ollama/gemma4:12b\n    kind: %s"
            % (model, kind)
        )
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "version: 1\n"
        "output_dir: %s\n"
        "generation:\n  repetitions: 2\n  thinking: off\n"
        "  timeout_seconds: 10\n"
        "prompts:\n  - id: current\n    path: prompt.md\n"
        "cases:\n  - id: case\n    transcript: transcript.md\n"
        "    golden: golden.md\n    split: development\n"
        "models:\n%s\n"
        "judge:\n  provider: homelab\n  model: titan/ollama/gemma4:12b\n"
        "  thinking: off\n  timeout_seconds: 10\n  pairwise_top_k: 2\n"
        % (tmp_path / "runs", "\n".join(model_lines)),
        encoding="utf-8",
    )
    return config


def _run_cli(args, env=None):
    environment = os.environ.copy()
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.meeting_summary"] + args,
        cwd=str(ROOT),
        env=environment,
        text=True,
        capture_output=True,
    )


def test_validate_does_not_execute_pi(tmp_path, monkeypatch, capsys):
    config = _config(tmp_path)
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(tmp_path / "missing-pi"))

    assert main(["validate", "--config", str(config)]) == 0
    assert "valid" in capsys.readouterr().out.lower()
    assert not list((tmp_path / "runs").glob("*"))


def test_generate_prints_run_directory_and_preserves_failures(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path, models=("candidate-a",))
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    monkeypatch.setenv("FAKE_PI_MODE", "error_stop")

    assert main(["generate", "--config", str(config)]) != 0
    output = capsys.readouterr().out
    assert str((tmp_path / "runs").resolve()) in output
    runs = list((tmp_path / "runs").glob("*"))
    assert len(runs) == 1
    artifacts = list((runs[0] / "generations").rglob("*.json"))
    assert artifacts
    assert all(
        json.loads(path.read_text())["status"] == "failed"
        for path in artifacts
    )


def test_judge_requires_existing_run_and_report_requires_complete_judgment(
    tmp_path, capsys
):
    config = _config(tmp_path)
    missing = tmp_path / "missing"
    assert (
        main(["judge", "--config", str(config), "--run-dir", str(missing)])
        != 0
    )
    assert "run" in capsys.readouterr().err.lower()
    assert main(["report", "--run-dir", str(missing)]) != 0


def test_judge_unknown_filters_fail_before_rpc(tmp_path, monkeypatch):
    config = _config(tmp_path)
    run_dir = cli_module.RunStore.create(tmp_path / "runs", config)
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    monkeypatch.setenv("FAKE_PI_CAPTURE", str(capture))

    for option in ("model", "prompt", "case", "split"):
        assert (
            main(
                [
                    "judge",
                    "--config",
                    str(config),
                    "--run-dir",
                    str(run_dir.run_dir),
                    "--%s" % option,
                    "missing",
                ]
            )
            != 0
        )
        assert not capture.exists()


def test_report_existing_run_without_complete_judgment_writes_files(
    tmp_path
):
    config = _config(tmp_path)
    run_dir = cli_module.RunStore.create(tmp_path / "runs", config)

    assert main(["report", "--run-dir", str(run_dir.run_dir)]) != 0
    assert all(
        (run_dir.run_dir / name).exists()
        for name in ("report.md", "report.json", "report.csv")
    )


def test_repeatable_filters_and_unknown_filter_fail_before_pi(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    monkeypatch.setenv("FAKE_PI_CAPTURE", str(capture))

    assert main(
        [
            "generate",
            "--config",
            str(config),
            "--model",
            "candidate-a",
            "--model",
            "candidate-b",
            "--prompt",
            "current",
            "--case",
            "case",
            "--split",
            "development",
        ]
    ) == 0
    assert capture.exists()
    capture.unlink()
    assert main(
        ["generate", "--config", str(config), "--model", "missing"]
    ) != 0
    assert not capture.exists()


def test_all_runs_generation_absolute_pairwise_and_report_in_order(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    monkeypatch.setenv("FAKE_PI_MODE", "valid_judgment")

    result = _run_cli(["all", "--config", str(config)])
    assert result.returncode == 0, result.stderr
    run_dirs = list((tmp_path / "runs").glob("*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert list((run_dir / "generations").rglob("*.md"))
    assert list((run_dir / "judgments").rglob("*.json"))
    assert list((run_dir / "pairwise").rglob("*.json"))
    assert all(
        (run_dir / name).exists()
        for name in ("report.md", "report.json", "report.csv")
    )
    assert "private transcript" not in "".join(
        path.read_text(encoding="utf-8")
        for path in run_dir.rglob("*")
        if path.is_file() and path.name.startswith("report")
    )


def test_all_calls_phases_in_order(tmp_path, monkeypatch):
    config = _config(tmp_path, models=("candidate-a",))
    store = cli_module.RunStore.create(tmp_path / "runs", config)
    phases = []

    monkeypatch.setattr(cli_module, "_open_or_create_store", lambda *_: store)
    monkeypatch.setattr(
        cli_module,
        "generate_candidates",
        lambda *args, **kwargs: phases.append("generation") or (),
    )
    monkeypatch.setattr(
        cli_module,
        "judge_generations",
        lambda *args, **kwargs: phases.append("absolute") or (),
    )
    monkeypatch.setattr(
        cli_module,
        "judge_pairwise_top_models",
        lambda *args, **kwargs: phases.append("pairwise") or (),
    )
    monkeypatch.setattr(
        cli_module,
        "build_report",
        lambda *_: phases.append("report") or SimpleNamespace(failures=[]),
    )
    monkeypatch.setattr(cli_module, "write_report", lambda *_: None)
    monkeypatch.setattr(cli_module, "_has_complete_judgment", lambda *_: True)

    assert main(["all", "--config", str(config)]) == 0
    assert phases == ["generation", "absolute", "pairwise", "report"]


def test_all_fail_fast_generation_skips_judging_but_writes_report(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, models=("candidate-a",))
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    monkeypatch.setenv("FAKE_PI_MODE", "error_stop")
    monkeypatch.setattr(
        cli_module,
        "_judge_store",
        lambda *args, **kwargs: pytest.fail("judging must not start"),
    )

    run_code = main(
        ["all", "--config", str(config), "--fail-fast"]
    )
    run_dir = next((tmp_path / "runs").glob("*"))
    assert run_code != 0
    assert all(
        (run_dir / name).exists()
        for name in ("report.md", "report.json", "report.csv")
    )


def test_resume_reuses_completed_artifacts_and_fail_fast_stops(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, models=("candidate-a",))
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    first = main(["generate", "--config", str(config)])
    assert first == 0
    run_dir = next((tmp_path / "runs").glob("*"))
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("FAKE_PI_CAPTURE", str(capture))
    assert (
        main(["generate", "--config", str(config), "--resume", str(run_dir)])
        == 0
    )
    assert not capture.exists()
    monkeypatch.setenv("FAKE_PI_MODE", "error_stop")
    assert (
        main(
            [
                "all",
                "--config",
                str(config),
                "--resume",
                str(run_dir),
                "--fail-fast",
            ]
        )
        != 0
    )
