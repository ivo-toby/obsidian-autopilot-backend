from pathlib import Path
from typing import Optional

import pytest

from benchmarks.meeting_summary.config import load_benchmark_config


def _write_config(
    tmp_path: Path,
    *,
    models: str = """\
  - id: gemma12
    provider: homelab
    model: titan/ollama/gemma4:12b
    kind: candidate
""",
    prompts: Optional[str] = None,
    cases: Optional[str] = None,
    generation: str = """\
  repetitions: 3
  thinking: off
  timeout_seconds: 900
""",
    judge: str = """\
  provider: openai-codex
  model: gpt-5.6-sol
  thinking: high
  timeout_seconds: 900
  pairwise_top_k: 3
""",
) -> Path:
    prompt = tmp_path / "prompt.md"
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    for path in (prompt, transcript, golden):
        path.write_text("content", encoding="utf-8")

    default_prompt = f"  - id: current\n    path: {prompt}"
    default_cases = (
        f"  - id: sync\n"
        f"    transcript: {transcript}\n"
        f"    golden: {golden}\n"
        "    split: development"
    )
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""\
version: 1
output_dir: {tmp_path / 'results'}
generation:
{generation}
prompts:
{prompts or default_prompt}
cases:
{cases or default_cases}
models:
{models}
judge:
{judge}
""",
        encoding="utf-8",
    )
    return config_path


def test_load_config_expands_paths_and_preserves_colons(tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    for path in (prompt, transcript, golden):
        path.write_text("content", encoding="utf-8")

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""
version: 1
output_dir: {tmp_path / 'results'}
generation:
  repetitions: 3
  thinking: off
  timeout_seconds: 900
prompts:
  - id: current
    path: {prompt}
cases:
  - id: sync
    transcript: {transcript}
    golden: {golden}
    split: development
models:
  - id: gemma12
    provider: homelab
    model: titan/ollama/gemma4:12b
    kind: candidate
judge:
  provider: openai-codex
  model: gpt-5.6-sol
  thinking: high
  timeout_seconds: 900
  pairwise_top_k: 3
""",
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.models[0].model == "titan/ollama/gemma4:12b"
    assert config.prompts[0].path == prompt
    assert config.cases[0].split == "development"
    assert config.output_dir == tmp_path / "results"
    assert isinstance(config.prompts, tuple)
    assert isinstance(config.cases, tuple)
    assert isinstance(config.models, tuple)


def test_load_config_resolves_relative_paths_and_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_dir = tmp_path / "configs"
    relative_dir = config_dir / "fixtures"
    home_dir = tmp_path / "home"
    config_dir.mkdir()
    relative_dir.mkdir()
    home_dir.mkdir()
    (relative_dir / "prompt.md").write_text("prompt", encoding="utf-8")
    (relative_dir / "transcript.md").write_text("transcript", encoding="utf-8")
    (home_dir / "golden.md").write_text("golden", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    config_path = config_dir / "benchmark.yaml"
    config_path.write_text(
        """\
version: 1
output_dir: ~/results
generation:
  repetitions: 1
  thinking: off
  timeout_seconds: 10
prompts:
  - id: current
    path: fixtures/prompt.md
cases:
  - id: sync
    transcript: fixtures/transcript.md
    golden: ~/golden.md
    split: development
models:
  - id: gemma12
    provider: homelab
    model: titan/ollama/gemma4:12b
    kind: candidate
judge:
  provider: openai-codex
  model: gpt-5.6-sol
  thinking: high
  timeout_seconds: 10
  pairwise_top_k: 2
""",
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.output_dir == home_dir / "results"
    assert config.prompts[0].path == relative_dir / "prompt.md"
    assert config.cases[0].transcript == relative_dir / "transcript.md"
    assert config.cases[0].golden == home_dir / "golden.md"


def test_duplicate_model_ids_are_rejected(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        models="""\
  - id: duplicate
    provider: homelab
    model: model-a
    kind: candidate
  - id: duplicate
    provider: homelab
    model: model-b
    kind: baseline
""",
    )
    with pytest.raises(ValueError, match="duplicate model id"):
        load_benchmark_config(config_path)


def test_duplicate_prompt_ids_are_rejected(tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    config_path = _write_config(
        tmp_path,
        prompts=f"""\
  - id: duplicate
    path: {prompt}
  - id: duplicate
    path: {prompt}
""",
    )
    with pytest.raises(ValueError, match="duplicate prompt id"):
        load_benchmark_config(config_path)


def test_duplicate_case_ids_are_rejected(tmp_path: Path):
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    config_path = _write_config(
        tmp_path,
        cases=f"""\
  - id: duplicate
    transcript: {transcript}
    golden: {golden}
    split: development
  - id: duplicate
    transcript: {transcript}
    golden: {golden}
    split: test
""",
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_benchmark_config(config_path)


def test_invalid_split_is_rejected(tmp_path: Path):
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    config_path = _write_config(
        tmp_path,
        cases=f"""\
  - id: sync
    transcript: {transcript}
    golden: {golden}
    split: staging
""",
    )
    with pytest.raises(ValueError, match="split"):
        load_benchmark_config(config_path)


def test_missing_prompt_is_rejected(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        prompts=f"""\
  - id: current
    path: {tmp_path / 'missing-prompt.md'}
""",
    )
    with pytest.raises(FileNotFoundError, match="prompt"):
        load_benchmark_config(config_path)


def test_missing_transcript_is_rejected(tmp_path: Path):
    golden = tmp_path / "golden.md"
    config_path = _write_config(
        tmp_path,
        cases=f"""\
  - id: sync
    transcript: {tmp_path / 'missing-transcript.md'}
    golden: {golden}
    split: development
""",
    )
    with pytest.raises(FileNotFoundError, match="transcript"):
        load_benchmark_config(config_path)


def test_missing_golden_is_rejected(tmp_path: Path):
    transcript = tmp_path / "transcript.md"
    config_path = _write_config(
        tmp_path,
        cases=f"""\
  - id: sync
    transcript: {transcript}
    golden: {tmp_path / 'missing-golden.md'}
    split: development
""",
    )
    with pytest.raises(FileNotFoundError, match="golden"):
        load_benchmark_config(config_path)


def test_repetitions_must_be_positive(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        generation="""\
  repetitions: 0
  thinking: off
  timeout_seconds: 900
""",
    )
    with pytest.raises(ValueError, match="repetitions"):
        load_benchmark_config(config_path)


def test_pairwise_top_k_must_be_at_least_two(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        judge="""\
  provider: openai-codex
  model: gpt-5.6-sol
  thinking: high
  timeout_seconds: 900
  pairwise_top_k: 1
""",
    )
    with pytest.raises(ValueError, match="pairwise_top_k"):
        load_benchmark_config(config_path)
