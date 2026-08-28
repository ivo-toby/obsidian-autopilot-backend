from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml  # type: ignore[import-untyped]

from .types import (
    BENCHMARK_SCHEMA_VERSION,
    VALID_MODEL_KINDS,
    VALID_SPLITS,
    BenchmarkConfig,
    CaseSpec,
    GenerationSpec,
    JudgeSpec,
    ModelSpec,
    PromptSpec,
)


def resolve_config_path(value: str, config_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    return candidate.resolve()


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    document = _mapping(raw, "configuration")
    version = document.get("version")
    if isinstance(version, bool) or version != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark schema version: {version!r}")

    config_dir = source.parent
    generation_data = _mapping(document.get("generation"), "generation")
    judge_data = _mapping(document.get("judge"), "judge")

    prompt_data = _collection(document.get("prompts"), "prompts")
    prompts = tuple(
        PromptSpec(
            id=_string(item, "id", "prompt"),
            path=resolve_config_path(
                _string(item, "path", "prompt"), config_dir
            ),
        )
        for item in prompt_data
    )

    case_data = _collection(document.get("cases"), "cases")
    cases = tuple(
        CaseSpec(
            id=_string(item, "id", "case"),
            transcript=resolve_config_path(
                _string(item, "transcript", "case"), config_dir
            ),
            golden=resolve_config_path(
                _string(item, "golden", "case"), config_dir
            ),
            split=_string(item, "split", "case"),
        )
        for item in case_data
    )

    model_data = _collection(document.get("models"), "models")
    models = tuple(
        ModelSpec(
            id=_string(item, "id", "model"),
            provider=_string(item, "provider", "model"),
            model=_string(item, "model", "model"),
            kind=_string(item, "kind", "model"),
        )
        for item in model_data
    )

    generation = GenerationSpec(
        repetitions=_integer(generation_data, "repetitions", "generation"),
        thinking=_thinking(generation_data, "thinking", "generation"),
        timeout_seconds=_integer(
            generation_data, "timeout_seconds", "generation"
        ),
    )
    judge = JudgeSpec(
        provider=_string(judge_data, "provider", "judge"),
        model=_string(judge_data, "model", "judge"),
        thinking=_thinking(judge_data, "thinking", "judge"),
        timeout_seconds=_integer(judge_data, "timeout_seconds", "judge"),
        pairwise_top_k=_integer(judge_data, "pairwise_top_k", "judge"),
    )
    config = BenchmarkConfig(
        source=source,
        output_dir=resolve_config_path(
            _string(document, "output_dir", "configuration"), config_dir
        ),
        generation=generation,
        prompts=prompts,
        cases=cases,
        models=models,
        judge=judge,
    )
    validate_benchmark_config(config)
    return config


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if config.generation.repetitions <= 0:
        raise ValueError("generation repetitions must be positive")
    if config.generation.timeout_seconds <= 0:
        raise ValueError("generation timeout_seconds must be positive")
    if config.judge.timeout_seconds <= 0:
        raise ValueError("judge timeout_seconds must be positive")
    if config.judge.pairwise_top_k < 2:
        raise ValueError("judge pairwise_top_k must be at least two")

    _require_nonempty(config.prompts, "prompts")
    _require_nonempty(config.cases, "cases")
    _require_nonempty(config.models, "models")
    _require_unique((prompt.id for prompt in config.prompts), "prompt")
    _require_unique((case.id for case in config.cases), "case")
    _require_unique((model.id for model in config.models), "model")

    for prompt in config.prompts:
        _require_path_safe_id(prompt.id, "prompt")
        _require_file(prompt.path, "prompt")
    for case in config.cases:
        _require_file(case.transcript, "transcript")
        _require_file(case.golden, "golden")
        if case.split not in VALID_SPLITS:
            raise ValueError(f"Invalid case split: {case.split!r}")
    for model in config.models:
        if not model.model:
            raise ValueError(f"Model {model.id!r} has an empty model name")
        if model.kind not in VALID_MODEL_KINDS:
            raise ValueError(f"Invalid model kind: {model.kind!r}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _collection(value: Any, name: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return tuple(_mapping(item, name) for item in value)


def _string(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} {key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, Any], key: str, name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} {key} must be an integer")
    return value


def _thinking(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if isinstance(value, bool):
        return "on" if value else "off"
    return _string(mapping, key, name)


def _require_nonempty(values: Sequence[Any], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def _require_unique(ids: Iterable[str], name: str) -> None:
    values = tuple(ids)
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name} id")


def _require_path_safe_id(value: str, name: str) -> None:
    if "/" in value or "\\" in value:
        raise ValueError(f"{name} id must not contain path separators")


def _require_file(path: Path, kind: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configured {kind} file does not exist: {path}"
        )
