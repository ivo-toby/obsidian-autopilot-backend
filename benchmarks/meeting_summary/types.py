from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

BENCHMARK_SCHEMA_VERSION = 1
VALID_SPLITS = frozenset({"development", "validation", "test"})
VALID_MODEL_KINDS = frozenset({"candidate", "baseline"})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    kind: str


@dataclass(frozen=True)
class PromptSpec:
    id: str
    path: Path


@dataclass(frozen=True)
class CaseSpec:
    id: str
    transcript: Path
    golden: Path
    split: str


@dataclass(frozen=True)
class GenerationSpec:
    repetitions: int
    thinking: str
    timeout_seconds: int


@dataclass(frozen=True)
class JudgeSpec:
    provider: str
    model: str
    thinking: str
    timeout_seconds: int
    pairwise_top_k: int


@dataclass(frozen=True)
class BenchmarkConfig:
    source: Path
    output_dir: Path
    generation: GenerationSpec
    prompts: Tuple[PromptSpec, ...]
    cases: Tuple[CaseSpec, ...]
    models: Tuple[ModelSpec, ...]
    judge: JudgeSpec


@dataclass(frozen=True)
class ScoreSet:
    factual_accuracy: int
    decisions_and_actions: int
    technical_detail_and_blockers: int
    structure_and_compliance: int
    concision_and_usefulness: int

    def weighted_total(self) -> float:
        values: Dict[str, float] = {
            "factual_accuracy": 0.35,
            "decisions_and_actions": 0.25,
            "technical_detail_and_blockers": 0.20,
            "structure_and_compliance": 0.10,
            "concision_and_usefulness": 0.10,
        }
        raw = sum(
            getattr(self, name) * weight for name, weight in values.items()
        )
        return round(raw * 20, 2)
