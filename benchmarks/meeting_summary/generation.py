"""Prompt composition and resumable benchmark generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Set, Tuple

from .pi_rpc import PiRequest, PiResponse, PiRpcClient
from .storage import RunStore, canonical_json_hash, sha256_text
from .types import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkConfig,
    CaseSpec,
    ModelSpec,
    PromptSpec,
)


@dataclass(frozen=True)
class GenerationFilters:
    model_ids: Optional[Set[str]] = None
    prompt_ids: Optional[Set[str]] = None
    case_ids: Optional[Set[str]] = None
    splits: Optional[Set[str]] = None


@dataclass(frozen=True)
class GenerationJob:
    case_id: str
    split: str
    prompt_id: str
    prompt_sha256: str
    transcript_sha256: str
    model_id: str
    provider: str
    model: str
    kind: str
    thinking: str
    repetition: int
    prompt: str
    transcript: str
    cache_key: str


@dataclass(frozen=True)
class GenerationArtifact:
    schema_version: int
    operation: str
    cache_key: str
    status: str
    case_id: str
    split: str
    prompt_id: str
    prompt_sha256: str
    transcript_sha256: str
    model_id: str
    provider: str
    model: str
    kind: str
    thinking: str
    repetition: int
    elapsed_seconds: float
    usage: Dict[str, object]
    session_tokens: Dict[str, object]
    stop_reason: str
    summary_path: str
    summary_sha256: str
    stderr: str
    artifact_path: Optional[Path] = None
    error: str = ""

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], path: Optional[Path] = None
    ) -> "GenerationArtifact":
        return cls(
            schema_version=_as_int(payload.get("schema_version", 1)),
            operation=str(payload.get("operation", "generation")),
            cache_key=str(payload["cache_key"]),
            status=str(payload["status"]),
            case_id=str(payload["case_id"]),
            split=str(payload["split"]),
            prompt_id=str(payload["prompt_id"]),
            prompt_sha256=str(payload["prompt_sha256"]),
            transcript_sha256=str(payload["transcript_sha256"]),
            model_id=str(payload["model_id"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            kind=str(payload["kind"]),
            thinking=str(payload["thinking"]),
            repetition=_as_int(payload["repetition"]),
            elapsed_seconds=_as_float(payload.get("elapsed_seconds", 0.0)),
            usage=_as_dict(payload.get("usage", {})),
            session_tokens=_as_dict(payload.get("session_tokens", {})),
            stop_reason=str(payload.get("stop_reason", "")),
            summary_path=str(payload.get("summary_path", "")),
            summary_sha256=str(payload.get("summary_sha256", "")),
            stderr=str(payload.get("stderr", "")),
            artifact_path=path,
            error=str(payload.get("error", "")),
        )

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "cache_key": self.cache_key,
            "status": self.status,
            "case_id": self.case_id,
            "split": self.split,
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
            "transcript_sha256": self.transcript_sha256,
            "model_id": self.model_id,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "thinking": self.thinking,
            "repetition": self.repetition,
            "elapsed_seconds": self.elapsed_seconds,
            "usage": self.usage,
            "session_tokens": self.session_tokens,
            "stop_reason": self.stop_reason,
            "summary_path": self.summary_path,
            "summary_sha256": self.summary_sha256,
            "stderr": self.stderr,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def compose_meeting_prompt(prompt: str, transcript: str) -> str:
    return "%s\n\n'''TRANSCRIPT'''\n%s" % (prompt, transcript)


def generate_candidates(
    config: BenchmarkConfig,
    store: RunStore,
    filters: Optional[object],
    client: PiRpcClient,
    fail_fast: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[GenerationArtifact, ...]:
    selected = _normalize_filters(filters)
    _validate_filters(config, selected)

    prompts = _selected(config.prompts, selected.prompt_ids)
    cases = _selected(config.cases, selected.case_ids)
    if selected.splits is not None:
        cases = tuple(case for case in cases if case.split in selected.splits)
    models = _selected(config.models, selected.model_ids)
    _record_inputs(store, prompts, cases)
    total = (
        len(models) * len(prompts) * len(cases) * config.generation.repetitions
    )

    artifacts = []
    index = 0
    for model in models:
        for prompt_spec in prompts:
            prompt = prompt_spec.path.read_text(encoding="utf-8")
            prompt_hash = sha256_text(prompt)
            for case in cases:
                transcript = case.transcript.read_text(encoding="utf-8")
                transcript_hash = sha256_text(transcript)
                for repetition in range(1, config.generation.repetitions + 1):
                    index += 1
                    job = _make_job(
                        model,
                        prompt_spec,
                        case,
                        prompt,
                        transcript,
                        prompt_hash,
                        transcript_hash,
                        config.generation.thinking,
                        repetition,
                    )
                    cached_path = store.find_completed(
                        "generation", job.cache_key
                    )
                    if cached_path is not None:
                        cached = GenerationArtifact.from_payload(
                            store.read_json(cached_path), cached_path
                        )
                        if _artifact_matches_job(cached, job, store):
                            artifacts.append(cached)
                            _emit_progress(
                                progress,
                                _job_progress(
                                    "generation",
                                    index,
                                    total,
                                    "cached",
                                    job,
                                ),
                            )
                            continue
                    try:
                        _emit_progress(
                            progress,
                            _job_progress(
                                "generation", index, total, "start", job
                            ),
                        )
                        response = client.run(
                            PiRequest(
                                provider=job.provider,
                                model=job.model,
                                thinking=job.thinking,
                                prompt=compose_meeting_prompt(
                                    job.prompt, job.transcript
                                ),
                                timeout_seconds=(
                                    config.generation.timeout_seconds
                                ),
                            )
                        )
                        _validate_response_identity(job, response)
                        artifact = _complete_artifact(job, response)
                        store.write_text(
                            Path(artifact.summary_path), response.text
                        )
                    except Exception as error:
                        artifact = _failed_artifact(job, str(error))
                        artifact_path = store.write_json(
                            _artifact_relative_path(job), artifact.to_payload()
                        )
                        artifacts.append(
                            GenerationArtifact.from_payload(
                                artifact.to_payload(), artifact_path
                            )
                        )
                        _emit_progress(
                            progress,
                            _job_progress(
                                "generation",
                                index,
                                total,
                                "failed",
                                job,
                            ),
                        )
                        if fail_fast:
                            raise
                        continue
                    artifact_path = store.write_json(
                        _artifact_relative_path(job), artifact.to_payload()
                    )
                    artifacts.append(
                        GenerationArtifact.from_payload(
                            artifact.to_payload(), artifact_path
                        )
                    )
                    _emit_progress(
                        progress,
                        _job_progress(
                            "generation",
                            index,
                            total,
                            "complete",
                            job,
                            artifact.elapsed_seconds,
                        ),
                    )
    return tuple(artifacts)


def _emit_progress(
    progress: Optional[Callable[[str], None]], message: str
) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        return


def _job_identity(job: GenerationJob) -> str:
    return "model=%s prompt=%s case=%s repetition=%d" % (
        job.model_id,
        job.prompt_id,
        job.case_id,
        job.repetition,
    )


def _job_progress(
    operation: str,
    index: int,
    total: int,
    status: str,
    job: GenerationJob,
    detail: object = "",
) -> str:
    suffix = _job_identity(job)
    if status == "complete":
        suffix = "elapsed=%.2fs %s" % (float(str(detail)), suffix)
    elif status == "failed":
        suffix = "error=see-artifact %s" % suffix
    return "[%s %d/%d] %s %s" % (
        operation,
        index,
        total,
        status,
        suffix,
    )


def _make_job(
    model: ModelSpec,
    prompt: PromptSpec,
    case: CaseSpec,
    prompt_text: str,
    transcript: str,
    prompt_hash: str,
    transcript_hash: str,
    thinking: str,
    repetition: int,
) -> GenerationJob:
    identity = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "operation": "generation",
        "case_id": case.id,
        "split": case.split,
        "prompt_id": prompt.id,
        "prompt_sha256": prompt_hash,
        "transcript_sha256": transcript_hash,
        "model_id": model.id,
        "provider": model.provider,
        "model": model.model,
        "kind": model.kind,
        "thinking": thinking,
        "repetition": repetition,
        "prompt": prompt_text,
        "transcript": transcript,
    }
    return GenerationJob(
        case_id=case.id,
        split=case.split,
        prompt_id=prompt.id,
        prompt_sha256=prompt_hash,
        transcript_sha256=transcript_hash,
        model_id=model.id,
        provider=model.provider,
        model=model.model,
        kind=model.kind,
        thinking=thinking,
        repetition=repetition,
        prompt=prompt_text,
        transcript=transcript,
        cache_key=canonical_json_hash(identity),
    )


def _validate_response_identity(
    job: GenerationJob, response: PiResponse
) -> None:
    if response.provider != job.provider or response.model != job.model:
        raise ValueError(
            "RPC response identity does not match requested provider/model: "
            "%s/%s != %s/%s"
            % (response.provider, response.model, job.provider, job.model)
        )


def _artifact_matches_job(
    artifact: GenerationArtifact, job: GenerationJob, store: RunStore
) -> bool:
    expected = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "operation": "generation",
        "cache_key": job.cache_key,
        "case_id": job.case_id,
        "split": job.split,
        "prompt_id": job.prompt_id,
        "prompt_sha256": job.prompt_sha256,
        "transcript_sha256": job.transcript_sha256,
        "model_id": job.model_id,
        "provider": job.provider,
        "model": job.model,
        "kind": job.kind,
        "thinking": job.thinking,
        "repetition": job.repetition,
    }
    if artifact.status != "complete" or not all(
        getattr(artifact, field) == value for field, value in expected.items()
    ):
        return False
    if not artifact.summary_sha256 or not artifact.summary_path:
        return False
    summary_path = store.run_dir / artifact.summary_path
    try:
        return (
            summary_path.is_file()
            and sha256_text(summary_path.read_text(encoding="utf-8"))
            == artifact.summary_sha256
        )
    except (OSError, UnicodeError):
        return False


def _complete_artifact(
    job: GenerationJob, response: PiResponse
) -> GenerationArtifact:
    return GenerationArtifact(
        schema_version=1,
        operation="generation",
        cache_key=job.cache_key,
        status="complete",
        case_id=job.case_id,
        split=job.split,
        prompt_id=job.prompt_id,
        prompt_sha256=job.prompt_sha256,
        transcript_sha256=job.transcript_sha256,
        model_id=job.model_id,
        provider=job.provider,
        model=job.model,
        kind=job.kind,
        thinking=job.thinking,
        repetition=job.repetition,
        elapsed_seconds=response.elapsed_seconds,
        usage=dict(response.usage),
        session_tokens=dict(response.session_tokens),
        stop_reason=response.stop_reason,
        summary_path=_summary_relative_path(job).as_posix(),
        summary_sha256=sha256_text(response.text),
        stderr=response.stderr,
    )


def _failed_artifact(job: GenerationJob, error: str) -> GenerationArtifact:
    return GenerationArtifact(
        schema_version=1,
        operation="generation",
        cache_key=job.cache_key,
        status="failed",
        case_id=job.case_id,
        split=job.split,
        prompt_id=job.prompt_id,
        prompt_sha256=job.prompt_sha256,
        transcript_sha256=job.transcript_sha256,
        model_id=job.model_id,
        provider=job.provider,
        model=job.model,
        kind=job.kind,
        thinking=job.thinking,
        repetition=job.repetition,
        elapsed_seconds=0.0,
        usage={},
        session_tokens={},
        stop_reason="",
        summary_path=_summary_relative_path(job).as_posix(),
        summary_sha256="",
        stderr=error,
        error=error,
    )


def _artifact_relative_path(job: GenerationJob) -> Path:
    return _generation_dir(job) / ("repetition-%d.json" % job.repetition)


def _summary_relative_path(job: GenerationJob) -> Path:
    return _generation_dir(job) / ("repetition-%d.md" % job.repetition)


def _generation_dir(job: GenerationJob) -> Path:
    return (
        Path("generations")
        / _safe(job.model_id)
        / _safe(job.prompt_id)
        / _safe(job.case_id)
        / job.cache_key
    )


def _safe(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _normalize_filters(filters: Optional[object]) -> GenerationFilters:
    if filters is None:
        return GenerationFilters()
    if isinstance(filters, Mapping):
        return GenerationFilters(
            model_ids=_as_set(filters.get("model_ids")),
            prompt_ids=_as_set(filters.get("prompt_ids")),
            case_ids=_as_set(filters.get("case_ids")),
            splits=_as_set(filters.get("splits")),
        )
    return GenerationFilters(
        model_ids=_as_set(getattr(filters, "model_ids", None)),
        prompt_ids=_as_set(getattr(filters, "prompt_ids", None)),
        case_ids=_as_set(getattr(filters, "case_ids", None)),
        splits=_as_set(getattr(filters, "splits", None)),
    )


def _as_set(value: object) -> Optional[Set[str]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("generation filter values must be iterable")
    return {str(item) for item in value}


def _validate_filters(
    config: BenchmarkConfig, filters: GenerationFilters
) -> None:
    valid = {
        "model_ids": {item.id for item in config.models},
        "prompt_ids": {item.id for item in config.prompts},
        "case_ids": {item.id for item in config.cases},
        "splits": {item.split for item in config.cases},
    }
    for name in ("model_ids", "prompt_ids", "case_ids", "splits"):
        values = getattr(filters, name)
        unknown = set() if values is None else values - valid[name]
        if unknown:
            raise ValueError(
                "unknown %s filter value(s): %s" % (name, sorted(unknown))
            )


def _selected(values, selected: Optional[Set[str]]):
    if selected is None:
        return values
    return tuple(value for value in values if value.id in selected)


def _as_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(str(value))


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(str(value))


def _as_dict(value: object) -> Dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _record_inputs(store: RunStore, prompts, cases) -> None:
    prompt_hashes = _as_dict(store.manifest.get("prompt_sha256", {}))
    transcript_hashes = _as_dict(store.manifest.get("transcript_sha256", {}))
    golden_hashes = _as_dict(store.manifest.get("golden_sha256", {}))
    for prompt in prompts:
        text = prompt.path.read_text(encoding="utf-8")
        prompt_hashes[prompt.id] = sha256_text(text)
        store.store_input("prompts", prompt.id, text)
    for case in cases:
        text = case.transcript.read_text(encoding="utf-8")
        transcript_hashes[case.id] = sha256_text(text)
        store.store_input("transcripts", case.id, text)
        golden = case.golden.read_text(encoding="utf-8")
        golden_hashes[case.id] = sha256_text(golden)
        store.store_input("goldens", case.id, golden)
    store.manifest["prompt_sha256"] = prompt_hashes
    store.manifest["transcript_sha256"] = transcript_hashes
    store.manifest["golden_sha256"] = golden_hashes
    store.write_json(Path("manifest.json"), store.manifest)
