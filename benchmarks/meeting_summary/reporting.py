"""Stable reports and prompt diagnostics for stored meeting benchmark runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .storage import RunStore

CSV_COLUMNS = (
    "split,prompt_id,model_id,kind,completed_runs,failed_runs,mean_score,"
    "score_stddev,critical_errors,mean_factual_accuracy,"
    "mean_decisions_and_actions,mean_technical_detail_and_blockers,"
    "mean_structure_and_compliance,mean_concision_and_usefulness,"
    "mean_latency_seconds,mean_input_tokens,mean_output_tokens,pairwise_wins,"
    "pairwise_losses,pairwise_ties,baseline_delta"
).split(",")
SCORE_FIELDS = (
    "factual_accuracy",
    "decisions_and_actions",
    "technical_detail_and_blockers",
    "structure_and_compliance",
    "concision_and_usefulness",
)


@dataclass(frozen=True)
class ReportRow:
    split: str
    prompt_id: str
    model_id: str
    kind: str
    completed_runs: int
    failed_runs: int
    mean_score: Optional[float]
    score_stddev: Optional[float]
    critical_errors: int
    mean_factual_accuracy: Optional[float]
    mean_decisions_and_actions: Optional[float]
    mean_technical_detail_and_blockers: Optional[float]
    mean_structure_and_compliance: Optional[float]
    mean_concision_and_usefulness: Optional[float]
    mean_latency_seconds: Optional[float]
    mean_input_tokens: Optional[float]
    mean_output_tokens: Optional[float]
    pairwise_wins: int
    pairwise_losses: int
    pairwise_ties: int
    baseline_delta: Optional[float]
    per_run_critical_errors: Tuple[int, ...] = ()
    score_values: Tuple[float, ...] = ()
    failure_tag_counts: Mapping[str, int] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        result = {name: getattr(self, name) for name in CSV_COLUMNS}
        result.update(
            {
                "per_run_critical_errors": list(self.per_run_critical_errors),
                "score_values": list(self.score_values),
                "failure_tag_counts": dict(self.failure_tag_counts or {}),
            }
        )
        return result


@dataclass(frozen=True)
class BenchmarkReport:
    rows: Tuple[ReportRow, ...]
    failures: Tuple[Mapping[str, Any], ...]
    failure_tags: Mapping[str, Mapping[str, Mapping[str, int]]]
    prompt_recommendations: Mapping[str, Tuple[str, ...]]
    pairwise_results: Tuple[Mapping[str, Any], ...]
    raw_artifacts: Mapping[str, Tuple[Mapping[str, Any], ...]]
    split_coverage: Mapping[str, Any]
    markdown: str = ""

    @property
    def split_warning(self) -> str:
        if self.split_coverage.get("development_only"):
            return "Prompt recommendations are development-only. Add at least one validation case and one held-out test case before promoting a prompt."
        missing = self.split_coverage.get("missing_splits", [])
        return "Missing completed splits: %s." % ", ".join(missing) if missing else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "rows": [row.to_dict() for row in self.rows],
            "leaderboard": _leaderboard(self.rows),
            "failures": list(self.failures),
            "failure_tags": {
                model: {prompt: dict(tags) for prompt, tags in prompts.items()}
                for model, prompts in self.failure_tags.items()
            },
            "prompt_recommendations": {
                tag: list(recommendations)
                for tag, recommendations in self.prompt_recommendations.items()
            },
            "pairwise_results": list(self.pairwise_results),
            "split_coverage": dict(self.split_coverage),
            "raw_artifacts": {
                kind: list(artifacts) for kind, artifacts in self.raw_artifacts.items()
            },
        }


def build_report(store: RunStore) -> BenchmarkReport:
    config = _load_config(store)
    raw = _read_artifacts(store)
    generations = _index_generations(raw["generation"])
    judgments = _index_judgments(raw["judgment"])
    configured = _configured_jobs(config, generations, store.manifest)
    valid_jobs: Dict[
        Tuple[str, str, str, int], Tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    failures: List[Mapping[str, Any]] = []

    for key in configured:
        generation = generations.get(key)
        if generation is None:
            failures.append(
                _failure(
                    key,
                    "generation",
                    "missing",
                    "generation artifact is missing",
                    config,
                )
            )
            continue
        if generation.get("status") != "complete":
            failures.append(
                _failure(
                    key,
                    "generation",
                    "failed",
                    str(
                        generation.get("error")
                        or generation.get("stderr")
                        or "generation failed"
                    ),
                    config,
                    generation,
                )
            )
            continue
        judgment = judgments.get(str(generation.get("cache_key")))
        if judgment is None:
            failures.append(
                _failure(
                    key,
                    "judgment",
                    "missing",
                    "judgment artifact is missing",
                    config,
                    generation,
                )
            )
            continue
        if judgment.get("status") != "complete" or not isinstance(
            judgment.get("result"), Mapping
        ):
            failures.append(
                _failure(
                    key,
                    "judgment",
                    "failed",
                    str(judgment.get("error") or "judgment failed"),
                    config,
                    generation,
                )
            )
            continue
        valid_jobs[key] = (generation, judgment)

    # Artifacts from a store without its original config are still reportable.
    for key, generation in generations.items():
        if key in configured:
            continue
        if _snapshot(store.manifest) is not None:
            continue
        if generation.get("status") != "complete":
            failures.append(
                _failure(
                    key,
                    "generation",
                    "failed",
                    str(generation.get("error") or "generation failed"),
                    config,
                    generation,
                )
            )
            continue
        judgment = judgments.get(str(generation.get("cache_key")))
        if judgment is None or judgment.get("status") != "complete":
            failures.append(
                _failure(
                    key,
                    "judgment",
                    "missing" if judgment is None else "failed",
                    "judgment artifact is unavailable",
                    config,
                    generation,
                )
            )
            continue
        if isinstance(judgment.get("result"), Mapping):
            valid_jobs[key] = (generation, judgment)

    pairwise_artifacts = _scoped_pairwise(
        raw["pairwise"], generations, store.manifest
    )
    pairwise = _complete_pairwise(pairwise_artifacts)
    failures.extend(_pairwise_failures(pairwise_artifacts, generations))
    failures.sort(
        key=lambda item: (
            str(item.get("split", "")),
            str(item.get("prompt_id", "")),
            str(item.get("model_id", "")),
            str(item.get("case_id", "")),
            int(item.get("repetition", 0)),
            str(item.get("stage", "")),
        )
    )
    keys = _row_keys(config, generations, valid_jobs, store.manifest)
    rows = _make_rows(keys, valid_jobs, failures, pairwise, config, store.manifest)
    tag_counts = _failure_tag_counts(valid_jobs)
    recommendations = _recommendations(valid_jobs)
    split_coverage = _split_coverage(config, rows, store.manifest)
    report = BenchmarkReport(
        rows=tuple(rows),
        failures=tuple(failures),
        failure_tags=tag_counts,
        prompt_recommendations=recommendations,
        pairwise_results=tuple(pairwise),
        raw_artifacts=raw,
        split_coverage=split_coverage,
    )
    return replace(report, markdown=_render_markdown(report))


def write_report(store: RunStore, report: BenchmarkReport) -> Tuple[Path, Path, Path]:
    """Write report.json, report.csv, and report.md through RunStore's atomic writer."""
    json_path = store.write_text(
        Path("report.json"),
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    csv_path = store.write_text(Path("report.csv"), _render_csv(report.rows))
    markdown_path = store.write_text(Path("report.md"), report.markdown)
    return json_path, csv_path, markdown_path


def _load_config(store: RunStore):
    config_path = store.manifest.get("config_path")
    if not config_path:
        return None
    try:
        from .config import load_benchmark_config

        return load_benchmark_config(Path(str(config_path)))
    except (ImportError, OSError, ValueError, TypeError):
        return _minimal_config(Path(str(config_path)))


def _minimal_config(path):
    """Read the small metadata subset needed for reporting without PyYAML."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    from types import SimpleNamespace

    sections = {"prompts": [], "cases": [], "models": []}
    section = ""
    current = None
    repetitions = 1
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "prompts:":
            section, current = "prompts", None
        elif stripped == "cases:":
            section, current = "cases", None
        elif stripped == "models:":
            section, current = "models", None
        elif stripped.startswith("repetitions:"):
            try:
                repetitions = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                repetitions = 1
        elif section in sections and stripped.startswith("- id:"):
            current = {"id": stripped.split(":", 1)[1].strip()}
            sections[section].append(current)
        elif (
            section in sections
            and current is not None
            and ":" in stripped
            and not stripped.startswith("-")
        ):
            name, value = stripped.split(":", 1)
            current[name.strip()] = value.strip().strip("\\\"'")
    prompts = tuple(SimpleNamespace(id=item["id"]) for item in sections["prompts"])
    cases = tuple(
        SimpleNamespace(id=item["id"], split=item.get("split", "development"))
        for item in sections["cases"]
    )
    models = tuple(
        SimpleNamespace(id=item["id"], kind=item.get("kind", "candidate"))
        for item in sections["models"]
    )
    return SimpleNamespace(
        prompts=prompts,
        cases=cases,
        models=models,
        generation=SimpleNamespace(repetitions=repetitions),
    )


def _read_artifacts(store: RunStore) -> Dict[str, Tuple[Mapping[str, Any], ...]]:
    result: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
    for kind, directory in (
        ("generation", "generations"),
        ("judgment", "judgments"),
        ("pairwise", "pairwise"),
    ):
        values: List[Mapping[str, Any]] = []
        root = store.run_dir / directory
        for path in sorted(root.rglob("*.json")) if root.is_dir() else ():
            try:
                payload = store.read_json(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            item = dict(payload)
            item["artifact_path"] = str(path.relative_to(store.run_dir))
            values.append(item)
        result[kind] = tuple(values)
    return result


def _index_generations(
    artifacts: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, int], Mapping[str, Any]]:
    return {
        (
            str(item.get("model_id", "")),
            str(item.get("prompt_id", "")),
            str(item.get("case_id", "")),
            int(item.get("repetition", 0)),
        ): item
        for item in artifacts
        if item.get("model_id") is not None
    }


def _index_judgments(
    artifacts: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(item.get("generation_cache_key")): item
        for item in artifacts
        if item.get("generation_cache_key") is not None
    }


def _snapshot(manifest):
    if not isinstance(manifest, Mapping):
        return None
    value = manifest.get("config_snapshot")
    return value if isinstance(value, Mapping) else None


def _configured_jobs(config, generations, manifest=None):
    snapshot = _snapshot(manifest)
    if snapshot is not None:
        models = [item["id"] for item in snapshot.get("models", [])]
        prompts = [item["id"] for item in snapshot.get("prompts", [])]
        cases = [item["id"] for item in snapshot.get("cases", [])]
        scope = manifest.get("scope", {}) if isinstance(manifest, Mapping) else {}
        if isinstance(scope, Mapping):
            models = [item for item in models if item in scope.get("model_ids", models)]
            prompts = [item for item in prompts if item in scope.get("prompt_ids", prompts)]
            cases = [item for item in cases if item in scope.get("case_ids", cases)]
            splits = set(scope.get("splits", []))
        else:
            splits = set()
        case_splits = {
            item["id"]: item.get("split", "")
            for item in snapshot.get("cases", [])
        }
        cases = [item for item in cases if not splits or case_splits.get(item) in splits]
        repetitions = int(snapshot.get("generation", {}).get("repetitions", 1))
        return {
            (model, prompt, case, repetition)
            for model in models
            for prompt in prompts
            for case in cases
            for repetition in range(1, repetitions + 1)
        }
    if config is None:
        return set(generations)
    return {
        (model.id, prompt.id, case.id, repetition)
        for model in config.models
        for prompt in config.prompts
        for case in config.cases
        for repetition in range(1, config.generation.repetitions + 1)
    }


def _row_keys(config, generations, valid_jobs, manifest=None):
    keys = set()
    for key, values in valid_jobs.items():
        keys.add((key[1], str(values[0].get("split", "")), key[0]))
    snapshot = _snapshot(manifest)
    if snapshot is not None:
        scope = manifest.get("scope", {}) if isinstance(manifest, Mapping) else {}
        model_ids = [item["id"] for item in snapshot.get("models", [])]
        prompt_ids = [item["id"] for item in snapshot.get("prompts", [])]
        cases = snapshot.get("cases", [])
        if isinstance(scope, Mapping):
            model_ids = [item for item in model_ids if item in scope.get("model_ids", model_ids)]
            prompt_ids = [item for item in prompt_ids if item in scope.get("prompt_ids", prompt_ids)]
            cases = [item for item in cases if item.get("id") in scope.get("case_ids", [item.get("id") for item in cases])]
            splits = set(scope.get("splits", []))
            cases = [item for item in cases if not splits or item.get("split") in splits]
        keys.update(
            (prompt, case.get("split", ""), model)
            for model in model_ids
            for prompt in prompt_ids
            for case in cases
        )
    elif config is not None:
        keys.update(
            (prompt.id, case.split, model.id)
            for model in config.models
            for prompt in config.prompts
            for case in config.cases
        )
    else:
        keys.update(
            (key[1], str(generations[key].get("split", "")), key[0])
            for key in generations
        )
    return sorted(keys, key=lambda item: (item[1], item[0], item[2]))


def _make_rows(keys, valid_jobs, failures, pairwise, config, manifest=None):
    snapshot = _snapshot(manifest)
    kinds = (
        {item["id"]: item.get("kind", "candidate") for item in snapshot.get("models", [])}
        if snapshot is not None
        else ({model.id: model.kind for model in config.models} if config is not None else {})
    )
    baseline_scores: Dict[Tuple[str, str], float] = {}
    grouped: Dict[
        Tuple[str, str, str], List[Tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = {}
    for key, values in valid_jobs.items():
        generation, judgment = values
        grouped.setdefault(
            (key[1], str(generation.get("split", "")), key[0]), []
        ).append(values)
    for row_key, values in grouped.items():
        if row_key[2] == "luna-control":
            scores = [_score(value[1]) for value in values]
            baseline_scores[(row_key[0], row_key[1])] = fmean(scores)

    rows = []
    for prompt_id, split, model_id in keys:
        values = grouped.get((prompt_id, split, model_id), [])
        scores = [_score(judgment) for _, judgment in values]
        means = {
            field: _mean([_dimension(judgment, field) for _, judgment in values])
            for field in SCORE_FIELDS
        }
        run_errors = tuple(
            len(_result(judgment).get("critical_errors", [])) for _, judgment in values
        )
        gen_by_key = {
            (
                str(g.get("model_id")),
                str(g.get("prompt_id")),
                str(g.get("case_id")),
                int(g.get("repetition", 0)),
            ): g
            for g, _ in values
        }
        wins, losses, ties = _pairwise_counts(
            model_id, prompt_id, split, gen_by_key, pairwise
        )
        failed = sum(
            1
            for item in failures
            if item.get("model_id") == model_id
            and item.get("prompt_id") == prompt_id
            and item.get("split") == split
        )
        baseline = baseline_scores.get((prompt_id, split))
        row = ReportRow(
            split=split,
            prompt_id=prompt_id,
            model_id=model_id,
            kind=kinds.get(model_id, _kind_from_values(values)),
            completed_runs=len(values),
            failed_runs=failed,
            mean_score=_mean(scores),
            score_stddev=_stddev(scores),
            critical_errors=sum(run_errors),
            mean_factual_accuracy=means["factual_accuracy"],
            mean_decisions_and_actions=means["decisions_and_actions"],
            mean_technical_detail_and_blockers=means["technical_detail_and_blockers"],
            mean_structure_and_compliance=means["structure_and_compliance"],
            mean_concision_and_usefulness=means["concision_and_usefulness"],
            mean_latency_seconds=_mean(
                [float(g.get("elapsed_seconds", 0.0)) for g, _ in values]
            ),
            mean_input_tokens=_mean([_tokens(g, "input") for g, _ in values]),
            mean_output_tokens=_mean([_tokens(g, "output") for g, _ in values]),
            pairwise_wins=wins,
            pairwise_losses=losses,
            pairwise_ties=ties,
            baseline_delta=(
                0.0
                if row_key_is_baseline(model_id, kinds, values)
                else (
                    _round(fmean(scores) - baseline)
                    if scores and baseline is not None
                    else None
                )
            ),
            per_run_critical_errors=run_errors,
            score_values=tuple(scores),
            failure_tag_counts=_row_tags(values),
        )
        rows.append(row)
    return rows


def row_key_is_baseline(model_id, kinds, values):
    return (
        kinds.get(model_id) == "baseline"
        or (not kinds and any(str(g.get("kind")) == "baseline" for g, _ in values))
        or model_id == "luna-control"
    )


def _pairwise_counts(model_id, prompt_id, split, jobs, pairwise):
    counts = [0, 0, 0]
    for item in pairwise:
        if str(item.get("prompt_id", "")) != prompt_id:
            continue
        case = str(item.get("case_id", ""))
        rep = int(item.get("repetition", 0))
        if not any(
            str(g.get("case_id")) == case and int(g.get("repetition", 0)) == rep
            for g in jobs.values()
        ):
            continue
        a, b = str(item.get("model_a_id", "")), str(item.get("model_b_id", ""))
        if model_id not in (a, b):
            continue
        winner = item.get("winner_model_id")
        result = item.get("result")
        if winner is None and isinstance(result, Mapping):
            label = result.get("winner")
            winner = a if label == "A" else b if label == "B" else None
        if winner == model_id:
            counts[0] += 1
        elif winner is None:
            counts[2] += 1
        else:
            counts[1] += 1
    return tuple(counts)


def _scoped_pairwise(artifacts, generations, manifest):
    if _snapshot(manifest) is None:
        return artifacts
    scope = manifest.get("scope", {})
    if not isinstance(scope, Mapping):
        return ()
    model_ids = set(str(item) for item in scope.get("model_ids", []))
    prompt_ids = set(str(item) for item in scope.get("prompt_ids", []))
    case_ids = set(str(item) for item in scope.get("case_ids", []))
    splits = set(str(item) for item in scope.get("splits", []))
    selected = []
    for item in artifacts:
        if (
            str(item.get("prompt_id", "")) not in prompt_ids
            or str(item.get("case_id", "")) not in case_ids
        ):
            continue
        if not {str(item.get("model_a_id", "")), str(item.get("model_b_id", ""))}.issubset(model_ids):
            continue
        split = next(
            (
                str(generation.get("split", ""))
                for generation in generations.values()
                if str(generation.get("case_id", "")) == str(item.get("case_id", ""))
                and str(generation.get("prompt_id", "")) == str(item.get("prompt_id", ""))
                and int(generation.get("repetition", 0)) == int(item.get("repetition", 0))
            ),
            "",
        )
        if splits and split not in splits:
            continue
        selected.append(item)
    return tuple(selected)


def _complete_pairwise(artifacts):
    return tuple(
        item
        for item in artifacts
        if item.get("status") == "complete" and isinstance(item.get("result"), Mapping)
    )


def _pairwise_failures(artifacts, generations):
    failures = []
    for item in artifacts:
        if item.get("status") == "complete" and isinstance(item.get("result"), Mapping):
            continue
        case_id = str(item.get("case_id", ""))
        prompt_id = str(item.get("prompt_id", ""))
        repetition = int(item.get("repetition", 0))
        split = next(
            (
                str(generation.get("split", ""))
                for generation in generations.values()
                if str(generation.get("case_id", "")) == case_id
                and str(generation.get("prompt_id", "")) == prompt_id
                and int(generation.get("repetition", 0)) == repetition
            ),
            "",
        )
        failures.append(
            {
                "model_id": "%s vs %s"
                % (item.get("model_a_id", ""), item.get("model_b_id", "")),
                "prompt_id": prompt_id,
                "case_id": case_id,
                "split": split,
                "repetition": repetition,
                "stage": "pairwise",
                "status": str(item.get("status", "failed")),
                "message": str(item.get("error") or "pairwise result unavailable"),
            }
        )
    return failures


def _leaderboard(rows):
    scores = {}
    for row in rows:
        if row.kind != "candidate":
            continue
        scores.setdefault((row.split, row.prompt_id, row.model_id), []).extend(
            row.score_values
        )
    populated = {key: values for key, values in scores.items() if values}
    return [
        {
            "split": split,
            "prompt_id": prompt,
            "model_id": model,
            "mean_score": _round(fmean(values)),
        }
        for (split, prompt, model), values in sorted(
            populated.items(), key=lambda item: (-fmean(item[1]), item[0])
        )
    ]


def _failure_tag_counts(valid_jobs):
    result: Dict[str, Dict[str, Dict[str, int]]] = {}
    for (model, prompt, _case, _rep), (generation, judgment) in valid_jobs.items():
        if str(generation.get("split", "")) == "test":
            continue
        tags = _result(judgment).get("failure_tags", [])
        model_result = result.setdefault(model, {})
        prompt_result = model_result.setdefault(prompt, {})
        for tag in tags if isinstance(tags, list) else []:
            prompt_result[str(tag)] = prompt_result.get(str(tag), 0) + 1
    return result


def _recommendations(valid_jobs):
    result: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    for _key, (generation, judgment) in valid_jobs.items():
        if str(generation.get("split", "")) == "test":
            continue
        data = _result(judgment)
        tags = data.get("failure_tags", [])
        recs = data.get("prompt_recommendations", [])
        if not isinstance(tags, list) or not isinstance(recs, list):
            continue
        for tag in tags:
            bucket = result.setdefault(str(tag), [])
            bucket_seen = seen.setdefault(str(tag), set())
            for recommendation in recs:
                if (
                    isinstance(recommendation, str)
                    and recommendation not in bucket_seen
                ):
                    bucket.append(recommendation)
                    bucket_seen.add(recommendation)
    return {tag: tuple(values) for tag, values in sorted(result.items())}


def _split_coverage(config, rows, manifest=None):
    snapshot = _snapshot(manifest)
    if snapshot is not None:
        cases = snapshot.get("cases", [])
        scope = manifest.get("scope", {}) if isinstance(manifest, Mapping) else {}
        case_ids = set(scope.get("case_ids", [])) if isinstance(scope, Mapping) else set()
        splits = set(scope.get("splits", [])) if isinstance(scope, Mapping) else set()
        configured = {
            str(item.get("split", ""))
            for item in cases
            if (not case_ids or item.get("id") in case_ids)
            and (not splits or item.get("split") in splits)
        }
    else:
        configured = (
            {case.split for case in config.cases}
            if config is not None
            else {row.split for row in rows}
        )
    completed = {row.split for row in rows if row.completed_runs}
    required_splits = {"validation", "test"}.intersection(configured)
    missing = sorted(required_splits - completed)
    development_only = configured == {"development"} or not required_splits.issubset(
        completed
    )
    return {
        "configured_splits": sorted(configured),
        "completed_splits": sorted(completed),
        "missing_splits": missing,
        "development_only": development_only,
    }


def _render_csv(rows):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(getattr(row, name)) for name in CSV_COLUMNS})
    return output.getvalue()


def _render_markdown(report):
    lines = ["# Meeting Summary Benchmark Report", "", "## Local model leaderboard", ""]
    local = [
        row for row in report.rows if row.kind == "candidate" and row.completed_runs
    ]
    lines.append(
        "| Split | Prompt | Model | Mean score | Stddev | Completed | Failed |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in sorted(
        local, key=lambda item: (-(item.mean_score or 0), item.model_id, item.split)
    ):
        lines.append(
            "| %s | %s | %s | %s | %s | %d | %d |"
            % (
                row.split,
                row.prompt_id,
                row.model_id,
                _display(row.mean_score),
                _display(row.score_stddev),
                row.completed_runs,
                row.failed_runs,
            )
        )
    if not local:
        lines.append("No completed local model runs.")
    lines += ["", "## Luna baseline comparison", ""]
    for row in report.rows:
        if row.kind == "candidate":
            lines.append(
                "- `%s` (%s/%s): baseline delta %s."
                % (row.model_id, row.prompt_id, row.split, _display(row.baseline_delta))
            )
    lines += ["", "## Critical factual failures by model", ""]
    for row in report.rows:
        if row.critical_errors:
            lines.append(
                "- `%s` (%s/%s): %d critical errors."
                % (row.model_id, row.prompt_id, row.split, row.critical_errors)
            )
    lines += ["", "## Consistency across repetitions", ""]
    for row in report.rows:
        if row.completed_runs:
            lines.append(
                "- `%s` (%s/%s): score standard deviation %s."
                % (row.model_id, row.prompt_id, row.split, _display(row.score_stddev))
            )
    lines += ["", "## Runtime/token trade-offs", ""]
    for row in report.rows:
        if row.completed_runs:
            lines.append(
                "- `%s` (%s/%s): %ss, %s input / %s output tokens."
                % (
                    row.model_id,
                    row.prompt_id,
                    row.split,
                    _display(row.mean_latency_seconds),
                    _display(row.mean_input_tokens),
                    _display(row.mean_output_tokens),
                )
            )
    lines += ["", "## Pairwise results", ""]
    for row in report.rows:
        if row.pairwise_wins or row.pairwise_losses or row.pairwise_ties:
            lines.append(
                "- `%s` (%s/%s): %d wins, %d losses, %d ties."
                % (
                    row.model_id,
                    row.prompt_id,
                    row.split,
                    row.pairwise_wins,
                    row.pairwise_losses,
                    row.pairwise_ties,
                )
            )
    lines += ["", "## Prompt failure categories", ""]
    for model, prompts in sorted(report.failure_tags.items()):
        for prompt, tags in sorted(prompts.items()):
            lines.append(
                "- `%s` / `%s`: %s"
                % (
                    model,
                    prompt,
                    ", ".join(
                        "%s (%d)" % (tag, count) for tag, count in sorted(tags.items())
                    ),
                )
            )
    lines += ["", "## Deduplicated prompt-edit recommendations", ""]
    for tag, recommendations in report.prompt_recommendations.items():
        lines.append("- `%s`: %s" % (tag, "; ".join(recommendations)))
    if not report.prompt_recommendations:
        lines.append("No prompt recommendations.")
    if report.split_coverage.get("development_only"):
        lines += [
            "",
            "Prompt recommendations are development-only. Add at least one validation case and one held-out test case before promoting a prompt.",
        ]
    lines += ["", "## Split coverage warning", ""]
    missing = report.split_coverage.get("missing_splits", [])
    lines.append(
        "Missing completed splits: %s." % (", ".join(missing) if missing else "none")
    )
    lines += ["", "## Failed or missing jobs", ""]
    if report.failures:
        for failure in report.failures:
            lines.append(
                "- `%s`/%s/%s repetition %s: %s (%s)."
                % (
                    failure.get("model_id"),
                    failure.get("prompt_id"),
                    failure.get("split"),
                    failure.get("repetition"),
                    failure.get("stage"),
                    failure.get("message"),
                )
            )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _failure(key, stage, status, message, config=None, generation=None):
    model, prompt, case, repetition = key
    split = str(generation.get("split", "")) if generation is not None else ""
    if not split and config is not None:
        split = next((item.split for item in config.cases if item.id == case), "")
    return {
        "model_id": model,
        "prompt_id": prompt,
        "case_id": case,
        "split": split,
        "repetition": repetition,
        "stage": stage,
        "status": status,
        "message": message,
    }


def _result(judgment):
    result = judgment.get("result")
    return result if isinstance(result, Mapping) else {}


def _score(judgment):
    scores = _result(judgment).get("scores", {})
    values = [float(scores.get(field, 0)) for field in SCORE_FIELDS]
    return round(
        sum(
            value * weight
            for value, weight in zip(values, (0.35, 0.25, 0.20, 0.10, 0.10))
        )
        * 20,
        2,
    )


def _dimension(judgment, field):
    return float(_result(judgment).get("scores", {}).get(field, 0))


def _mean(values):
    return _round(fmean(values)) if values else None


def _stddev(values):
    return _round(pstdev(values)) if values else None


def _round(value):
    return round(value, 2)


def _tokens(generation, direction):
    usage = generation.get("usage", {})
    session = generation.get("session_tokens", {})
    names = (
        direction + "_tokens",
        "prompt_tokens" if direction == "input" else "completion_tokens",
        direction,
    )
    for source in (usage, session):
        if isinstance(source, Mapping):
            for name in names:
                value = source.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
    return 0.0


def _row_tags(values):
    tags: Dict[str, int] = {}
    for _, judgment in values:
        failure_tags = _result(judgment).get("failure_tags", [])
        for tag in failure_tags if isinstance(failure_tags, list) else []:
            tags[str(tag)] = tags.get(str(tag), 0) + 1
    return tags


def _kind_from_values(values):
    return str(values[0][0].get("kind", "candidate")) if values else "candidate"


def _csv_value(value):
    return "" if value is None else value


def _display(value):
    return "n/a" if value is None else str(value)
