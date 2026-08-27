import json
from pathlib import Path

import pytest

from benchmarks.meeting_summary.reporting import build_report, write_report
from benchmarks.meeting_summary.storage import RunStore

FIELDS = (
    "factual_accuracy",
    "decisions_and_actions",
    "technical_detail_and_blockers",
    "structure_and_compliance",
    "concision_and_usefulness",
)


def _store(tmp_path, dev_only=False):
    config = tmp_path / "benchmark.yaml"
    cases = (
        ""
        if dev_only
        else """  - id: validation-case
    transcript: validation.md
    golden: golden.md
    split: validation
  - id: test-case
    transcript: test.md
    golden: golden.md
    split: test
"""
    )
    config.write_text(
        """version: 1
output_dir: runs
generation:
  repetitions: 3
  thinking: off
  timeout_seconds: 10
prompts:
  - id: current
    path: prompt.md
cases:
  - id: dev-case
    transcript: dev.md
    golden: golden.md
    split: development
"""
        + cases
        + """models:
  - id: local-a
    provider: test
    model: local-a
    kind: candidate
  - id: local-b
    provider: test
    model: local-b
    kind: candidate
  - id: luna-control
    provider: test
    model: luna
    kind: baseline
  - id: other-baseline
    provider: test
    model: other
    kind: baseline
judge:
  provider: test
  model: judge
  thinking: off
  timeout_seconds: 10
  pairwise_top_k: 3
""",
        encoding="utf-8",
    )
    for name in (
        "prompt.md",
        "dev.md",
        "validation.md",
        "test.md",
        "golden.md",
    ):
        (tmp_path / name).write_text(name, encoding="utf-8")
    return RunStore.create(tmp_path / "runs", config)


def _generation(
    store,
    model,
    repetition,
    split="development",
    prompt="current",
    case="dev-case",
    status="complete",
    elapsed=1.0,
    input_tokens=10,
    output_tokens=20,
):
    cache = "%s-%s-%s-%s" % (model, prompt, case, repetition)
    payload = {
        "operation": "generation",
        "status": status,
        "cache_key": cache,
        "case_id": case,
        "split": split,
        "prompt_id": prompt,
        "prompt_sha256": "p",
        "transcript_sha256": "t",
        "model_id": model,
        "provider": "test",
        "model": model,
        "kind": (
            "baseline"
            if model in ("luna-control", "other-baseline")
            else "candidate"
        ),
        "thinking": "off",
        "repetition": repetition,
        "elapsed_seconds": elapsed,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "session_tokens": {},
        "stop_reason": "stop",
        "summary_path": "summaries/%s-%s.md" % (model, repetition),
        "stderr": "",
    }
    store.write_json(
        Path("generations")
        / model
        / prompt
        / case
        / ("repetition-%d.json" % repetition),
        payload,
    )
    if status == "complete":
        store.write_text(Path(payload["summary_path"]), "summary")
    return cache


def _judgment(
    store,
    model,
    repetition,
    cache,
    split="development",
    prompt="current",
    case="dev-case",
    score=4,
    errors=0,
    tags=(),
    status="complete",
):
    payload = {
        "operation": "judgment",
        "status": status,
        "cache_key": "j-%s" % cache,
        "generation_cache_key": cache,
        "case_id": case,
        "split": split,
        "prompt_id": prompt,
        "model_id": model,
        "provider": "test",
        "model": model,
        "kind": "baseline" if model == "luna-control" else "candidate",
        "repetition": repetition,
        "usage": {},
        "elapsed_seconds": 0.1,
    }
    if status == "complete":
        payload["result"] = {
            "scores": dict((field, score) for field in FIELDS),
            "critical_errors": [
                {
                    "claim": "claim",
                    "transcript_evidence": "evidence",
                    "explanation": "explanation",
                }
                for _ in range(errors)
            ],
            "missed_items": [],
            "failure_tags": list(tags),
            "prompt_recommendations": (
                ["Require explicit owners"] if tags else []
            ),
            "verdict": "ok",
        }
    store.write_json(
        Path("judgments")
        / model
        / prompt
        / case
        / ("repetition-%d.json" % repetition),
        payload,
    )


def _pairwise(
    store, a, b, repetition, winner, prompt="current", case="dev-case"
):
    payload = {
        "operation": "pairwise",
        "status": "complete",
        "cache_key": "pw-%s-%s-%d" % (a, b, repetition),
        "model_a_id": a,
        "model_b_id": b,
        "case_id": case,
        "prompt_id": prompt,
        "repetition": repetition,
        "result": {
            "winner": winner,
            "reason": "reason",
            "critical_difference": "difference",
            "confidence": 4,
        },
    }
    store.write_json(
        Path("pairwise")
        / ("%s--%s" % tuple(sorted((a, b))))
        / prompt
        / case
        / ("repetition-%d.json" % repetition),
        payload,
    )


def test_aggregates_scores_failures_latency_tokens_pairwise_and_recommendations(  # noqa: E501
    tmp_path,
):
    store = _store(tmp_path)
    for repetition, score in enumerate((5, 4, 3), 1):
        cache = _generation(
            store,
            "local-a",
            repetition,
            elapsed=float(repetition),
            input_tokens=10 * repetition,
            output_tokens=20 * repetition,
        )
        _judgment(
            store,
            "local-a",
            repetition,
            cache,
            score=score,
            errors=1 if repetition == 1 else 0,
            tags=("missed_action",) if repetition < 3 else (),
        )
    failed_cache = _generation(store, "local-a", 4, status="failed")
    _judgment(store, "local-a", 4, failed_cache, status="failed")
    local_b_cache = _generation(store, "local-b", 1)
    _judgment(store, "local-b", 1, local_b_cache, score=5)
    for repetition in (1, 2, 3):
        cache = _generation(store, "luna-control", repetition)
        _judgment(store, "luna-control", repetition, cache, score=4)
    other_baseline_cache = _generation(store, "other-baseline", 1)
    _judgment(store, "other-baseline", 1, other_baseline_cache, score=5)
    _pairwise(store, "local-a", "luna-control", 1, "A")
    _pairwise(store, "local-a", "luna-control", 2, "tie")
    _pairwise(store, "local-a", "luna-control", 3, "B")

    report = build_report(store)
    row = next(
        row
        for row in report.rows
        if row.model_id == "local-a" and row.split == "development"
    )
    assert row.mean_score == 80.0
    assert row.score_stddev == pytest.approx(16.33)
    assert row.critical_errors == 1
    assert row.completed_runs == 3 and row.failed_runs == 1
    assert row.mean_latency_seconds == 2.0
    assert row.mean_input_tokens == 20.0 and row.mean_output_tokens == 40.0
    assert (row.pairwise_wins, row.pairwise_losses, row.pairwise_ties) == (
        1,
        1,
        1,
    )
    assert row.baseline_delta == 0.0
    assert report.to_dict()["leaderboard"][0]["model_id"] == "local-b"
    assert report.failure_tags["local-a"]["current"]["missed_action"] == 2
    assert len(report.prompt_recommendations) == 1


def test_split_coverage_warning_and_atomic_report_files(tmp_path):
    store = _store(tmp_path)
    cache = _generation(store, "local-a", 1)
    _judgment(store, "local-a", 1, cache, score=5)
    report = build_report(store)
    assert {row.split for row in report.rows} == {
        "development",
        "validation",
        "test",
    }
    assert all(
        row.completed_runs == 0
        for row in report.rows
        if row.split != "development"
    )
    paths = write_report(store, report)
    assert tuple(path.name for path in paths) == (
        "report.json",
        "report.csv",
        "report.md",
    )
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert "raw_artifacts" in data
    assert paths[1].read_text(encoding="utf-8").splitlines()[0] == (
        (
            "split,prompt_id,model_id,kind,completed_runs,failed_runs,"
            "mean_score,score_stddev,critical_errors,mean_factual_accuracy,"
            "mean_decisions_and_actions,mean_technical_detail_and_blockers,"
            "mean_structure_and_compliance,mean_concision_and_usefulness,"
            "mean_latency_seconds,mean_input_tokens,mean_output_tokens,"
            "pairwise_wins,pairwise_losses,pairwise_ties,baseline_delta"
        )
    )
    markdown = paths[2].read_text(encoding="utf-8")
    warning = (
        "Prompt recommendations are development-only. Add at least one "
        "validation case and one held-out test case before promoting a prompt."
    )
    assert warning in markdown
    validation_cache = _generation(
        store, "local-a", 1, split="validation", case="validation-case"
    )
    _judgment(
        store,
        "local-a",
        1,
        validation_cache,
        split="validation",
        case="validation-case",
        score=5,
    )
    test_cache = _generation(
        store, "local-a", 1, split="test", case="test-case"
    )
    _judgment(
        store,
        "local-a",
        1,
        test_cache,
        split="test",
        case="test-case",
        score=5,
    )
    report = build_report(store)
    assert warning not in report.markdown
    assert report.split_warning == ""
    assert not list(store.run_dir.glob(".*tmp-*"))


@pytest.mark.parametrize(
    "partial_split,partial_case,remaining_split,remaining_case",
    [
        ("validation", "validation-case", "test", "test-case"),
        ("test", "test-case", "validation", "validation-case"),
    ],
)
def test_partial_split_coverage_stays_development_only(
    tmp_path, partial_split, partial_case, remaining_split, remaining_case
):
    store = _store(tmp_path)
    warning = (
        "Prompt recommendations are development-only. Add at least one "
        "validation case and one held-out test case before promoting a prompt."
    )

    partial_cache = _generation(
        store, "local-a", 1, split=partial_split, case=partial_case
    )
    _judgment(
        store,
        "local-a",
        1,
        partial_cache,
        split=partial_split,
        case=partial_case,
        score=5,
    )
    report = build_report(store)
    assert warning in report.markdown
    assert report.split_warning == warning

    remaining_cache = _generation(
        store, "local-a", 1, split=remaining_split, case=remaining_case
    )
    _judgment(
        store,
        "local-a",
        1,
        remaining_cache,
        split=remaining_split,
        case=remaining_case,
        score=5,
    )
    report = build_report(store)
    assert warning not in report.markdown
    assert report.split_warning == ""


def test_development_only_recommendations_are_not_promoted(tmp_path):
    store = _store(tmp_path, dev_only=True)
    cache = _generation(store, "local-a", 1)
    _judgment(store, "local-a", 1, cache, tags=("missed_action",))
    report = build_report(store)
    assert "development-only" in report.markdown
