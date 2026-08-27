import json
from pathlib import Path

import pytest

import benchmarks.meeting_summary.judging as judging_module
from benchmarks.meeting_summary.generation import (
    GenerationFilters,
    generate_candidates,
)
from benchmarks.meeting_summary.judging import (
    VALID_SCORE_FIELDS,
    VALID_FAILURE_TAGS,
    choose_pairwise_order,
    judge_generations,
    judge_pairwise_top_models,
    parse_judge_result,
)
from benchmarks.meeting_summary.pi_rpc import PiResponse
from benchmarks.meeting_summary.storage import RunStore
from benchmarks.meeting_summary.types import (
    BenchmarkConfig,
    CaseSpec,
    GenerationSpec,
    JudgeSpec,
    ModelSpec,
    PromptSpec,
)

VALID_JUDGMENT = {
    "scores": {
        "factual_accuracy": 5,
        "decisions_and_actions": 4,
        "technical_detail_and_blockers": 4,
        "structure_and_compliance": 5,
        "concision_and_usefulness": 4,
    },
    "critical_errors": [],
    "missed_items": ["S3 partitioning remained unresolved"],
    "failure_tags": ["missed_blocker"],
    "prompt_recommendations": [
        "Require unresolved architecture choices to be listed under blockers"
    ],
    "verdict": "Strong and factually reliable, with one missed blocker",
}


class StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return PiResponse(
            text=value,
            provider=request.provider,
            model=request.model,
            stop_reason="stop",
            usage={"input": 10, "output": 20},
            session_tokens={"total": 30},
            elapsed_seconds=0.2,
            stderr="",
        )


def make_config(tmp_path: Path, model_count=3):
    prompt = tmp_path / "prompt.md"
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    prompt.write_text("meeting prompt", encoding="utf-8")
    transcript.write_text("authoritative transcript", encoding="utf-8")
    golden.write_text("golden summary", encoding="utf-8")
    models = tuple(
        ModelSpec("model-%d" % i, "local", "model-%d" % i, "candidate")
        for i in range(model_count)
    ) + (
        ModelSpec("luna-control", "openai-codex", "luna", "baseline"),
    )
    return BenchmarkConfig(
        source=tmp_path / "benchmark.yaml",
        output_dir=tmp_path / "runs",
        generation=GenerationSpec(2, "off", 10),
        prompts=(PromptSpec("current", prompt),),
        cases=(CaseSpec("case", transcript, golden, "development"),),
        models=models,
        judge=JudgeSpec("openai-codex", "gpt-5.6-sol", "high", 10, 2),
    )


def test_valid_result_is_strict_and_weighted_in_python():
    result = parse_judge_result(json.dumps(dict(VALID_JUDGMENT)))
    assert result.scores.weighted_total() == 89.0
    assert set(VALID_SCORE_FIELDS) == set(result.scores.__dataclass_fields__)


@pytest.mark.parametrize("field", list(VALID_SCORE_FIELDS))
def test_scores_must_be_integer_one_through_five(field):
    invalid = json.loads(json.dumps(VALID_JUDGMENT))
    invalid["scores"][field] = 5.0
    with pytest.raises(ValueError):
        parse_judge_result(json.dumps(invalid))


def test_unknown_and_missing_fields_are_rejected():
    unknown = json.loads(json.dumps(VALID_JUDGMENT))
    unknown["scores"]["extra"] = 4
    with pytest.raises(ValueError):
        parse_judge_result(json.dumps(unknown))
    missing = json.loads(json.dumps(VALID_JUDGMENT))
    del missing["scores"]["factual_accuracy"]
    with pytest.raises(ValueError):
        parse_judge_result(json.dumps(missing))


def test_unknown_failure_tags_are_rejected():
    invalid = json.loads(json.dumps(VALID_JUDGMENT))
    invalid["failure_tags"] = ["not-allowed"]
    with pytest.raises(ValueError):
        parse_judge_result(json.dumps(invalid))
    assert "missed_blocker" in VALID_FAILURE_TAGS


def test_fence_is_stripped_and_critical_errors_are_separate():
    judgment = json.loads(json.dumps(VALID_JUDGMENT))
    judgment["critical_errors"] = [
        {
            "claim": "The owner is Ada",
            "transcript_evidence": "Transcript assigns it to Lin",
            "explanation": "The candidate changed the owner",
        }
    ]
    result = parse_judge_result("```json\n%s\n```" % json.dumps(judgment))
    assert result.critical_errors[0].claim == "The owner is Ada"
    assert result.scores.weighted_total() == 89.0


def test_prompt_is_anonymous_and_authoritative(tmp_path: Path):
    config = make_config(tmp_path, model_count=1)
    store = RunStore.create(config.output_dir, config.source)
    generation = generate_candidates(
        config, store, None, StubClient(["summary", "summary"])
    )
    client = StubClient([json.dumps(VALID_JUDGMENT)] * len(generation))
    judge_generations(config, store, client)
    request = client.requests[0]
    assert "authoritative transcript" in request.prompt
    assert "golden summary" in request.prompt
    assert "summary" in request.prompt
    assert "model-0" not in request.prompt
    assert "local" not in request.prompt
    assert "The transcript is authoritative" in request.prompt
    assert "Decisions must be explicit outcomes" in request.prompt
    assert "Omit empty, generic, or fluff sections" in request.prompt


def test_invalid_json_gets_one_retry_and_failure_is_cached(tmp_path: Path):
    config = make_config(tmp_path, model_count=1)
    config = BenchmarkConfig(
        source=config.source,
        output_dir=config.output_dir,
        generation=GenerationSpec(1, "off", 10),
        prompts=config.prompts,
        cases=config.cases,
        models=config.models,
        judge=config.judge,
    )
    store = RunStore.create(config.output_dir, config.source)
    artifacts = generate_candidates(
        config,
        store,
        GenerationFilters(model_ids={"model-0"}),
        StubClient(["summary", "summary"]),
    )
    client = StubClient(["not json", "still not json"])
    assert judge_generations(config, store, client) == ()
    assert len(client.requests) == 2
    files = list((store.run_dir / "judgments").rglob("*.json"))
    assert len(files) == 1
    failure = store.read_json(
        [f for f in files if "repetition-1" in str(f)][0]
    )
    assert failure["status"] == "failed"
    assert len(failure["raw_attempts"]) == 2
    assert artifacts


def test_pairwise_placement_is_stable_and_balanced():
    placements = [
        choose_pairwise_order("a", "b", "case", "prompt", repetition)
        for repetition in range(40)
    ]
    assert placements == [
        choose_pairwise_order("a", "b", "case", "prompt", repetition)
        for repetition in range(40)
    ]
    assert set(placements) == {("a", "b"), ("b", "a")}
    counts = {
        placement: placements.count(placement) for placement in set(placements)
    }
    assert abs(counts[("a", "b")] - counts[("b", "a")]) <= 1


def test_judgment_cache_invalidates_when_judge_settings_change(tmp_path: Path):
    config = make_config(tmp_path, model_count=1)
    store = RunStore.create(config.output_dir, config.source)
    generate_candidates(
        config,
        store,
        GenerationFilters(model_ids={"model-0"}),
        StubClient(["summary", "summary"]),
    )
    first_client = StubClient([json.dumps(VALID_JUDGMENT)] * 2)
    judge_generations(config, store, first_client)
    changed = BenchmarkConfig(
        source=config.source,
        output_dir=config.output_dir,
        generation=config.generation,
        prompts=config.prompts,
        cases=config.cases,
        models=config.models,
        judge=JudgeSpec("other-judge", "other-model", "low", 10, 2),
    )
    second_client = StubClient([json.dumps(VALID_JUDGMENT)] * 2)
    judge_generations(changed, store, second_client)
    assert len(second_client.requests) == 2


def _pairwise_fixture(tmp_path):
    config = make_config(tmp_path, model_count=2)
    store = RunStore.create(config.output_dir, config.source)
    generation = generate_candidates(
        config, store, None, StubClient(["summary"] * 6)
    )
    judge_generations(
        config,
        store,
        StubClient([json.dumps(VALID_JUDGMENT)] * len(generation)),
    )
    response = json.dumps(
        {
            "winner": "A",
            "reason": "A is clearer",
            "critical_difference": "none",
            "confidence": 4,
        }
    )
    return config, store, response


def test_pairwise_cache_invalidates_on_judge_settings_change(tmp_path: Path):
    config, store, pairwise_response = _pairwise_fixture(tmp_path)
    judge_pairwise_top_models(
        config, store, StubClient([pairwise_response] * 4)
    )
    changed = BenchmarkConfig(
        source=config.source,
        output_dir=config.output_dir,
        generation=config.generation,
        prompts=config.prompts,
        cases=config.cases,
        models=config.models,
        judge=JudgeSpec("other-judge", "other-model", "low", 10, 2),
    )
    client = StubClient([pairwise_response] * 4)
    judge_pairwise_top_models(changed, store, client)
    assert len(client.requests) == 4


def test_pairwise_cache_invalidates_on_prompt_version_only(
    tmp_path: Path, monkeypatch
):
    config, store, pairwise_response = _pairwise_fixture(tmp_path)
    judge_pairwise_top_models(
        config, store, StubClient([pairwise_response] * 4)
    )
    monkeypatch.setattr(
        judging_module, "PAIRWISE_PROMPT_VERSION", "pairwise-v2"
    )
    client = StubClient([pairwise_response] * 4)
    judge_pairwise_top_models(config, store, client)
    assert len(client.requests) == 4

    original_template = judging_module._prompt_template
    monkeypatch.setattr(
        judging_module,
        "_prompt_template",
        lambda name: (
            original_template(name) + "\nversion change"
            if name == "pairwise-v1.md"
            else original_template(name)
        ),
    )
    client = StubClient([pairwise_response] * 4)
    judge_pairwise_top_models(config, store, client)
    assert len(client.requests) == 4


def test_pairwise_judge_has_no_identity_and_normalizes_winner(tmp_path: Path):
    config = make_config(tmp_path, model_count=2)
    store = RunStore.create(config.output_dir, config.source)
    generation = generate_candidates(
        config, store, None, StubClient(["summary"] * 6)
    )
    absolute = StubClient([json.dumps(VALID_JUDGMENT)] * len(generation))
    judge_generations(config, store, absolute)
    pairwise = StubClient(
        [
            json.dumps(
                {
                    "winner": "A",
                    "reason": "A is clearer",
                    "critical_difference": "none",
                    "confidence": 4,
                }
            )
        ]
        * 6
    )
    results = judge_pairwise_top_models(config, store, pairwise)
    assert results
    assert all(
        result.winner_model_id in {"model-0", "model-1", "luna-control"}
        for result in results
    )
    assert "model-0" not in pairwise.requests[0].prompt
    assert "local" not in pairwise.requests[0].prompt
