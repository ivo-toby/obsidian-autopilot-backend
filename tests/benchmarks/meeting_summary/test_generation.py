import json
from pathlib import Path
import pytest

from benchmarks.meeting_summary.generation import (
    GenerationFilters,
    compose_meeting_prompt,
    generate_candidates,
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


class StubPiRpcClient:
    def __init__(self, fail_models=None):
        self.requests = []
        self.fail_models = set(fail_models or ())

    def run(self, request):
        self.requests.append(request)
        if request.model in self.fail_models:
            raise RuntimeError("stub failure")
        return PiResponse(
            text="# Summary\nGenerated",
            provider=request.provider,
            model=request.model,
            stop_reason="stop",
            usage={"input": 3, "output": 2},
            session_tokens={"total": 5},
            elapsed_seconds=1.23,
            stderr="",
        )


def make_config(
    tmp_path: Path, prompt_text="PROMPT", transcript_text="TRANSCRIPT"
):
    prompt = tmp_path / "prompt.md"
    variant = tmp_path / "variant.md"
    transcript = tmp_path / "transcript.md"
    other_transcript = tmp_path / "other-transcript.md"
    golden = tmp_path / "golden.md"
    prompt.write_text(prompt_text, encoding="utf-8")
    variant.write_text("VARIANT", encoding="utf-8")
    transcript.write_text(transcript_text, encoding="utf-8")
    other_transcript.write_text("OTHER TRANSCRIPT", encoding="utf-8")
    golden.write_text("golden", encoding="utf-8")
    return BenchmarkConfig(
        source=tmp_path / "benchmark.yaml",
        output_dir=tmp_path / "runs",
        generation=GenerationSpec(
            repetitions=3, thinking="off", timeout_seconds=10
        ),
        prompts=(
            PromptSpec("current", prompt),
            PromptSpec("variant", variant),
        ),
        cases=(
            CaseSpec("sync", transcript, golden, "development"),
            CaseSpec("other", other_transcript, golden, "validation"),
        ),
        models=(
            ModelSpec("candidate", "homelab", "candidate-model", "candidate"),
            ModelSpec("baseline", "homelab", "baseline-model", "baseline"),
        ),
        judge=JudgeSpec("homelab", "judge", "off", 10, 2),
    )


def test_compose_meeting_prompt_matches_meeting_service():
    assert compose_meeting_prompt("PROMPT", "TRANSCRIPT") == (
        "PROMPT\n\n'''TRANSCRIPT'''\nTRANSCRIPT"
    )


def test_generation_creates_repetitions_and_candidate_and_baseline(
    tmp_path: Path,
):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    client = StubPiRpcClient()

    artifacts = generate_candidates(config, store, None, client)

    assert len(artifacts) == 2 * 2 * 2 * 3
    assert {artifact.model_id for artifact in artifacts} == {
        "candidate",
        "baseline",
    }
    assert {artifact.repetition for artifact in artifacts} == {1, 2, 3}
    assert len({artifact.cache_key for artifact in artifacts}) == len(
        artifacts
    )
    assert len(client.requests) == len(artifacts)


def test_filters_select_ids_without_changing_cache_keys(tmp_path: Path):
    config = make_config(tmp_path)
    all_store = RunStore.create(tmp_path / "all", config.source)
    all_artifacts = generate_candidates(
        config, all_store, None, StubPiRpcClient()
    )

    filtered_store = RunStore.create(tmp_path / "filtered", config.source)
    filters = GenerationFilters(
        model_ids={"candidate"},
        prompt_ids={"current"},
        splits={"development"},
    )
    filtered = generate_candidates(
        config, filtered_store, filters, StubPiRpcClient()
    )

    expected = {
        artifact.cache_key
        for artifact in all_artifacts
        if artifact.model_id == "candidate"
        and artifact.prompt_id == "current"
        and artifact.case_id == "sync"
    }
    assert {artifact.cache_key for artifact in filtered} == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_ids", {"missing"}),
        ("prompt_ids", {"missing"}),
        ("case_ids", {"missing"}),
        ("splits", {"missing"}),
    ],
)
def test_unknown_filter_fails_before_model_process(
    tmp_path: Path, field, value
):
    config = make_config(tmp_path)
    client = StubPiRpcClient()
    store = RunStore.create(config.output_dir, config.source)

    with pytest.raises(ValueError, match="filter"):
        generate_candidates(
            config, store, GenerationFilters(**{field: value}), client
        )
    assert client.requests == []


def test_completed_jobs_are_skipped_on_resume(tmp_path: Path):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    first_client = StubPiRpcClient()
    first = generate_candidates(
        config, store, GenerationFilters(case_ids={"sync"}), first_client
    )
    second_client = StubPiRpcClient()

    second = generate_candidates(
        config, store, GenerationFilters(case_ids={"sync"}), second_client
    )

    assert [a.cache_key for a in second] == [a.cache_key for a in first]
    assert second_client.requests == []


def test_prompt_and_transcript_content_changes_invalidate_only_related_jobs(
    tmp_path: Path,
):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    first = generate_candidates(config, store, None, StubPiRpcClient())
    config.prompts[0].path.write_text("CHANGED PROMPT", encoding="utf-8")
    config.cases[0].transcript.write_text(
        "CHANGED TRANSCRIPT", encoding="utf-8"
    )
    client = StubPiRpcClient()

    second = generate_candidates(config, store, None, client)

    assert len(client.requests) == 3 * 2 * 2 + 3 * 2
    old_by_job = {
        (a.prompt_id, a.case_id, a.model_id, a.repetition): a.cache_key
        for a in first
    }
    for artifact in second:
        key = (
            artifact.prompt_id,
            artifact.case_id,
            artifact.model_id,
            artifact.repetition,
        )
        if key == ("variant", "other", "candidate", artifact.repetition):
            assert artifact.cache_key == old_by_job[key]


def test_artifact_payload_and_composed_request_are_persisted(tmp_path: Path):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    client = StubPiRpcClient()

    artifacts = generate_candidates(
        config,
        store,
        GenerationFilters(
            model_ids={"candidate"},
            prompt_ids={"current"},
            case_ids={"sync"},
        ),
        client,
    )

    artifact = artifacts[0]
    payload = store.read_json(artifact.artifact_path)
    assert payload["schema_version"] == 1
    assert payload["operation"] == "generation"
    assert payload["status"] == "complete"
    assert payload["kind"] == "candidate"
    assert payload["summary_path"] == artifact.summary_path
    assert client.requests[0].prompt == compose_meeting_prompt(
        "PROMPT", "TRANSCRIPT"
    )


def test_fail_fast_persists_failure_before_raising(tmp_path: Path):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    client = StubPiRpcClient(fail_models={"candidate-model"})

    with pytest.raises(RuntimeError, match="stub failure"):
        generate_candidates(
            config,
            store,
            GenerationFilters(model_ids={"candidate"}),
            client,
            fail_fast=True,
        )

    failed = list((store.run_dir / "generations").rglob("*.json"))
    assert len(failed) == 1
    assert store.read_json(failed[0])["status"] == "failed"


def test_failed_artifact_is_retried_on_resume(tmp_path: Path):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    failing = StubPiRpcClient(fail_models={"candidate-model"})
    first = generate_candidates(
        config, store, GenerationFilters(model_ids={"candidate"}), failing
    )
    assert all(artifact.status == "failed" for artifact in first)

    retry = StubPiRpcClient()
    second = generate_candidates(
        config, store, GenerationFilters(model_ids={"candidate"}), retry
    )
    assert len(retry.requests) == len(first)
    assert all(artifact.status == "complete" for artifact in second)


def test_json_and_markdown_writes_are_atomic(tmp_path: Path):
    store = RunStore.create(tmp_path / "runs", tmp_path / "benchmark.yaml")
    json_path = store.write_json(Path("artifact.json"), {"ok": True})
    text_path = store.write_text(Path("summary.md"), "# Summary")

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert text_path.read_text(encoding="utf-8") == "# Summary"
    assert not list(store.run_dir.glob("*.tmp"))


def test_external_inputs_are_not_persisted_in_manifest(tmp_path: Path):
    config = make_config(
        tmp_path,
        prompt_text="PRIVATE PROMPT",
        transcript_text="PRIVATE TRANSCRIPT",
    )
    store = RunStore.create(config.output_dir, config.source)
    generate_candidates(
        config, store, GenerationFilters(case_ids={"sync"}), StubPiRpcClient()
    )

    manifest = (store.run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "PRIVATE PROMPT" not in manifest
    assert "PRIVATE TRANSCRIPT" not in manifest
    assert any(
        path.read_text(encoding="utf-8") == "PRIVATE PROMPT"
        for path in (store.run_dir / "inputs" / "prompts").glob("*")
    )
    assert any(
        path.read_text(encoding="utf-8") == "PRIVATE TRANSCRIPT"
        for path in (store.run_dir / "inputs" / "transcripts").glob("*")
    )
