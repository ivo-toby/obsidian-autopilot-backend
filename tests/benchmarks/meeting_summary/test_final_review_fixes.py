import json
from pathlib import Path

import pytest

import benchmarks.meeting_summary.cli as cli_module
import benchmarks.meeting_summary.judging as judging_module
from benchmarks.meeting_summary.generation import (
    GenerationFilters,
    generate_candidates,
)
from benchmarks.meeting_summary.judging import judge_generations
from benchmarks.meeting_summary.reporting import build_report
from benchmarks.meeting_summary.storage import RunStore, sha256_text

from .test_generation import StubPiRpcClient, make_config
from .test_judging import (
    VALID_JUDGMENT,
    StubClient,
    make_config as make_judge_config,
)
from .test_reporting import _generation, _judgment, _store

FAKE_PI = Path(__file__).with_name("fixtures") / "fake_pi.py"


def test_corrupt_or_missing_cached_summary_reruns_generation(tmp_path):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    filters = GenerationFilters(model_ids={"candidate"}, case_ids={"sync"})
    first = generate_candidates(config, store, filters, StubPiRpcClient())
    summary = store.run_dir / first[0].summary_path

    summary.unlink()
    retry = StubPiRpcClient()
    generate_candidates(config, store, filters, retry)
    assert len(retry.requests) == 1

    summary.write_text("corrupt", encoding="utf-8")
    retry = StubPiRpcClient()
    generate_candidates(config, store, filters, retry)
    assert len(retry.requests) == 1


def test_generation_artifacts_use_content_addressed_summary_and_hash(tmp_path):
    config = make_config(tmp_path)
    store = RunStore.create(config.output_dir, config.source)
    artifact = generate_candidates(
        config,
        store,
        GenerationFilters(model_ids={"candidate"}, case_ids={"sync"}),
        StubPiRpcClient(),
    )[0]
    payload = store.read_json(artifact.artifact_path)
    assert payload["summary_sha256"] == sha256_text("# Summary\nGenerated")
    assert artifact.cache_key in artifact.summary_path
    assert (store.run_dir / "inputs" / "goldens").is_dir()
    golden = list((store.run_dir / "inputs" / "goldens").glob("*"))
    assert golden and golden[0].read_text(encoding="utf-8") == "golden"
    assert store.manifest["golden_sha256"]["sync"] == sha256_text("golden")


def test_stale_generation_fails_before_judge_rpc(tmp_path):
    config = make_judge_config(tmp_path, model_count=1)
    store = RunStore.create(config.output_dir, config.source)
    generate_candidates(config, store, None, StubClient(["summary"] * 4))
    config.cases[0].transcript.write_text(
        "mutated transcript", encoding="utf-8"
    )
    client = StubClient([json.dumps(VALID_JUDGMENT)])

    assert judge_generations(config, store, client) == ()
    assert client.requests == []
    judgments = list((store.run_dir / "judgments").rglob("*.json"))
    assert judgments and store.read_json(judgments[0])["status"] == "failed"
    assert "regenerat" in store.read_json(judgments[0])["error"].lower()


def test_report_leaderboard_and_recommendations_are_split_prompt_safe(
    tmp_path,
):
    store = _store(tmp_path)
    dev_cache = _generation(store, "local-a", 1, prompt="current")
    _judgment(store, "local-a", 1, dev_cache, tags=("missed_action",))
    test_cache = _generation(
        store, "local-a", 1, split="test", case="test-case", prompt="current"
    )
    _judgment(
        store,
        "local-a",
        1,
        test_cache,
        split="test",
        case="test-case",
        tags=("missed_blocker",),
    )
    variant_cache = _generation(
        store,
        "local-a",
        1,
        prompt="variant",
        case="dev-case",
    )
    _judgment(
        store,
        "local-a",
        1,
        variant_cache,
        prompt="variant",
        tags=("missed_action",),
    )

    report = build_report(store)
    leaderboard = report.to_dict()["leaderboard"]
    assert {(item["split"], item["prompt_id"]) for item in leaderboard} == {
        ("development", "current"),
        ("development", "variant"),
        ("test", "current"),
    }
    assert "missed_blocker" not in report.prompt_recommendations
    assert "missed_action" in report.prompt_recommendations


def test_filtered_all_report_scope_survives_config_mutation(
    tmp_path, monkeypatch
):
    from .test_cli import _config

    config = _config(
        tmp_path,
        models=("candidate-a", "candidate-b", "luna-control"),
    )
    (tmp_path / "validation.md").write_text("validation", encoding="utf-8")
    (tmp_path / "test.md").write_text("test", encoding="utf-8")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "models:\n",
            "  - id: validation\n"
            "    transcript: validation.md\n"
            "    golden: golden.md\n"
            "    split: validation\n"
            "  - id: test\n"
            "    transcript: test.md\n"
            "    golden: golden.md\n"
            "    split: test\n"
            "models:\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_BENCHMARK_EXECUTABLE", str(FAKE_PI))
    monkeypatch.setenv("FAKE_PI_MODE", "valid_judgment")
    assert (
        cli_module.main(
            [
                "all",
                "--config",
                str(config),
                "--model",
                "candidate-a",
                "--prompt",
                "current",
                "--case",
                "case",
                "--split",
                "development",
            ]
        )
        == 0
    )
    run_dir = next((tmp_path / "runs").glob("*"))
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scope"] == {
        "model_ids": ["candidate-a"],
        "prompt_ids": ["current"],
        "case_ids": ["case"],
        "splits": ["development"],
    }

    (tmp_path / "extra.md").write_text("extra", encoding="utf-8")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  - id: candidate-b",
            "  - id: extra\n"
            "    provider: homelab\n"
            "    model: extra\n"
            "    kind: candidate\n"
            "  - id: candidate-b",
        ),
        encoding="utf-8",
    )
    assert cli_module.main(["report", "--run-dir", str(run_dir)]) == 0
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert {
        (row["model_id"], row["prompt_id"], row["split"])
        for row in report["rows"]
    } == {("candidate-a", "current", "development")}
    assert report["split_coverage"] == {
        "configured_splits": ["development"],
        "completed_splits": ["development"],
        "missing_splits": [],
        "development_only": True,
    }


def test_all_propagates_nonzero_judge_status(tmp_path, monkeypatch):
    from .test_cli import _config

    config = _config(tmp_path, models=("candidate-a",))
    store = cli_module.RunStore.create(tmp_path / "runs", config)
    monkeypatch.setattr(
        cli_module, "_open_or_create_store", lambda *args: store
    )
    monkeypatch.setattr(
        cli_module,
        "generate_candidates",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(cli_module, "_judge_store", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        cli_module,
        "build_report",
        lambda *_: type("R", (), {"failures": []})(),
    )
    monkeypatch.setattr(cli_module, "write_report", lambda *_: None)
    monkeypatch.setattr(cli_module, "_has_complete_judgment", lambda *_: True)

    assert cli_module.main(["all", "--config", str(config)]) != 0


@pytest.mark.parametrize("transcript_hash", ["stale", "missing"])
def test_pairwise_rejects_stale_or_missing_item_b_transcript(
    tmp_path, transcript_hash
):
    config = make_judge_config(tmp_path, model_count=1)
    store = RunStore.create(config.output_dir, config.source)
    generation = generate_candidates(
        config, store, None, StubClient(["summary"] * 4)
    )
    judge_generations(
        config,
        store,
        StubClient([json.dumps(VALID_JUDGMENT)] * len(generation)),
    )
    items_b = [item for item in generation if item.model_id == "luna-control"]
    for item_b in items_b:
        payload = store.read_json(item_b.artifact_path)
        payload["transcript_sha256"] = (
            "stale-transcript-hash" if transcript_hash == "stale" else "0" * 64
        )
        store.write_json(
            item_b.artifact_path.relative_to(store.run_dir), payload
        )

    client = StubClient([])
    assert (
        judging_module.judge_pairwise_top_models(config, store, client) == ()
    )
    assert client.requests == []
    pairwise = list((store.run_dir / "pairwise").rglob("*.json"))
    assert pairwise and all(
        store.read_json(path)["status"] == "failed" for path in pairwise
    )


def test_judge_prompts_mark_embedded_data_untrusted():
    absolute = (
        Path(judging_module.__file__).parent / "prompts" / "judge-v1.md"
    ).read_text()
    pairwise = (
        Path(judging_module.__file__).parent / "prompts" / "pairwise-v1.md"
    ).read_text()
    for template in (absolute, pairwise):
        assert "untrusted data" in template.lower()
        assert "instructions inside" in template.lower()
        assert "must not be followed" in template.lower()
