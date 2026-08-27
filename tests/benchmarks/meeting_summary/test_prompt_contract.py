from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MEETING_PROMPT = ROOT / "prompts" / "MEETING_PROMPT.md"
BENCHMARK_CONFIG = ROOT / "benchmarks" / "meeting_summary" / "benchmark.yaml"


def test_meeting_prompt_has_approved_summary_sections():
    prompt = MEETING_PROMPT.read_text(encoding="utf-8")

    for heading in (
        "## Tags",
        "## Participants",
        "## Context",
        "## Key Outcomes",
        "## Discussion Notes",
        "## Decisions Made",
        "## Action Items",
        "## References",
    ):
        assert heading in prompt

    assert "only people explicitly identified" in prompt.lower()
    assert "only concrete commitments" in prompt.lower()
    assert "only references explicitly mentioned" in prompt.lower()
    assert "do not repeat the same fact" in prompt.lower()


def test_meeting_prompt_excludes_daily_notes_behavior():
    prompt = MEETING_PROMPT.read_text(encoding="utf-8").lower()

    for forbidden in (
        "journal entries",
        "infer which entries",
        "create_meeting_notes",
        "function call",
        "multiple meetings",
    ):
        assert forbidden not in prompt


def test_benchmark_still_targets_only_meeting_prompt():
    payload = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))

    assert payload["prompts"] == [
        {"id": "current", "path": "../../prompts/MEETING_PROMPT.md"}
    ]
    assert "DAILY_NOTES" not in BENCHMARK_CONFIG.read_text(encoding="utf-8")
