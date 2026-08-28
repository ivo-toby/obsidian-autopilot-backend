from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MEETING_PROMPT = ROOT / "prompts" / "MEETING_PROMPT.md"
BENCHMARK_CONFIG = ROOT / "benchmarks" / "meeting_summary" / "benchmark.yaml"

PROMPT_CONFIG = [
    {"id": "current", "path": "../../prompts/MEETING_PROMPT.md"},
    {
        "id": "precision-first",
        "path": "../../prompts/MEETING_PROMPT_PRECISION_FIRST.md",
    },
    {
        "id": "balanced-coverage",
        "path": "../../prompts/MEETING_PROMPT_BALANCED_COVERAGE.md",
    },
]

PROMPT_VARIANTS = {
    "precision-first": (
        ROOT / "prompts" / "MEETING_PROMPT_PRECISION_FIRST.md"
    ),
    "balanced-coverage": (
        ROOT / "prompts" / "MEETING_PROMPT_BALANCED_COVERAGE.md"
    ),
}

REQUIRED_HEADINGS = (
    "## Tags",
    "## Participants",
    "## Context",
    "## Key Outcomes",
    "## Discussion Notes",
    "## Decisions Made",
    "## Action Items",
    "## References",
)

FORBIDDEN_DAILY_BEHAVIOR = (
    "journal entries",
    "infer which entries",
    "create_meeting_notes",
    "function call",
    "multiple meetings",
)


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


def test_benchmark_targets_only_meeting_prompts():
    payload = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))

    assert payload["prompts"] == PROMPT_CONFIG
    assert "DAILY_NOTES" not in BENCHMARK_CONFIG.read_text(encoding="utf-8")


def test_prompt_variants_preserve_meeting_contract():
    for prompt in PROMPT_VARIANTS.values():
        content = prompt.read_text(encoding="utf-8")
        lower_content = content.lower()

        for heading in REQUIRED_HEADINGS:
            assert heading in content
        for forbidden in FORBIDDEN_DAILY_BEHAVIOR:
            assert forbidden not in lower_content

        assert "silently" in lower_content
        assert "speaker labels" in lower_content
        assert "must not appear" in lower_content

    precision = PROMPT_VARIANTS["precision-first"].read_text(encoding="utf-8")
    balanced = PROMPT_VARIANTS["balanced-coverage"].read_text(encoding="utf-8")
    assert "prefer omission over inference" in precision.lower()
    assert "coverage ledger" in balanced.lower()
