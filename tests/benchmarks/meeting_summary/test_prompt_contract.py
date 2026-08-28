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

PRECISION_FIRST_AUDIT_BLOCK = (
    "## Silent Evidence Audit\n"
    "\n"
    "Before drafting, silently build an evidence table from the transcript. "
    "The table must not appear in the answer.\n"
    "\n"
    "- Derive Participants only from speaker labels, not from names mentioned "
    "in conversation.\n"
    "- Require direct transcript evidence for every decision and action.\n"
    "- Attach an owner or timing only to the exact commitment it qualifies.\n"
    "- Preserve proposals, uncertainty, disagreement, and unresolved status.\n"
    "- Omit or explicitly qualify corrupted or ambiguous terminology.\n"
    "- Keep each fact in its one most-specific section.\n"
    "- Prefer omission over inference when evidence is insufficient.\n"
    "\n"
    "Audit every final claim against that table before returning the summary. "
    "Output only the contracted summary; the evidence table and audit "
    "must not appear."
)

BALANCED_COVERAGE_AUDIT_BLOCK = (
    "## Silent Coverage and Evidence Audit\n"
    "\n"
    "Before drafting, silently build a coverage ledger from the transcript. "
    "The ledger must not appear in the answer.\n"
    "\n"
    "Check the ledger for:\n"
    "\n"
    "- major technical topic clusters;\n"
    "- concrete metrics and timing;\n"
    "- confirmed decisions;\n"
    "- explicit actions, owners, and timing;\n"
    "- blockers and dependencies;\n"
    "- unresolved alternatives or scope disagreements;\n"
    "- optional and stretch work separated from core scope.\n"
    "\n"
    "Then apply these evidence gates:\n"
    "\n"
    "- Derive Participants only from speaker labels, not from names mentioned "
    "in conversation.\n"
    "- Require direct transcript evidence for every decision and action.\n"
    "- Attach an owner or timing only to the exact commitment it qualifies.\n"
    "- Preserve proposals, uncertainty, disagreement, and unresolved status.\n"
    "- Omit or explicitly qualify corrupted or ambiguous terminology.\n"
    "- Keep each fact in its one most-specific section.\n"
    "- Prefer omission over inference when evidence is insufficient.\n"
    "\n"
    "Audit the final summary against the ledger before returning it. Output "
    "only the contracted summary; the coverage ledger and audit must not "
    "appear."
)

OUTPUT_CONTRACT_BOUNDARY = "\n\n## Output Contract\n"


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


def _expected_prompt_variant(audit_block):
    production = MEETING_PROMPT.read_text(encoding="utf-8")

    assert production.count(OUTPUT_CONTRACT_BOUNDARY) == 1
    prefix, boundary, suffix = production.partition(OUTPUT_CONTRACT_BOUNDARY)
    assert boundary
    return prefix + "\n\n" + audit_block + boundary + suffix


def test_prompt_variants_match_production_plus_approved_audit():
    expected_variants = {
        "precision-first": _expected_prompt_variant(
            PRECISION_FIRST_AUDIT_BLOCK
        ),
        "balanced-coverage": _expected_prompt_variant(
            BALANCED_COVERAGE_AUDIT_BLOCK
        ),
    }

    for prompt_id, expected in expected_variants.items():
        actual = PROMPT_VARIANTS[prompt_id].read_text(encoding="utf-8")
        assert actual == expected
