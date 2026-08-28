# Cross-Model Meeting Summary Prompt Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two standalone meeting-summary prompt variants and configure the 27-generation Muse-Glimmer, Qwen3.8, and Luna optimization experiment without running live models.

**Architecture:** Copy the approved production meeting prompt into two separate tracked variants and add one silent pre-drafting process to each. Register both variants in the existing benchmark YAML. Keep the production prompt, models, cases, and external benchmark artifacts unchanged.

**Tech Stack:** Markdown prompts, YAML, Python 3.8+, pytest, Flake8, Black, mypy

## Global Constraints

- Do not modify `prompts/MEETING_PROMPT.md`.
- Create complete standalone variants at `prompts/MEETING_PROMPT_PRECISION_FIRST.md` and `prompts/MEETING_PROMPT_BALANCED_COVERAGE.md`.
- Configure prompt IDs `precision-first` and `balanced-coverage` after `current`.
- Muse-Glimmer remains the primary optimization target.
- Qwen3.8 is benchmark model `qwen38`, routed to `homelab/titan/ollama/qwen3.8:latest`.
- Luna is baseline model `luna-control`, routed to `openai-codex/gpt-5.6-luna`.
- Development uses two variants × three models × three repetitions: 18 generations.
- Validation uses one winning variant × three models × three repetitions: nine generations.
- Do not invoke Pi, run live models, or modify external benchmark artifacts during implementation or verification.
- Do not copy transcript, golden, candidate, judgment, or raw error content into Git.
- The approved test change is limited to `tests/benchmarks/meeting_summary/test_prompt_contract.py`; create no new test files.
- Preserve Python 3.8 compatibility.
- Prompt promotion remains manual; this work must not change the production prompt.

---

### Task 1: Add and verify the prompt optimization variants

**Files:**
- Create: `prompts/MEETING_PROMPT_PRECISION_FIRST.md`
- Create: `prompts/MEETING_PROMPT_BALANCED_COVERAGE.md`
- Modify: `benchmarks/meeting_summary/benchmark.yaml`
- Modify: `tests/benchmarks/meeting_summary/test_prompt_contract.py`

**Interfaces:**
- Consumes: the complete output contract and evidence rules from `prompts/MEETING_PROMPT.md`.
- Produces: prompt IDs `precision-first` and `balanced-coverage`, selectable through the existing repeatable `--prompt` filter.

- [ ] **Step 1: Capture privacy and immutability baselines**

Run before editing:

```bash
sha256sum prompts/MEETING_PROMPT.md \
  > /tmp/meeting-prompt-before.sha256
find /home/ivo/cf-notes/benchmarks/meeting-summary \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/meeting-summary-runs-before-prompt-optimization.sha256
```

Expected: both manifests are created outside the repository.

- [ ] **Step 2: Update the approved prompt-contract test first**

In `tests/benchmarks/meeting_summary/test_prompt_contract.py`, add these constants below `BENCHMARK_CONFIG`:

```python
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
```

Replace `test_benchmark_still_targets_only_meeting_prompt` with:

```python
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

    precision = PROMPT_VARIANTS["precision-first"].read_text(
        encoding="utf-8"
    )
    balanced = PROMPT_VARIANTS["balanced-coverage"].read_text(
        encoding="utf-8"
    )
    assert "prefer omission over inference" in precision.lower()
    assert "coverage ledger" in balanced.lower()
```

Keep the existing production-prompt tests unchanged.

- [ ] **Step 3: Run the prompt-contract test and verify RED**

Run:

```bash
pytest tests/benchmarks/meeting_summary/test_prompt_contract.py -q --no-cov
```

Expected: FAIL because `benchmark.yaml` still contains only `current`; the variant files do not yet exist.

- [ ] **Step 4: Create the precision-first variant**

Copy `prompts/MEETING_PROMPT.md` to `prompts/MEETING_PROMPT_PRECISION_FIRST.md`. Insert this exact block between `## Evidence Rules` and `## Output Contract`, after the existing numbered evidence rules:

```markdown
## Silent Evidence Audit

Before drafting, silently build an evidence table from the transcript. The table must not appear in the answer.

- Derive Participants only from speaker labels, not from names mentioned in conversation.
- Require direct transcript evidence for every decision and action.
- Attach an owner or timing only to the exact commitment it qualifies.
- Preserve proposals, uncertainty, disagreement, and unresolved status.
- Omit or explicitly qualify corrupted or ambiguous terminology.
- Keep each fact in its one most-specific section.
- Prefer omission over inference when evidence is insufficient.

Audit every final claim against that table before returning the summary. Output only the contracted summary; the evidence table and audit must not appear.
```

Do not change any other production-prompt text in the copied file.

- [ ] **Step 5: Create the balanced-coverage variant**

Copy `prompts/MEETING_PROMPT.md` to `prompts/MEETING_PROMPT_BALANCED_COVERAGE.md`. Insert this exact block at the same location:

```markdown
## Silent Coverage and Evidence Audit

Before drafting, silently build a coverage ledger from the transcript. The ledger must not appear in the answer.

Check the ledger for:

- major technical topic clusters;
- concrete metrics and timing;
- confirmed decisions;
- explicit actions, owners, and timing;
- blockers and dependencies;
- unresolved alternatives or scope disagreements;
- optional and stretch work separated from core scope.

Then apply these evidence gates:

- Derive Participants only from speaker labels, not from names mentioned in conversation.
- Require direct transcript evidence for every decision and action.
- Attach an owner or timing only to the exact commitment it qualifies.
- Preserve proposals, uncertainty, disagreement, and unresolved status.
- Omit or explicitly qualify corrupted or ambiguous terminology.
- Keep each fact in its one most-specific section.
- Prefer omission over inference when evidence is insufficient.

Audit the final summary against the ledger before returning it. Output only the contracted summary; the coverage ledger and audit must not appear.
```

Do not change any other production-prompt text in the copied file.

- [ ] **Step 6: Register both prompt variants**

Add these entries after `current` in `benchmarks/meeting_summary/benchmark.yaml`:

```yaml
  - id: precision-first
    path: ../../prompts/MEETING_PROMPT_PRECISION_FIRST.md
  - id: balanced-coverage
    path: ../../prompts/MEETING_PROMPT_BALANCED_COVERAGE.md
```

Do not modify models, cases, generation settings, or judge settings.

- [ ] **Step 7: Run focused verification and verify GREEN**

Run:

```bash
python -m benchmarks.meeting_summary validate \
  --config benchmarks/meeting_summary/benchmark.yaml
pytest tests/benchmarks/meeting_summary/test_prompt_contract.py \
  tests/benchmarks/meeting_summary/test_config.py \
  tests/benchmarks/meeting_summary/test_cli.py -q --no-cov
```

Expected: configuration validation succeeds and all focused tests pass without invoking Pi.

- [ ] **Step 8: Run benchmark static verification**

Run:

```bash
pytest tests/benchmarks/meeting_summary -q --no-cov
flake8 benchmarks/meeting_summary tests/benchmarks/meeting_summary
black --check --line-length 79 \
  benchmarks/meeting_summary tests/benchmarks/meeting_summary
mypy benchmarks/meeting_summary
python -m compileall -q benchmarks/meeting_summary \
  tests/benchmarks/meeting_summary
python - <<'PY'
import ast
from pathlib import Path

roots = (
    Path("benchmarks/meeting_summary"),
    Path("tests/benchmarks/meeting_summary"),
)
for root in roots:
    for path in root.rglob("*.py"):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 8),
        )
print("Python 3.8 grammar parsing passed")
PY
```

Expected: all benchmark tests and static checks pass. The final command checks Python 3.8 grammar because an actual Python 3.8 runtime is unavailable.

- [ ] **Step 9: Verify production-prompt, privacy, and run immutability**

Run:

```bash
sha256sum -c /tmp/meeting-prompt-before.sha256
find /home/ivo/cf-notes/benchmarks/meeting-summary \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/meeting-summary-runs-after-prompt-optimization.sha256
cmp /tmp/meeting-summary-runs-before-prompt-optimization.sha256 \
  /tmp/meeting-summary-runs-after-prompt-optimization.sha256
git diff --check
git diff --stat
git status --short
```

Expected: the production prompt and every existing external run artifact are byte-identical. The Git diff contains only the two prompt variants, YAML registration, and approved test update. `.pi-subagents/` remains untracked.

- [ ] **Step 10: Commit**

Run:

```bash
git add prompts/MEETING_PROMPT_PRECISION_FIRST.md \
  prompts/MEETING_PROMPT_BALANCED_COVERAGE.md \
  benchmarks/meeting_summary/benchmark.yaml \
  tests/benchmarks/meeting_summary/test_prompt_contract.py
git commit -m "feat(benchmark): add prompt optimization variants"
```

Expected: one tracked implementation commit and no staged files.

## Development command after implementation

Do not run this during implementation. Ivo may run it after reviewing the prompt files:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model muse-glimmer \
  --model qwen38 \
  --model luna-control \
  --prompt precision-first \
  --prompt balanced-coverage \
  --split development
```
