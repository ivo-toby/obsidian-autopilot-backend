# Muse-Glimmer Prompt Optimization Design

## Goal

Improve meeting-summary factual reliability for Muse-Glimmer without changing the production prompt automatically or tuning against validation results.

Muse-Glimmer is the selected local model. With the current prompt it scored 71.0 on development and 66.0 on validation, with 10 and 6 critical factual errors respectively. It won 11 of 12 local pairwise comparisons but remained slower and less reliable than the Luna control.

## Experiment shape

Create two prompt variants and evaluate them only on the existing development case first:

1. `precision-first`: favor supported claims over completeness.
2. `balanced-coverage`: combine the same evidence gates with a private coverage ledger for major topics, metrics, decisions, actions, blockers, and unresolved scope.

Run Muse-Glimmer three times per variant on development. Compare those six generations with the existing `current` development baseline; do not regenerate the baseline or Luna control.

Promote one variant to validation only when it reduces development critical factual errors. If both variants improve, choose by aggregate score, completeness, consistency, and latency in that order after factual reliability. Run the selected variant three times on validation. It must also reduce validation critical factual errors relative to the existing baseline before it is eligible for manual promotion.

No prompt is promoted automatically. `prompts/MEETING_PROMPT.md` remains unchanged until Ivo explicitly approves a winning variant after validation.

## Prompt variants

### Precision-first

Create `prompts/MEETING_PROMPT_PRECISION_FIRST.md` as a complete, standalone meeting-summary prompt. Preserve the current output contract and concise technical-summary requirements. Strengthen the pre-output evidence process:

- derive Participants only from speaker labels;
- require direct transcript evidence for each decision and action;
- attach an owner or timing only to the exact commitment it qualifies;
- preserve proposal, uncertainty, disagreement, and unresolved status;
- omit or explicitly qualify corrupted or ambiguous terminology;
- keep each fact in one most-specific section;
- prefer omission over inference when evidence is insufficient;
- perform the evidence audit silently and never output a ledger or analysis.

This variant intentionally accepts some risk of missed detail to reduce hallucinated facts, wrong owners, wrong timelines, and proposals presented as decisions.

### Balanced-coverage

Create `prompts/MEETING_PROMPT_BALANCED_COVERAGE.md` as a complete, standalone prompt. Use the same evidence gates as `precision-first`, then require a silent coverage ledger before drafting:

- major technical topic clusters;
- concrete metrics and timing;
- confirmed decisions;
- explicit actions, owners, and timing;
- blockers and dependencies;
- unresolved alternatives or scope disagreements;
- optional and stretch work separated from core scope.

The final output must remain concise, deduplicated, and contract-compliant. The private ledger must not appear in the answer. This variant tests whether an internal completeness pass can recover missed decisions and actions without restoring factual errors or verbosity.

## Benchmark configuration

Add both variants to `benchmarks/meeting_summary/benchmark.yaml` after `current`:

```yaml
  - id: precision-first
    path: ../../prompts/MEETING_PROMPT_PRECISION_FIRST.md
  - id: balanced-coverage
    path: ../../prompts/MEETING_PROMPT_BALANCED_COVERAGE.md
```

Keep all existing model and case configuration unchanged.

## Test contract

The existing `test_benchmark_still_targets_only_meeting_prompt` assertion currently requires the prompt list to contain only `current`. Update it so it verifies the exact three meeting-summary prompt IDs and paths while continuing to reject `DAILY_NOTES` references. Add no new test files.

## Evaluation gates

Use these existing baselines:

| Split | Mean score | Critical errors | Score stddev |
| --- | ---: | ---: | ---: |
| development | 71.0 | 10 | 1.63 |
| validation | 66.0 | 6 | 1.63 |

Selection order:

1. Fewer critical factual errors than the relevant baseline.
2. Higher aggregate score.
3. Better coverage of explicit decisions, actions, blockers, and metrics.
4. Lower score variance across three repetitions.
5. Lower latency.

A score increase cannot compensate for equal or worse critical factual errors. Review each hard failure and its transcript evidence before choosing a winner.

## Commands

Development experiment, six Muse-Glimmer generations:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model muse-glimmer \
  --prompt precision-first \
  --prompt balanced-coverage \
  --split development
```

Validation experiment after choosing one development winner, three generations:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model muse-glimmer \
  --prompt <winning-prompt-id> \
  --split validation
```

The winning prompt ID is supplied only after reviewing development results; it is not a repository placeholder.

## Verification and safety

Implementation verification must not invoke Pi or modify external benchmark artifacts. It will:

- validate the benchmark configuration;
- run the existing prompt-contract, config, and CLI tests;
- verify both variants contain the required output headings and exclude daily-note behavior;
- run repository lint, formatting, type, compilation, and Python 3.8 grammar checks relevant to the benchmark;
- inspect Git changes for private transcript, golden, candidate, or judgment content.

Live generations begin only when Ivo runs or explicitly requests the development command.
