# Cross-Model Meeting Summary Prompt Optimization Design

## Goal

Improve meeting-summary factual reliability for Muse-Glimmer while checking that the prompt generalizes to Qwen3.8 and measuring the cloud ceiling with Luna. Do not change the production prompt automatically or tune against validation results.

Muse-Glimmer remains the selected local deployment target. Qwen3.8 uses the configured `qwen38` benchmark model, `homelab/titan/ollama/qwen3.8:latest`. Luna uses the configured `luna-control` baseline, `openai-codex/gpt-5.6-luna`.

## Experiment shape

Create two prompt variants and evaluate them only on the existing development case first:

1. `precision-first`: favor supported claims over completeness.
2. `balanced-coverage`: combine the same evidence gates with a private coverage ledger for major topics, metrics, decisions, actions, blockers, and unresolved scope.

Run each variant three times on Muse-Glimmer, Qwen3.8, and Luna for development: 18 generations. Compare them with the existing `current` development baselines; do not regenerate the current prompt results.

Promote one variant to validation only when it reduces Muse-Glimmer development critical factual errors without increasing critical errors for Qwen3.8 or Luna. If both variants qualify, choose using the evaluation gates below. Run the selected variant three times on all three models for validation: nine generations. It must reduce Muse-Glimmer validation critical factual errors without increasing critical errors for Qwen3.8 or Luna before it is eligible for manual promotion.

The complete experiment requires 27 generations. Muse-Glimmer is the primary optimization target, Qwen3.8 is the local-model generalization check, and Luna is the cloud baseline. No prompt is promoted automatically. `prompts/MEETING_PROMPT.md` remains unchanged until Ivo explicitly approves a winning variant after validation.

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

| Model | Split | Mean score | Critical errors | Score stddev |
| --- | --- | ---: | ---: | ---: |
| Muse-Glimmer | development | 71.0 | 10 | 1.63 |
| Muse-Glimmer | validation | 66.0 | 6 | 1.63 |
| Qwen3.8 | development | 56.33 | 19 | 0.94 |
| Qwen3.8 | validation | 55.33 | 14 | 2.05 |
| Luna | development | 73.0 | 7 | 1.63 |
| Luna | validation | 76.33 | 3 | 2.87 |

Development eligibility requires both conditions:

1. Fewer Muse-Glimmer critical factual errors than the development baseline of 10.
2. No increase in Qwen3.8 or Luna critical factual errors from their development baselines of 19 and 7.

When both variants are eligible, select in this order:

1. Lowest combined critical factual errors across all three models.
2. Highest Muse-Glimmer aggregate score.
3. Better coverage of explicit decisions, actions, blockers, and metrics across all three models.
4. Lower Muse-Glimmer score variance across three repetitions.
5. Lower Muse-Glimmer latency.

Validation uses the same guardrail: Muse-Glimmer must improve from 6 critical errors while Qwen3.8 and Luna must not regress from 14 and 3. A score increase cannot compensate for equal or worse Muse-Glimmer critical factual errors or a factual regression on either control model. Review each hard failure and its transcript evidence before choosing or promoting a winner. Pairwise model comparisons are secondary evidence; they do not override the factual-error gate.

## Commands

Development experiment, 18 generations:

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

Validation experiment after choosing one development winner, nine generations:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model muse-glimmer \
  --model qwen38 \
  --model luna-control \
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
