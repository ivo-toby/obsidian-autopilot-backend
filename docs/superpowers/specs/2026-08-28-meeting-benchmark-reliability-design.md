# Meeting Benchmark Reliability and Progress Design

## Context

The first full run looked stalled because the CLI emitted no progress while generation and judging ran sequentially. It was manually interrupted during an absolute judge request. The partial artifacts exposed two correctness problems:

- The absolute judge prompt duplicated an older meeting-summary section contract and therefore penalized sections now allowed by `prompts/MEETING_PROMPT.md`.
- The judge often returned objects inside `missed_items`, while the strict result schema requires strings. The retry prompt did not state the parser error or the element types clearly enough.

The interrupted run remains outside Git and must stay reusable after the fixes.

## Goals

- Judge every candidate against the exact prompt snapshot used to generate it.
- Preserve strict judge-result validation while making the one allowed retry actionable.
- Show continuous, flushed progress for generation, absolute judging, pairwise judging, and reporting.
- Preserve content-addressed caching, anonymous judging, input snapshots, resumability, and Python 3.8 compatibility.

## Non-goals

- Do not accept or silently normalize schema-invalid judge output.
- Do not change scoring weights, pairwise selection, model configuration, timeouts, or repetition counts.
- Do not fix the unrelated `gemma4-e4b` provider failure.
- Do not run live candidate or judge models during implementation or verification.
- Do not modify the transcript, golden summary, or existing run artifacts.

## Judge Contract

Absolute judging will load the content-addressed prompt snapshot identified by each generation artifact's `prompt_id` and `prompt_sha256`. The rendered judge request will include that snapshot in a dedicated trusted `SUMMARY_INSTRUCTIONS` block. Transcript, golden, and candidate blocks remain explicitly untrusted data.

`judge-v1.md` will stop duplicating the available summary headings. It will instruct the evaluator to apply the exact trusted summary instructions when scoring structure and compliance. This works for the current prompt and future prompt variants without another manually synchronized contract.

The rendered prompt remains part of the judgment cache key. Changing the judge template or generation prompt therefore invalidates old judgments while preserving valid generation artifacts. Resuming the interrupted run after this change will reuse summaries and regenerate absolute judgments under the corrected contract.

## Strict JSON Retry

The initial judge prompt and retry prompt will state the element types explicitly:

- `critical_errors` is an array of objects with exactly `claim`, `transcript_evidence`, and `explanation` string fields.
- `missed_items`, `failure_tags`, and `prompt_recommendations` are arrays of plain JSON strings, never objects.
- `verdict` is a non-empty string.

The retry request will include the exact `JudgmentParseError` message as trusted correction context alongside the previous response in an untrusted delimiter. Parsing remains strict. A second invalid response is persisted as a failed judgment exactly as it is today.

## CLI Progress

Core generation and judging functions will accept an optional progress callback and remain quiet when it is absent. CLI entry points will pass a callback that prints each message immediately with `flush=True`.

Messages will contain no transcript, golden, candidate, or judge content. They will use stable phase and job identities:

```text
[phase] generation
[generation 1/30] start model=qwen35-9b prompt=current case=core-ai-agents-sync repetition=1
[generation 1/30] complete elapsed=58.8s
[generation 2/30] cached
[generation 5/30] failed error=<concise error>

[phase] absolute judging
[absolute 1/27] start model=qwen35-9b prompt=current case=core-ai-agents-sync repetition=1
[absolute 1/27] complete elapsed=112.2s

[phase] pairwise judging
[pairwise 1/12] start models=qwen35-9b,gemma4-26b prompt=current case=core-ai-agents-sync repetition=1
[pairwise 1/12] complete elapsed=45.1s

[phase] reporting
[reporting] complete path=/external/run/directory
```

Cached and failed jobs count toward progress. Absolute totals include only complete, selected generation artifacts because failed generations cannot be judged. Pairwise totals are calculated after ranked models and matching repetition pairs are known.

The standalone `generate` and `judge` commands receive the same job-level progress. `all` adds phase boundaries and reporting status. Existing final run-directory messages and exit-code behavior remain unchanged.

## Error Handling and Resume

A failed job prints a concise error and remains persisted. Without `--fail-fast`, later jobs continue. With `--fail-fast`, the existing abort behavior remains.

On resume:

- Complete generation artifacts print `cached` and are reused.
- Failed generations run again under existing semantics.
- Judgments produced under the old rendered prompt are not cache matches and run again.
- Complete judgments produced under the corrected prompt are reused.
- Pairwise and reporting continue normally after absolute judging.

## Validation

Implementation uses strict TDD with fake Pi only:

- A prompt-contract regression test proves the exact generation prompt snapshot is embedded and newly allowed sections are not contradicted by stale judge instructions.
- A retry regression test proves the parser error and plain-string array contract appear in the second request while invalid output remains rejected.
- Generation tests prove ordered start/cached/complete/failed progress with correct totals.
- Absolute and pairwise tests prove ordered progress with selected-job totals.
- CLI tests prove phase messages are emitted and flushed behavior is wired without exposing private content.
- Existing benchmark tests, lint, formatting, mypy, compilation, and Python 3.8 grammar checks remain required.

No live model call or existing external run mutation is part of implementation verification.
