# Meeting-summary benchmark

This benchmark runs meeting-summary prompts through isolated Pi RPC processes, evaluates
outputs with an anonymous judge, and writes reproducible reports outside the repository.
Run commands from the repository root with the project Python environment.

## Prerequisites

* Confirm the available Pi models with `pi --list-models`.
* Authenticate every configured provider before an expensive run. The initial matrix uses
  `homelab`, `strongbad`, and `openai-codex` (for the control and offline judge).
* Install the project dependencies in a Python 3.8+ environment, including PyYAML and
  pytest for development.

Validate inputs without starting Pi or creating artifacts:

```bash
python -m benchmarks.meeting_summary validate \
  --config benchmarks/meeting_summary/benchmark.yaml
```

## Configuration and initial matrix

`benchmark.yaml` contains `version`, `output_dir`, `generation` (`repetitions`,
`thinking`, and timeout), `prompts` (id and file path), `cases` (id, transcript, golden
summary, and split), `models` (id, provider, model, and `kind`), and `judge` (provider,
model, thinking, timeout, and `pairwise_top_k`). Paths are resolved relative to the YAML
file, except `~` paths which use the home directory.

The exact initial model matrix is:

| id | provider | model | kind |
|---|---|---|---|
| `qwen35-9b` | `homelab` | `m5/omlx/Qwen3.5-9B-OptiQ-4bit` | candidate |
| `ternary-bonsai-27b` | `homelab` | `m5/omlx/Ternary-Bonsai-27B-mlx-2bit` | candidate |
| `gemma4-12b` | `homelab` | `titan/ollama/gemma4:12b` | candidate |
| `gemma4-26b` | `homelab` | `titan/ollama/gemma4:26b` | candidate |
| `gemma4-e4b` | `homelab` | `m5/omlx/gemma-4-e4b-it-OptiQ-4bit` | candidate |
| `magistral` | `homelab` | `titan/ollama/magistral:latest` | candidate |
| `muse-glimmer` | `homelab` | `titan/ollama/muse-glimmer:latest` | candidate |
| `qwen38` | `homelab` | `titan/ollama/qwen3.8:latest` | candidate |
| `strongbad-qwen36-35b` | `strongbad` | `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` | candidate |
| `luna-control` | `openai-codex` | `gpt-5.6-luna` | baseline |

Strongbad is a separate provider because it is a distinct endpoint/runtime. Keeping its
provider identity explicit prevents an accidentally routed request from looking like a
homelab result. The configured judge is `openai-codex` `gpt-5.6-sol`; its cloud calls are
used only as offline evaluation, never as the candidate under test.

## Isolation and artifacts

Each generation and judge request starts a fresh Pi RPC process with no session, tools,
extensions, skills, prompt templates, context files, or system/coding prompt. Override the
executable for a test or approved local harness with `PI_BENCHMARK_EXECUTABLE`; normal
runs use `pi`.

A run directory contains `manifest.json`, content-addressed transcript and golden-summary
snapshots, generation Markdown and JSON, judgment JSON, pairwise JSON, and `report.md`,
`report.json`, and `report.csv`. Judging uses the exact external golden snapshot and records
its SHA-256; a changed transcript or golden summary invalidates the run before any judge
RPC. Transcript, golden-summary, candidate, and judge content is written only to the
external run directory. Do not commit those artifacts. Delete old run directories
manually when retention is no longer needed; the benchmark never deletes them
automatically.

## Workflow

### Phase 1: select a model

Use the current prompt on development cases, with all three repetitions:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --prompt current \
  --split development
```

Select on factual reliability and hard factual failures first, then aggregate score,
consistency, latency, and tokens. A high mean score cannot hide a critical factual error.

### Phase 2: compare prompt variants

Run the selected local model against every configured prompt variant on development and
validation cases:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model <winning-model-id> \
  --split development \
  --split validation
```

Prompt recommendations derived only from development are explicitly overfitting-prone;
approve a prompt only after validation and held-out testing. To use a golden summary,
review every factual claim, owner, decision, action, and unresolved blocker against the
authoritative transcript, then approve the file as `golden-summary.md` outside Git.

### Held-out check

After choosing a prompt, run only the selected model and prompt on the test split:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model <winning-model-id> \
  --prompt <winning-prompt-id> \
  --split test
```

Keep development, validation, and test cases separate. Do not tune a prompt against test
content or use test results to choose a model.

### Resume and failures

Use `--resume` to reuse complete content-addressed artifacts after interruption:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --resume ~/cf-notes/benchmarks/meeting-summary/<run-id>
```

Changed prompt or transcript content invalidates only the affected jobs. `--model`,
`--prompt`, `--case`, and `--split` may each be supplied repeatedly. Add `--fail-fast`
to stop the current generation or judgment phase at its first failure. A nonzero exit code
means at least one requested job failed; completed artifacts and reports remain available.

Reports expose per-repetition quality, consistency, latency, tokens, baseline delta,
pairwise outcomes, and prompt failure categories. Inspect critical factual failures and
their transcript evidence separately from the aggregate score.

## Adding data safely

To add a transcript, place it in an external notes/transcripts directory and add only its
path, golden path, case id, and split to a local config. Do not paste transcript content
into YAML, tests, README files, or commits. Keep external runs until review is complete,
then manually delete them according to your retention policy.
