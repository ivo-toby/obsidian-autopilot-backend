# Meeting Summary Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable Python CLI that benchmarks meeting-summary models through Pi, evaluates candidates with an LLM judge, reports quality and runtime trade-offs, and diagnoses prompt weaknesses without changing the production prompt automatically.

**Architecture:** A small `benchmarks.meeting_summary` package will load a versioned YAML definition, invoke isolated Pi RPC processes for generation and judging, and persist content-addressed artifacts under `~/cf-notes/benchmarks/meeting-summary`. Generation, judging, and reporting remain separate commands so expensive work is resumable. Transcript content, golden summaries, candidate outputs, and judgments stay outside Git; the repository contains only the runner, prompt templates, configuration paths, documentation, and tests.

**Tech Stack:** Python 3.8+, standard library (`argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`, `statistics`, `subprocess`, `threading`, `queue`, `time`), PyYAML 6, pytest 7+, Pi RPC JSONL protocol.

## Global Constraints

- Implementation workers must use the builtin `worker` agent with model `openai-codex/gpt-5.6-luna`; do not use the existing custom Luna agent because it resolves to the unauthenticated `openai` provider.
- Do not copy meeting transcripts, golden summaries, candidate summaries, or judge results into Git.
- Do not modify `prompts/MEETING_PROMPT.md`; prompt variants are separate files and require explicit user promotion.
- Match the production prompt composition exactly: `<prompt>\n\n'''TRANSCRIPT'''\n<transcript>`.
- Every model run starts a new Pi RPC process with no session, tools, extensions, skills, prompt templates, project context, or coding system prompt.
- Use `thinking: off` for candidate generation. Judge thinking levels come from configuration.
- The transcript is authoritative during judging. The golden summary is a human-reviewed reference, not an independent source of truth.
- Candidate model names must not appear in judge prompts.
- A critical invented fact, owner, deadline, decision, or metric remains visible as a hard failure independent of the aggregate score.
- Cache keys must include runner schema version, provider, model, thinking level, prompt content, transcript content, operation type, repetition, and judge prompt version.
- Resume must never silently reuse artifacts whose cache key differs.
- Prompt recommendations are diagnostic output only. The CLI must never rewrite the production prompt.
- No new runtime dependency is needed; use the existing PyYAML dependency.

---

## File Map

**Repository files to create:**

- `benchmarks/__init__.py` — marks the benchmark namespace as a Python package.
- `benchmarks/meeting_summary/__init__.py` — exports the benchmark schema version.
- `benchmarks/meeting_summary/__main__.py` — module entry point.
- `benchmarks/meeting_summary/types.py` — immutable configuration and artifact dataclasses plus score calculation.
- `benchmarks/meeting_summary/config.py` — YAML loading, path expansion, semantic validation, and model definitions.
- `benchmarks/meeting_summary/pi_rpc.py` — one-shot Pi RPC subprocess client.
- `benchmarks/meeting_summary/storage.py` — run-directory creation, cache keys, atomic JSON/Markdown writes, and resume lookups.
- `benchmarks/meeting_summary/generation.py` — production-equivalent prompt composition and candidate execution.
- `benchmarks/meeting_summary/judging.py` — absolute judging, hard-failure extraction, pairwise comparison, and JSON validation.
- `benchmarks/meeting_summary/reporting.py` — aggregation, rankings, prompt diagnostics, Markdown, JSON, and CSV output.
- `benchmarks/meeting_summary/cli.py` — `validate`, `generate`, `judge`, `report`, and `all` commands.
- `benchmarks/meeting_summary/prompts/judge-v1.md` — absolute judge contract.
- `benchmarks/meeting_summary/prompts/pairwise-v1.md` — blinded pairwise judge contract.
- `benchmarks/meeting_summary/benchmark.yaml` — runnable model/case/prompt configuration containing paths only.
- `benchmarks/meeting_summary/README.md` — operating guide, artifact layout, and prompt-optimization workflow.

**Repository tests to create, after explicit test-file approval:**

- `tests/benchmarks/__init__.py`
- `tests/benchmarks/meeting_summary/__init__.py`
- `tests/benchmarks/meeting_summary/test_config.py`
- `tests/benchmarks/meeting_summary/test_pi_rpc.py`
- `tests/benchmarks/meeting_summary/test_generation.py`
- `tests/benchmarks/meeting_summary/test_judging.py`
- `tests/benchmarks/meeting_summary/test_reporting.py`
- `tests/benchmarks/meeting_summary/test_cli.py`
- `tests/benchmarks/meeting_summary/fixtures/fake_pi.py` — executable fake RPC server used instead of live models.

**External artifacts created only when the benchmark runs:**

- `~/cf-notes/benchmarks/meeting-summary/<run-id>/manifest.json`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/generations/**/*.json`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/generations/**/*.md`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/judgments/**/*.json`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/pairwise/**/*.json`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/report.json`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/report.csv`
- `~/cf-notes/benchmarks/meeting-summary/<run-id>/report.md`

---

### Task 1: Configuration, Domain Types, and Validation

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/meeting_summary/__init__.py`
- Create: `benchmarks/meeting_summary/types.py`
- Create: `benchmarks/meeting_summary/config.py`
- Create: `benchmarks/meeting_summary/benchmark.yaml`
- Create: `tests/benchmarks/__init__.py`
- Create: `tests/benchmarks/meeting_summary/__init__.py`
- Create: `tests/benchmarks/meeting_summary/test_config.py`

**Interfaces:**
- Produces: `BenchmarkConfig`, `ModelSpec`, `PromptSpec`, `CaseSpec`, `JudgeSpec`, `GenerationSpec`, `ScoreSet`, `load_benchmark_config(path: Path) -> BenchmarkConfig`, and `validate_benchmark_config(config: BenchmarkConfig) -> None`.
- Consumes: PyYAML and paths relative to the YAML file or expanded from `~`.

- [ ] **Step 1: Write configuration tests**

Create tests that prove path resolution, exact model loading, split validation, uniqueness rules, and missing-file failures:

```python
from pathlib import Path

import pytest

from benchmarks.meeting_summary.config import load_benchmark_config


def test_load_config_expands_paths_and_preserves_colons(tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    transcript = tmp_path / "transcript.md"
    golden = tmp_path / "golden.md"
    for path in (prompt, transcript, golden):
        path.write_text("content", encoding="utf-8")

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""
version: 1
output_dir: {tmp_path / 'results'}
generation:
  repetitions: 3
  thinking: off
  timeout_seconds: 900
prompts:
  - id: current
    path: {prompt}
cases:
  - id: sync
    transcript: {transcript}
    golden: {golden}
    split: development
models:
  - id: gemma12
    provider: homelab
    model: titan/ollama/gemma4:12b
    kind: candidate
judge:
  provider: openai-codex
  model: gpt-5.6-sol
  thinking: high
  timeout_seconds: 900
  pairwise_top_k: 3
""",
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.models[0].model == "titan/ollama/gemma4:12b"
    assert config.prompts[0].path == prompt
    assert config.cases[0].split == "development"
```

Also add tests named:

- `test_duplicate_model_ids_are_rejected`
- `test_duplicate_prompt_ids_are_rejected`
- `test_duplicate_case_ids_are_rejected`
- `test_invalid_split_is_rejected`
- `test_missing_prompt_is_rejected`
- `test_missing_transcript_is_rejected`
- `test_missing_golden_is_rejected`
- `test_repetitions_must_be_positive`
- `test_pairwise_top_k_must_be_at_least_two`

- [ ] **Step 2: Run the configuration tests and confirm the expected import failure**

Run:

```bash
pytest tests/benchmarks/meeting_summary/test_config.py -v
```

Expected: collection fails because `benchmarks.meeting_summary.config` does not exist.

- [ ] **Step 3: Add immutable domain types**

Implement frozen dataclasses in `types.py` with these exact fields:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

BENCHMARK_SCHEMA_VERSION = 1
VALID_SPLITS = frozenset({"development", "validation", "test"})
VALID_MODEL_KINDS = frozenset({"candidate", "baseline"})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    kind: str


@dataclass(frozen=True)
class PromptSpec:
    id: str
    path: Path


@dataclass(frozen=True)
class CaseSpec:
    id: str
    transcript: Path
    golden: Path
    split: str


@dataclass(frozen=True)
class GenerationSpec:
    repetitions: int
    thinking: str
    timeout_seconds: int


@dataclass(frozen=True)
class JudgeSpec:
    provider: str
    model: str
    thinking: str
    timeout_seconds: int
    pairwise_top_k: int


@dataclass(frozen=True)
class BenchmarkConfig:
    source: Path
    output_dir: Path
    generation: GenerationSpec
    prompts: Tuple[PromptSpec, ...]
    cases: Tuple[CaseSpec, ...]
    models: Tuple[ModelSpec, ...]
    judge: JudgeSpec


@dataclass(frozen=True)
class ScoreSet:
    factual_accuracy: int
    decisions_and_actions: int
    technical_detail_and_blockers: int
    structure_and_compliance: int
    concision_and_usefulness: int

    def weighted_total(self) -> float:
        values: Dict[str, float] = {
            "factual_accuracy": 0.35,
            "decisions_and_actions": 0.25,
            "technical_detail_and_blockers": 0.20,
            "structure_and_compliance": 0.10,
            "concision_and_usefulness": 0.10,
        }
        raw = sum(getattr(self, name) * weight for name, weight in values.items())
        return round(raw * 20, 2)
```

Each score must be an integer from 1 through 5; validate this when constructing judge results in Task 4.

- [ ] **Step 4: Implement YAML loading and semantic validation**

In `config.py`:

- Load with `yaml.safe_load`.
- Require `version: 1`.
- Resolve relative paths against `config_path.parent`.
- Apply `Path.expanduser()` before resolving.
- Preserve model IDs as strings, including `/`, `:`, and `.gguf`.
- Reject duplicate IDs in each collection.
- Reject empty model, prompt, or case collections.
- Require every configured prompt, transcript, and golden file to exist.
- Require positive timeouts and repetitions.
- Reject model kinds outside `candidate` and `baseline`.
- Reject case splits outside `development`, `validation`, and `test`.
- Return tuples rather than mutable lists.

Use explicit helper signatures:

```python
def load_benchmark_config(path: Path) -> BenchmarkConfig: ...
def validate_benchmark_config(config: BenchmarkConfig) -> None: ...
def resolve_config_path(value: str, config_dir: Path) -> Path: ...
```

- [ ] **Step 5: Add the runnable benchmark configuration**

Create `benchmark.yaml` with this exact initial matrix:

```yaml
version: 1
output_dir: ~/cf-notes/benchmarks/meeting-summary

generation:
  repetitions: 3
  thinking: off
  timeout_seconds: 900

prompts:
  - id: current
    path: ../../prompts/MEETING_PROMPT.md

cases:
  - id: core-ai-agents-sync
    transcript: ~/cf-notes/transcripts/2026-06-11-core-ai-agents-sync.md
    golden: ~/cf-notes/golden-summary.md
    split: development

models:
  - id: qwen35-9b
    provider: homelab
    model: m5/omlx/Qwen3.5-9B-OptiQ-4bit
    kind: candidate
  - id: ternary-bonsai-27b
    provider: homelab
    model: m5/omlx/Ternary-Bonsai-27B-mlx-2bit
    kind: candidate
  - id: gemma4-12b
    provider: homelab
    model: titan/ollama/gemma4:12b
    kind: candidate
  - id: gemma4-26b
    provider: homelab
    model: titan/ollama/gemma4:26b
    kind: candidate
  - id: gemma4-e4b
    provider: homelab
    model: m5/omlx/gemma-4-e4b-it-OptiQ-4bit
    kind: candidate
  - id: magistral
    provider: homelab
    model: titan/ollama/magistral:latest
    kind: candidate
  - id: muse-glimmer
    provider: homelab
    model: titan/ollama/muse-glimmer:latest
    kind: candidate
  - id: qwen38
    provider: homelab
    model: titan/ollama/qwen3.8:latest
    kind: candidate
  - id: strongbad-qwen36-35b
    provider: strongbad
    model: Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
    kind: candidate
  - id: luna-control
    provider: openai-codex
    model: gpt-5.6-luna
    kind: baseline

judge:
  provider: openai-codex
  model: gpt-5.6-sol
  thinking: high
  timeout_seconds: 900
  pairwise_top_k: 3
```

- [ ] **Step 6: Run configuration tests**

Run:

```bash
pytest tests/benchmarks/meeting_summary/test_config.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add benchmarks/__init__.py benchmarks/meeting_summary/__init__.py benchmarks/meeting_summary/types.py benchmarks/meeting_summary/config.py benchmarks/meeting_summary/benchmark.yaml tests/benchmarks

git commit -m "feat(benchmarks): define meeting summary benchmark"
```

---

### Task 2: Isolated Pi RPC Client

**Files:**
- Create: `benchmarks/meeting_summary/pi_rpc.py`
- Create: `tests/benchmarks/meeting_summary/fixtures/fake_pi.py`
- Create: `tests/benchmarks/meeting_summary/test_pi_rpc.py`

**Interfaces:**
- Produces: `PiRequest`, `PiResponse`, `PiRpcError`, and `PiRpcClient.run(request: PiRequest) -> PiResponse`.
- Consumes: one provider/model prompt and returns final assistant text, usage, elapsed time, model identity, stop reason, and captured stderr.

- [ ] **Step 1: Write RPC client tests using a fake Pi executable**

The fake server must read one JSONL `prompt` command, emit `response`, `agent_start`, `message_end`, and `agent_settled`, then answer `get_last_assistant_text` and `get_session_stats`. Tests must assert:

```python
from pathlib import Path

from benchmarks.meeting_summary.pi_rpc import PiRequest, PiRpcClient


def test_rpc_client_returns_text_usage_and_elapsed(fake_pi_path: Path):
    client = PiRpcClient(executable=str(fake_pi_path))
    response = client.run(
        PiRequest(
            provider="homelab",
            model="titan/ollama/gemma4:12b",
            thinking="off",
            prompt="summarize this",
            timeout_seconds=10,
        )
    )

    assert response.text == "# Summary"
    assert response.provider == "homelab"
    assert response.model == "titan/ollama/gemma4:12b"
    assert response.usage["input"] == 100
    assert response.usage["output"] == 20
    assert response.elapsed_seconds >= 0
```

Also add:

- `test_rpc_command_disables_all_ambient_context`
- `test_rpc_client_waits_for_agent_settled_not_agent_end`
- `test_rpc_client_raises_on_error_stop_reason`
- `test_rpc_client_raises_on_empty_assistant_text`
- `test_rpc_client_times_out_and_terminates_process`
- `test_rpc_client_surfaces_invalid_jsonl`
- `test_rpc_client_surfaces_early_process_exit`
- `test_rpc_client_captures_stderr_without_deadlock`

- [ ] **Step 2: Run the RPC tests and confirm the expected import failure**

```bash
pytest tests/benchmarks/meeting_summary/test_pi_rpc.py -v
```

Expected: collection fails because `pi_rpc.py` does not exist.

- [ ] **Step 3: Implement request and response types**

Use these exact shapes:

```python
@dataclass(frozen=True)
class PiRequest:
    provider: str
    model: str
    thinking: str
    prompt: str
    timeout_seconds: int


@dataclass(frozen=True)
class PiResponse:
    text: str
    provider: str
    model: str
    stop_reason: str
    usage: Dict[str, object]
    session_tokens: Dict[str, int]
    elapsed_seconds: float
    stderr: str
```

Define `PiRpcError(RuntimeError)` and include provider/model context in every raised message.

- [ ] **Step 4: Build the isolated Pi command**

`PiRpcClient.build_command()` must return:

```python
[
    executable,
    "--mode", "rpc",
    "--provider", request.provider,
    "--model", request.model,
    "--thinking", request.thinking,
    "--no-session",
    "--no-tools",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-context-files",
    "--system-prompt", "",
]
```

Set `PI_SKIP_VERSION_CHECK=1` and `PI_TELEMETRY=0` in the subprocess environment while preserving existing provider credentials and Pi configuration.

- [ ] **Step 5: Implement strict one-shot RPC execution**

The client must:

1. Start a fresh process for every request.
2. Drain stderr on a daemon thread into a bounded in-memory list.
3. Send `{"id":"benchmark-prompt","type":"prompt","message": request.prompt}` followed by LF.
4. Parse stdout as JSONL and wait for `agent_settled`.
5. Retain the completed assistant `message_end`, including `usage`, provider, model, and `stopReason`.
6. Send `get_last_assistant_text` and `get_session_stats` commands after settlement.
7. Reject `error`, `aborted`, and `length` stop reasons so truncated summaries do not score.
8. Reject empty text.
9. Enforce the request timeout using a reader queue and monotonic deadline.
10. Terminate and then kill the process if graceful shutdown exceeds two seconds.
11. Include captured stderr in raised errors and successful `PiResponse` artifacts.

Do not reuse a process between repetitions; that would contaminate context and usage.

- [ ] **Step 6: Run RPC client tests**

```bash
pytest tests/benchmarks/meeting_summary/test_pi_rpc.py -v
```

Expected: all tests pass without contacting a real model.

- [ ] **Step 7: Commit Task 2**

```bash
git add benchmarks/meeting_summary/pi_rpc.py tests/benchmarks/meeting_summary/fixtures/fake_pi.py tests/benchmarks/meeting_summary/test_pi_rpc.py

git commit -m "feat(benchmarks): add isolated Pi RPC runner"
```

---

### Task 3: Content-Addressed Generation and Resume

**Files:**
- Create: `benchmarks/meeting_summary/storage.py`
- Create: `benchmarks/meeting_summary/generation.py`
- Create: `tests/benchmarks/meeting_summary/test_generation.py`

**Interfaces:**
- Produces: `compose_meeting_prompt(prompt: str, transcript: str) -> str`, `GenerationJob`, `GenerationArtifact`, `RunStore`, and `generate_candidates(config, store, filters, client) -> Tuple[GenerationArtifact, ...]`.
- Consumes: validated config and `PiRpcClient`.

- [ ] **Step 1: Write production-equivalence and resume tests**

```python
from benchmarks.meeting_summary.generation import compose_meeting_prompt


def test_compose_meeting_prompt_matches_meeting_service():
    assert compose_meeting_prompt("PROMPT", "TRANSCRIPT") == (
        "PROMPT\n\n'''TRANSCRIPT'''\nTRANSCRIPT"
    )
```

Add tests that assert:

- Three repetitions create three distinct jobs.
- Candidate and baseline models are both generated.
- Filters select model, prompt, case, and split IDs without changing cache keys.
- Identical completed jobs are skipped on resume.
- Editing prompt content invalidates only jobs using that prompt.
- Editing transcript content invalidates only jobs using that transcript.
- A failed artifact is retried on resume.
- Markdown and JSON writes are atomic.
- Transcript and prompt content appear only under the external run directory, never in the repository manifest.

- [ ] **Step 2: Run generation tests and confirm expected failures**

```bash
pytest tests/benchmarks/meeting_summary/test_generation.py -v
```

Expected: collection fails because `generation.py` and `storage.py` do not exist.

- [ ] **Step 3: Implement run storage and stable cache keys**

`RunStore` must expose:

```python
class RunStore:
    @classmethod
    def create(cls, output_dir: Path, config_path: Path) -> "RunStore": ...

    @classmethod
    def open(cls, run_dir: Path) -> "RunStore": ...

    def find_completed(self, operation: str, cache_key: str) -> Optional[Path]: ...
    def write_json(self, relative_path: Path, payload: Dict[str, object]) -> Path: ...
    def write_text(self, relative_path: Path, content: str) -> Path: ...
    def read_json(self, path: Path) -> Dict[str, object]: ...
```

Create run IDs as UTC `YYYYMMDDTHHMMSSZ-<8-char-config-hash>`. Write through a temporary sibling followed by `Path.replace()`.

Hash canonical JSON with sorted keys and UTF-8 SHA-256. Include full prompt and transcript content in the hash input, but persist only their hashes in `manifest.json`.

- [ ] **Step 4: Implement generation jobs and artifacts**

Each `GenerationArtifact` JSON must contain:

```json
{
  "schema_version": 1,
  "operation": "generation",
  "cache_key": "sha256",
  "status": "complete",
  "case_id": "core-ai-agents-sync",
  "split": "development",
  "prompt_id": "current",
  "prompt_sha256": "sha256",
  "transcript_sha256": "sha256",
  "model_id": "gemma4-12b",
  "provider": "homelab",
  "model": "titan/ollama/gemma4:12b",
  "kind": "candidate",
  "thinking": "off",
  "repetition": 1,
  "elapsed_seconds": 1.23,
  "usage": {},
  "session_tokens": {},
  "stop_reason": "stop",
  "summary_path": "generations/.../summary.md",
  "stderr": ""
}
```

Store summary Markdown beside the JSON artifact. On errors, write a `status: failed` JSON artifact containing the exception text, then continue other jobs unless the caller passes `fail_fast=True`.

- [ ] **Step 5: Implement generation filtering**

Support optional sets for:

- `model_ids`
- `prompt_ids`
- `case_ids`
- `splits`

An unknown filter value must fail before any model process starts. This enables Phase 2 to run only the winning model against prompt variants and development/validation cases.

- [ ] **Step 6: Run generation tests**

```bash
pytest tests/benchmarks/meeting_summary/test_generation.py -v
```

Expected: all tests pass with a stub `PiRpcClient`.

- [ ] **Step 7: Commit Task 3**

```bash
git add benchmarks/meeting_summary/storage.py benchmarks/meeting_summary/generation.py tests/benchmarks/meeting_summary/test_generation.py

git commit -m "feat(benchmarks): generate resumable meeting summaries"
```

---

### Task 4: Absolute and Pairwise LLM Judging

**Files:**
- Create: `benchmarks/meeting_summary/judging.py`
- Create: `benchmarks/meeting_summary/prompts/judge-v1.md`
- Create: `benchmarks/meeting_summary/prompts/pairwise-v1.md`
- Create: `tests/benchmarks/meeting_summary/test_judging.py`

**Interfaces:**
- Produces: `JudgeResult`, `CriticalError`, `PairwiseResult`, `judge_generations(config, store, client)`, and `judge_pairwise_top_models(config, store, client)`.
- Consumes: complete generation artifacts, transcript, golden, and judge configuration.

- [ ] **Step 1: Write judge parsing and blindness tests**

Use a valid fixture result:

```python
VALID_JUDGMENT = {
    "scores": {
        "factual_accuracy": 5,
        "decisions_and_actions": 4,
        "technical_detail_and_blockers": 4,
        "structure_and_compliance": 5,
        "concision_and_usefulness": 4,
    },
    "critical_errors": [],
    "missed_items": ["S3 partitioning remained unresolved"],
    "failure_tags": ["missed_blocker"],
    "prompt_recommendations": [
        "Require unresolved architecture choices to be listed under blockers"
    ],
    "verdict": "Strong and factually reliable, with one missed blocker",
}
```

Tests must assert:

- Each score is an integer from 1 through 5.
- Unknown score fields and missing fields are rejected.
- Unknown failure tags are rejected.
- The weighted total is computed in Python, not trusted from model output.
- The judge prompt includes transcript, golden, and candidate.
- The judge prompt does not include candidate model/provider/ID.
- The judge instructions state that transcript evidence overrides the golden.
- Invalid JSON triggers one clean retry and then records failure.
- Critical errors remain separate from aggregate score.
- Pairwise A/B placement is deterministic from a stable hash and balanced across comparisons.
- Pairwise prompts contain no model identity.

- [ ] **Step 2: Run judge tests and confirm expected import failure**

```bash
pytest tests/benchmarks/meeting_summary/test_judging.py -v
```

Expected: collection fails because `judging.py` does not exist.

- [ ] **Step 3: Write the absolute judge prompt**

`judge-v1.md` must instruct the judge to return one JSON object and use this rubric:

- `factual_accuracy` — 35%; penalize unsupported claims and incorrect metrics, owners, dates, decisions, or terminology.
- `decisions_and_actions` — 25%; check explicit decisions, commitments, ownership, and timelines without converting suggestions into commitments.
- `technical_detail_and_blockers` — 20%; check concrete problems, constraints, dependencies, and relevant numbers.
- `structure_and_compliance` — 10%; check `MEETING_PROMPT.md` section rules and omission of empty/generic sections.
- `concision_and_usefulness` — 10%; reward information density and penalize repetition or transcript-like retelling.

Allow only these failure tags:

```text
hallucinated_fact
wrong_owner
wrong_timeline
proposal_as_decision
missed_decision
missed_action
missed_blocker
terminology_error
structure_violation
too_verbose
too_sparse
```

Require every critical error to contain `claim`, `transcript_evidence`, and `explanation`. Explicitly state: “The transcript is authoritative. The golden summary can be incomplete or wrong.”

Use unambiguous delimiter labels:

```text
<TRANSCRIPT>...</TRANSCRIPT>
<GOLDEN_SUMMARY>...</GOLDEN_SUMMARY>
<CANDIDATE_SUMMARY>...</CANDIDATE_SUMMARY>
```

- [ ] **Step 4: Implement strict judge-result parsing**

Define:

```python
@dataclass(frozen=True)
class CriticalError:
    claim: str
    transcript_evidence: str
    explanation: str


@dataclass(frozen=True)
class JudgeResult:
    scores: ScoreSet
    critical_errors: Tuple[CriticalError, ...]
    missed_items: Tuple[str, ...]
    failure_tags: Tuple[str, ...]
    prompt_recommendations: Tuple[str, ...]
    verdict: str
```

Strip a single outer Markdown code fence if present, parse JSON, reject extra top-level keys, and reject empty verdicts. One invalid response gets one fresh-process retry with a correction prefix that repeats the schema and includes the invalid response. Persist both raw attempts.

- [ ] **Step 5: Implement absolute judging and cache artifacts**

For every complete generation artifact:

- Re-read the source transcript and golden from config.
- Re-read the candidate summary from the external run directory.
- Render the anonymous judge prompt.
- Call configured `openai-codex/gpt-5.6-sol` with `thinking: high`.
- Compute weighted total in Python.
- Persist raw response, parsed result, judge usage, elapsed time, and cache key.
- Continue after individual judge failures.

- [ ] **Step 6: Implement top-model pairwise judging**

After absolute results exist:

1. Average totals per candidate model across repetitions, cases, and prompts within the selected filter.
2. Exclude baselines from the local-model top-K selection, but compare the top local model with `luna-control` once.
3. Select `pairwise_top_k` local models.
4. Compare every selected pair once per case/prompt using summary outputs matched by repetition.
5. Place each summary as A or B using the comparison cache hash.
6. Ask for `winner: A`, `winner: B`, or `winner: tie`, plus `reason`, `critical_difference`, and `confidence` from 1 through 5.
7. Persist normalized winner model IDs outside the raw judge response.

- [ ] **Step 7: Run judging tests**

```bash
pytest tests/benchmarks/meeting_summary/test_judging.py -v
```

Expected: all tests pass without live judge calls.

- [ ] **Step 8: Commit Task 4**

```bash
git add benchmarks/meeting_summary/judging.py benchmarks/meeting_summary/prompts tests/benchmarks/meeting_summary/test_judging.py

git commit -m "feat(benchmarks): add blinded LLM judging"
```

---

### Task 5: Reports and Prompt-Improvement Diagnostics

**Files:**
- Create: `benchmarks/meeting_summary/reporting.py`
- Create: `tests/benchmarks/meeting_summary/test_reporting.py`

**Interfaces:**
- Produces: `build_report(store: RunStore) -> BenchmarkReport` and `write_report(store: RunStore, report: BenchmarkReport) -> Tuple[Path, Path, Path]`.
- Consumes: generation, judgment, and pairwise artifacts.

- [ ] **Step 1: Write aggregation tests**

Create synthetic artifacts and assert:

- Model ranking uses mean weighted score.
- Standard deviation is reported for three repetitions.
- Critical error count is never averaged away.
- Mean latency and input/output tokens are reported.
- Pairwise wins and ties are included.
- Baseline delta is shown for each local model.
- Failed or missing jobs are listed and excluded from means.
- Failure tags are counted per model and prompt.
- Prompt recommendations are grouped by failure tag and deduplicated.
- The report never claims a prompt is better when only development cases exist.
- Validation and held-out test results are separated from development results.

- [ ] **Step 2: Run reporting tests and confirm expected import failure**

```bash
pytest tests/benchmarks/meeting_summary/test_reporting.py -v
```

Expected: collection fails because `reporting.py` does not exist.

- [ ] **Step 3: Implement report aggregation**

For each model/prompt/split combination calculate:

- completed and failed run counts
- mean and population standard deviation of weighted score
- mean of each rubric dimension
- total and per-run critical errors
- mean latency
- mean input and output tokens
- pairwise wins, losses, and ties
- delta from `luna-control`
- failure-tag counts

Keep all raw values in `report.json`. Use `statistics.fmean` and `statistics.pstdev`; report zero deviation for one sample.

- [ ] **Step 4: Implement prompt diagnostics**

The Markdown report must include:

1. Local model leaderboard.
2. Luna baseline comparison.
3. Critical factual failures by model.
4. Consistency across repetitions.
5. Runtime/token trade-offs.
6. Pairwise results.
7. Prompt failure categories.
8. Deduplicated prompt-edit recommendations.
9. Split coverage warning.

When only the development transcript is configured, print:

```text
Prompt recommendations are development-only. Add at least one validation case and one held-out test case before promoting a prompt.
```

Do not generate rewritten prompt text and do not modify prompt files.

- [ ] **Step 5: Write Markdown, JSON, and CSV atomically**

CSV columns must be stable:

```text
split,prompt_id,model_id,kind,completed_runs,failed_runs,mean_score,score_stddev,critical_errors,mean_factual_accuracy,mean_decisions_and_actions,mean_technical_detail_and_blockers,mean_structure_and_compliance,mean_concision_and_usefulness,mean_latency_seconds,mean_input_tokens,mean_output_tokens,pairwise_wins,pairwise_losses,pairwise_ties,baseline_delta
```

- [ ] **Step 6: Run reporting tests**

```bash
pytest tests/benchmarks/meeting_summary/test_reporting.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add benchmarks/meeting_summary/reporting.py tests/benchmarks/meeting_summary/test_reporting.py

git commit -m "feat(benchmarks): report quality and prompt diagnostics"
```

---

### Task 6: CLI, Fake End-to-End Test, and Operating Guide

**Files:**
- Create: `benchmarks/meeting_summary/cli.py`
- Create: `benchmarks/meeting_summary/__main__.py`
- Create: `benchmarks/meeting_summary/README.md`
- Create: `tests/benchmarks/meeting_summary/test_cli.py`

**Interfaces:**
- Produces: `python -m benchmarks.meeting_summary <command>`.
- Consumes: all prior task interfaces.

- [ ] **Step 1: Write CLI tests**

Required commands and assertions:

```bash
python -m benchmarks.meeting_summary validate --config benchmarks/meeting_summary/benchmark.yaml
python -m benchmarks.meeting_summary generate --config benchmarks/meeting_summary/benchmark.yaml
python -m benchmarks.meeting_summary judge --config benchmarks/meeting_summary/benchmark.yaml --run-dir <run-dir>
python -m benchmarks.meeting_summary report --run-dir <run-dir>
python -m benchmarks.meeting_summary all --config benchmarks/meeting_summary/benchmark.yaml
```

Tests must verify:

- `validate` performs no model calls.
- `generate` prints the run directory and preserves failed jobs.
- `judge` requires an existing run directory.
- `report` requires at least one complete judgment.
- `all` executes generation, absolute judging, pairwise judging, and reporting in order.
- `--resume <run-dir>` reuses completed artifacts.
- `--model`, `--prompt`, `--case`, and `--split` are repeatable filters.
- Unknown filter IDs fail before model execution.
- `--fail-fast` stops on first failed generation or judgment.
- Exit code is nonzero when any requested job fails, while completed artifacts and reports remain available.

- [ ] **Step 2: Run CLI tests and confirm expected import failure**

```bash
pytest tests/benchmarks/meeting_summary/test_cli.py -v
```

Expected: collection fails because `cli.py` does not exist.

- [ ] **Step 3: Implement argparse commands**

Use one parser with subcommands and shared options. Command handlers must return integer exit codes and `__main__.py` must call `raise SystemExit(main())`.

Exact filter behavior:

```bash
# Phase 1: current prompt, all development models, three repetitions
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --prompt current \
  --split development

# Resume an interrupted run
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --resume ~/cf-notes/benchmarks/meeting-summary/<run-id>

# Phase 2: winning local model against every configured prompt variant
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model <winning-model-id> \
  --split development \
  --split validation

# Final held-out check, run only after choosing a prompt
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --model <winning-model-id> \
  --prompt <winning-prompt-id> \
  --split test
```

The angle-bracket tokens above are documentation parameters supplied by the user at runtime, not code placeholders.

- [ ] **Step 4: Add a fake end-to-end CLI test**

Use `fake_pi.py` via a `PI_BENCHMARK_EXECUTABLE` environment override. Configure two fake candidate models, one fake baseline, one case, one prompt, and two repetitions. Run `all`, then assert that generation Markdown, judgment JSON, pairwise JSON, `report.md`, `report.json`, and `report.csv` exist and contain no source transcript outside the external temp run directory.

- [ ] **Step 5: Write the operating guide**

Document:

- prerequisites: `pi --list-models`, provider authentication, and Python environment
- config schema and the exact initial model matrix
- why Strongbad is a separate provider
- how candidate isolation works
- how to review and approve `golden-summary.md`
- Phase 1 model selection
- Phase 2 prompt variants
- development/validation/test split discipline
- resuming interrupted work
- interpreting hard factual failures versus aggregate score
- cloud judge use as offline evaluation only
- how to add a transcript without committing its content
- artifact retention and manual deletion

- [ ] **Step 6: Run focused benchmark tests**

```bash
pytest tests/benchmarks/meeting_summary -v
```

Expected: all tests pass and no live Pi model is contacted.

- [ ] **Step 7: Run repository lint and type checks when installed**

```bash
flake8 benchmarks/meeting_summary tests/benchmarks/meeting_summary
mypy benchmarks/meeting_summary
```

Expected: both pass. If either command is unavailable, record that fact and continue to the repository test suite.

- [ ] **Step 8: Run the full existing test suite**

```bash
pytest
```

Expected: all tests pass. Coverage remains scoped to existing `services` and `utils`; benchmark tests still execute.

- [ ] **Step 9: Perform a no-model smoke test**

```bash
python -m benchmarks.meeting_summary validate \
  --config benchmarks/meeting_summary/benchmark.yaml
```

Expected: configuration is valid and all current prompt/transcript/golden paths exist. This command must not launch Pi or write a run directory.

- [ ] **Step 10: Commit Task 6**

```bash
git add benchmarks/meeting_summary/cli.py benchmarks/meeting_summary/__main__.py benchmarks/meeting_summary/README.md tests/benchmarks/meeting_summary/test_cli.py

git commit -m "feat(benchmarks): add reusable benchmark CLI"
```

---

## Completion Audit

Before calling the implementation complete, map each requirement to evidence:

- All nine requested local models and the Luna control appear exactly in `benchmark.yaml`.
- Homelab and Strongbad providers resolve independently.
- Generation prompt bytes match `MeetingService.process_meeting_transcript()` composition.
- Every candidate call is isolated from Pi tools, skills, extensions, sessions, context files, and the coding prompt.
- Three repetitions are configured and reported separately.
- Raw summary, usage, latency, stop reason, and stderr are persisted.
- Resume invalidates changed prompts and transcripts by content hash.
- Absolute judging is anonymous and transcript-grounded.
- Critical factual failures cannot disappear into an average.
- Pairwise top-model comparison is blinded and A/B order is deterministic.
- Reports include quality, consistency, latency, tokens, baseline delta, and prompt failure categories.
- Development-only prompt recommendations carry an overfitting warning.
- The production meeting prompt is unchanged.
- No transcript or result content exists in Git-tracked files.
- Focused tests, full tests, available lint/type checks, and no-model validation all pass.

## Execution Strategy

Use subagent-driven implementation with one writer at a time:

1. Dispatch a fresh builtin `worker` for Task 1 with `model: openai-codex/gpt-5.6-luna`.
2. Parent reviews the diff and runs Task 1 checks.
3. Dispatch a fresh-context reviewer for correctness and scope.
4. Apply accepted fixes with one Luna worker if needed.
5. Repeat for Tasks 2–6.
6. Do not launch live benchmark models during implementation; the first real run is a separate, explicitly approved operation after the CLI passes its fake end-to-end tests.
