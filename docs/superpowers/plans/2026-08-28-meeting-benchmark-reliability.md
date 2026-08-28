# Meeting Benchmark Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make absolute judging follow the exact candidate prompt, make strict JSON retries actionable, and expose flushed per-job CLI progress throughout long benchmark runs.

**Architecture:** Absolute judging loads the content-addressed generation prompt snapshot and embeds it as the trusted summary contract, eliminating the duplicated section list. Generation and judging accept optional string progress callbacks; CLI commands supply a flushed stdout reporter while library callers remain quiet. Judge parsing stays strict, with the first parse error included in a more explicit single retry.

**Tech Stack:** Python 3.8+, argparse, pytest, fake Pi RPC fixture, content-addressed JSON/Markdown artifacts.

## Global Constraints

- Preserve Python 3.8 compatibility.
- Use strict TDD: write each regression test first, run it, and record the expected RED before modifying production code.
- Do not run live candidate or judge models.
- Do not modify `~/cf-notes/golden-summary.md`, transcripts, or existing external run directories.
- Preserve anonymous judging, strict result parsing, score weights, model configuration, timeouts, repetitions, pairwise selection, cache integrity, and exit codes.
- Do not normalize schema-invalid judge output.
- Keep core library calls quiet unless an optional progress callback is supplied.
- CLI progress must be flushed and must never print transcript, golden, candidate, or judge content.
- Do not fix the unrelated `gemma4-e4b` provider failure or the three unrelated `max_tokens` test failures.
- Keep `.pi-subagents/` untracked.

---

### Task 1: Prompt-snapshot judging and actionable strict retry

**Files:**
- Modify: `benchmarks/meeting_summary/prompts/judge-v1.md`
- Modify: `benchmarks/meeting_summary/judging.py`
- Modify: `tests/benchmarks/meeting_summary/test_judging.py`
- Modify: `tests/benchmarks/meeting_summary/test_final_review_fixes.py`

**Interfaces:**
- Consumes: generation prompt snapshots at `inputs/prompts/<prompt-id>-<prompt-sha256>.md` and `GenerationArtifact.prompt_id` / `prompt_sha256`.
- Produces: `render_judge_prompt(template: str, summary_instructions: str, transcript: str, golden: str, candidate: str) -> str`.
- Produces: `_retry_prompt(prompt: str, invalid_response: str, parse_error: str) -> str`.
- Preserves: `parse_judge_result(raw: str) -> JudgeResult` strict array and object contracts.

- [ ] **Step 1: Write failing prompt-snapshot contract tests**

Extend `test_prompt_is_anonymous_and_authoritative` or add a focused test that writes a generation prompt containing the newly allowed headings and then asserts the absolute judge request contains the exact prompt inside a trusted summary-instructions block:

```python
summary_prompt = (
    "## Tags\n## Participants\n## Discussion Notes\n## References"
)
config.prompts[0].path.write_text(summary_prompt, encoding="utf-8")
# Generate and judge with StubClient.
assert "<SUMMARY_INSTRUCTIONS>" in request.prompt
assert summary_prompt in request.prompt
assert "Available sections are `## Context`" not in request.prompt
```

Also assert a missing or corrupt prompt snapshot creates a persisted preflight judgment failure before any judge RPC, matching transcript/golden snapshot safety.

- [ ] **Step 2: Run the focused prompt tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_judging.py::test_prompt_is_anonymous_and_authoritative \
  tests/benchmarks/meeting_summary/test_final_review_fixes.py -k 'prompt or snapshot' -v
```

Expected: FAIL because the judge request does not include the generation prompt snapshot and the template still contains the stale hardcoded section list.

- [ ] **Step 3: Write failing retry-contract test**

Add a test whose first judge response uses object-valued `missed_items` and whose second response is valid. Assert the second request includes the exact parser error and explicit element-type rules:

```python
invalid = dict(VALID_JUDGMENT)
invalid["missed_items"] = [
    {"claim": "missing", "transcript_evidence": "evidence"}
]
client = StubClient([json.dumps(invalid), json.dumps(VALID_JUDGMENT)])
results = judge_generations(config, store, client)
retry = client.requests[1].prompt
assert "missed_items must be an array of strings" in retry
assert "plain JSON strings, never objects" in retry
assert results
```

Keep or extend the existing two-invalid-attempt test to prove the parser still rejects and persists invalid output.

- [ ] **Step 4: Run the retry tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_judging.py \
  -k 'retry or invalid_json' -v
```

Expected: FAIL because `_retry_prompt` does not receive the parser error and does not state string-array element types.

- [ ] **Step 5: Load and render the exact prompt snapshot**

In `judge_generations`, load the prompt snapshot before rendering:

```python
summary_instructions = _load_input_snapshot(
    store,
    "prompts",
    artifact.prompt_id,
    artifact.prompt_sha256,
)
prompt = render_judge_prompt(
    template,
    summary_instructions,
    transcript,
    golden,
    candidate,
)
```

Keep this read inside the existing preflight error boundary so missing, corrupt, or stale snapshots write a failed judgment and do not call Pi.

Update `render_judge_prompt` to substitute `summary_instructions`. Update `judge-v1.md` to:

- Treat only `SUMMARY_INSTRUCTIONS` as the trusted candidate contract.
- Keep transcript, golden, and candidate explicitly untrusted.
- Remove the duplicated list of allowed headings and duplicated decision/action wording.
- Include a non-empty example string in each string-array field so element types are unambiguous.
- State that `missed_items`, `failure_tags`, and `prompt_recommendations` must contain plain JSON strings, never objects.

- [ ] **Step 6: Include the exact parse failure in the one retry**

Track the first `JudgmentParseError` text and pass it to `_retry_prompt`:

```python
parse_error_message = ""
# On JudgmentParseError:
parse_error_message = str(parse_error)
# For attempt 2:
prompt=_retry_prompt(prompt, attempts[-1], parse_error_message)
```

The retry prompt must mark the parser error as trusted correction context and the previous response as untrusted data. It must repeat the exact top-level and nested element types without relaxing strict parsing.

- [ ] **Step 7: Run Task 1 tests GREEN**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_judging.py \
  tests/benchmarks/meeting_summary/test_final_review_fixes.py -v
```

Expected: all selected tests PASS with no live Pi process.

- [ ] **Step 8: Run Task 1 static checks and commit**

Run:

```bash
flake8 benchmarks/meeting_summary/judging.py \
  tests/benchmarks/meeting_summary/test_judging.py \
  tests/benchmarks/meeting_summary/test_final_review_fixes.py
black --check --line-length 79 benchmarks/meeting_summary/judging.py \
  tests/benchmarks/meeting_summary/test_judging.py \
  tests/benchmarks/meeting_summary/test_final_review_fixes.py
python -m py_compile benchmarks/meeting_summary/judging.py
```

Commit:

```bash
git add benchmarks/meeting_summary/prompts/judge-v1.md \
  benchmarks/meeting_summary/judging.py \
  tests/benchmarks/meeting_summary/test_judging.py \
  tests/benchmarks/meeting_summary/test_final_review_fixes.py
git commit -m "fix(benchmark): align judge with prompt snapshot"
```

---

### Task 2: Flushed per-job CLI progress

**Files:**
- Modify: `benchmarks/meeting_summary/generation.py`
- Modify: `benchmarks/meeting_summary/judging.py`
- Modify: `benchmarks/meeting_summary/cli.py`
- Modify: `benchmarks/meeting_summary/README.md`
- Modify: `tests/benchmarks/meeting_summary/test_generation.py`
- Modify: `tests/benchmarks/meeting_summary/test_judging.py`
- Modify: `tests/benchmarks/meeting_summary/test_cli.py`

**Interfaces:**
- Produces: optional `progress: Optional[Callable[[str], None]] = None` on `generate_candidates`, `judge_generations`, and `judge_pairwise_top_models`.
- Produces: CLI `_print_progress(message: str) -> None`, implemented with `print(message, flush=True)`.
- Message contract: `[phase] <name>` phase lines and `[<operation> <index>/<total>] <status> <identity-or-result>` job lines.
- Status values: `start`, `cached`, `complete`, and `failed`.

- [ ] **Step 1: Write failing generation progress tests**

Add focused tests using `messages = []` and `progress=messages.append`. Cover one uncached completion, one cached resume, and one failure. Assert stable identity and totals without private source text:

```python
assert messages[0] == (
    "[generation 1/2] start model=model-0 prompt=current "
    "case=case repetition=1"
)
assert messages[1].startswith("[generation 1/2] complete elapsed=")
assert "private transcript" not in "\n".join(messages)
```

For resume, assert each reused artifact emits one `cached` line and no `start` line. For failure, assert `failed error=` is emitted before continuing or raising under existing `fail_fast` behavior.

- [ ] **Step 2: Run generation progress tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_generation.py \
  -k progress -v
```

Expected: FAIL because `generate_candidates` has no progress callback.

- [ ] **Step 3: Implement optional generation progress**

Add a Python 3.8-safe callback type import and optional argument. Compute the selected job total from models, prompts, cases, and repetitions. Emit:

- `cached` for a verified cache hit.
- `start` immediately before `client.run`.
- `complete elapsed=<seconds>s` after persistence-ready success.
- `failed error=<single-line error>` after constructing the failure artifact.

Normalize embedded newlines in error text to spaces. Do not emit source content.

- [ ] **Step 4: Run generation progress tests GREEN**

Run the command from Step 2. Expected: all progress tests PASS.

- [ ] **Step 5: Write failing absolute and pairwise progress tests**

Add tests that pass `messages.append` to judging functions. Assert:

```python
assert messages[0].startswith("[absolute 1/2] start model=model-0")
assert messages[1].startswith("[absolute 1/2] complete elapsed=")
assert any("[absolute 2/2] cached" in item for item in resume_messages)
assert pairwise_messages[0].startswith("[pairwise 1/")
assert "models=" in pairwise_messages[0]
```

Cover failed absolute parsing and pairwise parsing so status is visible. Totals must reflect selected complete generation artifacts and actual matched pairwise comparison jobs.

- [ ] **Step 6: Run judging progress tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_judging.py \
  -k progress -v
```

Expected: FAIL because judging functions have no progress callback.

- [ ] **Step 7: Implement optional absolute and pairwise progress**

Add the optional callback to both public judging functions. For absolute judging, enumerate selected complete generation artifacts before the loop. For pairwise judging, build the matched comparison job list after ranking and before executing requests so the total is exact.

Emit `cached` without a preceding `start`; otherwise emit `start` and exactly one `complete` or `failed`. Use persisted payload elapsed time for completion/failure messages. Preserve all cache keys, ordering, anonymization, pairing, and fail-fast behavior.

- [ ] **Step 8: Run judging progress tests GREEN**

Run the command from Step 6. Expected: all progress tests PASS.

- [ ] **Step 9: Write failing CLI phase and flush tests**

Extend fake end-to-end CLI tests to assert phase order and representative job progress:

```python
assert result.stdout.index("[phase] generation") < result.stdout.index(
    "[phase] absolute judging"
)
assert result.stdout.index("[phase] absolute judging") < result.stdout.index(
    "[phase] pairwise judging"
)
assert result.stdout.index("[phase] pairwise judging") < result.stdout.index(
    "[phase] reporting"
)
assert "[generation 1/" in result.stdout
assert "[absolute 1/" in result.stdout
assert "[pairwise 1/" in result.stdout
assert "private transcript" not in result.stdout
```

Add a direct `_print_progress` test by monkeypatching `builtins.print` and asserting `flush=True`.

- [ ] **Step 10: Run CLI progress tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_cli.py -k 'progress or all_runs' -v
```

Expected: FAIL because CLI phases and flushed progress are not wired.

- [ ] **Step 11: Wire flushed CLI progress and document it**

Implement:

```python
def _print_progress(message: str) -> None:
    print(message, flush=True)
```

Pass `_print_progress` to generation and judging from `generate`, `judge`, and `all`. Emit phase lines before each phase. Emit `[reporting] complete path=<run-dir>` after report files are written. Preserve the existing final `Run directory`, `Judgments written`, and `Reports written` messages and exit statuses.

Update `benchmarks/meeting_summary/README.md` to explain that job progress is printed immediately, calls remain sequential, a job may remain active up to the configured timeout, and `--resume` reports cached work.

- [ ] **Step 12: Run Task 2 tests GREEN**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_generation.py \
  tests/benchmarks/meeting_summary/test_judging.py \
  tests/benchmarks/meeting_summary/test_cli.py -v
```

Expected: all selected tests PASS using fake/stub Pi only.

- [ ] **Step 13: Run all benchmark checks and commit**

Run:

```bash
python -m pytest -o addopts='' tests/benchmarks/meeting_summary -v
flake8 benchmarks/meeting_summary tests/benchmarks/meeting_summary
black --check --line-length 79 benchmarks/meeting_summary \
  tests/benchmarks/meeting_summary
mypy benchmarks/meeting_summary
find benchmarks/meeting_summary tests/benchmarks/meeting_summary \
  -type f -name '*.py' -print0 | xargs -0 python -m py_compile
python - <<'PY'
import ast
from pathlib import Path
for root in (Path('benchmarks/meeting_summary'), Path('tests/benchmarks/meeting_summary')):
    for path in root.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 8))
print('Python 3.8 grammar: PASS')
PY
```

Commit:

```bash
git add benchmarks/meeting_summary/generation.py \
  benchmarks/meeting_summary/judging.py \
  benchmarks/meeting_summary/cli.py \
  benchmarks/meeting_summary/README.md \
  tests/benchmarks/meeting_summary/test_generation.py \
  tests/benchmarks/meeting_summary/test_judging.py \
  tests/benchmarks/meeting_summary/test_cli.py
git commit -m "feat(benchmark): report live job progress"
```

---

## Final Verification

After both reviewed tasks:

```bash
python -m pytest -o addopts='' tests/benchmarks/meeting_summary -v
flake8 benchmarks/meeting_summary tests/benchmarks/meeting_summary
black --check --line-length 79 benchmarks/meeting_summary \
  tests/benchmarks/meeting_summary
mypy benchmarks/meeting_summary
find benchmarks/meeting_summary tests/benchmarks/meeting_summary \
  -type f -name '*.py' -print0 | xargs -0 python -m py_compile
python -m benchmarks.meeting_summary validate \
  --config benchmarks/meeting_summary/benchmark.yaml
```

Run the fake end-to-end workflow and inspect stdout for phase and job progress. Confirm no directory under `~/cf-notes/benchmarks/meeting-summary/` changed during implementation verification. Do not resume the live run until final review is clean and the user separately requests it.
