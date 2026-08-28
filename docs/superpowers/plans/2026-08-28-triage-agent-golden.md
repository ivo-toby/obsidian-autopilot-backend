# Triage Agent Golden Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an evidence-first validation golden for the August 20 triage-agent kickoff and make it selectable by the meeting-summary benchmark.

**Architecture:** Keep the transcript and golden outside Git. Add only their paths, the case ID, and the validation split to the tracked benchmark configuration. Verify content against the authoritative transcript without invoking Pi or modifying prior run directories.

**Tech Stack:** Markdown, YAML, Python benchmark CLI, pytest, Git

## Global Constraints

- The transcript is authoritative; never invent or silently normalize facts.
- Store the golden at `/home/ivo/cf-notes/golden-summary-triage-agent.md`.
- Use `/home/ivo/cf-notes/transcripts/2026-08-20-jethro-ivo-triage-agent-kickoff.md` as the source transcript.
- Keep transcript and golden content out of Git.
- Configure case ID `triage-agent-kickoff` with split `validation`.
- Do not invoke Pi, run live models, or modify existing benchmark run directories.
- Do not create or change automated tests.

---

### Task 1: Create and audit the external golden

**Files:**
- Create: `/home/ivo/cf-notes/golden-summary-triage-agent.md`
- Read: `/home/ivo/cf-notes/transcripts/2026-08-20-jethro-ivo-triage-agent-kickoff.md`
- Read: `prompts/MEETING_PROMPT.md`

**Interfaces:**
- Consumes: the authoritative transcript and the current meeting-summary output contract.
- Produces: an external Markdown golden usable by `BenchmarkCase.golden_path`.

- [ ] **Step 1: Read the complete transcript and build a private evidence ledger**

Record timestamps for every participant, technical claim, decision, action, owner, date, timebox, blocker, dependency, and unresolved option. Do not save the ledger in the repository.

- [ ] **Step 2: Draft the golden in the external file**

Use only applicable sections from `prompts/MEETING_PROMPT.md`. Include only Jethro Dew and Ivo Toby as participants. Preserve the staged scope of triage/research, explanation, code changes, and later browser/demo work; distinguish selected requirements from implementation ideas; retain the explicit governance task, blocking date, and roughly two-week timebox without claiming unsupported relationships between them.

- [ ] **Step 3: Perform a claim-by-claim evidence audit**

Re-read the full golden against the transcript. Remove or qualify every statement that lacks direct evidence. Confirm that Cloudflare Sandbox, Mastra/software-factory approaches, Playwright, Windows VM/Codex, Slack integration, and Salesforce capabilities are not presented as selected solutions unless the transcript records a decision.

- [ ] **Step 4: Verify the external artifact**

Run:

```bash
test -s /home/ivo/cf-notes/golden-summary-triage-agent.md
sha256sum /home/ivo/cf-notes/golden-summary-triage-agent.md
rg -n '^#|^- ' /home/ivo/cf-notes/golden-summary-triage-agent.md
```

Expected: the file is non-empty, has a stable SHA-256, and follows the meeting-summary section contract.

### Task 2: Register and verify the validation case

**Files:**
- Modify: `benchmarks/meeting_summary/benchmark.yaml`
- Read: `/home/ivo/cf-notes/golden-summary-triage-agent.md`

**Interfaces:**
- Consumes: the external golden created in Task 1.
- Produces: benchmark case `triage-agent-kickoff`, selectable with `--case triage-agent-kickoff`.

- [ ] **Step 1: Capture prior-run immutability evidence**

Run before changing the config:

```bash
find /home/ivo/cf-notes/benchmarks/meeting-summary \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/meeting-summary-runs-before.sha256
```

Expected: a checksum manifest is written outside the repository.

- [ ] **Step 2: Add the validation case**

Add this entry after `core-ai-agents-sync` in `benchmarks/meeting_summary/benchmark.yaml`:

```yaml
  - id: triage-agent-kickoff
    transcript: ~/cf-notes/transcripts/2026-08-20-jethro-ivo-triage-agent-kickoff.md
    golden: ~/cf-notes/golden-summary-triage-agent.md
    split: validation
```

- [ ] **Step 3: Run offline validation**

Run:

```bash
python -m benchmarks.meeting_summary validate \
  --config benchmarks/meeting_summary/benchmark.yaml
pytest tests/benchmarks/meeting_summary/test_config.py \
  tests/benchmarks/meeting_summary/test_cli.py -q
```

Expected: configuration validation succeeds without starting Pi, and the focused tests pass.

- [ ] **Step 4: Verify privacy and prior-run immutability**

Run:

```bash
git diff --check
git diff -- benchmarks/meeting_summary/benchmark.yaml
find /home/ivo/cf-notes/benchmarks/meeting-summary \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/meeting-summary-runs-after.sha256
cmp /tmp/meeting-summary-runs-before.sha256 \
  /tmp/meeting-summary-runs-after.sha256
```

Expected: the Git diff contains external paths only, and all existing benchmark run artifacts are byte-identical.

- [ ] **Step 5: Commit the tracked configuration**

Run:

```bash
git add benchmarks/meeting_summary/benchmark.yaml
git commit -m "chore(benchmark): add triage validation case"
```

Expected: only the YAML path configuration is committed; `.pi-subagents/` remains untracked.

## Approved run command

From the feature worktree with the project Python environment:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --case triage-agent-kickoff \
  --prompt current
```

This selects only the validation case and current prompt, while retaining all configured models and three repetitions.
