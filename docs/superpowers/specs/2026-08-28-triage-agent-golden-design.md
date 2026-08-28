# Triage Agent Golden Summary Design

## Goal

Add the August 20 triage-agent kickoff as a second meeting-summary benchmark case. Use it to validate model and prompt choices made against the existing Core AI Agents Sync development case.

## External artifact

Create `/home/ivo/cf-notes/golden-summary-triage-agent.md` from `/home/ivo/cf-notes/transcripts/2026-08-20-jethro-ivo-triage-agent-kickoff.md`.

The golden must follow `prompts/MEETING_PROMPT.md` and treat the transcript as authoritative. It must:

- identify only Jethro Dew and Ivo Toby as participants;
- preserve the distinction between confirmed decisions, proposals, examples, and unresolved options;
- include owners, dates, and timing only when explicit;
- retain useful technical detail about triage, research, code execution, testing, governance controls, and the staged scope;
- omit the personal discussion near the end unless it materially affects the meeting outcome;
- avoid normalizing corrupted product names unless the intended name is unambiguous from the transcript.

## Benchmark configuration

Add one case to `benchmarks/meeting_summary/benchmark.yaml`:

- ID: `triage-agent-kickoff`
- Transcript: `~/cf-notes/transcripts/2026-08-20-jethro-ivo-triage-agent-kickoff.md`
- Golden: `~/cf-notes/golden-summary-triage-agent.md`
- Split: `validation`

No transcript or golden content belongs in Git.

## Verification

Verification must not invoke Pi or modify an existing benchmark run directory. It will:

1. validate the benchmark configuration;
2. verify the new golden is non-empty and record its SHA-256;
3. confirm the external transcript and existing benchmark artifacts are unchanged;
4. inspect the Git diff to ensure it contains paths only, not private transcript or golden content.

## Run command

Run only the new case with the current prompt:

```bash
python -m benchmarks.meeting_summary all \
  --config benchmarks/meeting_summary/benchmark.yaml \
  --case triage-agent-kickoff \
  --prompt current
```

The command must be run from the feature worktree with the project Python environment. It runs the configured model matrix with three repetitions and judges the resulting summaries.
