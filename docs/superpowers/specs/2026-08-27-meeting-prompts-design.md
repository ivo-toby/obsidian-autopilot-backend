# Meeting Prompts Design

## Purpose

Separate two functionally different workflows and make each prompt maintainable:

1. Move the daily-notes processing prompt out of Python and into `prompts/DAILY_NOTES.md` without changing its behavior.
2. Strengthen `prompts/MEETING_PROMPT.md` for full-transcript summaries using only the relevant structural strengths of the existing inline daily-notes prompt.

The meeting-summary benchmark remains exclusively focused on `MEETING_PROMPT.md`. Its golden summary will be recreated after the updated meeting prompt is implemented.

## Workflow boundaries

### Daily notes

`DAILY_NOTES.md` processes logs kept throughout the day. It identifies meeting or call material and returns zero or more structured notes through the existing `create_meeting_notes` function call.

This workflow is not part of the meeting-summary benchmark.

### Meeting summary

`MEETING_PROMPT.md` summarizes one full meeting transcript as Markdown. It is used by the clipboard transcript workflow and by the meeting-summary benchmark.

It does not identify meetings inside daily logs and does not use the daily-notes function schema.

## Scope

### Daily-notes prompt externalization

- Add `prompts/DAILY_NOTES.md` containing the static content of the current hardcoded prompt verbatim.
- Preserve the existing `create_meeting_notes` function schema.
- Preserve current structured results, saved Markdown headings, and filenames.
- Make `LLMService.generate_meeting_notes()` load the new file.
- Make legacy `OpenAIService.generate_meeting_notes()` load the same file.
- Allow `--prompt-file` to override the prompt for either selected workflow.

### Meeting-summary prompt improvement

Keep the current factual-quality guidance in `MEETING_PROMPT.md` and add the useful structural coverage from the inline prompt:

- explicit participants;
- concise topical tags;
- explicit references mentioned in the transcript;
- a clear home for substantive discussion notes;
- strict separation between discussion notes, decisions, and action items;
- a more explicit output-section contract.

Do not copy daily-log detection, multi-meeting extraction, function-calling instructions, or the generic inline examples into `MEETING_PROMPT.md`.

## Non-goals

- Do not combine the two prompts or workflows.
- Do not rewrite or improve `DAILY_NOTES.md` in this change.
- Do not change providers, models, inference settings, or function-calling behavior.
- Do not change daily-note output fields, Markdown rendering, or filenames.
- Do not add daily-note cases to the meeting-summary benchmark.
- Do not create or approve the replacement golden summary during implementation.
- Do not run live models during implementation.

## Architecture

### Daily-notes prompt boundary

Add `services/daily_notes_prompt.py` to:

- resolve the default `prompts/DAILY_NOTES.md` path;
- accept an optional custom prompt path;
- read the prompt as UTF-8;
- append the supplied daily-note text after the existing `Journal entries:` label;
- raise a clear file error before any LLM request.

`DAILY_NOTES.md` will contain all static prose from the current hardcoded prompt, ending with `Journal entries:`. Runtime composition adds only a newline and the input text. It will not use template interpolation, preserving literal braces in the existing examples.

The function schema remains in the LLM services because it is a provider request contract rather than prompt prose.

### Production service changes

`LLMService.generate_meeting_notes()` and legacy `OpenAIService.generate_meeting_notes()` gain an optional `prompt_file` argument and use the shared daily-notes composer. Their messages, function schema, forced function call, and response parsing remain unchanged.

`MeetingService.process_meeting_notes()` gains an optional `prompt_file` argument and passes it through. Missing or invalid structured output writes no files and produces a concise failure message instead of dereferencing `None`.

`MeetingService.process_meeting_transcript()` continues to use `MEETING_PROMPT.md` and raw Markdown generation.

### CLI behavior

`main.py` passes `--prompt-file` to whichever workflow is selected:

```text
notes --meetingnotes       -> prompts/DAILY_NOTES.md by default
notes --from-clipboard     -> prompts/MEETING_PROMPT.md by default
```

The CLI help text will describe `--prompt-file` as an override for the selected workflow.

### Updated meeting-summary prompt

`MEETING_PROMPT.md` keeps its existing role, principles, quality standards, and red flags. Its output contract will be sharpened as follows:

- `# [Meeting Name/Purpose]` remains the title.
- `## Tags` contains a small set of useful topical tags, not invented facts.
- `## Participants` lists only people explicitly identified in the transcript.
- `## Context`, `## Key Outcomes`, and technical/work-update sections retain their current purpose.
- `## Discussion Notes` captures substantive details that do not belong in technical challenges, decisions, or actions.
- `## Decisions Made` contains only actual decisions.
- `## Action Items` contains only concrete commitments, with owner and timing only when stated.
- `## References` contains only links, documents, systems, or external resources explicitly mentioned.
- blocker, coordination, and follow-up sections remain available under the existing “only substantial content” rule.

Empty, generic, unsupported, or redundant sections must be omitted. The prompt must explicitly prevent duplication of the same fact across Key Outcomes, Discussion Notes, Decisions, and Action Items.

## Benchmark behavior

The benchmark continues loading `MEETING_PROMPT.md` from `benchmark.yaml` and composing it with a full transcript. No daily-notes prompt or function schema enters the benchmark.

The existing content-addressed prompt hash automatically invalidates generation artifacts when `MEETING_PROMPT.md` changes. Candidate isolation, judging, pairwise comparison, reporting, and external private-data storage remain unchanged.

The existing golden summary becomes obsolete when the prompt changes. A new golden will be created and human-reviewed after implementation, before any live benchmark run.

## Data flow

### Daily-notes processing

1. CLI selects `notes --meetingnotes` and an optional prompt override.
2. `MeetingService` extracts the requested daily-note text.
3. `LLMService` loads `DAILY_NOTES.md` and appends the daily-note text.
4. The provider receives the unchanged messages, function schema, and forced function call.
5. Parsed meetings are rendered and saved through the existing code.

### Transcript summarization

1. CLI selects `notes --from-clipboard` and an optional prompt override.
2. `MeetingService` loads the updated `MEETING_PROMPT.md`.
3. The transcript is appended and sent through `generate_text()`.
4. Existing Markdown saving and topic inference continue unchanged.

### Benchmark

1. The benchmark loads the updated `MEETING_PROMPT.md` and external transcript.
2. Isolated Pi generates a Markdown summary.
3. Existing judge and report phases evaluate that summary.
4. No daily-notes code is invoked.

## Error handling

- Missing or unreadable `DAILY_NOTES.md` fails before an LLM request.
- Empty daily-note or clipboard input continues to produce no LLM request.
- Missing or malformed daily-note function output writes no files.
- A custom prompt path follows the same validation rules as its selected workflow.
- Meeting-summary prompt failures continue through the existing benchmark failure artifacts and nonzero exits.
- No partial daily-note file is written after extraction failure.

## Compatibility

- Existing callers that do not pass `prompt_file` continue to work.
- Existing daily-note function name, schema, required fields, Markdown output, and filenames remain unchanged.
- The clipboard workflow still emits Markdown and uses the same save path.
- Benchmark configuration still points only to `MEETING_PROMPT.md`.
- Python 3.8 compatibility is required.

## Testing

Use test-driven development. Tests must cover:

### Daily notes

- default loading from `prompts/DAILY_NOTES.md`;
- custom prompt loading;
- exact composition of prompt file content and daily-note text;
- missing prompt failure before an LLM call;
- `LLMService` and legacy `OpenAIService` using the external prompt;
- unchanged function schema and structured response parsing;
- `MeetingService` passing the daily override;
- no files written for invalid structured output;
- unchanged saved Markdown content.

### Meeting summary

- the updated prompt contains the approved section contract;
- participants and references are restricted to transcript evidence;
- discussion notes, decisions, and action items are explicitly separated;
- daily-log detection and function-calling instructions are absent;
- clipboard composition still uses `MEETING_PROMPT.md` or its override.

### Benchmark boundary

- benchmark configuration still points to `MEETING_PROMPT.md`;
- benchmark composition remains full-transcript Markdown summarization;
- no reference to `DAILY_NOTES.md` exists in benchmark generation;
- fake-Pi benchmark tests remain green without live models.

## Verification

```bash
python -m pytest tests/services/test_meeting_service.py tests/services/test_llm_service.py tests/services/test_openai_service.py -v
python -m pytest tests/benchmarks/meeting_summary -v
flake8 services/daily_notes_prompt.py services/meeting_service.py services/llm_service.py services/openai_service.py main.py utils/cli.py tests/services benchmarks/meeting_summary tests/benchmarks/meeting_summary
mypy benchmarks/meeting_summary
python -m benchmarks.meeting_summary validate --config benchmarks/meeting_summary/benchmark.yaml
```

Run the full repository suite and report the three known unrelated `max_tokens` assertion failures separately if they still exist.

## Acceptance criteria

- `prompts/DAILY_NOTES.md` contains the old hardcoded daily-notes prompt without behavioral edits.
- Neither `LLMService` nor `OpenAIService` contains the daily-notes prompt prose.
- Editing `DAILY_NOTES.md` changes daily-note processing without editing Python.
- Daily-note function calling, structured output, rendering, and filenames remain unchanged.
- `MEETING_PROMPT.md` keeps its stronger factual guidance and gains the approved structural sections.
- `MEETING_PROMPT.md` contains no daily-log or function-calling behavior.
- The benchmark remains exclusively tied to the updated `MEETING_PROMPT.md`.
- A replacement golden summary is required before the next live benchmark.
- No live model calls occur during implementation or automated tests.
