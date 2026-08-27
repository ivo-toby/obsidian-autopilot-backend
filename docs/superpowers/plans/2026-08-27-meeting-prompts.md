# Meeting Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Externalize the daily-notes prompt without behavior changes and strengthen the separate full-transcript meeting prompt used by the existing benchmark.

**Architecture:** A focused `services/daily_notes_prompt.py` module loads and composes the new `prompts/DAILY_NOTES.md` file for both LLM service implementations. The clipboard summary path and benchmark remain tied only to `prompts/MEETING_PROMPT.md`, whose output contract gains participants, tags, references, and clearer separation of discussion, decisions, and actions.

**Tech Stack:** Python 3.8+, pathlib, pytest, unittest.mock, Markdown prompt files, existing provider function-calling abstraction, existing Pi RPC benchmark.

## Global Constraints

- `prompts/DAILY_NOTES.md` must contain the current hardcoded static prompt verbatim; its UTF-8 SHA-256 must be `f7cf47b7df86463d173832cdbbcb4e8f9cafe4480190c92e7317fd173291674e`.
- Daily-note function name, JSON schema, required fields, parsing, rendered Markdown headings, and filenames must not change.
- `prompts/MEETING_PROMPT.md` remains exclusively for one full meeting transcript and must not contain daily-log detection, multi-meeting extraction, or function-calling instructions.
- The benchmark remains exclusively tied to `prompts/MEETING_PROMPT.md`; do not add `DAILY_NOTES.md` to benchmark code or configuration.
- `--prompt-file` overrides the prompt for the selected workflow; defaults are `DAILY_NOTES.md` for `--meetingnotes` and `MEETING_PROMPT.md` for `--from-clipboard`.
- Keep Python 3.8 compatibility.
- Use strict TDD: add one focused test, observe the intended failure, implement minimally, then rerun.
- No automated test or implementation command may contact a live model.
- Do not create or modify the external golden summary in this plan.
- Do not fix the three unrelated `tests/services/test_openai_service.py` assertions that still expect removed `max_tokens` arguments.
- Do not stage or commit `.pi-subagents/`, transcripts, goldens, generated summaries, judgments, reports, or benchmark run directories.
- Preserve the current provider selection and inference configuration.

## File structure

- Create `prompts/DAILY_NOTES.md`: verbatim static daily-log processing instructions currently embedded in both LLM services.
- Create `services/daily_notes_prompt.py`: default path resolution, UTF-8 loading, and input composition only.
- Create `tests/services/test_daily_notes_prompt.py`: prompt-content, composition, custom path, and missing-file tests.
- Modify `services/llm_service.py`: consume the external daily prompt while preserving function calling.
- Modify `services/openai_service.py`: consume the same external daily prompt for legacy compatibility.
- Modify `services/meeting_service.py`: route a daily prompt override and handle absent structured output without writes.
- Modify `main.py`: pass `--prompt-file` to either selected workflow.
- Modify `utils/cli.py`: describe mode-specific prompt override behavior.
- Create `tests/test_main.py`: verify top-level workflow routing without live services.
- Modify `tests/services/test_meeting_service.py`: verify override propagation and invalid-output behavior.
- Create `tests/utils/test_cli.py`: verify parser behavior and help text.
- Modify `README.md`: document both default prompt files and mode-specific override behavior.
- Modify `CLAUDE.md`: document the external daily prompt boundary for future maintainers.
- Modify `prompts/MEETING_PROMPT.md`: strengthen the full-transcript output contract.
- Create `tests/benchmarks/meeting_summary/test_prompt_contract.py`: protect the prompt and benchmark boundary.

---

### Task 1: Externalize the Daily-Notes Prompt at the Service Layer

**Files:**
- Create: `prompts/DAILY_NOTES.md`
- Create: `services/daily_notes_prompt.py`
- Create: `tests/services/test_daily_notes_prompt.py`
- Modify: `services/llm_service.py:299-424`
- Modify: `services/openai_service.py:254-382`
- Modify: `tests/services/test_llm_service.py`
- Modify: `tests/services/test_openai_service.py:241-295`

**Interfaces:**
- Produces: `DEFAULT_DAILY_NOTES_PROMPT_PATH: pathlib.Path`
- Produces: `load_daily_notes_prompt(prompt_file: Optional[Union[str, pathlib.Path]] = None) -> str`
- Produces: `compose_daily_notes_prompt(notes: str, prompt_file: Optional[Union[str, pathlib.Path]] = None) -> str`
- Changes: `LLMService.generate_meeting_notes(notes: str, prompt_file: Optional[Union[str, pathlib.Path]] = None) -> Optional[Dict[str, List[Dict[str, Any]]]]`
- Changes: `OpenAIService.generate_meeting_notes(notes: str, prompt_file: Optional[Union[str, pathlib.Path]] = None) -> Optional[Dict[str, List[Dict[str, Any]]]]`
- Preserves: `create_meeting_notes` function name and existing schema in both services.

- [ ] **Step 1: Add failing prompt-loader tests**

Create `tests/services/test_daily_notes_prompt.py`:

```python
import hashlib
from pathlib import Path

import pytest

from services.daily_notes_prompt import (
    DEFAULT_DAILY_NOTES_PROMPT_PATH,
    compose_daily_notes_prompt,
    load_daily_notes_prompt,
)


EXPECTED_PROMPT_SHA256 = (
    "f7cf47b7df86463d173832cdbbcb4e8f"
    "9cafe4480190c92e7317fd173291674e"
)


def test_default_daily_notes_prompt_is_verbatim():
    content = load_daily_notes_prompt()

    assert DEFAULT_DAILY_NOTES_PROMPT_PATH.name == "DAILY_NOTES.md"
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == (
        EXPECTED_PROMPT_SHA256
    )
    assert content.endswith("Journal entries:\n")


def test_compose_daily_notes_prompt_appends_notes_without_reformatting():
    content = load_daily_notes_prompt()

    assert compose_daily_notes_prompt("[09:00] Log entry") == (
        content + "[09:00] Log entry"
    )


def test_custom_daily_notes_prompt_is_used_verbatim(tmp_path: Path):
    prompt = tmp_path / "daily.md"
    prompt.write_text("CUSTOM DAILY PREFIX\n", encoding="utf-8")

    assert load_daily_notes_prompt(prompt) == "CUSTOM DAILY PREFIX\n"
    assert compose_daily_notes_prompt("LOG", str(prompt)) == (
        "CUSTOM DAILY PREFIX\nLOG"
    )


def test_missing_daily_notes_prompt_fails_before_composition(tmp_path: Path):
    missing = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        compose_daily_notes_prompt("LOG", missing)
```

- [ ] **Step 2: Run the loader tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' tests/services/test_daily_notes_prompt.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'services.daily_notes_prompt'`.

- [ ] **Step 3: Create the verbatim prompt file**

Create `prompts/DAILY_NOTES.md` by extracting the static `JoinedStr` content from either existing `generate_meeting_notes()` implementation. The file must include the current leading newline, all current examples and literal braces, and the final newline after `Journal entries:`. Do not manually rewrite the prose.

Use this one-time extraction command before deleting the inline prompt:

```bash
python - <<'PY'
import ast
from pathlib import Path

source = Path("services/llm_service.py").read_text(encoding="utf-8")
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "generate_meeting_notes":
        assignment = next(
            item
            for item in node.body
            if isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prompt"
                for target in item.targets
            )
        )
        content = "".join(
            value.value
            for value in assignment.value.values
            if isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        )
        Path("prompts/DAILY_NOTES.md").write_text(
            content,
            encoding="utf-8",
        )
        break
else:
    raise SystemExit("generate_meeting_notes prompt not found")
PY
```

Verify:

```bash
sha256sum prompts/DAILY_NOTES.md
```

Expected first field: `f7cf47b7df86463d173832cdbbcb4e8f9cafe4480190c92e7317fd173291674e`.

- [ ] **Step 4: Implement the focused loader/composer**

Create `services/daily_notes_prompt.py`:

```python
from pathlib import Path
from typing import Optional, Union


PromptPath = Union[str, Path]
DEFAULT_DAILY_NOTES_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "DAILY_NOTES.md"
)


def load_daily_notes_prompt(
    prompt_file: Optional[PromptPath] = None,
) -> str:
    path = (
        Path(prompt_file).expanduser()
        if prompt_file is not None
        else DEFAULT_DAILY_NOTES_PROMPT_PATH
    )
    return path.read_text(encoding="utf-8")


def compose_daily_notes_prompt(
    notes: str,
    prompt_file: Optional[PromptPath] = None,
) -> str:
    return load_daily_notes_prompt(prompt_file) + notes
```

Do not strip whitespace, call `str.format`, or add another delimiter.

- [ ] **Step 5: Run loader tests and verify GREEN**

Run:

```bash
python -m pytest -o addopts='' tests/services/test_daily_notes_prompt.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Add failing service-level prompt tests**

Add this method inside `TestLLMService` in `tests/services/test_llm_service.py`. It creates a temporary custom prompt and verifies the provider receives exactly `CUSTOM DAILY PREFIX\nLOG`:

```python
    def test_generate_meeting_notes_uses_external_prompt(self, tmp_path):
        prompt = tmp_path / "daily.md"
        prompt.write_text("CUSTOM DAILY PREFIX\n", encoding="utf-8")
        config = {
            "inference": {
                "provider": "openai",
                "model": "gpt-4o",
                "openai": {"api_key": "test-key"},
            }
        }

        with patch("services.llm_service.OpenAIProvider") as provider_class:
            provider = Mock()
            provider.get_provider_name.return_value = "openai"
            provider.chat_completion_with_function.return_value = {
                "content": None,
                "function_call": {
                    "name": "create_meeting_notes",
                    "arguments": '{"meetings": []}',
                },
            }
            provider_class.return_value = provider
            service = LLMService(config)

            assert service.generate_meeting_notes(
                "LOG", prompt_file=prompt
            ) == {"meetings": []}

        messages, functions, function_call = (
            provider.chat_completion_with_function.call_args.args[:3]
        )
        assert messages[1]["content"] == "CUSTOM DAILY PREFIX\nLOG"
        assert functions[0]["name"] == "create_meeting_notes"
        assert function_call == {"name": "create_meeting_notes"}
```

Extend `test_generate_meeting_notes_success` in `tests/services/test_openai_service.py` by adding `tmp_path` to its fixture arguments and these exact setup/call/assertion changes while retaining its existing result and schema assertions:

```python
custom_prompt = tmp_path / "daily.md"
custom_prompt.write_text("CUSTOM DAILY PREFIX\n", encoding="utf-8")

result = openai_service.generate_meeting_notes(
    notes,
    prompt_file=custom_prompt,
)

call_args = mock_openai_client.chat.completions.create.call_args[1]
user_message = next(
    message["content"]
    for message in call_args["messages"]
    if message["role"] == "user"
)
assert user_message == "CUSTOM DAILY PREFIX\n" + notes
assert call_args["functions"][0]["name"] == "create_meeting_notes"
assert call_args["function_call"] == {"name": "create_meeting_notes"}
```

- [ ] **Step 7: Run the focused service tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/services/test_llm_service.py::TestLLMService::test_generate_meeting_notes_uses_external_prompt \
  tests/services/test_openai_service.py::test_generate_meeting_notes_success \
  -v
```

Expected: both tests fail because `prompt_file` is not accepted or the inline prompt is still used.

- [ ] **Step 8: Replace both inline prompts with the composer**

In `services/llm_service.py`:

```python
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from services.daily_notes_prompt import compose_daily_notes_prompt
```

Change the method signature and first statement:

```python
def generate_meeting_notes(
    self,
    notes: str,
    prompt_file: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    prompt = compose_daily_notes_prompt(notes, prompt_file)
```

Delete only the inline prompt f-string. Leave the system message, function list, forced function call, and response parsing unchanged.

Apply the same import, signature, and first statement in `services/openai_service.py`. Delete its duplicate inline prompt and leave its OpenAI request contract unchanged.

- [ ] **Step 9: Prove the hardcoded prompt prose is gone**

Run:

```bash
rg -n "From the following journal entries, infer which entries" \
  services/llm_service.py services/openai_service.py
```

Expected: no matches.

Run:

```bash
python -m pytest -o addopts='' \
  tests/services/test_daily_notes_prompt.py \
  tests/services/test_llm_service.py::TestLLMService::test_generate_meeting_notes_uses_external_prompt \
  tests/services/test_openai_service.py::test_generate_meeting_notes_success \
  tests/services/test_openai_service.py::test_generate_meeting_notes_error \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 10: Run focused quality checks**

Run:

```bash
flake8 services/daily_notes_prompt.py \
  tests/services/test_daily_notes_prompt.py
flake8 --ignore=E501 services/llm_service.py services/openai_service.py
python -m py_compile services/daily_notes_prompt.py \
  services/llm_service.py services/openai_service.py
python - <<'PY'
import ast
from pathlib import Path

for name in (
    "services/daily_notes_prompt.py",
    "services/llm_service.py",
    "services/openai_service.py",
):
    ast.parse(
        Path(name).read_text(encoding="utf-8"),
        filename=name,
        feature_version=(3, 8),
    )
print("Python 3.8 grammar passed")
PY
```

Expected: all commands exit 0.

- [ ] **Step 11: Commit Task 1**

```bash
git add prompts/DAILY_NOTES.md services/daily_notes_prompt.py \
  services/llm_service.py services/openai_service.py \
  tests/services/test_daily_notes_prompt.py \
  tests/services/test_llm_service.py tests/services/test_openai_service.py
git commit -m "refactor(prompts): externalize daily notes prompt"
```

---

### Task 2: Route Prompt Overrides Through the Daily-Notes Workflow

**Files:**
- Modify: `services/meeting_service.py:32-59`
- Modify: `main.py:50-63`
- Modify: `utils/cli.py:28-66`
- Modify: `tests/services/test_meeting_service.py:107-177`
- Create: `tests/test_main.py`
- Create: `tests/utils/test_cli.py`
- Modify: `README.md:40-60,190-205`
- Modify: `CLAUDE.md:89-105,154-162`

**Interfaces:**
- Consumes: `LLMService.generate_meeting_notes(notes: str, prompt_file: Optional[Union[str, Path]] = None)` from Task 1.
- Changes: `MeetingService.process_meeting_notes(date_str: Optional[str] = None, dry_run: bool = False, prompt_file: Optional[str] = None) -> None`.
- Preserves: `MeetingService.process_meeting_transcript(..., prompt_file: Optional[str] = None) -> None`.
- Produces: `main.process_meeting_notes()` passes the CLI override to exactly one selected workflow.

- [ ] **Step 1: Add failing MeetingService routing and failure tests**

In `tests/services/test_meeting_service.py`, update `test_process_meeting_notes` to call:

```python
meeting_service.process_meeting_notes(prompt_file="/tmp/daily.md")
```

Change its final assertion to:

```python
meeting_service.llm_service.generate_meeting_notes.assert_called_once_with(
    mock_notes,
    prompt_file="/tmp/daily.md",
)
```

Add:

```python
def test_process_meeting_notes_invalid_output_writes_nothing(
    meeting_service, temp_dir, capsys
):
    meeting_service.notes_service.load_notes = Mock(return_value="LOG")
    meeting_service.notes_service.extract_today_notes = Mock(
        return_value="LOG"
    )
    meeting_service.llm_service.generate_meeting_notes = Mock(
        return_value=None
    )

    meeting_service.process_meeting_notes()

    assert list(Path(temp_dir).glob("*.md")) == []
    assert "No structured notes were generated" in capsys.readouterr().out
```

- [ ] **Step 2: Add failing top-level routing tests**

Create `tests/test_main.py`:

```python
from argparse import Namespace
from unittest.mock import patch

from main import process_meeting_notes


def _args(from_clipboard):
    return Namespace(
        from_clipboard=from_clipboard,
        prompt_file="/tmp/custom.md",
        date="2024-02-18",
        dry_run=True,
    )


def test_process_meeting_notes_routes_override_to_daily_workflow():
    with patch("main.MeetingService") as service_class:
        process_meeting_notes({}, _args(from_clipboard=False))

    service_class.return_value.process_meeting_notes.assert_called_once_with(
        date_str="2024-02-18",
        dry_run=True,
        prompt_file="/tmp/custom.md",
    )
    service_class.return_value.process_meeting_transcript.assert_not_called()


def test_process_meeting_notes_routes_override_to_transcript_workflow():
    with patch("main.MeetingService") as service_class:
        process_meeting_notes({}, _args(from_clipboard=True))

    service_class.return_value.process_meeting_transcript.assert_called_once_with(
        date_str="2024-02-18",
        dry_run=True,
        prompt_file="/tmp/custom.md",
    )
    service_class.return_value.process_meeting_notes.assert_not_called()
```

- [ ] **Step 3: Add a failing CLI parser contract test**

Create `tests/utils/test_cli.py`:

```python
import pytest

from utils.cli import setup_argparser


def test_prompt_file_overrides_selected_notes_workflow():
    parser = setup_argparser()
    args = parser.parse_args(
        [
            "notes",
            "--meetingnotes",
            "--prompt-file",
            "/tmp/custom.md",
        ]
    )

    assert args.meetingnotes is True
    assert args.prompt_file == "/tmp/custom.md"


def test_notes_help_describes_selected_workflow_override(capsys):
    parser = setup_argparser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["notes", "--help"])

    assert error.value.code == 0
    assert "selected notes workflow" in capsys.readouterr().out.lower()
```

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/services/test_meeting_service.py::test_process_meeting_notes \
  tests/services/test_meeting_service.py::test_process_meeting_notes_invalid_output_writes_nothing \
  tests/test_main.py tests/utils/test_cli.py -v
```

Expected: failures show that daily `prompt_file` is not accepted or forwarded, `None` is dereferenced, and help text still describes only transcript mode.

- [ ] **Step 5: Implement daily override routing and safe empty output**

In `services/meeting_service.py`, change the signature:

```python
def process_meeting_notes(
    self,
    date_str: Optional[str] = None,
    dry_run: bool = False,
    prompt_file: Optional[str] = None,
) -> None:
```

Call the LLM service with:

```python
meeting_notes = self.llm_service.generate_meeting_notes(
    today_notes,
    prompt_file=prompt_file,
)
if meeting_notes is None:
    print("No structured notes were generated.")
    return
```

Do not treat `{"meetings": []}` as an error; zero extracted notes is valid.

In `main.py`, read `prompt_file` before branching and pass it to either `process_meeting_transcript()` or `process_meeting_notes()`.

In `utils/cli.py`, replace the current help text with:

```python
help="Path to a custom prompt file for the selected notes workflow"
```

Because `tests/services/test_meeting_service.py` is touched, remove its unused `os` and `datetime` imports and ensure the file ends with a newline. Do not reformat unrelated lines.

- [ ] **Step 6: Run the routing tests and verify GREEN**

Run:

```bash
python -m pytest -o addopts='' \
  tests/services/test_meeting_service.py::test_process_meeting_notes \
  tests/services/test_meeting_service.py::test_process_meeting_notes_invalid_output_writes_nothing \
  tests/test_main.py tests/utils/test_cli.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Update user and maintainer documentation**

In `README.md`, document these defaults next to the existing commands:

```text
notes --meetingnotes uses prompts/DAILY_NOTES.md to process daily logs into structured notes.
notes --from-clipboard uses prompts/MEETING_PROMPT.md to summarize one full transcript.
--prompt-file overrides the default prompt for whichever workflow is selected.
```

In `CLAUDE.md`, update the two meeting workflow descriptions with the same file boundary. Do not describe daily-note processing as part of the benchmark.

- [ ] **Step 8: Run Task 2 checks**

Run:

```bash
python -m pytest -o addopts='' \
  tests/services/test_meeting_service.py tests/test_main.py \
  tests/utils/test_cli.py -v
flake8 tests/test_main.py tests/utils/test_cli.py
flake8 --ignore=E501 services/meeting_service.py main.py utils/cli.py \
  tests/services/test_meeting_service.py
python -m py_compile services/meeting_service.py main.py utils/cli.py
```

Expected: all commands exit 0.

- [ ] **Step 9: Commit Task 2**

```bash
git add services/meeting_service.py main.py utils/cli.py \
  tests/services/test_meeting_service.py tests/test_main.py \
  tests/utils/test_cli.py README.md CLAUDE.md
git commit -m "fix(notes): route workflow prompt overrides"
```

---

### Task 3: Strengthen the Full-Transcript Meeting Prompt

**Files:**
- Modify: `prompts/MEETING_PROMPT.md`
- Create: `tests/benchmarks/meeting_summary/test_prompt_contract.py`
- Verify unchanged: `benchmarks/meeting_summary/benchmark.yaml`
- Verify unchanged: `benchmarks/meeting_summary/generation.py`

**Interfaces:**
- Consumes: existing benchmark prompt loading through `PromptSpec.path`.
- Preserves: `compose_meeting_prompt(prompt: str, transcript: str) -> str`.
- Produces: a Markdown-only prompt contract for exactly one full transcript.
- Produces: prompt-content tests that prevent daily-note behavior from entering the benchmark.

- [ ] **Step 1: Add failing prompt-contract tests**

Create `tests/benchmarks/meeting_summary/test_prompt_contract.py`:

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
MEETING_PROMPT = ROOT / "prompts" / "MEETING_PROMPT.md"
BENCHMARK_CONFIG = (
    ROOT / "benchmarks" / "meeting_summary" / "benchmark.yaml"
)


def test_meeting_prompt_has_approved_summary_sections():
    prompt = MEETING_PROMPT.read_text(encoding="utf-8")

    for heading in (
        "## Tags",
        "## Participants",
        "## Context",
        "## Key Outcomes",
        "## Discussion Notes",
        "## Decisions Made",
        "## Action Items",
        "## References",
    ):
        assert heading in prompt

    assert "only people explicitly identified" in prompt.lower()
    assert "only concrete commitments" in prompt.lower()
    assert "only references explicitly mentioned" in prompt.lower()
    assert "do not repeat the same fact" in prompt.lower()


def test_meeting_prompt_excludes_daily_notes_behavior():
    prompt = MEETING_PROMPT.read_text(encoding="utf-8").lower()

    for forbidden in (
        "journal entries",
        "infer which entries",
        "create_meeting_notes",
        "function call",
        "multiple meetings",
    ):
        assert forbidden not in prompt


def test_benchmark_still_targets_only_meeting_prompt():
    payload = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))

    assert payload["prompts"] == [
        {"id": "current", "path": "../../prompts/MEETING_PROMPT.md"}
    ]
    assert "DAILY_NOTES" not in BENCHMARK_CONFIG.read_text(
        encoding="utf-8"
    )
```

- [ ] **Step 2: Run prompt-contract tests and verify RED**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_prompt_contract.py -v
```

Expected: `test_meeting_prompt_has_approved_summary_sections` fails because Tags, Participants, Discussion Notes, and References are absent. Boundary assertions that already describe current behavior may pass.

- [ ] **Step 3: Replace `MEETING_PROMPT.md` with the approved summary contract**

Use the following complete prompt text. Preserve the Markdown headings and evidence language exactly because tests and judge expectations depend on them:

```markdown
You are an expert meeting summarizer specializing in technical team meetings. Create a precise, actionable summary of one full meeting transcript. Capture the real substance, technical challenges, and concrete outcomes without inventing missing details.

## Core Principles

- **Capture specifics, not generalities**: Include actual technical details, specific problems, and concrete solutions discussed.
- **Focus on what matters**: Prioritize decisions, concrete actions, blockers, dependencies, and meaningful discussion over process chatter.
- **Use precise language**: Preserve the terminology used in the transcript. Avoid vague business language.
- **Preserve technical context**: Include enough detail for someone who missed the meeting to understand the real constraints and proposals.
- **Include concrete data**: Preserve numbers, costs, success rates, timelines, capacity, and other metrics exactly as stated.
- **Stay grounded**: The transcript is authoritative. Never invent participants, owners, deadlines, decisions, metrics, references, or blockers.

## Evidence Rules

1. Distinguish a confirmed decision from an idea, suggestion, question, or unresolved discussion.
2. Include an action item only when the transcript contains a concrete commitment.
3. Include an owner or deadline only when it is explicitly stated.
4. List only people explicitly identified in the transcript. Do not infer attendance from context.
5. Include only references explicitly mentioned in the transcript, such as links, documents, repositories, tickets, systems, or external resources.
6. Tags may summarize explicit meeting topics, but they must not introduce unsupported claims.
7. Do not repeat the same fact across Key Outcomes, Discussion Notes, Decisions Made, and Action Items. Put it in the most specific section.
8. Omit any section that would be empty, generic, unsupported, or redundant.

## Output Contract

Use only the sections below that contain specific, useful information.

# [Meeting Name/Purpose]

## Tags
[Three to six concise topical tags, prefixed with `#`.]

## Participants
[Only people explicitly identified in the transcript. Add a role only when stated.]

## Context
[Why the meeting happened and what prompted it.]

## Key Outcomes
[The most important results: confirmed decisions, problems identified, and agreed next steps. Keep detailed decisions and actions in their dedicated sections.]

## Discussion Notes
[Substantive discussion details that do not belong in Technical Challenges, Decisions Made, or Action Items. Group by topic when useful.]

## Technical Challenges Discussed
[Specific technical problems, their impact, current state, and proposed solutions.]

### [Specific Challenge Name]
- **Problem**: [Exact issue described.]
- **Impact**: [Effect on the team or product, including stated metrics.]
- **Current State**: [Known status, costs, rates, capacity, or constraints.]
- **Proposed Solutions**: [Specific approaches discussed.]
- **Status**: [Current conclusion or unresolved next step.]

## Sprint/Work Updates
[Only significant completions, blockers, ownership changes, or changes in direction.]

## Decisions Made
[Only confirmed decisions.]
- [Decision] — [Rationale, when stated] — [Implementer, only when stated]

## Action Items
[Only concrete commitments.]
- [Specific action] — [Owner, only when stated] — [Timing, only when stated]

## Blockers & Dependencies
[What is preventing progress and what must happen to unblock it.]

## Team Coordination Notes
[Meaningful coordination changes, collaboration issues, friction, or alignment.]

## References
[Only links, documents, repositories, tickets, systems, or resources explicitly mentioned.]

## Follow-up Required
[Specific unresolved questions or topics requiring another decision or discussion.]

## Quality Standards

- Prefer exact details over generalized summaries.
- Preserve uncertainty. Do not turn tentative language into certainty.
- Preserve the difference between who proposed work and who committed to doing it.
- Use the actual technical terms from the transcript.
- Include stated numbers and metrics with their original context.
- Keep the summary concise by removing repetition and low-value process narration.
- Skip a section rather than filling it with generic text.

## Red Flags to Avoid

- Generic phrases such as “align on objectives,” “enhance workflow efficiency,” or “explored improvements.”
- Vague action items without a real commitment.
- Invented owners, dates, decisions, metrics, participants, or references.
- Treating proposals or questions as decisions.
- Sanitizing away technical difficulty, uncertainty, disagreement, or blockers.
- Repeating the same outcome in several sections.
```

- [ ] **Step 4: Run prompt-contract and composition tests**

Run:

```bash
python -m pytest -o addopts='' \
  tests/benchmarks/meeting_summary/test_prompt_contract.py \
  tests/benchmarks/meeting_summary/test_generation.py::test_compose_meeting_prompt_matches_meeting_service \
  -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the complete fake-only benchmark suite**

Run:

```bash
python -m pytest tests/benchmarks/meeting_summary -v
```

Expected: 85 tests pass: the previous 82 plus the 3 new prompt-contract tests. No live model process is contacted.

- [ ] **Step 6: Validate the benchmark without creating a run**

Run:

```bash
before=$(find ~/cf-notes/benchmarks/meeting-summary \
  -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
python -m benchmarks.meeting_summary validate \
  --config benchmarks/meeting_summary/benchmark.yaml
after=$(find ~/cf-notes/benchmarks/meeting-summary \
  -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
test "$before" = "$after"
```

Expected: configuration is valid and directory counts are equal.

- [ ] **Step 7: Run final focused checks**

Run:

```bash
flake8 services/daily_notes_prompt.py \
  tests/services/test_daily_notes_prompt.py tests/test_main.py \
  tests/utils/test_cli.py benchmarks/meeting_summary \
  tests/benchmarks/meeting_summary
flake8 --ignore=E501 services/meeting_service.py \
  services/llm_service.py services/openai_service.py main.py utils/cli.py \
  tests/services/test_meeting_service.py
black --check --line-length 79 benchmarks/meeting_summary \
  tests/benchmarks/meeting_summary
mypy benchmarks/meeting_summary
find services benchmarks/meeting_summary tests/benchmarks/meeting_summary \
  -type f -name '*.py' -print0 | xargs -0 python -m py_compile
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Run the full repository suite and classify only known failures**

Run:

```bash
python -m pytest
```

Expected current baseline: 3 failures in `tests/services/test_openai_service.py` because those tests still assert removed `max_tokens` values; all prompt and benchmark tests pass. Any additional failure blocks completion.

- [ ] **Step 9: Verify private-data and prompt boundaries**

Run:

```bash
test -z "$(git ls-files | grep -E \
  '(^|/)(\.pi-subagents|cf-notes|transcripts|golden-summary|runs|results)(/|$)' \
  || true)"
test -z "$(rg -l 'DAILY_NOTES' benchmarks/meeting_summary || true)"
git diff --name-only HEAD~2 -- prompts/MEETING_PROMPT.md \
  prompts/DAILY_NOTES.md services tests README.md CLAUDE.md main.py utils/cli.py
```

Expected: no forbidden tracked paths, no `DAILY_NOTES` reference in benchmark production code, and only planned files in the change list.

- [ ] **Step 10: Commit Task 3**

```bash
git add prompts/MEETING_PROMPT.md \
  tests/benchmarks/meeting_summary/test_prompt_contract.py
git commit -m "feat(prompts): strengthen meeting summary contract"
```

- [ ] **Step 11: Record the golden-summary handoff**

Do not edit `~/cf-notes/golden-summary.md`. Report that the old golden is obsolete and that a new human-reviewed golden must be created before running live candidate or judge models.
