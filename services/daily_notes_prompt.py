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
