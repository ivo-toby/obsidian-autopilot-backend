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
