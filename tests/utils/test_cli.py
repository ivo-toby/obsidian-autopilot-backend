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
