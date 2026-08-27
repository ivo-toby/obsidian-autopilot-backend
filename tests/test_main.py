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

    transcript_workflow = service_class.return_value.process_meeting_transcript
    transcript_workflow.assert_called_once_with(
        date_str="2024-02-18",
        dry_run=True,
        prompt_file="/tmp/custom.md",
    )
    service_class.return_value.process_meeting_notes.assert_not_called()
