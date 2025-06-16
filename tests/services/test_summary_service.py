"""
Tests for the summary service.

This module contains tests for the summary service functionality,
including daily and weekly note processing.
"""

import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest
from pathlib import Path

from services.summary_service import SummaryService


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = MagicMock()
    return mock


@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        "daily_notes_file": "~/notes/daily.md",
        "daily_output_dir": "~/notes/daily",
        "weekly_output_dir": "~/notes/weekly",
        "notes_base_dir": "~/notes",
        "api_key": "test-api-key",
        "model": "test-model",
        "knowledge_base": {
            "exclude_patterns": ["*.private/*", "archive/*"]
        }
    }


@pytest.fixture
def summary_service(sample_config, temp_dir, mock_openai_client):
    """Create a SummaryService instance with temporary directories."""
    config = sample_config.copy()
    config["daily_output_dir"] = str(temp_dir / "daily")
    config["weekly_output_dir"] = str(temp_dir / "weekly")
    config["notes_base_dir"] = str(temp_dir)

    with patch("services.openai_service.OpenAI", return_value=mock_openai_client):
        service = SummaryService(config)
        return service


@pytest.fixture
def sample_notes_structure(temp_dir):
    """Create a sample notes directory structure for testing."""
    # Create directories
    daily_dir = temp_dir / "daily"
    weekly_dir = temp_dir / "weekly"
    meeting_dir = temp_dir / "meetingnotes"
    learning_dir = temp_dir / "learnings"
    private_dir = temp_dir / ".private"
    archive_dir = temp_dir / "archive"

    for dir_path in [daily_dir, weekly_dir, meeting_dir, learning_dir, private_dir, archive_dir]:
        dir_path.mkdir(parents=True)

    # Create sample files
    (daily_dir / "2024-02-18.md").write_text("[2024-02-18] Daily note content")
    (weekly_dir / "2024-W07.md").write_text("Weekly note content")
    (meeting_dir / "2024-02-18_project_meeting.md").write_text("Meeting note content")
    (learning_dir / "python_tips.md").write_text("Learning note content")
    (private_dir / "private.md").write_text("Private note")
    (archive_dir / "old.md").write_text("Archived note")

    return temp_dir


def test_init_summary_service(sample_config):
    """Test summary service initialization."""
    with patch("services.openai_service.OpenAI") as mock_openai:
        service = SummaryService(sample_config)
        assert service.config == sample_config
        assert service.notes_service is not None
        assert service.openai_service is not None
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=None)


def test_get_all_notes(summary_service, sample_notes_structure):
    """Test getting all notes from the vault."""
    # Execute
    notes = summary_service.get_all_notes()

    # Assert
    assert len(notes) == 4  # Excludes private and archive notes

    # Verify note types
    note_types = {note["type"] for note in notes}
    assert note_types == {"daily", "weekly", "meeting", "learning"}

    # Verify content is loaded
    daily_note = next(note for note in notes if note["type"] == "daily")
    assert daily_note["content"] == "[2024-02-18] Daily note content"
    assert daily_note["date"] == "2024-02-18"


def test_get_all_notes_handles_errors(summary_service, sample_notes_structure):
    """Test error handling when reading notes."""
    # Create an unreadable file
    bad_file = sample_notes_structure / "daily" / "bad.md"
    bad_file.write_text("Some content")
    bad_file.chmod(0o000)  # Make file unreadable

    # Execute
    with patch("builtins.print") as mock_print:
        notes = summary_service.get_all_notes()

    # Assert
    assert len(notes) == 4  # Still gets other notes
    mock_print.assert_called_once()
    assert "Error processing file" in mock_print.call_args[0][0]

    # Cleanup
    bad_file.chmod(0o644)


def test_process_daily_notes_success(summary_service, temp_dir):
    """Test successful processing of daily notes."""
    # Setup
    mock_notes = "Sample daily notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    mock_result = {
        "summary": "Test summary",
        "actionable_items": ["Task 1", "Task 2"],
        "tags": ["tag1", "tag2"]
    }
    summary_service.openai_service.summarize_notes_and_identify_tasks = Mock(return_value=mock_result)

    with patch("services.reminder_service.ReminderService.add_to_reminders") as mock_add_reminder, \
         patch("builtins.print") as mock_print:
        # Execute
        summary_service.process_daily_notes(date_str="2024-02-18")

        # Assert
        assert mock_add_reminder.call_count == 2
        mock_print.assert_any_call("Summary:")
        mock_print.assert_any_call("Test summary")

        # Verify file was created
        year_month_week = "2024/02/07"  # Week 07 of February 2024
        output_file = Path(temp_dir) / "daily" / year_month_week / "2024-02-18.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert "Test summary" in content
        assert "Task 1" in content
        assert "Task 2" in content
        assert "#tag1" in content
        assert "#tag2" in content


def test_process_daily_notes_dry_run(summary_service, temp_dir):
    """Test daily notes processing in dry run mode."""
    # Setup
    mock_notes = "Sample daily notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    mock_result = {
        "summary": "Test summary",
        "actionable_items": ["Task 1"],
        "tags": ["tag1"]
    }
    summary_service.openai_service.summarize_notes_and_identify_tasks = Mock(return_value=mock_result)

    with patch("services.reminder_service.ReminderService.add_to_reminders") as mock_add_reminder:
        # Execute
        summary_service.process_daily_notes(date_str="2024-02-18", dry_run=True)

        # Assert
        mock_add_reminder.assert_not_called()
        assert not any(Path(temp_dir).glob("**/*.md"))


def test_process_daily_notes_skip_reminders(summary_service):
    """Test daily notes processing with reminders skipped."""
    # Setup
    mock_notes = "Sample daily notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    mock_result = {
        "summary": "Test summary",
        "actionable_items": ["Task 1"],
        "tags": ["tag1"]
    }
    summary_service.openai_service.summarize_notes_and_identify_tasks = Mock(return_value=mock_result)

    with patch("services.reminder_service.ReminderService.add_to_reminders") as mock_add_reminder:
        # Execute
        summary_service.process_daily_notes(skip_reminders=True)

        # Assert
        mock_add_reminder.assert_not_called()


def test_process_daily_notes_replace_summary(summary_service, temp_dir):
    """Test daily notes processing with summary replacement."""
    # Setup
    mock_notes = "Sample daily notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    # Create existing summary
    year_month_week = "2024/02/07"  # Week 07 of February 2024
    output_dir = Path(temp_dir) / "daily" / year_month_week
    output_dir.mkdir(parents=True)
    output_file = output_dir / "2024-02-18.md"
    output_file.write_text("Existing summary")

    mock_result = {
        "summary": "New summary",
        "actionable_items": ["Task 1"],
        "tags": ["tag1"]
    }
    summary_service.openai_service.summarize_notes_and_identify_tasks = Mock(return_value=mock_result)

    # Execute
    summary_service.process_daily_notes(date_str="2024-02-18", replace_summary=True)

    # Assert
    content = output_file.read_text()
    assert "New summary" in content
    assert "Existing summary" not in content


def test_process_weekly_notes_success(summary_service, temp_dir):
    """Test successful processing of weekly notes."""
    # Setup
    mock_notes = "Sample weekly notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_weekly_notes = Mock(return_value=mock_notes)

    mock_summary = "Test weekly summary"
    summary_service.openai_service.generate_weekly_summary = Mock(return_value=mock_summary)

    with patch("builtins.print") as mock_print:
        # Execute
        summary_service.process_weekly_notes(date_str="2024-02-18")

        # Assert
        mock_print.assert_any_call("Summary:")
        mock_print.assert_any_call(mock_summary)

        # Verify file was created
        year_month_week = "2024/02/07"  # Week 07 of February 2024
        output_file = Path(temp_dir) / "weekly" / year_month_week / "week_summary.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert mock_summary in content


def test_process_weekly_notes_dry_run(summary_service, temp_dir):
    """Test weekly notes processing in dry run mode."""
    # Setup
    mock_notes = "Sample weekly notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_weekly_notes = Mock(return_value=mock_notes)
    summary_service.openai_service.generate_weekly_summary = Mock(return_value="Test summary")

    # Execute
    summary_service.process_weekly_notes(dry_run=True)

    # Assert
    assert not any(Path(temp_dir).glob("**/*.md"))


def test_process_weekly_notes_replace_summary(summary_service, temp_dir):
    """Test weekly notes processing with summary replacement."""
    # Setup
    mock_notes = "Sample weekly notes"
    summary_service.notes_service.load_notes = Mock(return_value=mock_notes)
    summary_service.notes_service.extract_weekly_notes = Mock(return_value=mock_notes)

    # Create existing summary
    year_month_week = "2024/02/07"
    output_dir = Path(temp_dir) / "weekly" / year_month_week
    output_dir.mkdir(parents=True)
    output_file = output_dir / "week_summary.md"
    output_file.write_text("Existing summary")

    mock_summary = "New weekly summary"
    summary_service.openai_service.generate_weekly_summary = Mock(return_value=mock_summary)

    # Execute
    summary_service.process_weekly_notes(date_str="2024-02-18", replace_summary=True)

    # Assert
    content = output_file.read_text()
    assert mock_summary in content
    assert "Existing summary" not in content


def test_no_notes_found(summary_service):
    """Test handling when no notes are found."""
    # Setup
    summary_service.notes_service.load_notes = Mock(return_value="")
    summary_service.notes_service.extract_today_notes = Mock(return_value="")
    summary_service.notes_service.extract_weekly_notes = Mock(return_value="")

    with patch("builtins.print") as mock_print:
        # Test daily notes
        summary_service.process_daily_notes()
        mock_print.assert_called_with("No notes found for today.")

        # Test weekly notes
        summary_service.process_weekly_notes()
        mock_print.assert_called_with("No notes found for the past week.")