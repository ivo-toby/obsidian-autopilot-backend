"""
Tests for the meeting service.

This module contains tests for the meeting service functionality,
including processing and saving meeting notes.
"""

import os
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from pathlib import Path

from services.meeting_service import MeetingService


@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        "daily_notes_file": "~/notes/daily.md",
        "meeting_notes_output_dir": "~/notes/meetings",
        "api_key": "test-api-key",
        "model": "test-model"
    }


@pytest.fixture
def sample_meeting_data():
    """Return sample meeting data for testing."""
    return {
        "date": "2024-02-18",
        "meeting_subject": "Project Planning",
        "participants": ["Alice", "Bob", "Charlie"],
        "meeting_notes": "Discussed project timeline and milestones.",
        "decisions": "- Decided to use Python for backend\n- Weekly sprints",
        "action_items": [
            "Alice to create project structure",
            "Bob to set up CI/CD",
            "Charlie to write documentation"
        ],
        "tags": "#project #planning #development",
        "references": "- Project proposal doc\n- Technical specifications"
    }


@pytest.fixture
def mock_openai_service():
    """Create a mock OpenAI service."""
    with patch("services.meeting_service.OpenAIService") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def meeting_service(sample_config, temp_dir, mock_openai_service):
    """Create a MeetingService instance with temporary directories."""
    config = sample_config.copy()
    config["meeting_notes_output_dir"] = str(temp_dir)
    return MeetingService(config)


def test_init_meeting_service(sample_config, mock_openai_service):
    """Test meeting service initialization."""
    service = MeetingService(sample_config)
    assert service.config == sample_config
    assert service.notes_service is not None
    assert service.openai_service is not None


def test_save_meeting_notes(meeting_service, sample_meeting_data, temp_dir):
    """Test saving meeting notes to a file."""
    # Execute
    meeting_service._save_meeting_notes(sample_meeting_data)

    # Assert
    expected_filename = f"{sample_meeting_data['date']}_project_planning.md"
    output_file = Path(temp_dir) / expected_filename
    assert output_file.exists()

    content = output_file.read_text()
    assert sample_meeting_data["meeting_subject"] in content
    assert sample_meeting_data["meeting_notes"] in content
    assert all(participant in content for participant in sample_meeting_data["participants"])
    assert all(item in content for item in sample_meeting_data["action_items"])
    assert sample_meeting_data["decisions"] in content
    assert sample_meeting_data["tags"] in content
    assert sample_meeting_data["references"] in content


def test_save_meeting_notes_custom_output_dir(meeting_service, sample_meeting_data, temp_dir):
    """Test saving meeting notes to a custom output directory."""
    # Setup
    custom_dir = temp_dir / "custom"

    # Execute
    meeting_service._save_meeting_notes(sample_meeting_data, str(custom_dir))

    # Assert
    expected_filename = f"{sample_meeting_data['date']}_project_planning.md"
    output_file = custom_dir / expected_filename
    assert output_file.exists()


def test_process_meeting_notes(meeting_service, sample_meeting_data):
    """Test processing meeting notes with mocked dependencies."""
    # Setup
    mock_notes = "Sample daily notes with meeting content"
    meeting_service.notes_service.load_notes = Mock(return_value=mock_notes)
    meeting_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    mock_meeting_notes = {"meetings": [sample_meeting_data]}
    meeting_service.openai_service.generate_meeting_notes = Mock(return_value=mock_meeting_notes)

    # Execute
    meeting_service.process_meeting_notes()

    # Assert
    meeting_service.notes_service.load_notes.assert_called_once()
    meeting_service.notes_service.extract_today_notes.assert_called_once()
    meeting_service.openai_service.generate_meeting_notes.assert_called_once_with(mock_notes)


def test_process_meeting_notes_dry_run(meeting_service, sample_meeting_data, temp_dir):
    """Test processing meeting notes in dry run mode."""
    # Setup
    mock_notes = "Sample daily notes with meeting content"
    meeting_service.notes_service.load_notes = Mock(return_value=mock_notes)
    meeting_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    mock_meeting_notes = {"meetings": [sample_meeting_data]}
    meeting_service.openai_service.generate_meeting_notes = Mock(return_value=mock_meeting_notes)

    # Execute
    meeting_service.process_meeting_notes(dry_run=True)

    # Assert
    # Check that no files were created
    assert len(list(Path(temp_dir).glob("*.md"))) == 0


def test_process_meeting_notes_no_notes(meeting_service):
    """Test processing meeting notes when no notes are found."""
    # Setup
    meeting_service.notes_service.load_notes = Mock(return_value="")
    meeting_service.notes_service.extract_today_notes = Mock(return_value="")

    # Execute
    meeting_service.process_meeting_notes()

    # Assert
    meeting_service.notes_service.load_notes.assert_called_once()
    meeting_service.notes_service.extract_today_notes.assert_called_once()
    meeting_service.openai_service.generate_meeting_notes.assert_not_called()


def test_process_meeting_notes_with_date(meeting_service, sample_meeting_data):
    """Test processing meeting notes for a specific date."""
    # Setup
    mock_notes = "Sample daily notes with meeting content"
    meeting_service.notes_service.load_notes = Mock(return_value=mock_notes)
    meeting_service.notes_service.extract_today_notes = Mock(return_value=mock_notes)

    mock_meeting_notes = {"meetings": [sample_meeting_data]}
    meeting_service.openai_service.generate_meeting_notes = Mock(return_value=mock_meeting_notes)

    test_date = "2024-02-18"

    # Execute
    meeting_service.process_meeting_notes(date_str=test_date)

    # Assert
    meeting_service.notes_service.extract_today_notes.assert_called_once_with(mock_notes, test_date)


def test_process_meeting_transcript_writes_file(meeting_service, temp_dir, monkeypatch):
    """Test processing meeting transcript writes file correctly."""
    sample_summary = "# [Team Sync]\nThis is a summary."
    # Monkey-patch clipboard paste
    import services.meeting_service as ms
    monkeypatch.setattr(ms.pyperclip, "paste", lambda: "dummy transcript")
    # Monkey-patch the LLM call
    meeting_service.openai_service.generate_text = Mock(return_value=sample_summary)
    # Mock the topic inference to return a known value
    meeting_service._infer_topic_from_summary = Mock(return_value="team_sync")

    # Execute
    meeting_service.process_meeting_transcript(date_str="2024-03-01", dry_run=False)

    # Assert file exists with correct name and content
    expected_filename = "2024-03-01_team_sync.md"
    output_file = Path(temp_dir) / expected_filename
    assert output_file.exists()
    assert output_file.read_text() == sample_summary


def test_process_meeting_transcript_dry_run_prints(meeting_service, capsys, monkeypatch):
    """Test processing meeting transcript in dry-run prints summary without writing."""
    sample_summary = "# [Team Sync]\nThis is a summary."
    import services.meeting_service as ms
    monkeypatch.setattr(ms.pyperclip, "paste", lambda: "dummy transcript")
    meeting_service.openai_service.generate_text = Mock(return_value=sample_summary)
    # Mock the topic inference to return a known value
    meeting_service._infer_topic_from_summary = Mock(return_value="team_sync")

    # Execute in dry-run mode
    meeting_service.process_meeting_transcript(date_str="2024-03-02", dry_run=True)
    captured = capsys.readouterr()
    assert sample_summary in captured.out

    # Confirm no file created in output directory
    out_dir = Path(meeting_service.config["meeting_notes_output_dir"])
    assert not any(out_dir.iterdir())


def test_process_meeting_transcript_with_custom_prompt(meeting_service, temp_dir, monkeypatch, tmp_path):
    """Test processing meeting transcript with a custom prompt file."""
    # Create a custom prompt file
    custom_prompt = "# Custom Prompt Template"
    custom_prompt_file = tmp_path / "custom_prompt.md"
    custom_prompt_file.write_text(custom_prompt)

    # Monkey-patch clipboard paste
    import services.meeting_service as ms
    monkeypatch.setattr(ms.pyperclip, "paste", lambda: "dummy transcript")

    # Monkey-patch the LLM call
    expected_prompt = f"{custom_prompt}\n\n'''TRANSCRIPT'''\ndummy transcript"

    def check_prompt(prompt):
        assert prompt == expected_prompt
        return "# [Custom Meeting]\nSummary from custom prompt."

    meeting_service.openai_service.generate_text = Mock(side_effect=check_prompt)

    # Mock the topic inference to return a known value
    meeting_service._infer_topic_from_summary = Mock(return_value="custom_meeting")

    # Execute with custom prompt file
    meeting_service.process_meeting_transcript(
        date_str="2024-03-01",
        dry_run=False,
        prompt_file=str(custom_prompt_file)
    )

    # Assert correct prompt was used
    meeting_service.openai_service.generate_text.assert_called_once()

    # Assert file was created with correct content
    expected_filename = "2024-03-01_custom_meeting.md"
    output_file = Path(temp_dir) / expected_filename
    assert output_file.exists()
    assert "Summary from custom prompt" in output_file.read_text()


def test_infer_topic_from_summary(meeting_service):
    """Test inferring topic from a meeting summary."""
    # Test with a proper markdown heading
    summary_with_heading = "# [Team Sync]\nThis is a team sync meeting summary."
    meeting_service.openai_service.generate_text = Mock(return_value="Team Collaboration Discussion")

    topic = meeting_service._infer_topic_from_summary(summary_with_heading)
    assert topic == "team_collaboration_discussion"

    # Test fallback when LLM fails
    meeting_service.openai_service.generate_text = Mock(side_effect=Exception("API error"))
    topic = meeting_service._infer_topic_from_summary(summary_with_heading)
    assert topic == "team_sync"

    # Test with no markdown heading
    summary_without_heading = "This is a meeting summary without a proper heading."
    meeting_service.openai_service.generate_text = Mock(return_value="Project Status Update")

    topic = meeting_service._infer_topic_from_summary(summary_without_heading)
    assert topic == "project_status_update"

    # Test fallback to default when no heading and LLM fails
    meeting_service.openai_service.generate_text = Mock(side_effect=Exception("API error"))
    topic = meeting_service._infer_topic_from_summary(summary_without_heading)
    assert topic == "meeting"


def test_save_raw_summary_with_inferred_topic(meeting_service, temp_dir):
    """Test saving raw summary with an inferred topic."""
    sample_summary = "# [General Discussion]\nThis is a detailed project planning meeting."
    date_str = "2024-03-01"

    # Mock the infer_topic method to return a custom topic
    meeting_service._infer_topic_from_summary = Mock(return_value="project_planning_session")

    # Execute
    meeting_service._save_raw_summary(sample_summary, date_str)

    # Assert file exists with correct name using the inferred topic
    expected_filename = "2024-03-01_project_planning_session.md"
    output_file = Path(temp_dir) / expected_filename
    assert output_file.exists()
    assert output_file.read_text() == sample_summary