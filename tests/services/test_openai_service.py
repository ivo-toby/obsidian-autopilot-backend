"""
Tests for the OpenAI service.

This module contains tests for the OpenAI service functionality,
including various text generation and processing tasks.
"""

import json
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall

from services.openai_service import OpenAIService


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("services.openai_service.OpenAI") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def openai_service(mock_openai_client):
    """Create an OpenAIService instance with mocked client."""
    return OpenAIService(api_key="test-key", model="test-model", base_url=None)


@pytest.fixture
def sample_chat_response():
    """Create a sample chat completion response."""
    message = ChatCompletionMessage(
        content="Sample response",
        role="assistant",
        function_call=None
    )
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion"
    )


def test_init_openai_service():
    """Test OpenAI service initialization."""
    with patch("services.openai_service.OpenAI") as mock_openai:
        service = OpenAIService(api_key="test-key", model="test-model", base_url="http://localhost:11434/v1")
        assert service.model == "test-model"
        mock_openai.assert_called_once_with(api_key="test-key", base_url="http://localhost:11434/v1")


def test_generate_learning_title_success(openai_service, mock_openai_client, sample_chat_response):
    """Test successful learning title generation."""
    # Setup
    mock_openai_client.chat.completions.create.return_value = sample_chat_response
    learning = "Test learning content"

    # Execute
    title = openai_service.generate_learning_title(learning)

    # Assert
    assert title == "Sample response"
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "test-model"
    assert call_args["max_tokens"] == 50
    assert any(learning in msg["content"] for msg in call_args["messages"])


def test_generate_learning_title_error(openai_service, mock_openai_client):
    """Test learning title generation with API error."""
    # Setup
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    learning = "Test learning content"

    # Execute
    title = openai_service.generate_learning_title(learning)

    # Assert
    assert title == "Untitled Learning"


def test_generate_learning_tags_success(openai_service, mock_openai_client, sample_chat_response):
    """Test successful learning tags generation."""
    # Setup
    sample_chat_response.choices[0].message.content = "#tag1, #tag2, #tag3"
    mock_openai_client.chat.completions.create.return_value = sample_chat_response
    learning = "Test learning content"

    # Execute
    tags = openai_service.generate_learning_tags(learning)

    # Assert
    assert tags == ["#tag1", "#tag2", "#tag3"]
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "test-model"
    assert call_args["max_tokens"] == 50
    assert any(learning in msg["content"] for msg in call_args["messages"])


def test_generate_learning_tags_error(openai_service, mock_openai_client):
    """Test learning tags generation with API error."""
    # Setup
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    learning = "Test learning content"

    # Execute
    tags = openai_service.generate_learning_tags(learning)

    # Assert
    assert tags == []


def test_chat_completion_with_function_success(openai_service, mock_openai_client, sample_chat_response):
    """Test successful chat completion with function calling."""
    # Setup
    function_call = FunctionCall(
        name="test_function",
        arguments=json.dumps({"key": "value"})
    )
    sample_chat_response.choices[0].message.function_call = function_call
    mock_openai_client.chat.completions.create.return_value = sample_chat_response

    messages = [{"role": "user", "content": "Test message"}]
    functions = [{"name": "test_function", "parameters": {}}]
    function_call_param = {"name": "test_function"}

    # Execute
    response = openai_service.chat_completion_with_function(messages, functions, function_call_param)

    # Assert
    assert response.function_call == function_call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "test-model"
    assert call_args["messages"] == messages
    assert call_args["functions"] == functions
    assert call_args["function_call"] == function_call_param


def test_chat_completion_with_function_error(openai_service, mock_openai_client):
    """Test chat completion with function calling error."""
    # Setup
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    messages = [{"role": "user", "content": "Test message"}]
    functions = [{"name": "test_function", "parameters": {}}]
    function_call = {"name": "test_function"}

    # Execute
    response = openai_service.chat_completion_with_function(messages, functions, function_call)

    # Assert
    assert response is None


def test_summarize_notes_and_identify_tasks_success(openai_service, mock_openai_client, sample_chat_response):
    """Test successful note summarization and task identification."""
    # Setup
    function_call = FunctionCall(
        name="create_meeting_notes",
        arguments=json.dumps({
            "summary": "Test summary",
            "actionable_items": ["Task 1", "Task 2"],
            "tags": ["tag1", "tag2"]
        })
    )
    sample_chat_response.choices[0].message.function_call = function_call
    mock_openai_client.chat.completions.create.return_value = sample_chat_response

    notes = "Test notes content"

    # Execute
    result = openai_service.summarize_notes_and_identify_tasks(notes)

    # Assert
    assert result == {
        "summary": "Test summary",
        "actionable_items": ["Task 1", "Task 2"],
        "tags": ["tag1", "tag2"]
    }
    mock_openai_client.chat.completions.create.assert_called_once()


def test_summarize_notes_and_identify_tasks_error(openai_service, mock_openai_client):
    """Test note summarization with API error."""
    # Setup
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    notes = "Test notes content"

    # Execute
    result = openai_service.summarize_notes_and_identify_tasks(notes)

    # Assert
    assert result is None


def test_generate_weekly_summary_success(openai_service, mock_openai_client, sample_chat_response):
    """Test successful weekly summary generation."""
    # Setup
    mock_openai_client.chat.completions.create.return_value = sample_chat_response
    notes = "Test weekly notes content"

    # Execute
    summary = openai_service.generate_weekly_summary(notes)

    # Assert
    assert summary == "Sample response"
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "test-model"
    assert call_args["max_tokens"] == 1500
    assert any(notes in msg["content"] for msg in call_args["messages"])


def test_generate_weekly_summary_error(openai_service, mock_openai_client):
    """Test weekly summary generation with API error."""
    # Setup
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    notes = "Test weekly notes content"

    # Execute
    summary = openai_service.generate_weekly_summary(notes)

    # Assert
    assert summary is None


def test_generate_meeting_notes_success(openai_service, mock_openai_client, sample_chat_response):
    """Test successful meeting notes generation."""
    # Setup
    function_call = FunctionCall(
        name="create_meeting_notes",
        arguments=json.dumps({
            "meetings": [{
                "date": "2024-02-18",
                "meeting_subject": "Test Meeting",
                "participants": ["Alice", "Bob"],
                "meeting_notes": "Test meeting content",
                "decisions": "Test decisions",
                "action_items": ["Task 1", "Task 2"],
                "tags": "#meeting #test",
                "references": "Test references"
            }]
        })
    )
    sample_chat_response.choices[0].message.function_call = function_call
    mock_openai_client.chat.completions.create.return_value = sample_chat_response

    notes = "Test meeting notes content"

    # Execute
    result = openai_service.generate_meeting_notes(notes)

    # Assert
    assert result == {
        "meetings": [{
            "date": "2024-02-18",
            "meeting_subject": "Test Meeting",
            "participants": ["Alice", "Bob"],
            "meeting_notes": "Test meeting content",
            "decisions": "Test decisions",
            "action_items": ["Task 1", "Task 2"],
            "tags": "#meeting #test",
            "references": "Test references"
        }]
    }
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "test-model"
    assert any(notes in msg["content"] for msg in call_args["messages"])


def test_generate_meeting_notes_error(openai_service, mock_openai_client):
    """Test meeting notes generation with API error."""
    # Setup
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
    notes = "Test meeting notes content"

    # Execute
    result = openai_service.generate_meeting_notes(notes)

    # Assert
    assert result is None