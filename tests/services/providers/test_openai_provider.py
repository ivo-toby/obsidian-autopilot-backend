"""Tests for the OpenAI provider."""

from unittest.mock import Mock, patch

import pytest

from services.providers.openai_provider import OpenAIProvider


@pytest.fixture
def mock_openai_client():
    with patch("services.providers.openai_provider.OpenAI") as mock_openai:
        client = Mock()
        mock_openai.return_value = client
        yield client


@pytest.fixture
def provider(mock_openai_client):
    return OpenAIProvider(
        {
            "inference": {
                "provider": "openai",
                "openai": {
                    "api_key": "test-key",
                    "model": "gpt-5.6-luna",
                    "temperature": 0.7,
                },
            }
        }
    )


@pytest.fixture
def sample_response():
    message = Mock(content="Sample response", role="assistant", function_call=None)
    choice = Mock(message=message, finish_reason="stop")
    return Mock(choices=[choice])


def test_generate_text_does_not_send_temperature(
    provider, mock_openai_client, sample_response
):
    mock_openai_client.chat.completions.create.return_value = sample_response

    provider.generate_text("Test prompt")

    call_args = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_args


def test_chat_completion_does_not_send_temperature(
    provider, mock_openai_client, sample_response
):
    mock_openai_client.chat.completions.create.return_value = sample_response

    provider.chat_completion([{"role": "user", "content": "Test message"}])

    call_args = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_args


def test_function_call_does_not_send_temperature(
    provider, mock_openai_client, sample_response
):
    mock_openai_client.chat.completions.create.return_value = sample_response

    provider.chat_completion_with_function(
        [{"role": "user", "content": "Test message"}],
        [{"name": "test_function", "parameters": {}}],
        {"name": "test_function"},
    )

    call_args = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_args
