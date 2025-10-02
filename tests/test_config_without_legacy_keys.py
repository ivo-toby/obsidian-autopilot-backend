"""
Test that the system works without legacy config keys.

This test verifies that when users remove the deprecated api_key, model,
and base_url keys, the system still functions correctly using only the
inference configuration section.
"""

import pytest
from unittest.mock import Mock, patch

from services.llm_service import LLMService


def test_llm_service_works_without_legacy_keys():
    """Test LLMService initialization without legacy config keys."""
    # Config with only new inference section, no legacy keys
    config = {
        "daily_notes_file": "~/notes/daily.md",
        "inference": {
            "provider": "openai",
            "model": "gpt-4o",
            "openai": {
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }
        }
    }

    with patch("services.llm_service.OpenAIProvider") as MockProvider:
        mock_provider = Mock()
        mock_provider.get_provider_name.return_value = "openai"
        MockProvider.return_value = mock_provider

        # Should not raise any KeyError
        service = LLMService(config)
        assert service.provider is not None


def test_llm_service_ollama_without_legacy_keys():
    """Test LLMService with Ollama provider without legacy keys."""
    config = {
        "daily_notes_file": "~/notes/daily.md",
        "inference": {
            "provider": "ollama",
            "model": "llama3.2",
            "ollama": {
                "base_url": "http://localhost:11434",
                "num_ctx": 8192
            }
        }
    }

    with patch("services.llm_service.OllamaProvider") as MockProvider:
        mock_provider = Mock()
        mock_provider.get_provider_name.return_value = "ollama"
        MockProvider.return_value = mock_provider

        # Should not raise any KeyError
        service = LLMService(config)
        assert service.provider is not None


def test_embedding_service_without_legacy_keys():
    """Test EmbeddingService works without legacy api_key."""
    from services.vector_store.embedding_service import EmbeddingService

    config = {
        "inference": {
            "provider": "openai",
            "model": "gpt-4o",
            "openai": {
                "api_key": "test-key"
            }
        },
        "embeddings": {
            "model_type": "openai",
            "model_name": "text-embedding-3-small",
            "batch_size": 100
        }
    }

    with patch("services.vector_store.embedding_service.OpenAIEmbeddings"):
        # Should not raise KeyError when accessing api_key
        service = EmbeddingService(config)
        assert service.model_type == "openai"


def test_openai_provider_without_legacy_keys():
    """Test OpenAIProvider initialization without legacy keys."""
    from services.providers.openai_provider import OpenAIProvider

    config = {
        "inference": {
            "provider": "openai",
            "model": "gpt-4o",
            "openai": {
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1"
            }
        }
    }

    with patch("services.providers.openai_provider.OpenAI"):
        # Should not raise KeyError
        provider = OpenAIProvider(config)
        assert provider.model == "gpt-4o"
        assert provider.api_key == "test-key"


def test_ollama_provider_without_any_legacy_keys():
    """Test OllamaProvider initialization without any legacy keys."""
    from services.providers.ollama_provider import OllamaProvider

    config = {
        "inference": {
            "provider": "ollama",
            "model": "llama3.2",
            "ollama": {
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "num_ctx": 8192,
                "num_thread": 4,
                "timeout": 120
            }
        }
    }

    with patch("services.providers.ollama_provider.ollama.Client"):
        # Should not raise KeyError
        provider = OllamaProvider(config)
        assert provider.model == "llama3.2"
        assert provider.base_url == "http://localhost:11434"
