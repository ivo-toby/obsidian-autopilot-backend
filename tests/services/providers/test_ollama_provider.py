"""
Unit tests for OllamaProvider with focus on reasoning mode functionality.
"""

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from services.providers.ollama_provider import OllamaProvider


class TestOllamaProviderReasoning(unittest.TestCase):
    """Test OllamaProvider reasoning mode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            "inference": {
                "provider": "ollama",
                "ollama": {
                    "model": "qwen3:14b",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.7,
                    "num_ctx": 8192,
                    "num_thread": 4,
                    "timeout": 120,
                    "reasoning": {
                        "enabled": False,
                        "save_thinking": False,
                        "log_thinking": False,
                        "models": ["qwen3", "qwen2.5", "deepseek-r1", "qwq", "smallthinker"]
                    }
                }
            }
        }

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_initialization_with_reasoning_disabled(self, mock_client_class):
        """Test provider initialization with reasoning disabled."""
        provider = OllamaProvider(self.base_config)

        self.assertEqual(provider.model, "qwen3:14b")
        self.assertFalse(provider.reasoning_enabled)
        self.assertFalse(provider.save_thinking)
        self.assertFalse(provider.log_thinking)
        self.assertIn("qwen3", provider.reasoning_models)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_initialization_with_reasoning_enabled(self, mock_client_class):
        """Test provider initialization with reasoning enabled."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["reasoning"]["enabled"] = True
        config["inference"]["ollama"]["reasoning"]["save_thinking"] = True
        config["inference"]["ollama"]["reasoning"]["log_thinking"] = True

        provider = OllamaProvider(config)

        self.assertTrue(provider.reasoning_enabled)
        self.assertTrue(provider.save_thinking)
        self.assertTrue(provider.log_thinking)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_is_reasoning_model_detection(self, mock_client_class):
        """Test reasoning model detection using programmatic template checking."""
        # Template with reasoning support (contains IsThinkSet or .Thinking)
        reasoning_template = """{{- if .IsThinkSet }}
{{- if .Thinking }}
<think>{{ .Thinking }}</think>
{{- end }}
{{- end }}"""

        # Template without reasoning support
        no_reasoning_template = """{{- range .Messages }}
{{ .Content }}
{{- end }}"""

        test_cases = [
            ("qwen3:14b", reasoning_template, True),
            ("deepseek-r1:8b", reasoning_template, True),
            ("gpt-oss:20b", reasoning_template, True),
            ("magistral:24b", reasoning_template, True),
            ("llama3.2:3b", no_reasoning_template, False),
            ("mistral:7b", no_reasoning_template, False),
        ]

        for model, template, expected in test_cases:
            config = self.base_config.copy()
            config["inference"]["ollama"]["model"] = model

            # Mock the client.show() response
            mock_instance = Mock()
            mock_instance.show.return_value = {"template": template}
            mock_client_class.return_value = mock_instance

            provider = OllamaProvider(config)

            self.assertEqual(
                provider._is_reasoning_model(),
                expected,
                f"Model {model} should {'support' if expected else 'not support'} reasoning"
            )

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_is_reasoning_model_fallback(self, mock_client_class):
        """Test reasoning model detection fallback when API call fails."""
        # Mock client.show() to raise an exception
        mock_instance = Mock()
        mock_instance.show.side_effect = Exception("API error")
        mock_client_class.return_value = mock_instance

        test_cases = [
            ("qwen3:14b", True),  # In fallback list
            ("gpt-oss:20b", True),  # In fallback list
            ("deepseek-r1:8b", True),  # In fallback list
            ("llama3.2:3b", False),  # Not in fallback list
        ]

        for model, expected in test_cases:
            config = self.base_config.copy()
            config["inference"]["ollama"]["model"] = model
            provider = OllamaProvider(config)

            self.assertEqual(
                provider._is_reasoning_model(),
                expected,
                f"Model {model} should {'support' if expected else 'not support'} reasoning (fallback)"
            )

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_is_reasoning_model_caching(self, mock_client_class):
        """Test that reasoning model detection is cached."""
        mock_instance = Mock()
        mock_instance.show.return_value = {"template": "{{- if .IsThinkSet }}thinking{{- end }}"}
        mock_client_class.return_value = mock_instance

        config = self.base_config.copy()
        config["inference"]["ollama"]["model"] = "qwen3:14b"
        provider = OllamaProvider(config)

        # Call twice
        result1 = provider._is_reasoning_model()
        result2 = provider._is_reasoning_model()

        # Should only call show() once due to caching
        self.assertEqual(mock_instance.show.call_count, 1)
        self.assertEqual(result1, result2)
        self.assertTrue(result1)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_strip_thinking_tags(self, mock_client_class):
        """Test thinking tag removal."""
        provider = OllamaProvider(self.base_config)

        test_cases = [
            (
                "<think>Some thinking process</think>Final answer",
                "Final answer"
            ),
            (
                "<thinking>Deep thoughts here</thinking>Response text",
                "Response text"
            ),
            (
                "No tags here",
                "No tags here"
            ),
            (
                "<think>First thought</think>Middle<think>Second thought</think>End",
                "MiddleEnd"
            ),
        ]

        for input_text, expected in test_cases:
            result = provider._strip_thinking_tags(input_text)
            self.assertEqual(result, expected)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_process_response_with_reasoning_suppress(self, mock_client_class):
        """Test response processing with reasoning enabled and thinking suppressed."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["reasoning"]["enabled"] = True
        config["inference"]["ollama"]["reasoning"]["save_thinking"] = False

        provider = OllamaProvider(config)

        response = {
            "message": {
                "content": "Final answer",
                "thinking": "This is my reasoning process..."
            }
        }

        result = provider._process_response(response, reasoning_enabled=True)

        # Should return only content, not thinking
        self.assertEqual(result, "Final answer")
        self.assertNotIn("thinking", result)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_process_response_with_reasoning_save(self, mock_client_class):
        """Test response processing with reasoning enabled and thinking saved."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["reasoning"]["enabled"] = True
        config["inference"]["ollama"]["reasoning"]["save_thinking"] = True

        provider = OllamaProvider(config)

        response = {
            "message": {
                "content": "Final answer",
                "thinking": "This is my reasoning process..."
            }
        }

        result = provider._process_response(response, reasoning_enabled=True)

        # Should include thinking in output
        self.assertIn("thinking", result.lower())
        self.assertIn("This is my reasoning process...", result)
        self.assertIn("Final answer", result)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_process_response_without_reasoning(self, mock_client_class):
        """Test response processing with reasoning disabled."""
        provider = OllamaProvider(self.base_config)

        response = {
            "message": {
                "content": "Simple response"
            }
        }

        result = provider._process_response(response, reasoning_enabled=False)

        self.assertEqual(result, "Simple response")

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_generate_text_with_reasoning_enabled(self, mock_client_class):
        """Test generate_text with reasoning mode enabled."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["reasoning"]["enabled"] = True

        # Mock client.show() to return a reasoning template
        mock_instance = MagicMock()
        mock_instance.show.return_value = {"template": "{{- if .IsThinkSet }}reasoning{{- end }}"}
        mock_client_class.return_value = mock_instance

        provider = OllamaProvider(config)

        # Mock the client.chat response
        mock_response = {
            "message": {
                "content": "The answer is 42",
                "thinking": "Let me think about this...",
                "role": "assistant"
            }
        }
        provider.client.chat = MagicMock(return_value=mock_response)

        result = provider.generate_text("What is the meaning of life?")

        # Verify client.chat was called with think=True
        provider.client.chat.assert_called_once()
        call_args = provider.client.chat.call_args
        self.assertTrue(call_args.kwargs.get("think"))

        # Verify response is processed (thinking suppressed by default)
        self.assertEqual(result, "The answer is 42")

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_generate_text_with_reasoning_override(self, mock_client_class):
        """Test generate_text with reasoning parameter override."""
        # Config has reasoning disabled
        config = self.base_config.copy()
        config["inference"]["ollama"]["reasoning"]["enabled"] = False

        # Mock client.show() to return a reasoning template
        mock_instance = MagicMock()
        mock_instance.show.return_value = {"template": "{{- if .IsThinkSet }}reasoning{{- end }}"}
        mock_client_class.return_value = mock_instance

        provider = OllamaProvider(config)

        mock_response = {
            "message": {
                "content": "Response",
                "thinking": "Thinking...",
                "role": "assistant"
            }
        }
        provider.client.chat = MagicMock(return_value=mock_response)

        # Override reasoning to True
        result = provider.generate_text("Prompt", reasoning=True)

        # Verify think=True was used
        call_args = provider.client.chat.call_args
        self.assertTrue(call_args.kwargs.get("think"))

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_generate_text_reasoning_disabled_for_non_reasoning_model(self, mock_client_class):
        """Test that reasoning is not enabled for non-reasoning models."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["model"] = "llama3.2:3b"
        config["inference"]["ollama"]["reasoning"]["enabled"] = True

        provider = OllamaProvider(config)

        mock_response = {
            "message": {
                "content": "Response",
                "role": "assistant"
            }
        }
        provider.client.chat = MagicMock(return_value=mock_response)

        result = provider.generate_text("Prompt")

        # Verify think parameter was NOT added
        call_args = provider.client.chat.call_args
        self.assertNotIn("think", call_args.kwargs)

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_chat_completion_with_reasoning(self, mock_client_class):
        """Test chat_completion with reasoning mode."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["reasoning"]["enabled"] = True

        # Mock client.show() to return a reasoning template
        mock_instance = MagicMock()
        mock_instance.show.return_value = {"template": "{{- if .IsThinkSet }}reasoning{{- end }}"}
        mock_client_class.return_value = mock_instance

        provider = OllamaProvider(config)

        mock_response = {
            "message": {
                "content": "Chat response",
                "thinking": "Thinking about the question...",
                "role": "assistant"
            }
        }
        provider.client.chat = MagicMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Hello"}
        ]

        result = provider.chat_completion(messages)

        # Verify think=True was used
        call_args = provider.client.chat.call_args
        self.assertTrue(call_args.kwargs.get("think"))

        # Verify response
        self.assertEqual(result["content"], "Chat response")
        self.assertEqual(result["role"], "assistant")

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_chat_completion_with_function_reasoning(self, mock_client_class):
        """Test chat_completion_with_function with reasoning mode."""
        config = self.base_config.copy()
        config["inference"]["ollama"]["model"] = "qwen3:14b"  # Reasoning model
        config["inference"]["ollama"]["reasoning"]["enabled"] = False

        # Mock client.show() to return a reasoning template
        mock_instance = MagicMock()
        mock_instance.show.return_value = {"template": "{{- if .IsThinkSet }}reasoning{{- end }}"}
        mock_client_class.return_value = mock_instance

        provider = OllamaProvider(config)

        # Mock response for structured output (qwen3 doesn't support native tools)
        mock_response = {
            "message": {
                "content": '{"arg": "value"}',
                "thinking": "Let me think...",
                "role": "assistant"
            }
        }
        provider.client.chat = MagicMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Test"}]
        functions = [
            {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string"}
                    }
                }
            }
        ]
        function_call = {"name": "test_function"}

        # Override reasoning to True
        result = provider.chat_completion_with_function(
            messages, functions, function_call, reasoning=True
        )

        # Verify think=True was used
        call_args = provider.client.chat.call_args
        self.assertTrue(call_args.kwargs.get("think"))

        # Verify function call extraction
        self.assertIn("function_call", result)
        self.assertEqual(result["function_call"]["name"], "test_function")


class TestOllamaProviderBasics(unittest.TestCase):
    """Test basic OllamaProvider functionality."""

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_get_provider_name(self, mock_client_class):
        """Test provider name."""
        config = {
            "inference": {
                "provider": "ollama",
                "ollama": {
                    "model": "llama3.2"
                }
            }
        }
        provider = OllamaProvider(config)
        self.assertEqual(provider.get_provider_name(), "ollama")

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_model_supports_tools(self, mock_client_class):
        """Test tool support detection using programmatic template checking."""
        # Template with tool support (contains .Tools)
        tools_template = """{{- if .Tools }}
# Tools
You may call one or more functions...
{{- end }}"""

        # Template without tool support
        no_tools_template = """{{- if .Messages }}
{{- range .Messages }}
{{ .Content }}
{{- end }}
{{- end }}"""

        test_cases = [
            ("llama3.1:8b", tools_template, True),
            ("llama3.2:3b", tools_template, True),
            ("mistral:7b", tools_template, True),
            ("gpt-oss:20b", tools_template, True),
            ("qwen3:14b", tools_template, True),  # qwen3 actually supports tools
            ("gemma:7b", no_tools_template, False),
        ]

        for model, template, expected in test_cases:
            config = {
                "inference": {
                    "provider": "ollama",
                    "ollama": {
                        "model": model
                    }
                }
            }

            # Mock the client.show() response
            mock_instance = Mock()
            mock_instance.show.return_value = {"template": template}
            mock_client_class.return_value = mock_instance

            provider = OllamaProvider(config)

            self.assertEqual(
                provider._model_supports_tools(),
                expected,
                f"Model {model} should {'support' if expected else 'not support'} tools"
            )

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_model_supports_tools_fallback(self, mock_client_class):
        """Test tool support detection fallback when API call fails."""
        # Mock client.show() to raise an exception
        mock_instance = Mock()
        mock_instance.show.side_effect = Exception("API error")
        mock_client_class.return_value = mock_instance

        test_cases = [
            ("llama3.1:8b", True),  # In fallback list
            ("gpt-oss:20b", True),  # In fallback list
            ("unknown-model:1b", False),  # Not in fallback list
        ]

        for model, expected in test_cases:
            config = {
                "inference": {
                    "provider": "ollama",
                    "ollama": {
                        "model": model
                    }
                }
            }
            provider = OllamaProvider(config)

            self.assertEqual(
                provider._model_supports_tools(),
                expected,
                f"Model {model} should {'support' if expected else 'not support'} tools (fallback)"
            )

    @patch('services.providers.ollama_provider.ollama.Client')
    def test_model_supports_tools_caching(self, mock_client_class):
        """Test that tool support detection is cached."""
        mock_instance = Mock()
        mock_instance.show.return_value = {"template": "{{- if .Tools }}tools{{- end }}"}
        mock_client_class.return_value = mock_instance

        config = {
            "inference": {
                "provider": "ollama",
                "ollama": {
                    "model": "llama3.2"
                }
            }
        }
        provider = OllamaProvider(config)

        # Call twice
        result1 = provider._model_supports_tools()
        result2 = provider._model_supports_tools()

        # Should only call show() once due to caching
        self.assertEqual(mock_instance.show.call_count, 1)
        self.assertEqual(result1, result2)
        self.assertTrue(result1)


if __name__ == "__main__":
    unittest.main()
