"""
Ollama provider implementation.

This module provides the Ollama implementation of the LLM provider interface
using the native Ollama Python SDK.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import ollama

from services.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama provider for LLM operations using native Ollama SDK."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ollama provider.

        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        inference_config = config.get("inference", {})
        ollama_config = inference_config.get("ollama", {})

        # Get model from ollama config first, then inference root, then default
        self.model = (
            ollama_config.get("model") or
            inference_config.get("model") or
            "llama3.2"
        )
        self.base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.temperature = ollama_config.get("temperature", 0.7)
        self.num_ctx = ollama_config.get("num_ctx", 8192)
        self.num_thread = ollama_config.get("num_thread", 4)
        self.timeout = ollama_config.get("timeout", 120)

        # Reasoning mode configuration
        reasoning_config = ollama_config.get("reasoning", {})
        self.reasoning_enabled = reasoning_config.get("enabled", False)
        self.save_thinking = reasoning_config.get("save_thinking", False)
        self.log_thinking = reasoning_config.get("log_thinking", False)
        self.reasoning_models = reasoning_config.get("models", [
            "qwen3", "qwen2.5", "deepseek-r1", "qwq", "smallthinker"
        ])

        # Initialize Ollama client
        self.client = ollama.Client(host=self.base_url)

        # Cache for capability detection (avoid repeated API calls)
        self._tool_support_cache = None
        self._reasoning_support_cache = None

        logger.info(f"Initialized Ollama provider with model: {self.model}")
        if self.reasoning_enabled:
            logger.info(f"Reasoning mode enabled (save_thinking={self.save_thinking}, log_thinking={self.log_thinking})")

    def _is_reasoning_model(self) -> bool:
        """
        Check if current model supports reasoning mode.

        Uses programmatic detection by checking the model's template for reasoning
        logic. Falls back to a hardcoded list if the API call fails.

        Returns:
            bool: True if model supports reasoning, False otherwise
        """
        # Return cached result if available
        if self._reasoning_support_cache is not None:
            return self._reasoning_support_cache

        try:
            # Get model information from Ollama
            logger.debug(f"Checking reasoning support for model: {self.model}")
            response = self.client.show(self.model)
            template = response.get("template", "")

            # Check if template contains reasoning logic
            # Models with reasoning support have IsThinkSet or .Thinking in their template
            supports_reasoning = "IsThinkSet" in template or ".Thinking" in template

            logger.info(f"Model {self.model} reasoning support: {supports_reasoning} (detected programmatically)")
            self._reasoning_support_cache = supports_reasoning
            return supports_reasoning

        except Exception as e:
            logger.warning(f"Failed to detect reasoning support programmatically: {e}")
            logger.warning("Falling back to hardcoded model list")

            # Fallback to expanded hardcoded list if API call fails
            reasoning_models = [
                "qwen3", "qwen2.5",
                "deepseek-r1", "deepseek-v3",
                "qwq",
                "smallthinker",
                "gpt-oss",
                "magistral"
            ]
            supports_reasoning = any(name in self.model.lower() for name in reasoning_models)
            logger.info(f"Model {self.model} reasoning support: {supports_reasoning} (fallback detection)")
            self._reasoning_support_cache = supports_reasoning
            return supports_reasoning

    def _strip_thinking_tags(self, text: str) -> str:
        """
        Remove <think> or <thinking> tags from text.

        Args:
            text (str): Text potentially containing thinking tags

        Returns:
            str: Text with thinking tags removed
        """
        import re
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove <thinking>...</thinking> blocks
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        return text.strip()

    def _process_response(self, response: Dict[str, Any], reasoning_enabled: bool) -> str:
        """
        Process response and handle thinking content.

        Args:
            response (Dict[str, Any]): Raw response from Ollama API
            reasoning_enabled (bool): Whether reasoning was enabled for this request

        Returns:
            str: Processed response content
        """
        message = response.get("message", {})
        content = message.get("content", "")

        if reasoning_enabled and "thinking" in message:
            thinking = message["thinking"]

            # Log thinking if configured
            if self.log_thinking:
                logger.debug(f"Model thinking:\n{thinking}")

            # Optionally include thinking in output
            if self.save_thinking:
                return f"<thinking>\n{thinking}\n</thinking>\n\n{content}"

            # Default: return only final content (suppress thinking)
            return content

        # Fallback: strip thinking tags if present in content
        return self._strip_thinking_tags(content)

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using Ollama.

        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters (temperature, reasoning, etc.)
                reasoning (bool): Override global reasoning setting

        Returns:
            str: Generated text response
        """
        try:
            temperature = kwargs.get("temperature", self.temperature)

            # Check if reasoning should be enabled for this request
            use_reasoning = kwargs.get("reasoning", self.reasoning_enabled)
            use_reasoning = use_reasoning and self._is_reasoning_model()

            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_thread": self.num_thread,
                },
            }

            # Add think parameter if reasoning is enabled
            if use_reasoning:
                api_params["think"] = True
                logger.debug(f"Reasoning enabled for request (model: {self.model})")

            response = self.client.chat(**api_params)

            # Process response to handle thinking content
            return self._process_response(response, use_reasoning)
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return ""

    def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion from messages.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            **kwargs: Additional parameters (temperature, reasoning, etc.)
                reasoning (bool): Override global reasoning setting

        Returns:
            Dict[str, Any]: Response dictionary
        """
        try:
            temperature = kwargs.get("temperature", self.temperature)

            # Check if reasoning should be enabled for this request
            use_reasoning = kwargs.get("reasoning", self.reasoning_enabled)
            use_reasoning = use_reasoning and self._is_reasoning_model()

            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_thread": self.num_thread,
                },
            }

            # Add think parameter if reasoning is enabled
            if use_reasoning:
                api_params["think"] = True
                logger.debug(f"Reasoning enabled for chat completion (model: {self.model})")

            response = self.client.chat(**api_params)

            # Process response to handle thinking content
            processed_content = self._process_response(response, use_reasoning)

            return {
                "content": processed_content,
                "role": response["message"]["role"],
                "finish_reason": "stop",
            }
        except Exception as e:
            logger.error(f"Error in chat completion with Ollama: {e}")
            return {"content": "", "role": "assistant", "finish_reason": "error"}

    def chat_completion_with_function(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a chat completion with function calling using Ollama.

        For models that support tools (llama3.1+), uses native tool support.
        For other models, falls back to structured output parsing.

        Args:
            messages (List[Dict[str, str]]): Chat messages
            functions (List[Dict[str, Any]]): Function definitions (OpenAI format)
            function_call (Dict[str, str]): Function to call
            **kwargs: Additional parameters

        Returns:
            Optional[Dict[str, Any]]: Response message with function call or None
        """
        try:
            # Check if model supports tools (llama3.1, llama3.2, etc.)
            supports_tools = self._model_supports_tools()

            if supports_tools:
                return self._chat_with_native_tools(messages, functions, function_call, **kwargs)
            else:
                return self._chat_with_structured_output(messages, functions, function_call, **kwargs)

        except Exception as e:
            logger.error(f"Error in function calling with Ollama: {e}")
            return None

    def _model_supports_tools(self) -> bool:
        """
        Check if the current model supports native tool calling.

        Uses programmatic detection by checking the model's template for tool calling
        logic. Falls back to a hardcoded list if the API call fails.

        Returns:
            bool: True if model supports tools, False otherwise
        """
        # Return cached result if available
        if self._tool_support_cache is not None:
            return self._tool_support_cache

        try:
            # Get model information from Ollama
            logger.debug(f"Checking tool support for model: {self.model}")
            response = self.client.show(self.model)
            template = response.get("template", "")

            # Check if template contains tool calling logic
            # Models with tool support have .Tools or .ToolCalls in their template
            supports_tools = ".Tools" in template or ".ToolCalls" in template

            logger.info(f"Model {self.model} tool support: {supports_tools} (detected programmatically)")
            self._tool_support_cache = supports_tools
            return supports_tools

        except Exception as e:
            logger.warning(f"Failed to detect tool support programmatically: {e}")
            logger.warning("Falling back to hardcoded model list")

            # Fallback to hardcoded list if API call fails
            tool_supported_models = [
                "llama3.1", "llama3.2", "llama3.3",
                "mistral", "mistral-nemo", "mistral-small", "mistral-large",
                "mixtral",
                "gpt-oss",
                "qwen2", "qwen2.5", "qwen3",
                "command-r", "command-r-plus",
                "firefunction",
                "granite3"
            ]
            supports_tools = any(model in self.model.lower() for model in tool_supported_models)
            logger.info(f"Model {self.model} tool support: {supports_tools} (fallback detection)")
            self._tool_support_cache = supports_tools
            return supports_tools

    def _chat_with_native_tools(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Use native Ollama tools for function calling.

        Args:
            messages: Chat messages
            functions: OpenAI-format function definitions
            function_call: Function to call
            **kwargs: Additional parameters
                reasoning (bool): Override global reasoning setting

        Returns:
            Optional[Dict[str, Any]]: Response with function call
        """
        try:
            # Convert OpenAI function format to Ollama tools format
            tools = self._convert_functions_to_tools(functions)

            temperature = kwargs.get("temperature", self.temperature)

            # Check if reasoning should be enabled for this request
            use_reasoning = kwargs.get("reasoning", self.reasoning_enabled)
            use_reasoning = use_reasoning and self._is_reasoning_model()

            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_thread": self.num_thread,
                },
            }

            # Add think parameter if reasoning is enabled
            if use_reasoning:
                api_params["think"] = True
                logger.debug(f"Reasoning enabled for function calling (model: {self.model})")

            response = self.client.chat(**api_params)

            message = response["message"]

            # Process content to handle thinking
            processed_content = self._process_response(response, use_reasoning)

            result = {
                "content": processed_content,
                "role": message["role"],
                "finish_reason": "stop",
            }

            # Check if there are tool calls
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                result["function_call"] = {
                    "name": tool_call["function"]["name"],
                    "arguments": json.dumps(tool_call["function"]["arguments"]),
                }

            return result

        except Exception as e:
            logger.error(f"Error in native tools calling: {e}")
            return None

    def _chat_with_structured_output(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback to structured output parsing for models without tool support.

        Args:
            messages: Chat messages
            functions: OpenAI-format function definitions
            function_call: Function to call
            **kwargs: Additional parameters
                reasoning (bool): Override global reasoning setting

        Returns:
            Optional[Dict[str, Any]]: Response with parsed function call
        """
        try:
            # Find the target function
            target_function_name = function_call.get("name")
            target_function = next(
                (f for f in functions if f["name"] == target_function_name),
                None
            )

            if not target_function:
                logger.error(f"Function {target_function_name} not found")
                return None

            # Create a structured prompt with JSON schema
            schema_prompt = self._create_schema_prompt(target_function)
            enhanced_messages = messages.copy()
            enhanced_messages.append({
                "role": "system",
                "content": schema_prompt
            })

            temperature = kwargs.get("temperature", 0.3)  # Lower temp for structured output

            # Check if reasoning should be enabled for this request
            use_reasoning = kwargs.get("reasoning", self.reasoning_enabled)
            use_reasoning = use_reasoning and self._is_reasoning_model()

            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": enhanced_messages,
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_thread": self.num_thread,
                },
                "format": "json",  # Request JSON output
            }

            # Add think parameter if reasoning is enabled
            if use_reasoning:
                api_params["think"] = True
                logger.debug(f"Reasoning enabled for structured output (model: {self.model})")

            response = self.client.chat(**api_params)

            # Process content to handle thinking
            processed_content = self._process_response(response, use_reasoning)

            # Try to parse JSON from response
            parsed_args = self._extract_json(processed_content)

            if parsed_args:
                return {
                    "content": processed_content,
                    "role": "assistant",
                    "finish_reason": "stop",
                    "function_call": {
                        "name": target_function_name,
                        "arguments": json.dumps(parsed_args),
                    }
                }

            logger.warning(f"Could not parse function arguments from response: {processed_content}")
            return None

        except Exception as e:
            logger.error(f"Error in structured output parsing: {e}")
            return None

    def _convert_functions_to_tools(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI function format to Ollama tools format.

        Args:
            functions: OpenAI-format function definitions

        Returns:
            List[Dict[str, Any]]: Ollama tools format
        """
        tools = []
        for func in functions:
            tool = {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"],
                }
            }
            tools.append(tool)
        return tools

    def _create_schema_prompt(self, function: Dict[str, Any]) -> str:
        """
        Create a prompt with JSON schema for structured output.

        Args:
            function: Function definition with schema

        Returns:
            str: Formatted prompt
        """
        schema = function["parameters"]
        required = schema.get("required", [])

        prompt = f"""You must respond with valid JSON matching this exact schema for the function '{function['name']}':

Function: {function['name']}
Description: {function['description']}

JSON Schema:
{json.dumps(schema, indent=2)}

Required fields: {', '.join(required)}

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text response.

        Args:
            text: Text potentially containing JSON

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON or None
        """
        try:
            # Try to parse the entire text as JSON first
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text using regex
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        return None

    def get_provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: "ollama"
        """
        return "ollama"
