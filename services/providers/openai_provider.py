"""
OpenAI provider implementation.

This module provides the OpenAI implementation of the LLM provider interface.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from services.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider for LLM operations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI provider.

        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        inference_config = config.get("inference", {})
        openai_config = inference_config.get("openai", {})

        # Support both new and legacy config formats
        if "openai" in inference_config:
            # New config format - read from openai section first, then inference root, then legacy, then env
            self.api_key = (
                openai_config.get("api_key")
                or config.get("api_key")
                or os.environ.get("OPENAI_API_KEY", "")
            )
            self.base_url = openai_config.get("base_url")
            self.model = (
                openai_config.get("model") or inference_config.get("model") or "gpt-4o"
            )
        else:
            # Legacy config format - fall back to env if not in config
            self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
            self.base_url = config.get("base_url")
            self.model = config.get("model", "gpt-4o")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"Initialized OpenAI provider with model: {self.model}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using OpenAI.

        Args:
            prompt (str): The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return ""

    def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion from messages.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Response dictionary
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"Error in chat completion with OpenAI: {e}")
            return {"content": "", "role": "assistant", "finish_reason": "error"}

    def chat_completion_with_function(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a chat completion with function calling using OpenAI.

        Args:
            messages (List[Dict[str, str]]): Chat messages
            functions (List[Dict[str, Any]]): Function definitions (OpenAI format)
            function_call (Dict[str, str]): Function to call
            **kwargs: Additional parameters

        Returns:
            Optional[Dict[str, Any]]: Response message with function call or None
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call=function_call,
            )

            message = response.choices[0].message

            # Return structured response
            result = {
                "content": message.content,
                "role": message.role,
                "finish_reason": response.choices[0].finish_reason,
            }

            if message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                }

            return result

        except Exception as e:
            logger.error(f"Error in function calling with OpenAI: {e}")
            return None

    def get_provider_name(self) -> str:
        """
        Get the provider name.

        Returns:
            str: "openai"
        """
        return "openai"
