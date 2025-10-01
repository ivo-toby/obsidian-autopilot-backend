"""
Base provider interface for LLM implementations.

This module defines the abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt (str): The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Generated text response
        """
        pass

    @abstractmethod
    def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion from messages.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Response dictionary containing the completion
        """
        pass

    @abstractmethod
    def chat_completion_with_function(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a chat completion with function calling.

        Args:
            messages (List[Dict[str, str]]): Chat messages
            functions (List[Dict[str, Any]]): Function definitions
            function_call (Dict[str, str]): Function to call
            **kwargs: Additional provider-specific parameters

        Returns:
            Optional[Dict[str, Any]]: Response message or None if error occurs
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.

        Returns:
            str: Provider name (e.g., "openai", "ollama")
        """
        pass
