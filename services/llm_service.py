"""
LLM service providing unified interface to multiple providers.

This module provides a unified interface for LLM operations across different
providers (OpenAI, Ollama). It maintains API compatibility with the legacy
OpenAIService while supporting the new provider abstraction.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from services.providers import OpenAIProvider, OllamaProvider
from services.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class LLMService:
    """
    Unified LLM service supporting multiple providers.

    This service provides a consistent interface for text generation and chat
    operations regardless of the underlying provider (OpenAI, Ollama, etc.).
    It maintains backward compatibility with the OpenAIService API.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM service with appropriate provider.

        Args:
            config (Dict[str, Any]): Configuration dictionary

        Raises:
            ValueError: If provider type is unsupported
        """
        self.config = config
        self.provider = self._initialize_provider(config)
        logger.info(
            f"Initialized LLM service with provider: {self.provider.get_provider_name()}"
        )

    def _initialize_provider(self, config: Dict[str, Any]) -> BaseProvider:
        """
        Initialize the appropriate provider based on configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary

        Returns:
            BaseProvider: Initialized provider instance

        Raises:
            ValueError: If provider type is unsupported
        """
        # Check for new inference configuration
        if "inference" in config and "provider" in config["inference"]:
            provider_type = config["inference"]["provider"]
        else:
            # Legacy config: detect based on base_url
            base_url = config.get("base_url", "")
            if "localhost:11434" in base_url or "ollama" in base_url:
                logger.warning(
                    "Detected Ollama usage via base_url. "
                    "Consider updating config to use 'inference.provider: ollama'"
                )
                provider_type = "openai"  # Keep using OpenAI SDK for now
            else:
                provider_type = "openai"

        # Initialize provider
        if provider_type == "openai":
            return OpenAIProvider(config)
        elif provider_type == "ollama":
            return OllamaProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    # Backward-compatible API methods matching OpenAIService

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt (str): Input prompt
            **kwargs: Additional provider-specific parameters
                reasoning (bool): Enable reasoning mode (for supported models)

        Returns:
            str: Generated text
        """
        return self.provider.generate_text(prompt, **kwargs)

    def generate_learning_title(self, learning: str) -> str:
        """
        Generate a title for a learning entry.

        Args:
            learning (str): Learning entry content

        Returns:
            str: Generated title
        """
        prompt = (
            f"Generate a concise short title for the following learning:\n\n{learning}"
        )
        try:
            response = self.provider.generate_text(prompt)
            return response.strip() if response else "Untitled Learning"
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "Untitled Learning"

    def generate_learning_tags(self, learning: str) -> List[str]:
        """
        Generate relevant tags for a learning entry.

        Args:
            learning (str): Learning entry content

        Returns:
            List[str]: List of generated tags
        """
        prompt = (
            "Generate relevant tags for the following learning, formatted in snake-case, "
            "each tag should be prefixed with a #-sign, split the tags with a , :\n\n"
            f"{learning}"
        )
        try:
            response = self.provider.generate_text(prompt)
            if response:
                return [tag.strip() for tag in response.split(",")]
            return []
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []

    def chat_completion_with_function(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Make a chat completion request with function calling.

        This method maintains compatibility with OpenAI's function calling format
        while supporting multiple providers.

        Args:
            messages (List[Dict[str, str]]): Chat messages
            functions (List[Dict[str, Any]]): Function definitions
            function_call (Dict[str, str]): Function to call
            **kwargs: Additional provider-specific parameters
                reasoning (bool): Enable reasoning mode (for supported models)

        Returns:
            Optional[Dict[str, Any]]: Response with function call or None if error
        """
        result = self.provider.chat_completion_with_function(
            messages, functions, function_call, **kwargs
        )

        if result and "function_call" in result:
            # Convert to legacy format expected by existing code
            # Create a mock object that mimics OpenAI's response structure
            class FunctionCall:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = arguments

            class Message:
                def __init__(self, content, function_call_dict):
                    self.content = content
                    if function_call_dict:
                        self.function_call = FunctionCall(
                            function_call_dict["name"], function_call_dict["arguments"]
                        )
                    else:
                        self.function_call = None

            return Message(result.get("content"), result.get("function_call"))

        return None

    def summarize_notes_and_identify_tasks(
        self, notes: str
    ) -> Optional[Dict[str, Any]]:
        """
        Summarize notes and extract tasks.

        Args:
            notes (str): Notes content to process

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing summary, tasks, and tags
        """
        prompt = f"""Analyze the following journal entries and use the create_meeting_notes function to provide a structured summary.

Journal entries:
{notes}

Use the function to extract:
- summary: An easy-to-read daily summary in Markdown format capturing all knowledge, links, and facts
- actionable_items: List of tasks or actions identified that are actionable by the note owner
- tags: Relevant tags in snake_case format that categorize the content themes"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that structures information using available tools. Always use the provided function to structure your response.",
            },
            {"role": "user", "content": prompt},
        ]

        functions = [
            {
                "name": "create_meeting_notes",
                "description": "Create meeting notes from the journal entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "actionable_items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "actionable_items", "tags"],
                },
            }
        ]
        function_call = {"name": "create_meeting_notes"}

        response = self.chat_completion_with_function(
            messages, functions, function_call
        )
        if response and response.function_call and response.function_call.arguments:
            return json.loads(response.function_call.arguments)
        return None

    def generate_weekly_summary(self, notes: str) -> Optional[str]:
        """
        Generate a weekly summary from notes.

        Args:
            notes (str): Notes content to summarize

        Returns:
            Optional[str]: Generated summary or None if processing fails
        """
        prompt = f"""
        Given the provided journal entries, please generate an easy-to-read weekly journal in Markdown format, which captures all the knowledge, links, and facts from the journal entries for future reference.
        Following the summary, create a section that enumerates accomplishments based on the journal entries.
        Following the accomplishments, create a section called Learnings, and list any learnings identified within the journal entries.

        Conclude with a list of links extracted from the journal entries, formatted in Markdown and infer a title for each link based on the URL or context in which the link was originally found.

        Example:
        Journal entry: "[2024-05-21 02:38:09 PM] The team discussed the upcoming project launch, [focusing on the marketing strategy](http://www.link.com), budget allocations, and the final review of the product design. Tasks were assigned to finalize the promotional materials and secure additional funding."

        Summary:
        - [2024-05-21 02:38:09 PM] Discussed upcoming product launch, focusing on the marketing strategy, budget allocations, and product design finalization.

        Accomplishments:
        - Finalized promotional materials.
        - Secured additional funding.

        Learnings:
        - Importance of clear communication in marketing strategies.
        - Budget allocation challenges.

        Links:
        - [Marketing Strategy](http://www.link.com)

        Weekly journal entries:
        {notes}
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and a genius summarizer.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            result = self.provider.chat_completion(messages)
            return result.get("content")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def generate_meeting_notes(
        self, notes: str
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Extract and format meeting notes from general notes.

        Args:
            notes (str): Notes content to process

        Returns:
            Optional[Dict[str, List[Dict[str, Any]]]]: Dictionary containing list of meeting notes
        """
        prompt = f"""
From the following journal entries, infer which entries may have been taken during a meeting or call. For each meeting or call, extract details to create meeting notes in Markdown format based on this template:
# {{date}} Meeting Notes - {{meeting_subject}}
## Tags
{{tags}}
## Participants
- {{participant_1}}
- {{participant_2}}
## Meeting notes
{{meeting_notes}}
## Decisions
## Action items
## References

Example:
Journal entry: "[2024-05-22 01:00:00 PM] Meeting on Project X. Participants: Alice, Bob. Discussed project timelines, potential risks, and mitigation strategies. Decisions made to accelerate phase 1 and review phase 2 next week. Action items: Alice to draft phase 1 report, Bob to set up a client meeting. Reference: [Project docs](http://www.link.com)."
Journal entry: "[2024-05-22 04:00:00 PM] Call on Project Y. Participants: John. Discussed project budget, marketing strategies. Decisions made to accelerate phase 1 and review phase 2 next week. Action items: Alice to draft phase 1 report, Bob to set up a client meeting. Reference: [Project docs](http://www.link.com)."

# 2024-05-22 Meeting Notes - Project X
## Tags
project_x, timeline, risks
## Participants
- Alice
- Bob
## Meeting notes
Discussed project timelines, potential risks, and mitigation strategies.
## Decisions
Accelerate phase 1 and review phase 2 next week.
## Action items
- Alice to draft phase 1 report.
- Bob to set up a client meeting.
## References
[Project docs](http://www.link.com)

# 2024-05-22 Meeting Notes - Project Y
## Tags
project_y, marketing, budget
## Participants
- John
## Meeting notes
Discussed project budget, marketing strategies.
## Decisions
Accelerate phase 1 and review phase 2 next week.
## Action items
- Alice to draft phase 1 report.
- Bob to set up a client meeting.
## References
[Project docs](http://www.link.com)

Journal entries:\n{notes}"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and a genius summarizer.",
            },
            {"role": "user", "content": prompt},
        ]

        functions = [
            {
                "name": "create_meeting_notes",
                "description": "Generate meeting notes from provided journal entries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "meetings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string"},
                                    "meeting_subject": {"type": "string"},
                                    "tags": {"type": "string"},
                                    "participants": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "meeting_notes": {"type": "string"},
                                    "decisions": {"type": "string"},
                                    "action_items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "references": {"type": "string"},
                                },
                                "required": [
                                    "date",
                                    "meeting_subject",
                                    "tags",
                                    "participants",
                                    "meeting_notes",
                                ],
                            },
                        }
                    },
                    "required": ["meetings"],
                },
            }
        ]
        function_call = {"name": "create_meeting_notes"}

        response = self.chat_completion_with_function(
            messages, functions, function_call
        )

        if response and response.function_call and response.function_call.arguments:
            try:
                return json.loads(response.function_call.arguments)
            except Exception as e:
                logger.error(f"Error parsing meeting notes: {e}")
                return None
        return None
