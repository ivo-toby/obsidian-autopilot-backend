"""
OpenAI service integration.

This module provides a service wrapper for OpenAI API interactions,
handling various text generation tasks including summarization,
meeting notes extraction, and learning processing.
"""

from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam


class OpenAIService:
    """
    Service for interacting with OpenAI's API.

    This service handles all OpenAI API calls, providing methods for
    various text generation and processing tasks.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        """
        Initialize the OpenAI service.

        Args:
            api_key (str): OpenAI API key
            model (str, optional): Model to use for completions. Defaults to "gpt-4o-mini"
            base_url (str, optional): Base URL for the API. Defaults to None (uses OpenAI default)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_learning_title(self, learning: str) -> str:
        """
        Generate a title for a learning entry using AI.

        Args:
            learning (str): Learning entry content

        Returns:
            str: Generated title
        """
        prompt = (
            f"Generate a concise short title for the following learning:\n\n{learning}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            if response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "Untitled Learning"
        except Exception as e:
            print(f"Error generating title: {e}")
            return "Untitled Learning"

    def generate_learning_tags(self, learning: str) -> List[str]:
        """
        Generate relevant tags for a learning entry using AI.

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            if response.choices[0].message.content:
                return [
                    tag.strip()
                    for tag in response.choices[0].message.content.split(",")
                ]
            return []
        except Exception as e:
            print(f"Error generating tags: {e}")
            return []

    def chat_completion_with_function(
        self,
        messages: List[ChatCompletionMessageParam],
        functions: List[Dict[str, Any]],
        function_call: Dict[str, str],
    ) -> Optional[ChatCompletionMessage]:
        """
        Make a chat completion request with function calling.

        Args:
            messages (List[Dict[str, str]]): Chat messages
            functions (List[Dict[str, Any]]): Function definitions
            function_call (Dict[str, str]): Function to call

        Returns:
            Optional[Any]: Response message or None if error occurs

        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.6,
                messages=messages,
                functions=functions,
                function_call=function_call,
            )
            return response.choices[0].message
        except Exception as e:
            print(f"An error occurred during API call: {e}")
            return None

    def summarize_notes_and_identify_tasks(
        self, notes: str
    ) -> Optional[Dict[str, Any]]:
        """
        Summarize notes and extract tasks using AI.

        Args:
            notes (str): Notes content to process

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing summary, tasks, and tags
                                    or None if processing fails
        """
        prompt = f"""
        Given the provided journal entries, please generate an easy-to-read daily journal in Markdown format, which captures all the knowledge, links, and facts from the journal entries for future reference.
        Following the summary, enumerate any actionable items identified within the journal entries that are actionable by the owner of the notes.
        Conclude with a list of relevant tags, formatted in snake-case, that categorize the content or themes of the notes.

        Example:
        Journal entry: "[2024-05-21 02:38:09 PM] The team discussed the upcoming project launch, [focusing on the marketing strategy](http://www.link.com), budget allocations, and the final review of the product design. Tasks were assigned to finalize the promotional materials and secure additional funding."

        Summary: "[02:38:09 PM] Discussed upcoming product launch, [marketing strategies](http://www.link.com), budgeting, and product design finalization."

        Actionable Items:
        1. Finalize promotional materials.
        2. Secure additional funding.

        Tags: project_launch, marketing_strategy, budget_allocation, product_design

        Journal entries:\\n{notes}"""

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
        if response:
            if response and response.function_call and response.function_call.arguments:
                import json

                return json.loads(response.function_call.arguments)
            return None

    def generate_weekly_summary(self, notes: str) -> Optional[str]:
        """
        Generate a weekly summary from notes using AI.

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
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=1500
            )
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def generate_meeting_notes(
        self, notes: str
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Extract and format meeting notes from general notes using AI.

        Args:
            notes (str): Notes content to process

        Returns:
            Optional[Dict[str, List[Dict[str, Any]]]]: Dictionary containing list of meeting notes
                                                      or None if processing fails
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

        # Extract the arguments from the response function call
        if response and response.function_call and response.function_call.arguments:
            try:
                import json
                return json.loads(response.function_call.arguments)
            except Exception as e:
                print(f"Error parsing meeting notes: {e}")
                return None
        return None

    def generate_text(self, prompt: str) -> str:
        """Generate raw text from the given prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
