"""
Learning processing service.

This module handles the extraction, processing and storage of learning entries
from notes, including generating titles and tags using AI services.
"""

import os
import re
from typing import List, Tuple

from services.llm_service import LLMService
from utils.file_handler import load_notes, write_summary_to_file


class LearningService:
    """
    Service for processing and managing learning entries.

    This service handles loading learning entries from files, processing them
    with AI services to generate titles and tags, and saving them as individual
    markdown files.
    """

    def __init__(self, learnings_file: str, learnings_output_dir: str):
        """
        Initialize the learning service.

        Args:
            learnings_file (str): Path to the file containing learning entries
            learnings_output_dir (str): Directory to save processed learning files
        """
        self.learnings_file = os.path.expanduser(learnings_file)
        self.learnings_output_dir = os.path.expanduser(learnings_output_dir)

    def load_learnings(self) -> str:
        """
        Load learning entries from the configured file.

        Returns:
            str: Content of the learnings file
        """
        return load_notes(self.learnings_file)

    def identify_new_learnings(self, content: str) -> List[Tuple[str, str, str]]:
        """
        Extract learning entries from content using regex.

        Args:
            content (str): Raw content containing learning entries

        Returns:
            List[Tuple[str, str, str]]: List of (full_match, timestamp, learning) tuples
        """
        pattern = r"(\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [AP]M)\](.*?)(?=\n\[|\Z))"
        matches = re.findall(pattern, content, re.DOTALL)
        return [
            (full_match.strip(), timestamp, learning.strip())
            for full_match, timestamp, learning in matches
        ]

    def generate_markdown_file(
        self, timestamp: str, learning: str, title: str, tags: List[str]
    ) -> str:
        """
        Generate a markdown file for a learning entry.

        Args:
            timestamp (str): Timestamp of the learning entry
            learning (str): Content of the learning
            title (str): Generated title for the learning
            tags (List[str]): List of relevant tags

        Returns:
            str: Name of the generated file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.learnings_output_dir, exist_ok=True)

        # Clean the title and create a unique filename using timestamp
        clean_title = re.sub(r"[^\w\s-]", "", title.lower())
        clean_title = re.sub(r"[-\s]+", "_", clean_title).strip("-_")
        # Add timestamp to ensure uniqueness (remove spaces and : from timestamp)
        clean_timestamp = re.sub(r"[\s:]", "", timestamp)
        filename = f"{clean_title}_{clean_timestamp}.md"

        content = f"# {title}\n\n"
        content += f"Date: {timestamp}\n\n"
        content += f"## Learning\n{learning}\n\n"
        content += f"## Tags\n{' '.join(tags)}\n\n"

        file_path = os.path.join(self.learnings_output_dir, filename)
        write_summary_to_file(file_path, content)
        return filename

    def process_new_learnings(self, llm_service: LLMService) -> None:
        """
        Process all new learning entries.

        Loads learnings, generates titles and tags using AI, saves them as
        markdown files, and removes processed entries from the source file.

        Args:
            llm_service (LLMService): Service for AI-powered text generation

        Returns:
            None
        """
        content = self.load_learnings()
        learnings = self.identify_new_learnings(content)

        print(f"Processing {len(learnings)} learnings...")
        for full_match, timestamp, learning in learnings:
            print(f"Processing learning: {learning[:100]}...")
            title = llm_service.generate_learning_title(learning)
            print(f"Title: {title}")
            tags = llm_service.generate_learning_tags(learning)
            print(f"Tags: {tags}")

            self.generate_markdown_file(timestamp, learning, title, tags)

            # Remove the processed learning from the content
            content = content.replace(full_match, "").strip()

        # Remove any consecutive newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Write the updated content back to the file
        write_summary_to_file(self.learnings_file, content)
        print(
            "Processing complete. Processed learnings have been removed from the source file."
        )
