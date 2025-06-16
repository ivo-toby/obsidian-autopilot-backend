"""
Meeting notes service.

This module handles the generation and storage of meeting notes extracted from daily notes.
"""

import os
from datetime import datetime
from typing import Dict, Optional

from services.notes_service import NotesService
from services.openai_service import OpenAIService
from utils.file_handler import create_output_dir, write_summary_to_file
from utils.markdown import create_meeting_notes_content
import pyperclip
import re
from pathlib import Path


class MeetingService:
    def __init__(self, config: Dict):
        """
        Initialize the meeting service.

        Args:
            config (Dict): Application configuration
        """
        self.config = config
        self.notes_service = NotesService(config["daily_notes_file"])
        self.openai_service = OpenAIService(
            api_key=config["api_key"], model=config["model"], base_url=config.get("base_url")
        )

    def process_meeting_notes(
        self, date_str: Optional[str] = None, dry_run: bool = False
    ) -> None:
        """
        Process daily notes to generate structured meeting notes.

        Args:
            date_str (str, optional): Date to process notes for. Defaults to None.
            dry_run (bool): If True, don't write files

        Returns:
            None
        """
        notes = self.notes_service.load_notes()
        today_notes = self.notes_service.extract_today_notes(notes, date_str)

        if not today_notes:
            print("No notes found for today.")
            return

        meeting_notes = self.openai_service.generate_meeting_notes(today_notes)

        if not dry_run:
            for meeting in meeting_notes.get("meetings", []):
                self._save_meeting_notes(meeting)
        else:
            for meeting in meeting_notes.get("meetings", []):
                print(meeting)

    def process_meeting_transcript(
        self, date_str: Optional[str] = None, dry_run: bool = False,
        prompt_file: Optional[str] = None
    ) -> None:
        """
        Process a full meeting transcript from the clipboard to generate a summary.

        Args:
            date_str (str, optional): Date to use for the filename. Defaults to None (today).
            dry_run (bool): If True, don't write files. Defaults to False.
            prompt_file (str, optional): Path to the prompt template file. Defaults to None
                                        (uses prompts/MEETING_PROMPT.md).
        """
        transcript = pyperclip.paste()
        if not transcript.strip():
            print("Clipboard is empty or not text.")
            return

        # Load prompt template
        if prompt_file is None:
            prompt_file = Path(__file__).parent.parent / "prompts" / "MEETING_PROMPT.md"
        else:
            prompt_file = Path(prompt_file)

        if not prompt_file.exists():
            print(f"Prompt file not found: {prompt_file}")
            return

        template = prompt_file.read_text()

        # Combine template and transcript
        full_prompt = f"{template}\n\n'''TRANSCRIPT'''\n{transcript}"

        # Call the LLM
        summary_text = self.openai_service.generate_text(full_prompt)

        # Output or save
        if dry_run:
            print(summary_text)
        else:
            self._save_raw_summary(summary_text, date_str)

    def _infer_topic_from_summary(self, summary_text: str) -> str:
        """
        Infer a descriptive topic name from the full meeting summary.

        Uses the LLM to generate a concise topic suitable for a filename.
        Falls back to extracting from the heading if LLM inference fails.

        Args:
            summary_text (str): The full meeting summary text

        Returns:
            str: A descriptive topic name (lowercased with underscores)
        """
        # First try to extract from heading as a fallback
        match = re.match(r"#\s*\[(.*?)\]", summary_text)
        extracted_topic = match.group(1).strip() if match else "meeting"

        try:
            # Ask the LLM to infer a better topic name
            prompt = f"""Based on the following meeting summary, create a short, descriptive topic name (3-5 words maximum):

{summary_text}

Return ONLY the topic name, nothing else. This will be used as a filename."""

            inferred_topic = self.openai_service.generate_text(prompt).strip()

            # Clean up the inferred topic - remove any markdown, quotes, etc.
            inferred_topic = re.sub(r'[^\w\s-]', '', inferred_topic)
            inferred_topic = inferred_topic.strip()

            # If we got something reasonable, use it; otherwise fall back to extracted topic
            if inferred_topic and len(inferred_topic) > 2 and len(inferred_topic) < 50:
                return inferred_topic.replace(" ", "_").lower()
            else:
                return extracted_topic.replace(" ", "_").lower()
        except Exception as e:
            print(f"Error inferring topic: {e}")
            return extracted_topic.replace(" ", "_").lower()

    def _save_raw_summary(
        self, summary_text: str, date_str: Optional[str] = None
    ) -> None:
        """
        Save raw summary markdown text using configured output directory and naming.
        """
        ds = date_str or datetime.now().strftime("%Y-%m-%d")

        # Infer a descriptive topic from the summary
        topic = self._infer_topic_from_summary(summary_text)

        filename = f"{ds}_{topic}.md"
        outdir = create_output_dir(
            os.path.expanduser(self.config["meeting_notes_output_dir"])
        )
        target_path = os.path.join(outdir, filename)
        write_summary_to_file(target_path, summary_text)

    def _save_meeting_notes(
        self, meeting_data: Dict, output_dir: Optional[str] = None
    ) -> None:
        """
        Save meeting notes to a markdown file.

        Args:
            meeting_data (Dict): Meeting information including subject, participants, etc.
            output_dir (str, optional): Override default output directory. Defaults to None.

        Returns:
            None
        """
        output_dir = output_dir or self.config["meeting_notes_output_dir"]
        date_str = meeting_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        subject = (
            meeting_data.get("meeting_subject", "meeting").replace(" ", "_").lower()
        )
        file_name = f"{date_str}_{subject}"

        content = create_meeting_notes_content(meeting_data)

        output_dir = create_output_dir(os.path.expanduser(f"{output_dir}"))
        output_file = os.path.join(output_dir, f"{file_name}.md")

        write_summary_to_file(output_file, content)
