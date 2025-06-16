"""
Summary service for processing daily and weekly notes.

This module handles the generation and storage of note summaries.
"""

import fnmatch
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from services.notes_service import NotesService
from services.openai_service import OpenAIService
from services.reminder_service import ReminderService
from utils.date_utils import get_date_str
from utils.file_handler import create_output_dir, write_summary_to_file
from utils.markdown import create_daily_summary_content, create_weekly_summary_content


class SummaryService:
    def __init__(self, config: Dict):
        """
        Initialize the summary service.

        Args:
            config (Dict): Application configuration
        """
        self.config = config
        self.notes_service = NotesService(config["daily_notes_file"])
        self.openai_service = OpenAIService(
            api_key=config["api_key"], model=config["model"], base_url=config.get("base_url")
        )

    def get_all_notes(self) -> List[Dict[str, str]]:
        """
        Get all notes from the entire vault.

        Returns:
            List[Dict[str, str]]: List of notes with their metadata
        """
        entries = []
        notes_dir = os.path.expanduser(
            self.config.get("notes_base_dir", "~/Documents/notes")
        )
        exclude_patterns = self.config.get("knowledge_base", {}).get(
            "exclude_patterns", []
        )

        def should_exclude(path: str) -> bool:
            """Check if path matches any exclude pattern."""
            rel_path = os.path.relpath(path, notes_dir)
            return any(
                fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns
            )

        def get_note_type(filepath: str) -> str:
            """Determine the type of note based on its location and content."""
            filepath_lower = filepath.lower()
            if "daily" in filepath_lower:
                return "daily"
            elif "weekly" in filepath_lower:
                return "weekly"
            elif "meetingnotes" in filepath_lower:
                return "meeting"
            elif "learnings" in filepath_lower:
                return "learning"
            return "note"  # default type

        # Recursively walk through all directories
        for root, dirs, files in os.walk(notes_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]

            for file in files:
                if not file.endswith(".md") or should_exclude(os.path.join(root, file)):
                    continue

                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, notes_dir)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Try to extract date from filename or content
                    date = None
                    if (
                        file.startswith("20") and len(file) >= 10
                    ):  # Filename like 2024-02-16
                        date = file[:10]
                    else:
                        # Look for date in first line of content
                        first_line = content.split("\n")[0]
                        if first_line.startswith("[20"):  # [2024-02-16
                            date = first_line[1:11]

                    note_type = get_note_type(filepath)

                    entries.append(
                        {
                            "id": rel_path,  # Use relative path as ID
                            "content": content,
                            "date": date,
                            "source": rel_path,
                            "type": note_type,
                            "filename": file,
                            "path": filepath,
                            "modified_time": os.path.getmtime(filepath),
                        }
                    )
                except Exception as e:
                    print(f"Error processing file {filepath}: {str(e)}")
                    continue

        return entries

    def process_daily_notes(
        self,
        date_str: Optional[str] = None,
        dry_run: bool = False,
        skip_reminders: bool = False,
        replace_summary: bool = False,
    ) -> None:
        """
        Process daily notes to generate summaries and extract tasks.

        Args:
            date_str (str, optional): Date to process notes for. Defaults to None.
            dry_run (bool): If True, don't write files or create reminders
            skip_reminders (bool): If True, don't create reminders
            replace_summary (bool): If True, replace existing summary

        Returns:
            None
        """
        notes = self.notes_service.load_notes()
        today_notes = self.notes_service.extract_today_notes(notes, date_str)

        if not today_notes:
            print("No notes found for today.")
            return

        result = self.openai_service.summarize_notes_and_identify_tasks(today_notes)
        summary = result.get("summary", "No summary available")
        tasks = result.get("actionable_items", [])
        tags = result.get("tags", [])

        self._display_results(summary, tasks, tags)

        if not dry_run:
            if not skip_reminders:
                self._add_tasks_to_reminders(tasks)

            self._write_daily_summary(
                summary, tasks, tags, today_notes, replace_summary, date_str
            )

    def process_weekly_notes(
        self,
        date_str: Optional[str] = None,
        dry_run: bool = False,
        replace_summary: bool = False,
    ) -> None:
        """
        Process and generate weekly note summaries.

        Args:
            date_str (str, optional): Date to process notes for. Defaults to None.
            dry_run (bool): If True, don't write files
            replace_summary (bool): If True, replace existing summary

        Returns:
            None
        """
        notes = self.notes_service.load_notes()
        weekly_notes = self.notes_service.extract_weekly_notes(notes, date_str)

        if not weekly_notes:
            print("No notes found for the past week.")
            return

        weekly_summary = self.openai_service.generate_weekly_summary(weekly_notes)
        self._display_results(weekly_summary, [], [])

        if not dry_run:
            self._write_weekly_summary(
                weekly_summary, weekly_notes, replace_summary, date_str
            )

    def _display_results(self, summary: str, tasks: List[str], tags: List[str]) -> None:
        """Display the processed results to the console."""
        print("Summary:")
        print(summary)
        print("Tags:")
        print(" ".join([f"#{tag}" for tag in tags]))
        print("Action Items:")
        for task in tasks:
            print(f"- {task}")

    def _add_tasks_to_reminders(self, tasks: List[str]) -> None:
        """Add identified tasks to the system reminders."""
        for task in tasks:
            ReminderService.add_to_reminders(task)

    def _write_daily_summary(
        self,
        summary: str,
        tasks: List[str],
        tags: List[str],
        today_notes: str,
        replace_summary: bool,
        today_str: Optional[str],
    ) -> None:
        """Write the daily summary to a file."""
        if today_str is not None:
            try:
                datetime.strptime(today_str, "%Y-%m-%d")
            except ValueError:
                print("Invalid date string. Using today's date instead.")
                today_str = get_date_str()
        else:
            today_str = get_date_str()

        now = datetime.strptime(today_str, "%Y-%m-%d")
        year = now.strftime("%Y")
        month = now.strftime("%m")
        week_number = now.strftime("%U")

        output_dir = create_output_dir(
            os.path.expanduser(
                f"{self.config['daily_output_dir']}/{year}/{month}/{week_number}"
            )
        )
        output_file = os.path.join(output_dir, f"{today_str}.md")

        if replace_summary or not os.path.exists(output_file):
            content = create_daily_summary_content(summary, tasks, tags, today_notes)
        else:
            with open(output_file, "r", encoding="utf-8") as file:
                existing_content = file.read()
            additional_content = create_daily_summary_content(
                summary, tasks, tags, today_notes
            )
            content = existing_content + "\n\n" + additional_content

        write_summary_to_file(output_file, content)

    def _write_weekly_summary(
        self,
        weekly_summary: str,
        weekly_notes: str,
        replace_summary: bool,
        date_str: Optional[str] = None,
    ) -> None:
        """Write the weekly summary to a file."""
        current_date = (
            datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
        )

        year = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        week_number = current_date.strftime("%U")

        output_dir = create_output_dir(
            os.path.expanduser(
                f"{self.config['weekly_output_dir']}/{year}/{month}/{week_number}"
            )
        )
        output_file = os.path.join(output_dir, "week_summary.md")

        if replace_summary or not os.path.exists(output_file):
            content = create_weekly_summary_content(weekly_summary, weekly_notes)
        else:
            with open(output_file, "r", encoding="utf-8") as file:
                existing_content = file.read()
            additional_content = create_weekly_summary_content(
                weekly_summary, weekly_notes
            )
            content = existing_content + "\n\n" + additional_content

        start_date = current_date - timedelta(days=current_date.weekday())
        end_date = start_date + timedelta(days=6)
        date_range = (
            f"# Weekly Summary: {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')}\n\n"
        )
        content = date_range + content

        write_summary_to_file(output_file, content)
