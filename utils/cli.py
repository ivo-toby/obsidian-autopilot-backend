"""
Command line interface configuration.

This module handles the setup and parsing of command line arguments.
"""

import argparse


def setup_argparser() -> argparse.ArgumentParser:
    """
    Configure and return the argument parser for the application.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Note Summarizer and Knowledge Base Manager"
    )

    # Basic arguments
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to files or create reminders",
    )

    # Create subparsers for different command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Notes processing commands
    notes_parser = subparsers.add_parser("notes", help="Process daily and weekly notes")
    notes_parser.add_argument(
        "--date",
        type=str,
        help="Date to fetch notes from (or start date for weekly notes)",
    )
    notes_parser.add_argument(
        "--weekly", action="store_true", help="Process weekly notes"
    )
    notes_parser.add_argument(
        "--skip-reminders", action="store_true", help="Do not create reminders"
    )
    notes_parser.add_argument(
        "--replace-summary",
        action="store_true",
        help="Replace existing summary instead of appending",
    )
    notes_parser.add_argument(
        "--meetingnotes",
        action="store_true",
        help="Generate and save meeting notes",
    )
    notes_parser.add_argument(
        "--process-learnings", action="store_true", help="Process new learnings"
    )
    notes_parser.add_argument(
        "--from-clipboard",
        action="store_true",
        help="Read a full meeting transcript from the system clipboard instead of daily notes"
    )
    notes_parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a custom prompt file for the meeting transcript feature"
    )

    # Vector store commands
    kb_parser = subparsers.add_parser("kb", help="Knowledge base operations")
    kb_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex of all notes in the vector store",
    )
    kb_parser.add_argument(
        "--update",
        action="store_true",
        help="Update only modified notes in the vector store (faster than full reindex)",
    )
    kb_parser.add_argument(
        "--query", type=str, help="Search for similar content in the vector store"
    )
    kb_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return for vector store queries",
    )
    kb_parser.add_argument(
        "--show-connections",
        type=str,
        metavar="NOTE_PATH",
        help="Show connections for a specific note",
    )
    kb_parser.add_argument(
        "--find-by-tag", type=str, metavar="TAG", help="Find notes with specific tag"
    )
    kb_parser.add_argument(
        "--find-by-date",
        type=str,
        metavar="DATE",
        help="Find notes from specific date (YYYY-MM-DD)",
    )
    kb_parser.add_argument(
        "--note-structure",
        type=str,
        metavar="NOTE_PATH",
        help="Display semantic structure of a note",
    )
    kb_parser.add_argument(
        "--analyze-links",
        type=str,
        metavar="NOTE_PATH",
        help="Analyze links and relationships for a note",
    )
    kb_parser.add_argument(
        "--auto-link",
        action="store_true",
        help="Automatically add suggested links without prompting",
    )
    kb_parser.add_argument(
        "--analyze-all",
        action="store_true",
        help="Analyze and auto-link all notes in the knowledge base",
    )
    kb_parser.add_argument(
        "--analyze-updated",
        action="store_true",
        help="Analyze and auto-link only recently modified notes",
    )
    kb_parser.add_argument(
        "--note-type",
        type=str,
        choices=["daily", "weekly", "meeting", "learning", "note"],
        help="Filter results by note type",
    )
    kb_parser.add_argument(
        "--graph",
        action="store_true",
        help="Output note connections in Mermaid graph format",
    )
    kb_parser.add_argument(
        "--debug-store",
        action="store_true",
        help="Display debug information about the vector store contents",
    )

    return parser
