"""
File handling utilities.

This module provides functions for reading and writing files,
with support for home directory expansion and error handling.
"""

import os
from pathlib import Path


def load_notes(filename: str) -> str:
    """
    Load and read contents from a file.

    Args:
        filename (str): Path to the file to read, supports ~ expansion

    Returns:
        str: Contents of the file, or empty string if file not found
    """
    expanded_filename = os.path.expanduser(filename)
    try:
        with open(expanded_filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""
    except PermissionError:
        raise PermissionError(f"Permission denied reading file: {expanded_filename}")


def write_summary_to_file(filename: str, content: str) -> None:
    """
    Write content to a file, creating parent directories if needed.

    Args:
        filename (str): Path to the file to write, supports ~ expansion
        content (str): Content to write to the file

    Raises:
        PermissionError: If the file can't be written to
        OSError: If there's an error writing to the file
    """
    expanded_filename = os.path.expanduser(filename)
    try:
        # Create parent directories if they don't exist
        Path(os.path.dirname(expanded_filename)).mkdir(parents=True, exist_ok=True)

        # Write the content
        with open(expanded_filename, "w", encoding="utf-8") as file:
            file.write(content)
    except PermissionError:
        raise PermissionError(f"Permission denied writing to file: {expanded_filename}")
    except OSError as e:
        raise OSError(f"Error writing to file {expanded_filename}: {str(e)}")


def create_output_dir(output_dir: str) -> str:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir (str): Path to create, supports ~ expansion

    Returns:
        str: Expanded path to the created directory

    Raises:
        PermissionError: If the directory can't be created
        OSError: If there's an error creating the directory
    """
    expanded_dir = os.path.expanduser(output_dir)
    try:
        Path(expanded_dir).mkdir(parents=True, exist_ok=True)
        return expanded_dir
    except PermissionError:
        raise PermissionError(f"Permission denied creating directory: {expanded_dir}")
    except OSError as e:
        raise OSError(f"Error creating directory {expanded_dir}: {str(e)}")
