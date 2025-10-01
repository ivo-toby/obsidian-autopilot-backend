"""
Tests for configuration loading utilities.

This module contains tests for configuration loading functions.
"""

import os
from unittest.mock import patch, mock_open

import pytest
import yaml

from utils.config_loader import load_config


@pytest.fixture
def sample_config_yaml():
    """Return sample configuration YAML content."""
    return """
daily_notes_file: ~/notes/daily.md
daily_output_dir: ~/notes/daily
weekly_output_dir: ~/notes/weekly
notes_base_dir: ~/notes
api_key: test-api-key
model: test-model
knowledge_base:
  exclude_patterns:
    - "*.private/*"
    - "archive/*"
"""


def test_load_config_success(temp_dir, sample_config_yaml):
    """Test successful loading of configuration."""
    # Setup
    config_file = temp_dir / "config.yaml"
    config_file.write_text(sample_config_yaml)

    # Execute
    config = load_config(str(config_file))

    # Assert
    assert config["daily_notes_file"] == "~/notes/daily.md"
    assert config["daily_output_dir"] == "~/notes/daily"
    assert config["weekly_output_dir"] == "~/notes/weekly"
    assert config["notes_base_dir"] == "~/notes"
    assert config["api_key"] == "test-api-key"
    assert config["model"] == "test-model"
    assert config["knowledge_base"]["exclude_patterns"] == ["*.private/*", "archive/*"]


def test_load_config_file_not_found():
    """Test loading configuration when file doesn't exist."""
    # Setup
    non_existent_file = "/path/to/nonexistent/config.yaml"

    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        load_config(non_existent_file)


def test_load_config_invalid_yaml(temp_dir):
    """Test loading configuration with invalid YAML."""
    # Setup
    config_file = temp_dir / "config.yaml"
    config_file.write_text("invalid: yaml: content:")

    # Execute and Assert
    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))


def test_load_config_expands_home():
    """Test that load_config expands home directory."""
    # Setup
    config_content = """
    daily_notes_file: ~/notes/daily.md
    daily_output_dir: ~/notes/daily
    weekly_output_dir: ~/notes/weekly
    notes_base_dir: ~/notes
    api_key: test-api-key
    model: test-model
    """

    with patch("os.path.expanduser") as mock_expanduser, \
         patch("builtins.open", mock_open(read_data=config_content)):
        mock_expanduser.return_value = "/home/user/config.yaml"

        # Execute
        load_config("~/config.yaml")

        # Assert
        mock_expanduser.assert_called_once_with("~/config.yaml")


def test_load_config_missing_required_fields(temp_dir):
    """Test loading configuration with missing required fields."""
    # Setup
    config_file = temp_dir / "config.yaml"
    config_file.write_text("optional_field: value")

    # Execute and Assert
    with pytest.raises(KeyError, match="Required field missing"):
        load_config(str(config_file))


def test_load_config_with_new_inference_format(temp_dir):
    """Test loading configuration with new inference format (no legacy keys)."""
    config_content = """
daily_notes_file: ~/notes/daily.md
daily_output_dir: ~/notes/daily
weekly_output_dir: ~/notes/weekly
notes_base_dir: ~/notes
inference:
  provider: openai
  model: gpt-4o
  openai:
    api_key: test-api-key
    base_url: https://api.openai.com/v1
"""
    config_file = temp_dir / "config.yaml"
    config_file.write_text(config_content)

    # Execute
    config = load_config(str(config_file))

    # Assert
    assert config["inference"]["provider"] == "openai"
    assert config["inference"]["model"] == "gpt-4o"
    assert config["inference"]["openai"]["api_key"] == "test-api-key"
    # Should not have legacy keys
    assert "api_key" not in config or config.get("api_key") is None


def test_load_config_missing_llm_configuration(temp_dir):
    """Test that missing LLM configuration raises helpful error."""
    config_content = """
daily_notes_file: ~/notes/daily.md
daily_output_dir: ~/notes/daily
weekly_output_dir: ~/notes/weekly
notes_base_dir: ~/notes
"""
    config_file = temp_dir / "config.yaml"
    config_file.write_text(config_content)

    # Execute and Assert
    with pytest.raises(KeyError, match="Missing LLM configuration"):
        load_config(str(config_file))