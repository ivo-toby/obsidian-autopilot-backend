"""
Configuration loader module.

Provides functionality to load YAML configuration files with error handling.
"""

import os
import yaml
from yaml.parser import ParserError


def load_config(config_file: str) -> dict:
    """
    Load and parse a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file

    Returns:
        dict: Parsed configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ParserError: If the YAML file is invalid
        ValueError: If the configuration is empty
        KeyError: If required fields are missing
    """
    # Expand home directory in config file path
    config_file = os.path.expanduser(config_file)

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            if config is None:
                raise ValueError(f"Empty configuration file: {config_file}")

            # Validate required fields (note: api_key and model are now optional)
            required_fields = [
                "daily_notes_file",
                "daily_output_dir",
                "weekly_output_dir",
                "notes_base_dir",
            ]
            for field in required_fields:
                if field not in config:
                    raise KeyError(f"Required field missing in config: {field}")

            # Validate LLM configuration (either legacy or new format)
            has_legacy_config = "api_key" in config and "model" in config
            has_inference_config = "inference" in config and "provider" in config.get("inference", {})

            if not has_legacy_config and not has_inference_config:
                raise KeyError(
                    "Missing LLM configuration. Please provide either:\n"
                    "  - New format: inference.provider and inference.openai/ollama settings, OR\n"
                    "  - Legacy format: api_key and model"
                )

            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except ParserError as e:
        raise ParserError(f"Invalid YAML in configuration file {config_file}: {str(e)}")
