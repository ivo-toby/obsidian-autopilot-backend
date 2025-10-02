#!/usr/bin/env python3
"""
Manual test script for Ollama reasoning mode with Qwen3.

This script tests the reasoning mode implementation with a real Qwen3 model.
"""

import sys
import yaml
from services.llm_service import LLMService


def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_reasoning_disabled():
    """Test with reasoning disabled (should not include thinking)."""
    print("\n" + "="*80)
    print("TEST 1: Reasoning Disabled (Global)")
    print("="*80)

    config = load_config()
    # Ensure reasoning is disabled in config
    if "reasoning" not in config["inference"]["ollama"]:
        config["inference"]["ollama"]["reasoning"] = {}
    config["inference"]["ollama"]["reasoning"]["enabled"] = False

    llm_service = LLMService(config)

    prompt = "What is 15 * 24? Think step by step."
    print(f"\nPrompt: {prompt}")
    print("\nExpected: Direct answer without thinking process")
    print("\nResponse:")
    print("-" * 80)

    response = llm_service.generate_text(prompt)
    print(response)
    print("-" * 80)

    return response


def test_reasoning_enabled_suppressed():
    """Test with reasoning enabled but thinking suppressed."""
    print("\n" + "="*80)
    print("TEST 2: Reasoning Enabled, Thinking Suppressed (Default)")
    print("="*80)

    config = load_config()
    config["inference"]["ollama"]["reasoning"] = {
        "enabled": True,
        "save_thinking": False,
        "log_thinking": True,  # Log to console for verification
        "models": ["qwen3", "qwen2.5", "deepseek-r1", "qwq", "smallthinker"]
    }

    llm_service = LLMService(config)

    prompt = "What is 15 * 24? Think step by step."
    print(f"\nPrompt: {prompt}")
    print("\nExpected: Direct answer (thinking logged but not in output)")
    print("\nResponse:")
    print("-" * 80)

    response = llm_service.generate_text(prompt)
    print(response)
    print("-" * 80)

    return response


def test_reasoning_enabled_saved():
    """Test with reasoning enabled and thinking saved in output."""
    print("\n" + "="*80)
    print("TEST 3: Reasoning Enabled, Thinking Saved in Output")
    print("="*80)

    config = load_config()
    config["inference"]["ollama"]["reasoning"] = {
        "enabled": True,
        "save_thinking": True,
        "log_thinking": False,
        "models": ["qwen3", "qwen2.5", "deepseek-r1", "qwq", "smallthinker"]
    }

    llm_service = LLMService(config)

    prompt = "What is 15 * 24? Think step by step."
    print(f"\nPrompt: {prompt}")
    print("\nExpected: Output includes <thinking>...</thinking> tags with reasoning process")
    print("\nResponse:")
    print("-" * 80)

    response = llm_service.generate_text(prompt)
    print(response)
    print("-" * 80)

    return response


def test_reasoning_override():
    """Test per-request reasoning override."""
    print("\n" + "="*80)
    print("TEST 4: Reasoning Override (Global Disabled, Override Enabled)")
    print("="*80)

    config = load_config()
    config["inference"]["ollama"]["reasoning"] = {
        "enabled": False,
        "save_thinking": True,
        "log_thinking": False,
        "models": ["qwen3", "qwen2.5", "deepseek-r1", "qwq", "smallthinker"]
    }

    llm_service = LLMService(config)

    prompt = "What is 15 * 24? Think step by step."
    print(f"\nPrompt: {prompt}")
    print("\nExpected: Reasoning enabled via override, thinking should be included")
    print("\nResponse:")
    print("-" * 80)

    response = llm_service.generate_text(prompt, reasoning=True)
    print(response)
    print("-" * 80)

    return response


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Ollama Reasoning Mode - Manual Test Suite")
    print("="*80)
    print("\nThis script tests the reasoning mode implementation with Qwen3.")
    print("Make sure Ollama is running and qwen3:14b model is available.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    try:
        # Run all tests
        test_reasoning_disabled()
        test_reasoning_enabled_suppressed()
        test_reasoning_enabled_saved()
        test_reasoning_override()

        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80)
        print("\nVerification checklist:")
        print("✓ Test 1: Response should be direct without thinking process")
        print("✓ Test 2: Response should be direct, check logs for thinking content")
        print("✓ Test 3: Response should include <thinking>...</thinking> tags")
        print("✓ Test 4: Response should include thinking despite global setting")

    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
