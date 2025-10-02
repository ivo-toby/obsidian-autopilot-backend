#!/usr/bin/env python3
"""
Test script for programmatic tool calling detection.

This script tests the new programmatic tool detection mechanism
with real Ollama models.
"""

import sys
from services.providers.ollama_provider import OllamaProvider


def test_model_tool_support(model_name):
    """Test tool support detection for a specific model."""
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print('='*80)

    config = {
        "inference": {
            "ollama": {
                "model": model_name,
                "base_url": "http://localhost:11434"
            }
        }
    }

    try:
        provider = OllamaProvider(config)
        supports_tools = provider._model_supports_tools()

        print(f"✓ Model loaded successfully")
        print(f"✓ Tool support: {'YES' if supports_tools else 'NO'}")

        # Show template snippet for verification
        try:
            response = provider.client.show(model_name)
            template = response.get("template", "")
            has_tools = ".Tools" in template
            has_toolcalls = ".ToolCalls" in template

            print(f"✓ Template contains '.Tools': {has_tools}")
            print(f"✓ Template contains '.ToolCalls': {has_toolcalls}")

            if has_tools or has_toolcalls:
                # Show a snippet of the tool-related template
                lines = template.split('\n')
                tool_lines = [line for line in lines if '.Tools' in line or '.ToolCalls' in line]
                if tool_lines:
                    print(f"\nTemplate snippet:")
                    for line in tool_lines[:5]:  # Show first 5 relevant lines
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"✗ Error getting template: {e}")

        return supports_tools

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Run tool detection tests on available models."""
    print("\n" + "="*80)
    print("Ollama Tool Calling Detection - Test Suite")
    print("="*80)
    print("\nTesting programmatic detection of tool calling support...")

    # Test models (only test models that exist locally)
    test_models = [
        "qwen3:14b",
        "mistral:7b",
        "gpt-oss:20b",  # User's model
    ]

    results = {}
    for model in test_models:
        result = test_model_tool_support(model)
        results[model] = result

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)

    for model, supports in results.items():
        if supports is None:
            status = "ERROR"
        elif supports:
            status = "✓ Supports tools"
        else:
            status = "✗ No tool support"
        print(f"{model:30s} {status}")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
