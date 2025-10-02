#!/usr/bin/env python3
"""
Test script for programmatic reasoning detection.

This script tests the new programmatic reasoning detection mechanism
with real Ollama models.
"""

import sys
from services.providers.ollama_provider import OllamaProvider


def test_model_reasoning_support(model_name):
    """Test reasoning support detection for a specific model."""
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
        supports_reasoning = provider._is_reasoning_model()

        print(f"✓ Model loaded successfully")
        print(f"✓ Reasoning support: {'YES' if supports_reasoning else 'NO'}")

        # Show template snippet for verification
        try:
            response = provider.client.show(model_name)
            template = response.get("template", "")
            has_think_set = "IsThinkSet" in template
            has_thinking = ".Thinking" in template

            print(f"✓ Template contains 'IsThinkSet': {has_think_set}")
            print(f"✓ Template contains '.Thinking': {has_thinking}")

            if has_think_set or has_thinking:
                # Show a snippet of the reasoning-related template
                lines = template.split('\n')
                reasoning_lines = [line for line in lines if 'IsThinkSet' in line or '.Thinking' in line or '<think>' in line]
                if reasoning_lines:
                    print(f"\nTemplate snippet:")
                    for line in reasoning_lines[:5]:  # Show first 5 relevant lines
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"✗ Error getting template: {e}")

        return supports_reasoning

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Run reasoning detection tests on available models."""
    print("\n" + "="*80)
    print("Ollama Reasoning Detection - Test Suite")
    print("="*80)
    print("\nTesting programmatic detection of reasoning support...")

    # Test models (only test models that exist locally)
    test_models = [
        "qwen3:14b",
        "mistral:7b",
        "gpt-oss:20b",  # User's model
    ]

    results = {}
    for model in test_models:
        result = test_model_reasoning_support(model)
        results[model] = result

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)

    for model, supports in results.items():
        if supports is None:
            status = "ERROR"
        elif supports:
            status = "✓ Supports reasoning"
        else:
            status = "✗ No reasoning support"
        print(f"{model:30s} {status}")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
