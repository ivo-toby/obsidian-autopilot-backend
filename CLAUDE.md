# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Obsidian Autopilot Backend is a Python tool for processing and enriching Obsidian notes. It provides automatic summarization, task extraction, semantic knowledge base features, and meeting transcript processing.

## Development Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
pip install -r requirements.txt
cp config.template.yaml config.yaml  # Edit with your settings
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/services/test_notes_service.py

# Run specific test
pytest tests/services/test_notes_service.py::test_function_name

# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit
```

### Development Tools (if installed via setup.py extras)
```bash
pip install -e ".[dev]"  # Install with dev dependencies
black .                  # Code formatting
flake8                   # Linting
mypy .                   # Type checking
isort .                  # Import sorting
```

## Architecture

### Service Layer Organization

**Core Processing Services:**
- `NotesService`: Basic note operations and reading from Obsidian vault
- `SummaryService`: Generates daily/weekly summaries, orchestrates note processing
- `MeetingService`: Processes meeting notes from daily notes or clipboard transcripts
- `LearningService`: Extracts and tags learning entries
- `ReminderService`: Integrates with Apple Reminders for task management
- `LLMService`: Unified LLM interface supporting multiple providers (OpenAI, Ollama)
  - `services/providers/OpenAIProvider`: OpenAI API integration
  - `services/providers/OllamaProvider`: Native Ollama SDK integration with tool support

**Knowledge Base Services:**
- `services/vector_store/`:
  - `EmbeddingService`: Generates embeddings (OpenAI or Ollama)
  - `ChunkingService`: Semantic document chunking
  - `VectorStoreService`: ChromaDB-based vector storage and similarity search
- `services/knowledge/`:
  - `LinkService`: Analyzes note relationships and manages Obsidian wiki-links and backlinks

### Key Workflows

**Daily Notes Processing** (`main.py:process_daily_notes`):
1. Reads daily notes file
2. Generates summary via LLMService (uses configured provider: OpenAI or Ollama)
3. Extracts tasks and creates Apple Reminders
4. Saves formatted output

**Knowledge Base Indexing** (`main.py:process_knowledge_base`):
1. `SummaryService.get_all_notes()` scans vault for markdown files
2. `ChunkingService` breaks documents into semantic chunks
3. `EmbeddingService` generates embeddings
4. `VectorStoreService` stores in ChromaDB with metadata (note type, tags, dates)

**Link Analysis** (`LinkService.analyze_relationships`):
1. Finds existing wiki-links and backlinks in note
2. Searches vector store for semantically similar content
3. Filters and ranks suggestions
4. Optionally updates notes with new links and backlinks

### Configuration

Uses `config.yaml` for all settings:
- File paths for notes directories (daily, weekly, meetings, learnings)
- **Inference provider selection** (`inference.provider`: "openai" or "ollama")
  - OpenAI: `inference.openai.api_key`, `inference.openai.model`, `inference.openai.base_url`
  - Ollama: `inference.ollama.model`, `inference.ollama.base_url`, temperature, num_ctx, num_thread, timeout
  - Each provider has its own model setting in its own section
- Embedding configuration (model_type: "openai" or "ollama") - separate from inference
- Vector store settings (path, similarity thresholds, HNSW index parameters)
- Search thresholds for different query types
- **Legacy config keys** (api_key, model, base_url) maintained for backward compatibility

## Common Tasks

### Processing clipboard meeting transcripts
```bash
# Use default meetingnotes prompt
python main.py notes --from-clipboard

# Use custom prompt file
python main.py notes --from-clipboard --prompt-file /path/to/prompt.md
```

### Working with the knowledge base
```bash
# Initial setup (first time only)
python main.py kb --reindex

# Daily workflow - update with new/modified notes
python main.py kb --update
python main.py kb --analyze-updated --auto-link

# Search knowledge base
python main.py kb --query "search term" --limit 10

# Analyze specific note
python main.py kb --analyze-links "path/to/note.md"

# Debug vector store
python main.py kb --debug-store
```

### Testing specific services
When modifying services, run focused tests:
```bash
# Test vector store functionality
pytest tests/services/test_store_service.py -v

# Test embedding service
pytest tests/services/test_embedding_service.py -v

# Test link service
pytest tests/services/test_link_service.py -v
```

## Important Implementation Details

### LLM Provider Architecture
The codebase uses a provider abstraction pattern for LLM operations:
- `LLMService` provides a unified interface for all text generation
- Providers are selected via `inference.provider` config setting
- Each provider implements `BaseProvider` interface with methods:
  - `generate_text()` - Simple text generation
  - `chat_completion()` - Chat-based completions
  - `chat_completion_with_function()` - Function calling support

### OpenAI Provider
- Uses OpenAI Python SDK
- Supports official OpenAI API and OpenAI-compatible APIs (via base_url)
- Native function calling support

### Ollama Provider
- Uses **native Ollama Python SDK** (not OpenAI SDK with base_url hack)
- Two function calling modes:
  - **Native tools**: For llama3.1+, llama3.2+, mistral models (uses Ollama tools API)
  - **Structured output fallback**: For older models (uses JSON format with schema prompting)
- Configurable Ollama-specific options: num_ctx, num_thread, timeout
- Requires Ollama service running (`ollama serve`)

**Reasoning Mode Support:**
- Supports Ollama's reasoning mode for compatible models (Qwen3, DeepSeek-R1, QwQ, etc.)
- Configuration in `config.yaml`:
  ```yaml
  inference:
    ollama:
      reasoning:
        enabled: false        # Global toggle for reasoning mode
        save_thinking: false  # Include thinking tokens in output
        log_thinking: false   # Log thinking content for debugging
        models:               # Models that support reasoning
          - "qwen3"
          - "qwen2.5"
          - "deepseek-r1"
          - "qwq"
          - "smallthinker"
  ```
- When enabled, adds `think=True` parameter to Ollama API calls
- Response processing:
  - Thinking tokens available in `response["message"]["thinking"]`
  - By default, thinking is suppressed (only final answer returned)
  - Set `save_thinking: true` to include thinking in output as `<thinking>...</thinking>`
  - Set `log_thinking: true` to log thinking content for debugging
- Per-request override: Pass `reasoning=True/False` to any LLM method
- Auto-detection: Only enables reasoning for models in the `models` list
- Works with all LLM operations: `generate_text()`, `chat_completion()`, `chat_completion_with_function()`

### Embedding Support (Separate from Inference)
Embeddings use `EmbeddingService` with separate configuration:
- Embeddings: Set `embeddings.model_type: "ollama"` and `embeddings.model_name: "mxbai-embed-large"`
- Text generation and embeddings can use different providers/models

### Vector Store Persistence
- ChromaDB stores embeddings in `vector_store.path` (default: `~/Documents/notes/.vector_store`)
- Metadata includes: doc_id (note path), type (daily/weekly/meeting/learning/note), tags, dates, modified_time
- Last update timestamp tracked to enable incremental updates
- HNSW index parameters configurable for large note collections

### Note Type Detection
`SummaryService.get_all_notes()` infers note types from file paths:
- Files in daily_output_dir → "daily"
- Files in weekly_output_dir → "weekly"
- Files in meeting_notes_output_dir → "meeting"
- Files in learnings_output_dir → "learning"
- All others → "note"

### Link Management
`LinkService` distinguishes between:
- **Direct links**: Wiki-links found in note content (`[[target]]`)
- **Backlinks**: References from other notes pointing to current note
- **Semantic links**: Similar notes found via vector search
- **Suggested links**: Filtered semantic links not already directly linked
