# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Obsidian Autopilot Backend is a Python tool for processing and enriching Obsidian notes. It provides automatic summarization, task extraction, semantic knowledge base features, meeting transcript processing, and audio recording/transcription capabilities (macOS only).

## Development Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
pip install -r requirements.txt
cp config.template.yaml config.yaml  # Edit with your settings

# Optional: Build Swift audio capture helper (macOS only)
cd swift && xcrun swiftc -parse-as-library -O -o sckit-capture sckit-capture.swift

# Optional: Install transcription dependencies
pip install faster-whisper  # Default local ASR (recommended for macOS)
# For Parakeet (heavy dependencies):
# pip install soundfile librosa resampy torch torchvision torchaudio 'nemo_toolkit[asr]'
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
- `AudioCaptureService`: macOS per-app audio recording via ScreenCaptureKit (Zoom, Teams, Slack, Discord, Rekordbox)

**Knowledge Base Services:**
- `services/vector_store/`:
  - `EmbeddingService`: Generates embeddings (OpenAI or Ollama)
  - `ChunkingService`: Semantic document chunking
  - `VectorStoreService`: ChromaDB-based vector storage and similarity search
- `services/knowledge/`:
  - `LinkService`: Analyzes note relationships and manages Obsidian wiki-links and backlinks

**Transcription Services:**
- `services/transcribers/`:
  - `faster_whisper.py`: Default local ASR using CTranslate2 (lightweight, CPU-friendly)
  - `parakeet.py`: Optional NVIDIA NeMo Parakeet TDT 0.6B v2 (requires heavy dependencies)

### Key Workflows

**Daily Notes Processing** (`main.py:process_daily_notes`):
1. Reads daily notes file
2. Calls `summarize_notes_and_identify_tasks()` with **function calling** to extract structured data
3. Returns: `{summary: str, actionable_items: list, tags: list}`
4. Creates Apple Reminders from actionable items
5. Saves formatted output to daily notes directory

**Meeting Notes Processing** - Two modes:

1. **From Clipboard** (`main.py notes --from-clipboard`):
   - Uses **simple text generation** (no function calling)
   - Reads transcript from clipboard
   - Uses prompt template `prompts/MEETING_PROMPT.md` by default
   - `--prompt-file` overrides the default prompt for this workflow
   - Returns markdown-formatted meeting notes directly
   - Saves to meeting notes directory with inferred topic name

2. **From Daily Notes** (`main.py notes --meetingnotes`):
   - Uses `prompts/DAILY_NOTES.md` by default to process daily logs into structured notes
   - `--prompt-file` overrides the default prompt for this workflow
   - Uses **function calling** to extract structured meeting data
   - Parses daily notes to identify meeting entries
   - Returns: `{meetings: [{date, subject, participants, notes, decisions, action_items}]}`
   - Saves individual markdown files per meeting

**Weekly Summary Processing** (`main.py:process_weekly_notes`):
- Uses **chat completion** (no function calling)
- Generates narrative summary of weekly activities
- Includes accomplishments, learnings, and extracted links

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
  - OpenAI: `inference.openai.api_key`, `inference.openai.model`, `inference.openai.base_url`, `inference.openai.temperature`
  - Ollama: `inference.ollama.model`, `inference.ollama.base_url`, temperature, num_ctx, num_thread, timeout
  - Each provider has its own model setting and temperature in its own section
- Embedding configuration (model_type: "openai" or "ollama") - separate from inference
- Vector store settings (path, similarity thresholds, HNSW index parameters)
- Search thresholds for different query types
- **Legacy config keys** (api_key, model, base_url) maintained for backward compatibility

## Common Tasks

### Recording and transcribing meetings
```bash
# Build macOS audio capture helper (one-time setup)
cd swift && xcrun swiftc -parse-as-library -O -o sckit-capture sckit-capture.swift

# Record application audio (macOS only, requires Screen Recording permission)
python main.py notes --record-audio --app zoom
# Press 'q' then Enter to stop recording

# Transcribe existing audio file (default: faster-whisper)
python main.py notes --audio-file ./recordings/meeting_audio.wav

# Record with custom settings
python main.py notes --record-audio --app teams --stop-key s --audio-out ~/custom/path.wav
```

### Processing clipboard meeting transcripts
```bash
# Use the default clipboard meeting prompt
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

# Note: Audio capture and transcription services currently lack automated tests
# (require macOS runtime, ScreenCaptureKit permissions, and running applications)
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
- Configurable temperature (default: 0.7) via `inference.openai.temperature`

### Ollama Provider
- Uses **native Ollama Python SDK** (not OpenAI SDK with base_url hack)
- **Automatic tool calling with fallback**:
  - **Native tools**: For models with tool support (llama3.1+, llama3.2+, mistral, gpt-oss, qwen2+, etc.)
  - **Automatic fallback**: If native tools don't return function calls, automatically falls back to structured output
  - **Structured output mode**: Uses JSON format with schema prompting for models without native tool support
  - **Programmatic detection**: Checks model templates via Ollama API to detect tool calling capability
- Configurable Ollama-specific options: num_ctx, num_thread, timeout
- Requires Ollama service running (`ollama serve`)

**Tool Calling Best Practices:**
- Keep prompts concise and tool-focused - avoid verbose formatting examples
- Explicitly mention the function name in the prompt when you want it called
- Use direct, imperative language: "Use the X function to..." rather than "Please generate..."
- System message should emphasize tool usage: "Always use the provided function to structure your response"
- Lower temperature (0.3-0.5) improves structured output reliability

**Troubleshooting Tool Calling:**
If a model with tool support doesn't return function calls:
1. Check the prompt - remove verbose examples and formatting instructions
2. Verify the function name matches what's mentioned in the prompt
3. Ensure the system message emphasizes tool usage
4. The automatic fallback will handle it, but fixing the prompt is preferred
5. Check logs with `logging.level: "DEBUG"` in config.yaml for detailed diagnostics

**Reasoning Mode Support:**
- Supports Ollama's reasoning mode for compatible models (automatically detected)
- **Automatic capability detection**: Models are checked programmatically via Ollama API
  - Detects reasoning support by inspecting model templates for `IsThinkSet` or `.Thinking` markers
  - Detects tool calling support by checking for `.Tools` or `.ToolCalls` in templates
  - Falls back to known model lists if API detection fails
  - Results are cached to avoid repeated API calls
- Configuration in `config.yaml`:
  ```yaml
  inference:
    ollama:
      reasoning:
        enabled: false        # Global toggle for reasoning mode
        save_thinking: false  # Include thinking tokens in output
        log_thinking: false   # Log thinking content for debugging
        # No need to specify models - automatic detection via API!
  ```
- When enabled, adds `think=True` parameter to Ollama API calls
- Response processing:
  - Thinking tokens available in `response["message"]["thinking"]`
  - By default, thinking is suppressed (only final answer returned)
  - Set `save_thinking: true` to include thinking in output as `<thinking>...</thinking>`
  - Set `log_thinking: true` to log thinking content for debugging
- Per-request override: Pass `reasoning=True/False` to any LLM method
- Programmatic detection: Automatically checks model capabilities via template inspection
- Works with all LLM operations: `generate_text()`, `chat_completion()`, `chat_completion_with_function()`

### Embedding Support (Separate from Inference)
Embeddings use `EmbeddingService` with separate configuration:
- Embeddings: Set `embeddings.model_type: "ollama"` and `embeddings.model_name: "mxbai-embed-large"`
- Text generation and embeddings can use different providers/models

### Recent Improvements

**Tool Calling Reliability (2025-10):**
- **Issue**: Models like gpt-oss:20b were returning text instead of tool calls despite having tool support
- **Root cause**: Prompts were too instruction-heavy with formatting examples, confusing the model
- **Fix**: Refactored `summarize_notes_and_identify_tasks()` prompt to be concise and tool-focused
- **Added**: Automatic fallback from native tools to structured output if tool calls aren't returned
- **Added**: Defensive error handling in `SummaryService` to prevent crashes on None responses
- **Result**: All tool-capable models now work reliably, with graceful degradation for edge cases

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

### Audio Capture and Transcription (macOS only)

**Architecture:**
- Uses ScreenCaptureKit (macOS 13+) for per-application audio capture
- Swift helper binary (`swift/sckit-capture`) preferred for robustness
- PyObjC fallback available (requires additional dependencies)
- Records to WAV with optional segmentation (default: 1-hour segments)

**Supported Applications:**
Bundle IDs mapped in `AudioCaptureService.BUNDLE_IDS`:
- `zoom`: us.zoom.xos
- `teams`: com.microsoft.teams2
- `slack`: com.tinyspeck.slackmacgap
- `discord`: com.hnc.Discord
- `rekordbox`: com.pioneerdj.rekordbox

**Recording Configuration** (`config.yaml`):
```yaml
recording:
  output_dir: "./recordings"  # WAV output directory
  segment_seconds: 3600       # Rotate files every N seconds
  sample_rate: 16000         # Target sample rate (capture is 48kHz, resampled)
  channels: 1                # 1=mono, 2=stereo
```

**Transcription Backends:**

1. **faster-whisper** (default):
   - CTranslate2-based implementation
   - Lightweight, CPU-friendly (works well on Apple Silicon)
   - Supports: wav, m4a, mp3 (via ffmpeg)
   - Models: small, medium, large-v3
   - Configuration: model_size, device (auto/cpu), compute_type (int8/float16)

2. **Parakeet** (optional):
   - NVIDIA NeMo Parakeet TDT 0.6B v2
   - Requires: torch, nemo_toolkit[asr], soundfile, librosa
   - Auto-converts to 16kHz mono
   - Better accuracy but heavier dependencies

**Swift Helper Details:**
- Location: `swift/sckit-capture.swift`
- Build: `xcrun swiftc -parse-as-library -O -o sckit-capture sckit-capture.swift`
- Features:
  - Per-app audio capture via ScreenCaptureKit
  - WAV segmentation support
  - Microphone ringbuffer (captures both app audio and user's microphone)
  - Handles both system audio and input devices
- Output: Timestamped WAV files in format `meeting_audio_YYYY-MM-DDTHH-MM-SSZ_NNN.wav`

**Permissions:**
- First run prompts for Screen Recording permission
- Location: System Settings → Privacy & Security → Screen Recording
- Grant permission to Terminal or your IDE

**Implementation Flow** (`main.py:notes --record-audio`):
1. Check if Swift helper exists (`swift/sckit-capture`)
2. If found, use Swift helper for recording
3. Fallback to PyObjC-based `_record_using_screencapturekit()` if helper not found
4. Save WAV to configured output directory
5. For transcription: Pass WAV to faster-whisper by default
6. Save transcript via `MeetingService._save_raw_summary()`

**Phase Status:**
- ✅ Phase 1: Audio capture to WAV (completed)
- ✅ Phase 2: Local transcription with faster-whisper (completed)
- ⏳ Phase 3: Transcriber abstraction layer (in progress)
- ⏳ Phase 4: End-to-end orchestration with secure deletion (planned)

See `PLAN-TRANSCRIBE.md` for detailed implementation roadmap.
