"""
faster-whisper transcriber (Phase 2 default).

Lightweight local ASR using CTranslate2. Handles common formats via ffmpeg.
"""

from __future__ import annotations

from typing import Optional


def transcribe(path: str, model_size: str = "large-v3", device: str = "auto", compute_type: str = "int8") -> str:
    """Transcribe an audio file to text using faster-whisper.

    Args:
        path: Input audio path (wav/m4a/mp3, etc.).
        model_size: Whisper model size (e.g., small, medium, large-v3).
        device: Device selection (auto/cpu).
        compute_type: Quantization (int8/float16/auto) to reduce memory.

    Returns:
        Transcript text.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            path,
            beam_size=5,
            vad_filter=True,
        )
        texts = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text.strip())
        return "\n".join(t for t in texts if t)
    except Exception as e:
        return f"[faster-whisper error] {e}"


