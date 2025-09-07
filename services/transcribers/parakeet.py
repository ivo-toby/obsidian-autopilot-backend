"""
Parakeet transcriber (Phase 2).

Provides a simple `transcribe(path: str) -> str` using NVIDIA NeMo Parakeet.
Heavy dependencies are imported lazily to keep base install lean.
"""

from __future__ import annotations

import os
from typing import Optional


def _ensure_16k_mono(input_path: str) -> str:
    """Return a path to a 16 kHz mono WAV; copy/convert if needed.

    Dependencies are optional; if conversion fails or libraries are missing,
    we fall back to returning the original path.
    """
    # Preferred path: librosa can read m4a/mp3/wav via audioread and resample/mono in one step
    try:
        import librosa  # type: ignore
        import soundfile as sf  # type: ignore

        target_sr = 16_000
        data, sr = librosa.load(input_path, sr=target_sr, mono=True)
        out_path = os.path.splitext(input_path)[0] + "_16k.wav"
        sf.write(out_path, data, target_sr, subtype="PCM_16")
        return out_path
    except Exception:
        # Fallback: try soundfile for WAV/AIFF and do manual resample if needed
        try:
            import soundfile as sf  # type: ignore
            import numpy as np  # type: ignore

            data, sr = sf.read(input_path, dtype="float32")
            if hasattr(data, "ndim") and data.ndim == 2:
                data = data.mean(axis=1)
            target_sr = 16_000
            if sr != target_sr:
                try:
                    import librosa  # type: ignore

                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                except Exception:
                    ratio = target_sr / float(sr)
                    idx = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
                    data = data[idx]
            out_path = os.path.splitext(input_path)[0] + "_16k.wav"
            sf.write(out_path, data, target_sr, subtype="PCM_16")
            return out_path
        except Exception:
            # Last resort: return original path (may fail for Parakeet if stereo)
            return input_path


def transcribe(path: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v2") -> str:
    """Transcribe an audio file to text using Parakeet.

    Args:
        path: Input WAV path (48k or 16k). Will be downsampled to 16k mono if possible.
        model_name: NeMo model hub name.

    Returns:
        Transcript text.
    """
    try:
        # Lazy imports to keep base install lean
        from nemo.collections.asr import models  # type: ignore

        wav_16k = _ensure_16k_mono(path)
        m = models.ASRModel.from_pretrained(model_name)
        result = m.transcribe([wav_16k], batch_size=1, return_hypotheses=False)
        if isinstance(result, list) and result:
            return result[0] or ""
        return ""
    except Exception as e:
        return f"[Parakeet error] {e}"


