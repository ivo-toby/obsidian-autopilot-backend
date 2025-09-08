## Goal

Add a `notes --record-audio` capability, delivered in phases:

1. Record macOS app audio to WAV (no deletion yet).
2. Local transcription with faster-whisper (default).
3. Add a transcriber abstraction to support faster-whisper, Parakeet, and OpenAI Whisper.
4. Enable secure deletion of audio and wire end‑to‑end summarization.

Default to local, per‑app capture via ScreenCaptureKit on macOS. Zoom Cloud/webhooks are deferred.

Alignment with current architecture
• Keep changes inside `services/` and reuse existing `utils/cli.py`, `services/meeting_service.py`, and `services/openai_service.py`.
• Orchestration belongs in a new `services/transcription_service.py` rather than a top‑level `pipelines/`.
• Summarization continues to use `OpenAIService` and `MeetingService` (no new `summarize/` package).

Architecture
• services/audio_capture.py — per‑app audio capture via ScreenCaptureKit (PyObjC).
• services/transcription_service.py — orchestrates capture → ASR (and later delete).
• services/transcribers/faster_whisper.py — local Whisper (CTranslate2) default backend.
• services/transcribers/parakeet.py — optional Parakeet TDT 0.6B v2 via NeMo/HF.
• services/transcribers/openai_whisper.py — OpenAI Whisper (optional backend).
• services/meeting_service.py — reuse to save summaries and filenames.
• utils/cli.py — extend existing `notes` subcommand with recording/transcribe flags.

UX & Modes
• CLI (extend notes):

- `notes --record-audio [--app zoom|teams|slack|discord] [--stop-key q] [--silence-stop N]`
- `notes --audio-file /path/to.wav` (transcribe existing file)
  • Phase 1 saves WAV locally (no deletion).
  • Stop conditions: keypress (q), optional silence gate, or SIGINT.
  • Cloud mode (Zoom) is deferred to a later phase.

Permissions & Platform
• macOS 13+: ScreenCaptureKit supports per‑app audio capture; set SCStreamConfiguration.capturesAudio = true. First run will prompt for Screen Recording permission (TCC).
• OBS’s “macOS Screen Capture” does per‑app audio on Ventura+ via ScreenCaptureKit (no drivers).
• PyObjC bindings: pyobjc-framework-ScreenCaptureKit (current on PyPI).

Implementation Steps (phased)
Phase 1 — Audio capture (macOS)

1. Deps: `pyobjc-core`, `pyobjc-framework-ScreenCaptureKit`, `pyobjc-framework-AVFAudio`, `numpy`, `soundfile`.
2. Enumerate capturable apps with `SCShareableContent.shareableContent()`, pick by bundle id (e.g., `us.zoom.xos`).
3. Configure stream: `SCStreamConfiguration.capturesAudio=True`, sample rate 48000, PCM 16‑bit; `SCContentFilter` with selected `SCApplication`.
4. Receive audio buffers: implement `SCStreamOutput` to get `CMSampleBufferRef`; extract `AudioBufferList` → `numpy` → write to WAV via `soundfile`.
5. Stop logic: stdin reader for `q`, optional silence gate (RMS < threshold for N sec), SIGINT.
6. CLI: `notes --record-audio --app zoom` writes WAV to an output directory; no deletion.

Phase 2 — Local ASR (default: faster-whisper)

1. Deps: `faster-whisper` (CTranslate2). Optional: `ffmpeg`.
2. Implement `services/transcribers/faster_whisper.py` with `transcribe(path:str)->str`.
3. CLI: `notes --audio-file /path/to.wav` routes to faster-whisper and prints/saves transcript.

Phase 3 — Transcriber abstraction

1. Define `Transcriber` interface in `services/transcription_service.py` and register backends.
2. Add `services/transcribers/openai_whisper.py` as the cloud backend.
3. Config: `transcription.provider: faster_whisper|parakeet|openai`.

Phase 4 — End‑to‑end + deletion

1. Orchestrate capture → transcribe → summarize via existing `MeetingService`.
2. Enable secure deletion of WAV only after summary is written; toggle via config.

Swift alternative (optional)

If PyObjC event loops get fussy, ship a 200-line Swift helper (sckit-capture) that writes WAV to stdout; spawn it from Python and read a stream until stop. Same API calls (SCStream, SCContentFilter). This keeps the Python side simple and avoids GIL/RunLoop juggling. ￼

Zoom Cloud mode (deferred, optional)
• Webhook: subscribe to recording.completed. Payload includes download token/URLs. Download M4A → ASR → summary → DELETE /meetings/{meetingId}/recordings?action=trash. Document the permanent-delete toggle.

CLI sketch

python main.py notes --record-audio --app zoom --stop-key q
python main.py notes --audio-file /path/to.wav
python main.py notes --record-audio --silence-stop 8

Python skeletons (key bits)

# services/audio_capture.py

from Cocoa import NSObject, NSLog
import ScreenCaptureKit as SCK
import AVFoundation as AVF
import CoreMedia as CM
import numpy as np, soundfile as sf

BUNDLE_IDS = {"zoom": "us.zoom.xos", "rekordbox": "com.pioneerdj.rekordbox"}

class AudioSink(NSObject):
def initWithPath*(self, path): ...
def stream_didOutputSampleBuffer_ofType*(self, stream, sbuf, stype):
if stype != SCK.SCStreamOutputTypeAudio: return # extract PCM from CMSampleBuffer, append to ring/buffer, flush to WAV file

def record*app_audio(app_key, out_wav, mono=True, samplerate=48000):
cfg = SCK.SCStreamConfiguration.alloc().init()
cfg.capturesAudio = True # <-- audio on
cfg.sampleRate = samplerate # pick app
content = SCK.SCShareableContent.shareableContent()
app = next(a for a in content.applications() if a.bundleIdentifier() == BUNDLE_IDS[app_key])
filter* = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow*(app) # or initWithApplication*
stream = SCK.SCStream.alloc().initWithFilter*configuration_delegate*(filter*, cfg, None)
sink = AudioSink.alloc().initWithPath*(out*wav)
stream.addStreamOutput_type_minimumFrameTime_error*(sink, SCK.SCStreamOutputTypeAudio, CM.CMTimeMake(1,10), None)
stream.startCaptureWithCompletionHandler\_(lambda err: NSLog("capturing")) # block until stop key; then stop and teardown

# services/transcribers/faster_whisper.py

from faster_whisper import WhisperModel
def transcribe(path:str)->str:
model = WhisperModel("large-v3", device="auto", compute_type="int8")
segments, info = model.transcribe(path)
return "\n".join(s.text for s in segments if s.text)

Acceptance criteria (per phase)
• Phase 1: `notes --record-audio --app zoom` writes a valid WAV from app audio on macOS 13+.
• Phase 2: `notes --audio-file file.wav` returns a transcript using faster-whisper locally.
• Phase 3: Config switch selects faster-whisper, Parakeet (optional), or OpenAI Whisper; both paths work on the same input.

Stretch goal
• Parakeet on MPS via a small Swift helper leveraging Metal; expose as optional backend.
• Phase 4: End‑to‑end: record → transcribe → summarize; audio deletion happens after successful write.

Edge cases
• No audio device / app not running: fail fast with actionable error.
• Sample-rate mismatch: normalize to 48 kHz on write.
• TCC blocked: detect and instruct user to grant Screen Recording permission (once).
• Older macOS: if ScreenCaptureKit unavailable, command errors with hint to use BlackHole+ffmpeg (fallback module). ￼

This keeps user setup near-zero, leans on first-party APIs, and slots straight into your backend’s meetingnotes flow.
