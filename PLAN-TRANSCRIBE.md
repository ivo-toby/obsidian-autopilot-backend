## Goal

Add a meetingnotes.record() feature that:
(1) records app audio (Zoom, Discord, Teams, Slack),
(2) transcribes with OpenAI Whisper, Parakeet,
(3) summarizes using existing meetingnotes function,
(4) deletes audio immediately after summarization

Default to local, per-app capture via ScreenCaptureKit; offer Zoom Cloud mode if hosts enable cloud recordings.

Architecture
• capture/
• sckit.py — per-app audio capture via ScreenCaptureKit (PyObjC). ￼
• fallback_ffmpeg.py — optional ffmpeg+BlackHole path for macOS < 13.
• zoom_cloud.py — webhook handler to fetch/delete cloud recording. ￼ ￼
• asr/parakeet.py — batch transcribe (Parakeet TDT 0.6B v2 via NeMo/HF). ￼ ￼
• summarize/llm.py — runs your existing LLM stack to produce action items/decisions.
• pipelines/record_transcribe_summarize.py — orchestrates capture → ASR → summary → secure delete.
• cli.py — adds autopilot meetingnotes record --app zoom|rekordbox [--cloud] [--stop-key q].

UX & Modes
• Local (default): --app zoom captures Zoom.app audio; --app rekordbox captures Rekordbox audio.
• Stop conditions: keypress (q), silence N seconds, or SIGINT.
• Ephemeral storage: write WAV to tempfile then unlink after ASR; optionally RAM disk is supported but not required.
• Cloud mode (optional): auto-download M4A on recording.completed, transcribe, then call Delete meeting recordings API with action=trash or permanent delete as configured. ￼ ￼ ￼

Permissions & Platform
• macOS 13+: ScreenCaptureKit supports per-app audio capture; set SCStreamConfiguration.capturesAudio = true. First run will prompt for Screen Recording permission (TCC). ￼
• Proof it works: OBS’s “macOS Screen Capture” does per-app audio on Ventura+ via ScreenCaptureKit (no drivers). ￼ ￼
• PyObjC bindings: pyobjc-framework-ScreenCaptureKit (current on PyPI). ￼

Implementation Steps (local capture) 1. Deps
• pip install pyobjc-core pyobjc-framework-ScreenCaptureKit pyobjc-framework-AVFAudio numpy soundfile
• Your ASR deps (nemo*toolkit, torch) and LLM client. 2. Enumerate capturable apps
• SCShareableContent.getShareableContentWithCompletionHandler* → pick app by bundle id (us.zoom.xos, com.pioneerdj.rekordbox). ￼ 3. Configure stream
• SCStreamConfiguration: capturesAudio=True, sample rate 48000, 1ch for Zoom, 2ch for Rekordbox; format PCM 16-bit.
• SCContentFilter with the selected SCApplication. 4. Receive audio buffers
• Implement SCStreamOutput (PyObjC) to get CMSampleBufferRef; extract AudioBufferList → numpy → write to wav using AVAudioFile/soundfile. ￼ 5. Stop logic
• A small stdin reader that watches for q\n, or a silence gate (RMS < threshold for N sec). 6. Transcribe
• Load nvidia/parakeet-tdt-0.6b-v2; downmix/resample to 16 kHz mono for Zoom; keep stereo for Rekordbox if you want DJ set analysis then sum to mono for ASR. ￼ 7. Summarize
• Feed transcript to your existing meetingnotes summarizer; produce decisions, actions, topics. 8. Shred
• os.remove(wav_path) immediately after ASR finishes and transcript is persisted to your notes graph.

Swift alternative (optional)

If PyObjC event loops get fussy, ship a 200-line Swift helper (sckit-capture) that writes WAV to stdout; spawn it from Python and read a stream until stop. Same API calls (SCStream, SCContentFilter). This keeps the Python side simple and avoids GIL/RunLoop juggling. ￼

Zoom Cloud mode (optional)
• Webhook: subscribe to recording.completed. Payload includes download token/URLs. Download M4A → ASR → summary → DELETE /meetings/{meetingId}/recordings?action=trash. Document the permanent-delete toggle. ￼ ￼ ￼

CLI sketch

autopilot meetingnotes record --app zoom --stop-key q
autopilot meetingnotes record --app rekordbox --silence-stop 8
autopilot meetingnotes record --cloud --meeting-id 123456789

Python skeletons (key bits)

# capture/sckit.py

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

# asr/parakeet.py

from nemo.collections.asr import models
def transcribe(path:str)->str:
m = models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
return m.transcribe([path], batch_size=1, return_hypotheses=False)[0]

Acceptance criteria
• autopilot meetingnotes record --app zoom produces a transcript + summary and deletes the WAV within 2 seconds of finishing.
• Works on macOS 13+ with only Screen Recording permission; no third-party audio drivers.
• Optional: --cloud path downloads M4A after webhook, then deletes the cloud recording via API.
• Rekordbox capture preserves stereo to file; ASR handles mono resample internally.

Edge cases
• No audio device / app not running: fail fast with actionable error.
• Sample-rate mismatch: normalize to 48 kHz on write.
• TCC blocked: detect and instruct user to grant Screen Recording permission (once).
• Older macOS: if ScreenCaptureKit unavailable, command errors with hint to use BlackHole+ffmpeg (fallback module). ￼

This keeps user setup near-zero, leans on first-party APIs, and slots straight into your backend’s meetingnotes flow.
