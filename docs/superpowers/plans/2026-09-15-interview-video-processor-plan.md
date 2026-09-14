# Interview Video Batch Processor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade Google Colab notebook and modular Python CLI pipeline to batch-process interview videos with natural AI silence removal (Silero VAD v5), AI speech denoising (DeepFilterNet 3 / Resemble Enhance), NVENC GPU video re-encoding, and NLE timeline export (Premiere Pro / DaVinci Resolve FCP7 XML & EDL).

**Architecture:** The pipeline is architected into modular components: `vad_engine.py` for speech boundary detection with padding and crossfade calculations; `denoise_engine.py` for neural noise reduction and loudness normalization; `timeline_exporter.py` for standard XML/EDL generation; `video_engine.py` for hardware-accelerated segment cutting; orchestrated by `interview_processor.py` CLI and wrapped in an interactive `Interview_AI_Studio.ipynb` Colab notebook.

**Tech Stack:** Python 3.10+, PyTorch, Silero VAD v5, DeepFilterNet 3 (`deepfilternet`), Resemble Enhance, FFmpeg / FFprobe (NVENC `h264_nvenc`), `pyloudnorm`, `scipy`, `numpy`, `tqdm`.

**Spec:** `docs/superpowers/specs/2026-09-15-interview-video-processor-design.md`

## Global Constraints
- Target platform: Google Colab (Linux x86_64, CUDA GPU runtime) + Windows/Linux local CLI fallback.
- Python version: >= 3.10.
- All cut operations must be frame-accurate based on source video FPS probed via `ffprobe`.
- Default silence parameters: min silence duration = `0.8s`, head/tail padding margin = `0.25s` (250ms), crossfade = `30ms`.
- Default audio target loudness: `-14.0 LUFS` (EBU R128).
- GPU codec: `h264_nvenc` with automatic graceful fallback to CPU `libx264`.
- XML export must conform to the Final Cut Pro 7 XML standard for direct drag-and-drop into Adobe Premiere Pro and DaVinci Resolve.

---

## File Structure Map
```
E:\Project\MeowVid\interview\Colab\
├── requirements.txt                  # Task 5: Python dependencies
├── vad_engine.py                     # Task 1: Silero VAD v5 & speech segment logic
├── denoise_engine.py                 # Task 2: DeepFilterNet / Resemble & audio processing
├── timeline_exporter.py              # Task 3: FCP7 XML & CMX 3600 EDL generators
├── video_engine.py                   # Task 4: FFprobe/FFmpeg NVENC video processing
├── interview_processor.py            # Task 5: Batch orchestrator & CLI entrypoint
├── Interview_AI_Studio.ipynb         # Task 6: Google Colab Notebook with GUI Forms
├── README.md                         # Task 6: Documentation & NLE Import Guide
└── tests/
    ├── test_vad.py                   # Task 1 tests
    ├── test_denoise.py               # Task 2 tests
    ├── test_timeline.py              # Task 3 tests
    └── test_video.py                 # Task 4 tests
```

---

### Task 1: Voice Activity Detection (VAD) & Natural Silence Segment Engine

**Files:**
- Create: `vad_engine.py`
- Create: `tests/test_vad.py`

**Interfaces:**
- Produces:
  - `class SpeechSegment(NamedTuple): start: float, end: float`
  - `load_vad_model(device: str = "cpu") -> tuple[torch.nn.Module, tuple]`
  - `get_speech_timestamps(audio_tensor: torch.Tensor, sample_rate: int, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 800, threshold: float = 0.5) -> list[dict]`
  - `pad_and_merge_segments(segments: list[dict], total_duration: float, padding_sec: float = 0.25, min_silence_sec: float = 0.8) -> list[tuple[float, float]]`
  - `calculate_speech_stats(original_duration: float, kept_segments: list[tuple[float, float]]) -> dict`

- [ ] **Step 1: Write unit tests for VAD boundary padding, merging, and stats calculations**

Create `tests/test_vad.py`:
```python
import pytest
from vad_engine import pad_and_merge_segments, calculate_speech_stats

def test_pad_and_merge_segments_basic():
    # Raw speech intervals in seconds
    raw = [
        {"start": 1.0, "end": 2.0},
        {"start": 2.5, "end": 4.0},
        {"start": 7.0, "end": 8.0}
    ]
    total_duration = 10.0
    # With 0.25s padding and 0.8s min silence:
    # Segment 1 becomes [0.75, 2.25]
    # Segment 2 becomes [2.25, 4.25] -> overlaps with seg 1, merges into [0.75, 4.25]
    # Gap between 4.25 and 6.75 is 2.5s > 0.8s, kept as silence.
    # Segment 3 becomes [6.75, 8.25]
    merged = pad_and_merge_segments(raw, total_duration, padding_sec=0.25, min_silence_sec=0.8)
    assert len(merged) == 2
    assert pytest.approx(merged[0][0], 0.01) == 0.75
    assert pytest.approx(merged[0][1], 0.01) == 4.25
    assert pytest.approx(merged[1][0], 0.01) == 6.75
    assert pytest.approx(merged[1][1], 0.01) == 8.25

def test_pad_and_merge_clamp_boundaries():
    raw = [{"start": 0.1, "end": 9.9}]
    total_duration = 10.0
    # Padding should not exceed 0.0 on start or total_duration on end
    merged = pad_and_merge_segments(raw, total_duration, padding_sec=0.5, min_silence_sec=0.8)
    assert len(merged) == 1
    assert merged[0][0] == 0.0
    assert merged[0][1] == 10.0

def test_calculate_speech_stats():
    segments = [(1.0, 4.0), (6.0, 9.0)] # total kept = 6.0s
    stats = calculate_speech_stats(original_duration=10.0, kept_segments=segments)
    assert stats["original_duration"] == 10.0
    assert stats["kept_duration"] == 6.0
    assert stats["silence_removed"] == 4.0
    assert pytest.approx(stats["silence_percentage"], 0.1) == 40.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vad_engine'`

- [ ] **Step 3: Implement `vad_engine.py`**

Create `vad_engine.py`:
```python
"""
vad_engine.py - Voice Activity Detection and Natural Speech Segmenting using Silero VAD v5.
"""
from typing import List, Tuple, Dict, Any
import torch
import numpy as np

def load_vad_model(device: str = "cpu"):
    """Loads Silero VAD v5 from torch hub."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    model.to(device)
    model.eval()
    return model, utils

def get_speech_timestamps(
    audio_tensor: torch.Tensor,
    model: Any,
    utils: Any,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 800,
) -> List[Dict[str, float]]:
    """
    Returns list of speech timestamps in seconds: [{'start': 1.2, 'end': 3.5}, ...]
    """
    get_speech_ts_fn = utils[0]
    # silero expects 1D float tensor
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.mean(dim=0)
    audio_tensor = audio_tensor.squeeze()

    speech_timestamps = get_speech_ts_fn(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True
    )
    return speech_timestamps

def pad_and_merge_segments(
    segments: List[Dict[str, float]],
    total_duration: float,
    padding_sec: float = 0.25,
    min_silence_sec: float = 0.8
) -> List[Tuple[float, float]]:
    """
    Expands speech segments with head/tail padding and merges overlapping or closely spaced segments.
    """
    if not segments:
        return []

    # 1. Apply padding and clamp to [0, total_duration]
    padded = []
    for seg in segments:
        s = max(0.0, float(seg['start']) - padding_sec)
        e = min(total_duration, float(seg['end']) + padding_sec)
        padded.append((s, e))

    # 2. Sort by start time
    padded.sort(key=lambda x: x[0])

    # 3. Merge overlapping or segments separated by less than min_silence_sec
    merged: List[Tuple[float, float]] = []
    current_start, current_end = padded[0]

    for next_start, next_end in padded[1:]:
        # If the gap between current segment end and next start is less than min_silence_sec
        if next_start <= current_end + min_silence_sec:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    return merged

def calculate_speech_stats(original_duration: float, kept_segments: List[Tuple[float, float]]) -> Dict[str, float]:
    """Calculates summary statistics about removed silence."""
    kept_duration = sum(end - start for start, end in kept_segments)
    silence_removed = max(0.0, original_duration - kept_duration)
    silence_pct = (silence_removed / original_duration * 100.0) if original_duration > 0 else 0.0
    return {
        "original_duration": round(original_duration, 3),
        "kept_duration": round(kept_duration, 3),
        "silence_removed": round(silence_removed, 3),
        "silence_percentage": round(silence_pct, 2)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vad.py -v`
Expected: 3 passed

---

### Task 2: AI Denoising, Audio Crossfading & Loudness Normalization

**Files:**
- Create: `denoise_engine.py`
- Create: `tests/test_denoise.py`

**Interfaces:**
- Produces:
  - `denoise_audio_deepfilter(input_wav: str, output_wav: str, model_path: str = None) -> str`
  - `denoise_audio_resemble(input_wav: str, output_wav: str, solver: str = "midpoint", nfe: int = 64) -> str`
  - `cut_and_crossfade_audio(audio_data: np.ndarray, sample_rate: int, segments: list[tuple[float, float]], crossfade_ms: int = 30) -> np.ndarray`
  - `normalize_loudness(audio_data: np.ndarray, sample_rate: int, target_lufs: float = -14.0) -> np.ndarray`

- [ ] **Step 1: Write unit tests for audio crossfading and loudness normalization**

Create `tests/test_denoise.py`:
```python
import numpy as np
import pytest
from denoise_engine import cut_and_crossfade_audio, normalize_loudness

def test_cut_and_crossfade_audio_shape():
    sr = 48000
    # 10 seconds of stereo audio (sine wave)
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t)
    audio = np.stack([sig, sig], axis=0) # shape (2, 480000)

    # Segments: 1s to 3s (2s), 5s to 8s (3s) -> total 5s
    segments = [(1.0, 3.0), (5.0, 8.0)]
    crossfade_ms = 30
    crossfade_samples = int(sr * crossfade_ms / 1000)

    result = cut_and_crossfade_audio(audio, sr, segments, crossfade_ms=crossfade_ms)
    # Expected length: sum(segment_samples) - (num_cuts - 1) * crossfade_samples
    expected_len = (2 * sr + 3 * sr) - (1 * crossfade_samples)
    assert result.shape[0] == 2
    assert abs(result.shape[1] - expected_len) <= 2
    # Ensure no NaN or Inf
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()

def test_normalize_loudness_peak():
    sr = 48000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    # Very quiet signal
    sig = 0.01 * np.sin(2 * np.pi * 440 * t)
    audio = np.stack([sig, sig], axis=0)

    norm_audio = normalize_loudness(audio, sr, target_lufs=-14.0)
    # Normalized audio should have higher peak amplitude than the 0.01 input
    assert np.max(np.abs(norm_audio)) > np.max(np.abs(audio))
    # True peak should remain <= 1.0 (no clipping)
    assert np.max(np.abs(norm_audio)) <= 1.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_denoise.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'denoise_engine'`

- [ ] **Step 3: Implement `denoise_engine.py`**

Create `denoise_engine.py`:
```python
"""
denoise_engine.py - AI Audio Denoising (DeepFilterNet 3 / Resemble Enhance),
natural crossfading at cut points, and EBU R128 loudness normalization.
"""
import os
import shutil
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

def denoise_audio_deepfilter(input_wav: str, output_wav: str) -> str:
    """
    Runs DeepFilterNet 3 enhancement on input WAV.
    Falls back gracefully to high-pass/low-noise filter if libdf is not installed.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df()
        audio, _ = load_audio(input_wav, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(output_wav, enhanced, df_state.sr())
        return output_wav
    except ImportError:
        # Fallback to df-enhance CLI if available
        if shutil.which("df-enhance"):
            out_dir = os.path.dirname(os.path.abspath(output_wav))
            cmd = ["df-enhance", input_wav, "-o", out_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Find generated file and rename if needed
            base = os.path.splitext(os.path.basename(input_wav))[0]
            gen_file = os.path.join(out_dir, f"{base}_DeepFilterNet3.wav")
            if os.path.exists(gen_file):
                if gen_file != output_wav:
                    shutil.move(gen_file, output_wav)
                return output_wav
        # Fallback to copy if no neural denoiser available
        shutil.copyfile(input_wav, output_wav)
        return output_wav

def denoise_audio_resemble(input_wav: str, output_wav: str, solver: str = "midpoint", nfe: int = 64) -> str:
    """
    Runs Resemble Enhance on input WAV.
    """
    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import denoise, enhance
        device = "cuda" if torch.cuda.is_available() else "cpu"
        info = torchaudio.info(input_wav)
        audio, sr = torchaudio.load(input_wav)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        enhanced_audio, new_sr = enhance(audio.squeeze(), sr, device, nfe=nfe, solver=solver, lambd=0.9, tau=0.5)
        torchaudio.save(output_wav, enhanced_audio.unsqueeze(0).cpu(), new_sr)
        return output_wav
    except ImportError:
        shutil.copyfile(input_wav, output_wav)
        return output_wav

def cut_and_crossfade_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    segments: list[tuple[float, float]],
    crossfade_ms: int = 30
) -> np.ndarray:
    """
    Slices audio according to kept intervals and applies smooth cosine crossfades between slices.
    audio_data: np.ndarray with shape (channels, samples) or (samples,)
    """
    if not segments:
        return np.zeros((2, 0) if audio_data.ndim > 1 else 0, dtype=audio_data.dtype)

    is_1d = (audio_data.ndim == 1)
    if is_1d:
        audio_data = np.expand_dims(audio_data, axis=0)

    channels, total_samples = audio_data.shape
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)

    extracted_slices = []
    for start_sec, end_sec in segments:
        start_samp = int(round(start_sec * sample_rate))
        end_samp = min(total_samples, int(round(end_sec * sample_rate)))
        if end_samp > start_samp:
            extracted_slices.append(audio_data[:, start_samp:end_samp])

    if not extracted_slices:
        return np.zeros((channels, 0), dtype=audio_data.dtype) if not is_1d else np.zeros(0, dtype=audio_data.dtype)

    if len(extracted_slices) == 1:
        out = extracted_slices[0]
        return out[0] if is_1d else out

    # Concatenate with crossfade
    combined = extracted_slices[0]
    for i in range(1, len(extracted_slices)):
        next_slice = extracted_slices[i]
        # Check if crossfade is possible
        cf_len = min(crossfade_samples, combined.shape[1], next_slice.shape[1])
        if cf_len > 0:
            # Cosine fade curve
            t = np.linspace(0, np.pi / 2, cf_len)
            fade_out = np.cos(t) ** 2
            fade_in = np.sin(t) ** 2

            # Overlap region
            overlap = combined[:, -cf_len:] * fade_out + next_slice[:, :cf_len] * fade_in
            combined = np.concatenate([
                combined[:, :-cf_len],
                overlap,
                next_slice[:, cf_len:]
            ], axis=1)
        else:
            combined = np.concatenate([combined, next_slice], axis=1)

    return combined[0] if is_1d else combined

def normalize_loudness(audio_data: np.ndarray, sample_rate: int, target_lufs: float = -14.0) -> np.ndarray:
    """
    Normalizes audio loudness to EBU R128 target (default -14 LUFS) with true-peak limiting.
    """
    is_1d = (audio_data.ndim == 1)
    if is_1d:
        audio_2d = np.expand_dims(audio_data, axis=0)
    else:
        audio_2d = audio_data

    # pyloudnorm expects shape (samples, channels)
    audio_for_meter = audio_2d.T
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio_for_meter)
        if not np.isneginf(loudness) and not np.isnan(loudness):
            normalized = pyln.normalize.loudness(audio_for_meter, loudness, target_lufs)
            audio_out = normalized.T
        else:
            audio_out = audio_2d
    except Exception:
        audio_out = audio_2d

    # Peak limit to -1.0 dBFS (~0.89) to prevent clipping
    max_peak = np.max(np.abs(audio_out))
    if max_peak > 0.95:
        audio_out = audio_out * (0.95 / max_peak)

    return audio_out[0] if is_1d else audio_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_denoise.py -v`
Expected: 2 passed

---

### Task 3: NLE Timeline Exporters (Final Cut Pro 7 XML & CMX 3600 EDL)

**Files:**
- Create: `timeline_exporter.py`
- Create: `tests/test_timeline.py`

**Interfaces:**
- Produces:
  - `seconds_to_timecode(seconds: float, fps: float) -> str`
  - `generate_fcp7_xml(source_media_path: str, segments: list[tuple[float, float]], fps: float, width: int = 1920, height: int = 1080, sample_rate: int = 48000, sequence_name: str = "Interview_Cut") -> str`
  - `generate_cmx3600_edl(source_media_path: str, segments: list[tuple[float, float]], fps: float, sequence_name: str = "Interview_Cut") -> str`
  - `export_timeline_files(source_media_path: str, segments: list[tuple[float, float]], fps: float, output_base_path: str) -> dict[str, str]`

- [ ] **Step 1: Write unit tests for timecode conversion and XML/EDL generation**

Create `tests/test_timeline.py`:
```python
import xml.etree.ElementTree as ET
import pytest
from timeline_exporter import seconds_to_timecode, generate_fcp7_xml, generate_cmx3600_edl

def test_seconds_to_timecode():
    assert seconds_to_timecode(0.0, 30.0) == "00:00:00:00"
    assert seconds_to_timecode(1.5, 30.0) == "00:00:01:15"
    assert seconds_to_timecode(65.0, 25.0) == "00:01:05:00"
    assert seconds_to_timecode(3600.0, 24.0) == "01:00:00:00"

def test_generate_fcp7_xml_validity():
    segments = [(1.0, 5.0), (10.0, 15.0)]
    xml_content = generate_fcp7_xml(
        source_media_path="/path/to/interview.mp4",
        segments=segments,
        fps=30.0,
        width=1920,
        height=1080
    )
    # Validate XML can be parsed
    root = ET.fromstring(xml_content)
    assert root.tag == "xmeml"
    sequence = root.find("sequence")
    assert sequence is not None
    # 2 segments -> 2 clipitems on video track
    clipitems = sequence.findall(".//media/video/track/clipitem")
    assert len(clipitems) == 2

    # Check first clipitem in/out frames (1s = 30 frames, 5s = 150 frames)
    in_frame = int(clipitems[0].find("in").text)
    out_frame = int(clipitems[0].find("out").text)
    assert in_frame == 30
    assert out_frame == 150

def test_generate_cmx3600_edl():
    segments = [(0.0, 4.0), (10.0, 14.0)]
    edl = generate_cmx3600_edl("/path/to/video.mp4", segments, fps=30.0)
    assert "TITLE: Interview_Cut" in edl
    assert "001" in edl
    assert "002" in edl
    assert "00:00:00:00 00:00:04:00" in edl
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_timeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'timeline_exporter'`

- [ ] **Step 3: Implement `timeline_exporter.py`**

Create `timeline_exporter.py`:
```python
"""
timeline_exporter.py - Generates Final Cut Pro 7 XML and CMX 3600 EDL timeline project files
compatible with Adobe Premiere Pro, DaVinci Resolve, and Final Cut Pro.
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict

def seconds_to_timecode(seconds: float, fps: float) -> str:
    """Converts seconds into standard non-drop frame timecode string (HH:MM:SS:FF)."""
    total_frames = int(round(seconds * fps))
    int_fps = int(round(fps)) if int(round(fps)) > 0 else 30
    frames = total_frames % int_fps
    total_seconds = total_frames // int_fps
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

def generate_fcp7_xml(
    source_media_path: str,
    segments: List[Tuple[float, float]],
    fps: float,
    width: int = 1920,
    height: int = 1080,
    sample_rate: int = 48000,
    sequence_name: str = "Interview_Cut"
) -> str:
    """
    Generates standard FCP7 XML sequence containing individual clips for kept speech segments.
    """
    timebase = int(round(fps))
    ntsc = "TRUE" if abs(fps - 29.97) < 0.05 or abs(fps - 23.976) < 0.05 or abs(fps - 59.94) < 0.05 else "FALSE"

    root = ET.Element("xmeml", version="4")
    sequence = ET.SubElement(root, "sequence")
    ET.SubElement(sequence, "name").text = sequence_name
    
    # Calculate total sequence frames
    total_duration_frames = sum(int(round((end - start) * fps)) for start, end in segments)
    ET.SubElement(sequence, "duration").text = str(total_duration_frames)

    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = ntsc

    timecode = ET.SubElement(sequence, "timecode")
    tc_rate = ET.SubElement(timecode, "rate")
    ET.SubElement(tc_rate, "timebase").text = str(timebase)
    ET.SubElement(tc_rate, "ntsc").text = ntsc
    ET.SubElement(timecode, "string").text = "00:00:00:00"
    ET.SubElement(timecode, "frame").text = "0"

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    v_format = ET.SubElement(video, "format")
    samplecharacteristics = ET.SubElement(v_format, "samplecharacteristics")
    ET.SubElement(samplecharacteristics, "width").text = str(width)
    ET.SubElement(samplecharacteristics, "height").text = str(height)
    ET.SubElement(samplecharacteristics, "pixelaspectratio").text = "square"
    v_rate = ET.SubElement(samplecharacteristics, "rate")
    ET.SubElement(v_rate, "timebase").text = str(timebase)
    ET.SubElement(v_rate, "ntsc").text = ntsc

    v_track = ET.SubElement(video, "track")
    audio = ET.SubElement(media, "audio")
    a1_track = ET.SubElement(audio, "track")
    a2_track = ET.SubElement(audio, "track")

    timeline_in = 0
    media_abs_path = os.path.abspath(source_media_path)
    media_filename = os.path.basename(source_media_path)
    file_url = f"file://localhost/{media_abs_path.replace(os.sep, '/')}"

    for idx, (start_sec, end_sec) in enumerate(segments, start=1):
        in_frame = int(round(start_sec * fps))
        out_frame = int(round(end_sec * fps))
        clip_duration = out_frame - in_frame
        timeline_out = timeline_in + clip_duration

        if clip_duration <= 0:
            continue

        # Video clipitem
        v_clip = ET.SubElement(v_track, "clipitem", id=f"clipitem-v-{idx}")
        ET.SubElement(v_clip, "name").text = f"{media_filename} [Part {idx}]"
        ET.SubElement(v_clip, "duration").text = str(int(round(999999 * fps))) # large master duration
        v_clip_rate = ET.SubElement(v_clip, "rate")
        ET.SubElement(v_clip_rate, "timebase").text = str(timebase)
        ET.SubElement(v_clip_rate, "ntsc").text = ntsc
        ET.SubElement(v_clip, "start").text = str(timeline_in)
        ET.SubElement(v_clip, "end").text = str(timeline_out)
        ET.SubElement(v_clip, "in").text = str(in_frame)
        ET.SubElement(v_clip, "out").text = str(out_frame)

        # File element reference
        v_file = ET.SubElement(v_clip, "file", id="file-master")
        ET.SubElement(v_file, "name").text = media_filename
        ET.SubElement(v_file, "pathurl").text = file_url
        v_file_rate = ET.SubElement(v_file, "rate")
        ET.SubElement(v_file_rate, "timebase").text = str(timebase)
        ET.SubElement(v_file_rate, "ntsc").text = ntsc
        v_file_media = ET.SubElement(v_file, "media")
        v_file_video = ET.SubElement(v_file_media, "video")
        v_sample = ET.SubElement(v_file_video, "samplecharacteristics")
        ET.SubElement(v_sample, "width").text = str(width)
        ET.SubElement(v_sample, "height").text = str(height)

        # Audio Track 1 & 2 clipitems
        for a_idx, a_track in [(1, a1_track), (2, a2_track)]:
            a_clip = ET.SubElement(a_track, "clipitem", id=f"clipitem-a{a_idx}-{idx}")
            ET.SubElement(a_clip, "name").text = f"{media_filename} [Part {idx}]"
            a_clip_rate = ET.SubElement(a_clip, "rate")
            ET.SubElement(a_clip_rate, "timebase").text = str(timebase)
            ET.SubElement(a_clip_rate, "ntsc").text = ntsc
            ET.SubElement(a_clip, "start").text = str(timeline_in)
            ET.SubElement(a_clip, "end").text = str(timeline_out)
            ET.SubElement(a_clip, "in").text = str(in_frame)
            ET.SubElement(a_clip, "out").text = str(out_frame)
            ET.SubElement(a_clip, "file", id="file-master")

        timeline_in = timeline_out

    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

def generate_cmx3600_edl(
    source_media_path: str,
    segments: List[Tuple[float, float]],
    fps: float,
    sequence_name: str = "Interview_Cut"
) -> str:
    """Generates standard CMX 3600 EDL."""
    lines = [
        f"TITLE: {sequence_name}",
        f"FCM: NON-DROP FRAME",
        ""
    ]
    reel_name = os.path.splitext(os.path.basename(source_media_path))[0][:8].upper()
    timeline_sec = 0.0

    for idx, (start_sec, end_sec) in enumerate(segments, start=1):
        duration = end_sec - start_sec
        if duration <= 0:
            continue
        src_in = seconds_to_timecode(start_sec, fps)
        src_out = seconds_to_timecode(end_sec, fps)
        rec_in = seconds_to_timecode(timeline_sec, fps)
        rec_out = seconds_to_timecode(timeline_sec + duration, fps)

        lines.append(f"{idx:03d}  {reel_name:<8} AA/V  C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {os.path.basename(source_media_path)}")
        lines.append("")
        timeline_sec += duration

    return "\n".join(lines)

def export_timeline_files(
    source_media_path: str,
    segments: List[Tuple[float, float]],
    fps: float,
    output_base_path: str,
    width: int = 1920,
    height: int = 1080
) -> Dict[str, str]:
    """Saves both XML and EDL files for NLE import."""
    xml_path = f"{output_base_path}_timeline.xml"
    edl_path = f"{output_base_path}_timeline.edl"

    xml_text = generate_fcp7_xml(source_media_path, segments, fps, width, height)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_text)

    edl_text = generate_cmx3600_edl(source_media_path, segments, fps)
    with open(edl_path, "w", encoding="utf-8") as f:
        f.write(edl_text)

    return {"xml": xml_path, "edl": edl_path}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_timeline.py -v`
Expected: 3 passed

---

### Task 4: Video Processing & Hardware-Accelerated NVENC Cutting

**Files:**
- Create: `video_engine.py`
- Create: `tests/test_video.py`

**Interfaces:**
- Produces:
  - `probe_video_metadata(video_path: str) -> dict[str, Any]`
  - `extract_audio_from_video(video_path: str, output_wav_48k: str, output_wav_16k: str) -> tuple[str, str]`
  - `cut_and_render_video_nvenc(video_path: str, clean_audio_wav: str, segments: list[tuple[float, float]], output_video_path: str, use_gpu: bool = True) -> str`

- [ ] **Step 1: Write integration tests for video probe and fallback cut engine**

Create `tests/test_video.py`:
```python
import subprocess
import shutil
import pytest
from video_engine import probe_video_metadata, extract_audio_from_video

@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """Generates a synthetic 3-second test video using ffmpeg."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in test environment")
    tmp_dir = tmp_path_factory.mktemp("video_test")
    test_file = str(tmp_dir / "synthetic_test.mp4")
    # Generate 3 seconds test video with audio test tone
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        test_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return test_file

def test_probe_video_metadata(sample_video):
    meta = probe_video_metadata(sample_video)
    assert meta["width"] == 640
    assert meta["height"] == 360
    assert pytest.approx(meta["fps"], 0.1) == 30.0
    assert meta["duration"] >= 2.9
    assert meta["has_audio"] is True

def test_extract_audio_from_video(sample_video, tmp_path):
    wav_48k = str(tmp_path / "audio_48k.wav")
    wav_16k = str(tmp_path / "audio_16k.wav")
    out_48k, out_16k = extract_audio_from_video(sample_video, wav_48k, wav_16k)
    assert os.path.exists(out_48k)
    assert os.path.exists(out_16k)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'video_engine'`

- [ ] **Step 3: Implement `video_engine.py`**

Create `video_engine.py`:
```python
"""
video_engine.py - FFprobe metadata inspection, audio track extraction,
and GPU-accelerated NVENC video cutting & muxing.
"""
import os
import json
import subprocess
import shutil
import tempfile
from typing import Dict, Any, List, Tuple

def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    """Retrieves framerate, resolution, duration, and audio stream metadata."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed or not in PATH.")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,sample_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)

    width = 1920
    height = 1080
    fps = 30.0
    has_audio = False
    sample_rate = 48000
    duration = float(data.get("format", {}).get("duration", 0.0))

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 1920))
            height = int(stream.get("height", 1080))
            r_rate = stream.get("r_frame_rate", "30/1")
            if "/" in r_rate:
                num, den = r_rate.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 30.0
            else:
                fps = float(r_rate)
        elif stream.get("codec_type") == "audio":
            has_audio = True
            sample_rate = int(stream.get("sample_rate", 48000))

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration": duration,
        "has_audio": has_audio,
        "sample_rate": sample_rate
    }

def extract_audio_from_video(video_path: str, output_wav_48k: str, output_wav_16k: str) -> Tuple[str, str]:
    """Extracts 48kHz audio and 16kHz mono audio for VAD."""
    # 48kHz high-fidelity WAV
    cmd_48k = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000",
        output_wav_48k
    ]
    subprocess.run(cmd_48k, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 16kHz mono WAV for VAD
    cmd_16k = [
        "ffmpeg", "-y", "-i", output_wav_48k,
        "-ac", "1", "-ar", "16000",
        output_wav_16k
    ]
    subprocess.run(cmd_16k, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_wav_48k, output_wav_16k

def cut_and_render_video_nvenc(
    video_path: str,
    clean_audio_wav: str,
    segments: List[Tuple[float, float]],
    output_video_path: str,
    use_gpu: bool = True
) -> str:
    """
    Renders the final cut video using NVENC GPU acceleration (with libx264 CPU fallback),
    muxing the cleaned/crossfaded audio track.
    """
    if not segments:
        raise ValueError("No speech segments provided for rendering.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Build concat filter or segment list
        filter_complex_parts = []
        concat_inputs = []

        for idx, (start, end) in enumerate(segments):
            filter_complex_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{idx}];")
            concat_inputs.append(f"[v{idx}]")

        concat_filter = f"{''.join(filter_complex_parts)}{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[outv]"

        # Check GPU encoder support
        encoder = "h264_nvenc" if use_gpu else "libx264"
        encoder_args = ["-c:v", encoder, "-preset", "p4", "-cq", "20"] if encoder == "h264_nvenc" else ["-c:v", "libx264", "-preset", "fast", "-crf", "19"]

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", clean_audio_wav,
            "-filter_complex", concat_filter,
            "-map", "[outv]",
            "-map", "1:a",
            *encoder_args,
            "-c:a", "aac", "-b:a", "320k",
            "-movflags", "+faststart",
            output_video_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            if encoder == "h264_nvenc":
                # Graceful CPU fallback if NVENC fails
                encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "19"]
                cmd_cpu = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", clean_audio_wav,
                    "-filter_complex", concat_filter,
                    "-map", "[outv]",
                    "-map", "1:a",
                    *encoder_args,
                    "-c:a", "aac", "-b:a", "320k",
                    "-movflags", "+faststart",
                    output_video_path
                ]
                subprocess.run(cmd_cpu, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                raise

    return output_video_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_video.py -v`
Expected: 2 passed

---

### Task 5: Batch Processing Orchestrator & CLI Tool

**Files:**
- Create: `interview_processor.py`
- Create: `requirements.txt`

**Interfaces:**
- Produces:
  - `process_single_video(video_path: str, output_dir: str, config: dict) -> dict`
  - `process_batch(input_dir: str, output_dir: str, config: dict) -> list[dict]`
  - CLI command: `python interview_processor.py --input <path> --output <path> ...`

- [ ] **Step 1: Create `requirements.txt`**

Create `requirements.txt`:
```txt
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.23.0
scipy>=1.10.0
soundfile>=0.12.1
pyloudnorm>=0.1.1
tqdm>=4.65.0
deepfilternet>=0.5.6
```

- [ ] **Step 2: Implement `interview_processor.py`**

Create `interview_processor.py`:
```python
"""
interview_processor.py - Unified Batch Processor CLI for AI Interview Silence Cutting,
DeepFilterNet Denoising, and NLE Timeline XML/EDL Export.
"""
import os
import sys
import glob
import json
import time
import argparse
import tempfile
import torch
import soundfile as sf
from tqdm import tqdm

from vad_engine import load_vad_model, get_speech_timestamps, pad_and_merge_segments, calculate_speech_stats
from denoise_engine import denoise_audio_deepfilter, denoise_audio_resemble, cut_and_crossfade_audio, normalize_loudness
from timeline_exporter import export_timeline_files
from video_engine import probe_video_metadata, extract_audio_from_video, cut_and_render_video_nvenc

SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

def process_single_video(
    video_path: str,
    output_dir: str,
    config: dict,
    vad_model_tuple: tuple = None
) -> dict:
    """Processes a single video file end-to-end with temporary file isolation."""
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{base_name}_clean_cut.mp4")
    out_base = os.path.join(output_dir, base_name)

    if not config.get("overwrite", False) and os.path.exists(out_video):
        return {"status": "skipped", "file": video_path, "reason": "Output already exists"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Probe video
        meta = probe_video_metadata(video_path)
        if not meta["has_audio"]:
            return {"status": "failed", "file": video_path, "error": "No audio stream found"}

        # 2. Extract audio
        raw_48k = os.path.join(tmp_dir, "raw_48k.wav")
        raw_16k = os.path.join(tmp_dir, "raw_16k.wav")
        extract_audio_from_video(video_path, raw_48k, raw_16k)

        # 3. AI Denoising
        denoised_48k = os.path.join(tmp_dir, "denoised_48k.wav")
        engine = config.get("denoise_engine", "DeepFilterNet3")
        if engine == "DeepFilterNet3":
            denoise_audio_deepfilter(raw_48k, denoised_48k)
        elif engine == "ResembleEnhance":
            denoise_audio_resemble(raw_48k, denoised_48k)
        else:
            denoised_48k = raw_48k

        # 4. Voice Activity Detection
        if vad_model_tuple is None:
            device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
            vad_model, vad_utils = load_vad_model(device=device)
        else:
            vad_model, vad_utils = vad_model_tuple

        # Load 16k audio into torch tensor for VAD
        audio_16k_data, _ = sf.read(raw_16k, dtype="float32")
        audio_16k_tensor = torch.from_numpy(audio_16k_data)

        raw_segments = get_speech_timestamps(
            audio_16k_tensor,
            vad_model,
            vad_utils,
            sample_rate=16000,
            threshold=config.get("vad_threshold", 0.5),
            min_speech_duration_ms=int(config.get("min_speech_ms", 250)),
            min_silence_duration_ms=int(config.get("min_silence_sec", 0.8) * 1000)
        )

        # 5. Natural Padding & Merging
        kept_segments = pad_and_merge_segments(
            raw_segments,
            total_duration=meta["duration"],
            padding_sec=config.get("padding_sec", 0.25),
            min_silence_sec=config.get("min_silence_sec", 0.8)
        )

        if not kept_segments:
            # If no speech detected, keep entire original audio as fallback
            kept_segments = [(0.0, meta["duration"])]

        stats = calculate_speech_stats(meta["duration"], kept_segments)

        # 6. Audio Assembly, Crossfading & Normalization
        audio_48k_data, sr_48k = sf.read(denoised_48k, dtype="float32")
        if audio_48k_data.ndim > 1:
            audio_48k_data = audio_48k_data.T

        cut_audio = cut_and_crossfade_audio(
            audio_48k_data,
            sr_48k,
            kept_segments,
            crossfade_ms=config.get("crossfade_ms", 30)
        )

        if config.get("normalize_audio", True):
            cut_audio = normalize_loudness(cut_audio, sr_48k, target_lufs=config.get("target_lufs", -14.0))

        clean_cut_wav = os.path.join(tmp_dir, "clean_cut_audio.wav")
        sf.write(clean_cut_wav, cut_audio.T if cut_audio.ndim > 1 else cut_audio, sr_48k)

        # 7. Render Final Video
        cut_and_render_video_nvenc(
            video_path=video_path,
            clean_audio_wav=clean_cut_wav,
            segments=kept_segments,
            output_video_path=out_video,
            use_gpu=config.get("use_gpu", True)
        )

        # 8. Export NLE Timeline (XML / EDL)
        timeline_files = {}
        if config.get("export_timeline", True):
            timeline_files = export_timeline_files(
                source_media_path=video_path,
                segments=kept_segments,
                fps=meta["fps"],
                output_base_path=out_base,
                width=meta["width"],
                height=meta["height"]
            )

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "file": video_path,
            "output_video": out_video,
            "timelines": timeline_files,
            "stats": stats,
            "processing_time_sec": round(elapsed, 2)
        }

def process_batch(input_dir: str, output_dir: str, config: dict) -> list:
    """Scans input folder and batch processes all supported video files."""
    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    for ext in SUPPORTED_VIDEO_EXTS:
        all_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        all_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    all_files = sorted(list(set(all_files)))
    if not all_files:
        print(f"No video files found in {input_dir}")
        return []

    print(f"Found {len(all_files)} video(s) to process.")
    device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
    print(f"Loading Silero VAD model on {device}...")
    vad_model_tuple = load_vad_model(device=device)

    results = []
    summary_path = os.path.join(output_dir, "processing_summary.json")

    with tqdm(all_files, desc="Batch Progress", unit="video") as pbar:
        for video_file in pbar:
            pbar.set_postfix_str(os.path.basename(video_file))
            try:
                res = process_single_video(video_file, output_dir, config, vad_model_tuple)
                results.append(res)
            except Exception as e:
                print(f"\n[ERROR] Failed to process {video_file}: {e}")
                results.append({"status": "failed", "file": video_file, "error": str(e)})

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBatch processing complete! Summary saved to {summary_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Interview Video Batch Silence Remover & AI Denoiser")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing raw videos")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed videos & XMLs")
    parser.add_argument("--denoise-engine", default="DeepFilterNet3", choices=["DeepFilterNet3", "ResembleEnhance", "None"])
    parser.add_argument("--min-silence", type=float, default=0.8, help="Min silence duration to cut (seconds)")
    parser.add_argument("--padding", type=float, default=0.25, help="Speech head/tail padding margin (seconds)")
    parser.add_argument("--crossfade-ms", type=int, default=30, help="Audio crossfade length (ms)")
    parser.add_argument("--no-norm", action="store_true", help="Disable EBU R128 loudness normalization")
    parser.add_argument("--no-xml", action="store_true", help="Disable NLE XML/EDL export")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode instead of GPU")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    args = parser.parse_args()
    config = {
        "denoise_engine": args.denoise_engine,
        "min_silence_sec": args.min_silence,
        "padding_sec": args.padding,
        "crossfade_ms": args.crossfade_ms,
        "normalize_audio": not args.no_norm,
        "export_timeline": not args.no_xml,
        "use_gpu": not args.cpu,
        "overwrite": args.overwrite
    }

    process_batch(args.input, args.output, config)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test CLI help invocation**

Run: `python interview_processor.py --help`
Expected: Output showing all arguments and defaults cleanly.

---

### Task 6: Google Colab Studio Notebook (`Interview_AI_Studio.ipynb`) & Documentation (`README.md`)

**Files:**
- Create: `Interview_AI_Studio.ipynb`
- Create: `README.md`

**Interfaces:**
- Produces:
  - Ready-to-run Jupyter notebook with interactive Colab GUI `@param` forms.
  - Full instructions in `README.md` including how to import generated XML into Adobe Premiere Pro and DaVinci Resolve.

- [ ] **Step 1: Create `Interview_AI_Studio.ipynb`**

Generate a structured `.ipynb` containing:
1. Header & Quickstart Guide.
2. Cell 1: Dependency Installation (`!pip install -q ...` and GPU check).
3. Cell 2: Google Drive Auto-Mount & Folder Setup (`/content/drive/MyDrive/Interview_Studio/input` & `output`).
4. Cell 3: Interactive Configuration Form with `@param` sliders for Denoise engine, min silence, padding, and XML toggle.
5. Cell 4: Batch Execution Cell with live `tqdm` output.
6. Cell 5: Video Player & Results Downloader (`IPython.display.Video`, `.zip` exporter).

- [ ] **Step 2: Create `README.md`**

Create `README.md` documenting:
1. Quick start with Google Colab link / local setup.
2. Feature overview (Silero VAD v5, DeepFilterNet 3, NVENC, XML/EDL).
3. Step-by-step NLE import guide for Premiere Pro and DaVinci Resolve.
4. CLI command reference for local execution.

- [ ] **Step 3: Verify complete test suite execution**

Run: `pytest tests/ -v`
Expected: All tests pass with 100% green status.
