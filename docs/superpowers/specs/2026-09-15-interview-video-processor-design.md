# Design Specification: AI-Powered Interview Video Batch Processor

- **Date:** 2026-09-15
- **Status:** Approved / Ready for Implementation
- **Target Platform:** Google Colab (with standalone local Python CLI execution support)

---

## 1. Overview & Objective

The goal of this project is to build an open-source, automated pipeline and Google Colab Jupyter Notebook that batch processes raw interview video recordings to:
1. **Remove Silence & Dead Air Naturally**: Using deep-learning Voice Activity Detection (**Silero VAD v5**) with configurable speech margin padding (head/tail preservation) and audio crossfading to avoid jarring clips or cut-off words.
2. **AI Noise Reduction & Speech Enhancement**: Suppressing background hiss, hum, room reverberation, and air conditioning noise using **DeepFilterNet 3** (real-time neural filtering) with an optional switchable **Resemble Enhance** mode.
3. **High-Speed GPU Video Rendering**: Leveraging NVIDIA GPU NVENC (`h264_nvenc`) in Google Colab for fast frame-accurate cutting and re-encoding.
4. **NLE Timeline Export**: Generating standard **Final Cut Pro 7 XML (`.xml`)** and **CMX 3600 EDL (`.edl`)** files alongside each rendered video, allowing editors to import cuts directly into Adobe Premiere Pro or DaVinci Resolve for manual fine-tuning.
5. **Robust Batch Folder Workflow**: Processing all video files from an input folder with automatic Google Drive mounting, disk-safe temp file cleanup, and summary metrics.

---

## 2. Architecture & Pipeline Components

```
                +-------------------------------------------------+
                |   Input Video File (.mp4, .mov, .mkv, etc.)     |
                +-------------------------------------------------+
                                        |
                 (1) ffprobe metadata + ffmpeg audio extract
                                        v
                 +-----------------------------------------------+
                 |  Extracted Audio: 48kHz WAV & 16kHz VAD Mono  |
                 +-----------------------------------------------+
                                        |
                 (2) AI Speech Enhancement & Denoising
                     - DeepFilterNet 3 (libDF) [Default]
                     - Resemble Enhance [Optional]
                                        v
                 +-----------------------------------------------+
                 |              Clean Denoised Audio             |
                 +-----------------------------------------------+
                                        |
                 (3) Silero VAD v5 Speech Segment Detection
                     - Frame-level speech probability
                     - Apply Head/Tail Padding (e.g. 250ms)
                     - Merge pauses < Min Silence Threshold (0.8s)
                     - Compute [start_sec, end_sec] speech intervals
                                        v
                 +-----------------------------------------------+
                 |           Valid Speech Time Intervals         |
                 +-----------------------------------------------+
                         /                             \
    (4A) Audio Assembly & Crossfade      (4B) Frame Mapping & Timeline Gen
                 |                                      |
                 v                                      v
    +--------------------------+           +--------------------------+
    | Concatenated & Normalized|           | - FCP7 XML (Premiere/DR) |
    | Audio (EBU R128 -14 LUFS)|           | - EDL (CMX 3600)         |
    +--------------------------+           +--------------------------+
                         \                             /
                 (5) Video Cut & NVENC GPU Remux (FFmpeg)
                                        v
                 +-----------------------------------------------+
                 |  Output Batch Folder:                         |
                 |   - [filename]_clean_cut.mp4                  |
                 |   - [filename]_timeline.xml                   |
                 |   - [filename]_timeline.edl                   |
                 |   - processing_summary.json                   |
                 +-----------------------------------------------+
```

---

## 3. Detailed Component Specifications

### 3.1. Audio Extraction & Pre-Processing
- **Tool**: FFmpeg / `torchaudio`.
- **Extraction**: Extract the primary audio track to uncompressed PCM WAV:
  - High-res audio: 48,000 Hz, 24-bit / 32-bit float, Stereo/Mono.
  - VAD guidance track: 16,000 Hz, 16-bit Mono.
- **Probe**: `ffprobe` retrieves exact video stream metadata:
  - Video framerate (e.g., `23.976`, `24`, `25`, `29.97`, `30`, `59.94`, `60`).
  - Total frame count, duration, codec, width, and height.

### 3.2. AI Noise Reduction & Speech Enhancement
- **Primary Engine (Default): DeepFilterNet 3 (`df-enhance` / `libdf`)**:
  - Open-source state-of-the-art neural noise suppression.
  - Processes full-band 48kHz audio directly with low latency and high fidelity.
  - Effectively eliminates stationary and non-stationary noise (fans, traffic, room hum, clicks) without making speech sound muffled or hollow.
- **Secondary Engine (Toggle): Resemble Enhance**:
  - Uses deep generative speech upsampling + denoiser.
  - Ideal for lower-quality microphone restoration.
- **Bypass Mode**: Option to keep original audio without denoising if only silence cutting is desired.

### 3.3. Natural Silence Removal with Silero VAD v5
- **Model**: Silero VAD v5 (loaded via PyTorch Hub or local cached weights).
- **Detection Algorithm**:
  - Audio chunking in 32ms / 512-sample blocks.
  - Computes raw speech probability curve (threshold default: `0.5`).
- **Naturalness Smoothing Rules**:
  1. **Head & Tail Margins (Padding)**: Expands each speech boundary by adding `+250ms` at speech onset (lead-in) and `+250ms` at speech offset (lead-out). This guarantees breaths, subtle consonants, and trailing syllables are never abruptly cut.
  2. **Pause Thresholding**: Pauses shorter than `0.8s` (configurable `0.3s` - `2.0s`) are bridged and kept intact, preserving natural conversational pauses between sentences.
  3. **Audio Crossfading**: Between consecutive kept audio segments, applies a `30ms` smooth cosine/linear crossfade to guarantee zero DC offset pops or clicks.
- **Loudness Normalization**:
  - Optional 2-pass EBU R128 normalization (`ffmpeg-normalize` / `pyloudnorm`) targeting `-14.0 LUFS` integrated loudness and `-1.0 dBTP` true peak.

### 3.4. Video Segment Cutting & Hardware Acceleration
- **Frame Snapping**:
  - Converts timestamp intervals `[start_time, end_time]` into exact integer frame indices:
    $$\text{start\_frame} = \text{round}(\text{start\_time} \times \text{fps})$$
    $$\text{end\_frame} = \text{round}(\text{end\_time} \times \text{fps})$$
- **GPU Re-encoding**:
  - On Colab GPU runtime: Uses `ffmpeg` with `-c:v h264_nvenc -preset p4 -cq 20 -profile:v high`.
  - Concat filter complex or demux list ensures frame-accurate segment assembly.
  - Muxes the denoised, crossfaded audio track directly into the output container with `-c:a aac -b:a 320k`.
- **CPU Fallback**: Automatic fallback to `libx264 -crf 19 -preset fast` if no NVIDIA CUDA GPU is detected.

### 3.5. NLE Timeline Export (Final Cut Pro 7 XML & CMX 3600 EDL)
- **FCP7 XML (`.xml`)**:
  - Generates a fully formatted FCP7 XML sequence.
  - Contains sequence framerate, timecode, tracks (V1, A1, A2), and individual `<clipitem>` elements for each speech segment referencing the source media file.
  - Editors can open the XML directly in **Adobe Premiere Pro** (`File -> Import`) or **DaVinci Resolve** (`File -> Import Timeline -> Import AAF/EDL/XML`) to review every cut with full handle flexibility.
- **CMX 3600 EDL (`.edl`)**:
  - Generates standard EDL edit list entries with Source In/Out and Record In/Out timecodes in `HH:MM:SS:FF` format.

### 3.6. Google Colab Environment & Batch Interface
- **Notebook (`Interview_AI_Studio.ipynb`) Structure**:
  - **Cell 1: Environment & Dependency Setup**:
    - Clones repository / installs `deepfilternet`, `silero-vad`, `torchaudio`, `pyloudnorm`, `tqdm`.
    - Checks GPU availability (`torch.cuda.is_available()`).
  - **Cell 2: Storage & Google Drive Mount**:
    - Auto-mounts Google Drive at `/content/drive/MyDrive/Interview_Studio/`.
    - Creates `input/` and `output/` folders.
  - **Cell 3: Processing Configuration Form (Colab `@param` GUI)**:
    - `Input Directory` / `Output Directory`
    - `Denoise Engine`: Dropdown (`"DeepFilterNet3"`, `"ResembleEnhance"`, `"None"`)
    - `Min Silence Duration (seconds)`: Slider `[0.2, 2.0]` (default `0.8`)
    - `Speech Padding Margin (seconds)`: Slider `[0.05, 0.5]` (default `0.25`)
    - `Loudness Normalization`: Checkbox (default `True`)
    - `Export Timeline XML/EDL`: Checkbox (default `True`)
    - `Overwrite Existing Outputs`: Checkbox (default `False`)
  - **Cell 4: Run Batch Queue**:
    - Iterates through all videos with `tqdm` progress bars.
    - Prints real-time statistics (Original duration, Clean duration, Silence cut %, Processing FPS).
  - **Cell 5: Download & Review Summary**:
    - Displays interactive audio/video player for output preview.
    - Option to download all outputs as a `.zip` archive or directly access via Google Drive.

---

## 4. File Structure

```
E:\Project\MeowVid\interview\Colab\
├── Interview_AI_Studio.ipynb         # Interactive Google Colab Notebook
├── interview_processor.py            # Core Python pipeline & CLI script
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation & NLE Import Guide
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-09-15-interview-video-processor-design.md
└── tests/
    └── test_processor.py             # Unit tests for VAD logic, frame conversion, XML builder
```

---

## 5. Error Handling & Edge Cases

| Scenario | Handling Strategy |
| :--- | :--- |
| **Corrupted / unreadable video file** | Catch FFprobe/FFmpeg error, record in `processing_summary.json`, continue to next file in batch. |
| **Video with no speech detected** | Warn user, preserve original file with notice, avoid creating empty 0-second video. |
| **Video with continuous speech (no silence)** | Copy clean denoised video without cuts, avoiding empty concat errors. |
| **Variable Frame Rate (VFR) source video** | FFmpeg normalizes timestamps to constant frame rate (CFR) to prevent A/V sync drift. |
| **Colab GPU Disconnection / Low Disk** | Intermediate audio cuts and temporary files are cleaned up immediately per video; progress can resume without reprocessing completed files. |

---

## 6. Verification & Testing Strategy

1. **Unit Tests (`tests/test_processor.py`)**:
   - Verify speech segment margin expansion and overlap merging logic.
   - Verify timestamp-to-timecode conversion (`HH:MM:SS:FF`) for 24fps, 25fps, 29.97fps, and 30fps.
   - Verify XML builder creates valid parseable FCP7 XML conforming to xml schema.
2. **End-to-End Processing Test**:
   - Generate synthetic sample test video with alternating speech and silence tone.
   - Execute `interview_processor.py` against test video.
   - Confirm silence removal, crossfade continuity, and timeline XML generation.
3. **Colab Validation**:
   - Ensure `Interview_AI_Studio.ipynb` executes cleanly in Colab CPU/GPU environments.
