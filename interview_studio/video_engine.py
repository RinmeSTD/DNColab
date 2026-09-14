"""
interview_studio.video_engine - FFprobe metadata inspection, audio track extraction,
and GPU-accelerated NVENC video cutting & muxing with CPU fallback.
"""
import os
import json
import logging
import subprocess
import shutil
import tempfile
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Inspects source video file using ffprobe to extract resolution,
    frame rate, duration, and audio stream properties.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed or not in PATH.")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,sample_rate,duration",
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

    format_duration_str = data.get("format", {}).get("duration")
    duration = float(format_duration_str) if format_duration_str and format_duration_str != "N/A" else 0.0

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            if "width" in stream and stream["width"] is not None:
                width = int(stream["width"])
            if "height" in stream and stream["height"] is not None:
                height = int(stream["height"])

            r_rate = stream.get("r_frame_rate", "")
            avg_rate = stream.get("avg_frame_rate", "")
            rate_to_parse = r_rate if r_rate and r_rate != "0/0" else avg_rate

            if rate_to_parse:
                if "/" in rate_to_parse:
                    num_str, den_str = rate_to_parse.split("/", 1)
                    num, den = float(num_str), float(den_str)
                    if den > 0:
                        fps = num / den
                else:
                    fps = float(rate_to_parse)

            stream_dur = stream.get("duration")
            if duration == 0.0 and stream_dur and stream_dur != "N/A":
                duration = float(stream_dur)

        elif codec_type == "audio":
            has_audio = True
            sr_val = stream.get("sample_rate")
            if sr_val:
                sample_rate = int(sr_val)

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration": float(duration),
        "has_audio": has_audio,
        "sample_rate": sample_rate
    }


def extract_audio_from_video(video_path: str, output_wav_48k: str, output_wav_16k: str) -> Tuple[str, str]:
    """
    Extracts two WAV audio streams from the source video:
      1. 48kHz PCM WAV (-vn -acodec pcm_s16le -ar 48000)
      2. 16kHz mono WAV (-ac 1 -ar 16000) for Silero VAD
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not in PATH.")

    # Ensure output directories exist
    out_dir_48k = os.path.dirname(os.path.abspath(output_wav_48k))
    if out_dir_48k:
        os.makedirs(out_dir_48k, exist_ok=True)

    out_dir_16k = os.path.dirname(os.path.abspath(output_wav_16k))
    if out_dir_16k:
        os.makedirs(out_dir_16k, exist_ok=True)

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
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
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
    Renders the final cut video using NVENC GPU acceleration (with graceful libx264 CPU fallback),
    muxing the clean audio track with AAC encoding.
    """
    if not segments:
        raise ValueError("No speech segments provided for rendering.")

    valid_segments = [(max(0.0, float(s)), float(e)) for s, e in segments if float(e) > max(0.0, float(s))]
    if not valid_segments:
        raise ValueError("No valid speech segments provided for rendering.")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(clean_audio_wav):
        raise FileNotFoundError(f"Clean audio file not found: {clean_audio_wav}")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not in PATH.")

    out_dir = os.path.dirname(os.path.abspath(output_video_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Build concat filter for specified video segments
    filter_complex_parts = []
    concat_inputs = []

    for idx, (start, end) in enumerate(valid_segments):
        filter_complex_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{idx}];")
        concat_inputs.append(f"[v{idx}]")

    concat_filter = f"{''.join(filter_complex_parts)}{''.join(concat_inputs)}concat=n={len(valid_segments)}:v=1:a=0[outv]"

    with tempfile.TemporaryDirectory() as tmp_dir:
        filter_script_path = os.path.join(tmp_dir, "filter_complex.txt")
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(concat_filter)

        def _render(encoder_args: list[str]) -> None:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", clean_audio_wav,
                "-filter_complex_script", filter_script_path,
                "-map", "[outv]",
                "-map", "1:a",
                *encoder_args,
                "-c:a", "aac", "-b:a", "320k",
                "-shortest",
                "-movflags", "+faststart",
                output_video_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        cpu_encoder_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p"]

        if use_gpu:
            nvenc_encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-pix_fmt", "yuv420p"]
            try:
                _render(nvenc_encoder_args)
                return output_video_path
            except subprocess.CalledProcessError as exc:
                err_msg = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
                logger.warning(
                    "NVENC GPU encoding failed; falling back cleanly to CPU libx264 encoding. Stderr: %s",
                    err_msg
                )
                _render(cpu_encoder_args)
                return output_video_path
            except Exception as exc:
                logger.warning(
                    "NVENC GPU encoding failed (%s); falling back cleanly to CPU libx264 encoding.",
                    exc
                )
                _render(cpu_encoder_args)
                return output_video_path
        else:
            _render(cpu_encoder_args)
            return output_video_path
