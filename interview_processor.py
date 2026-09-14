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
import warnings
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from vad_engine import (
    load_vad_model,
    get_speech_timestamps,
    pad_and_merge_segments,
    calculate_speech_stats,
)
from denoise_engine import (
    denoise_audio_deepfilter,
    denoise_audio_resemble,
    cut_and_crossfade_audio,
    normalize_loudness,
)
from timeline_exporter import export_timeline_files
from video_engine import (
    probe_video_metadata,
    extract_audio_from_video,
    cut_and_render_video_nvenc,
)

SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def process_single_video(
    video_path: str,
    output_dir: str,
    config: Optional[dict] = None,
    vad_model_tuple: Optional[tuple] = None,
) -> dict:
    """
    Processes a single video file end-to-end with temporary file isolation.
    Pipeline:
      1. Probe video metadata.
      2. Extract 48kHz audio and 16kHz VAD audio into temporary directory.
      3. Apply AI Denoising (DeepFilterNet3, ResembleEnhance, or None).
      4. Detect speech segments with Silero VAD.
      5. Apply natural margin padding (default 250ms) and merge pauses < min_silence (default 0.8s).
      6. Assemble audio with anti-pop micro edge fades and apply EBU R128 loudness normalization (-14 LUFS).
      7. Render final cut video with GPU NVENC (with CPU libx264 fallback).
      8. Export FCP7 XML and CMX 3600 EDL timeline project files.
    """
    if config is None:
        config = {}

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{base_name}_clean_cut.mp4")
    out_base = os.path.join(output_dir, base_name)

    if not config.get("overwrite", False) and os.path.exists(out_video):
        return {"status": "skipped", "file": video_path, "reason": "Output already exists"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Probe video
        meta = probe_video_metadata(video_path)
        if not meta.get("has_audio", False):
            return {"status": "failed", "file": video_path, "error": "No audio stream found"}

        total_duration = meta.get("duration", 0.0)
        if total_duration <= 0.0:
            return {"status": "failed", "file": video_path, "error": "Invalid video duration"}

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
            if torch is not None:
                try:
                    device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
                    vad_model, vad_utils = load_vad_model(device=device)
                except Exception as e:
                    warnings.warn(f"Failed to load VAD model ({e}); fallback to full video.", UserWarning)
                    vad_model, vad_utils = None, None
            else:
                vad_model, vad_utils = None, None
        else:
            vad_model, vad_utils = vad_model_tuple

        raw_segments = []
        if vad_model is not None and vad_utils is not None:
            audio_16k_data, _ = sf.read(raw_16k, dtype="float32")
            if torch is not None:
                audio_16k_tensor = torch.from_numpy(audio_16k_data)
            else:
                audio_16k_tensor = audio_16k_data

            raw_segments = get_speech_timestamps(
                audio_16k_tensor,
                vad_model,
                vad_utils,
                sample_rate=16000,
                threshold=config.get("vad_threshold", 0.5),
                min_speech_duration_ms=int(config.get("min_speech_ms", 250)),
                min_silence_duration_ms=int(config.get("min_silence_sec", 0.8) * 1000),
            )

        # 5. Natural Padding & Merging
        kept_segments = pad_and_merge_segments(
            raw_segments,
            total_duration=total_duration,
            padding_sec=config.get("padding_sec", 0.25),
            min_silence_sec=config.get("min_silence_sec", 0.8),
        )

        if not kept_segments:
            # If no speech detected, keep entire original audio as fallback
            kept_segments = [(0.0, total_duration)]

        stats = calculate_speech_stats(total_duration, kept_segments)

        # 6. Audio Assembly, Crossfading & Normalization
        audio_48k_data, sr_48k = sf.read(denoised_48k, dtype="float32")
        if audio_48k_data.ndim > 1:
            audio_48k_data = audio_48k_data.T

        cut_audio = cut_and_crossfade_audio(
            audio_48k_data,
            sr_48k,
            kept_segments,
            crossfade_ms=config.get("crossfade_ms", 30),
        )

        if config.get("normalize_audio", True):
            cut_audio = normalize_loudness(
                cut_audio,
                sr_48k,
                target_lufs=config.get("target_lufs", -14.0),
            )

        clean_cut_wav = os.path.join(tmp_dir, "clean_cut_audio.wav")
        sf.write(clean_cut_wav, cut_audio.T if cut_audio.ndim > 1 else cut_audio, sr_48k)

        # 7. Render Final Video
        cut_and_render_video_nvenc(
            video_path=video_path,
            clean_audio_wav=clean_cut_wav,
            segments=kept_segments,
            output_video_path=out_video,
            use_gpu=config.get("use_gpu", True),
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
                height=meta["height"],
            )

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "file": video_path,
            "output_video": out_video,
            "timelines": timeline_files,
            "stats": stats,
            "processing_time_sec": round(elapsed, 2),
        }


def process_batch(
    input_dir: str,
    output_dir: str,
    config: Optional[dict] = None,
) -> List[dict]:
    """Scans input folder and batch processes all supported video files."""
    if config is None:
        config = {}

    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    if os.path.exists(input_dir):
        for fname in os.listdir(input_dir):
            fpath = os.path.join(input_dir, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTS:
                    all_files.append(fpath)

    all_files = sorted(list(set(all_files)))
    summary_path = os.path.join(output_dir, "processing_summary.json")

    if not all_files:
        print(f"No video files found in {input_dir}")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    print(f"Found {len(all_files)} video(s) to process.")
    vad_model_tuple = config.get("vad_model_tuple", None)
    if vad_model_tuple is None and torch is not None:
        try:
            device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
            print(f"Loading Silero VAD model on {device}...")
            vad_model_tuple = load_vad_model(device=device)
        except Exception as e:
            print(f"Warning: Could not preload VAD model ({e}). Fallback mode enabled.")

    results = []
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


def build_parser() -> argparse.ArgumentParser:
    """Creates argparse CLI argument parser for interview_processor."""
    parser = argparse.ArgumentParser(
        description="Interview Video Batch Silence Remover & AI Denoiser"
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory containing raw videos")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed videos & XMLs")
    parser.add_argument(
        "--denoise-engine",
        default="DeepFilterNet3",
        choices=["DeepFilterNet3", "ResembleEnhance", "None"],
        help="AI Denoising engine (default: DeepFilterNet3)",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.8,
        help="Min silence duration to cut in seconds (default: 0.8)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Speech head/tail padding margin in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=30,
        help="Audio edge crossfade length in ms (default: 30)",
    )
    parser.add_argument(
        "--no-norm",
        action="store_true",
        help="Disable EBU R128 loudness normalization (-14 LUFS)",
    )
    parser.add_argument(
        "--no-xml",
        action="store_true",
        help="Disable NLE XML/EDL timeline export",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU encoding instead of GPU NVENC",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed outputs",
    )
    return parser


def parse_cli_args(argv: Optional[List[str]] = None) -> dict:
    """Parses command line arguments and returns configuration dictionary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return {
        "input": args.input,
        "output": args.output,
        "denoise_engine": args.denoise_engine,
        "min_silence_sec": args.min_silence,
        "padding_sec": args.padding,
        "crossfade_ms": args.crossfade_ms,
        "normalize_audio": not args.no_norm,
        "export_timeline": not args.no_xml,
        "use_gpu": not args.cpu,
        "overwrite": args.overwrite,
    }


def main(argv: Optional[List[str]] = None) -> List[dict]:
    """CLI entrypoint for batch interview processing."""
    config = parse_cli_args(argv)
    input_dir = config.pop("input")
    output_dir = config.pop("output")
    return process_batch(input_dir, output_dir, config)


if __name__ == "__main__":
    main()
