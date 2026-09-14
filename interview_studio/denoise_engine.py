"""
denoise_engine.py - AI Denoising, Audio Crossfading & Loudness Normalization.
"""
import os
import shutil
import subprocess
import warnings
from typing import List, Tuple

import numpy as np


def denoise_audio_deepfilter(input_wav: str, output_wav: str) -> str:
    """
    Denoises audio using DeepFilterNet 3.
    1. Checks for standalone precompiled `deep-filter` or `df-enhance` CLI binaries.
    2. Falls back to DeepFilterNet Python API (df.enhance).
    3. Falls back gracefully to file copy with warning if neither is installed.
    """
    out_dir = os.path.dirname(os.path.abspath(output_wav)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fallback_reason = None

    # 1. Attempt using standalone CLI command (`deep-filter` or `df-enhance`)
    cli_candidates = ["deep-filter", "df-enhance"]
    local_bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".bin")
    if os.path.isdir(local_bin_dir):
        for name in ["deep-filter", "deep-filter.exe", "df-enhance", "df-enhance.exe"]:
            p = os.path.join(local_bin_dir, name)
            if os.path.isfile(p):
                cli_candidates.insert(0, p)

    for cli_cmd in cli_candidates:
        cli_path = shutil.which(cli_cmd) if not os.path.isabs(cli_cmd) else (cli_cmd if os.path.isfile(cli_cmd) else None)
        if cli_path:
            try:
                res = subprocess.run([cli_path, input_wav, "-o", out_dir], capture_output=True, text=True)
                if res.returncode == 0:
                    if os.path.exists(output_wav):
                        return output_wav
                    base = os.path.splitext(os.path.basename(input_wav))[0]
                    candidates = [
                        os.path.join(out_dir, f"{base}.wav"),
                        os.path.join(out_dir, f"{base}_enhanced.wav"),
                        os.path.join(out_dir, f"{base}_DeepFilterNet3.wav"),
                        os.path.join(out_dir, f"{base}_df.wav"),
                    ]
                    for cand in candidates:
                        if os.path.exists(cand):
                            if os.path.abspath(cand) != os.path.abspath(output_wav):
                                shutil.move(cand, output_wav)
                            return output_wav
            except Exception as e:
                fallback_reason = f"{cli_cmd} CLI error: {e}"

    # 2. Attempt using DeepFilterNet Python API (df.enhance)
    try:
        import df.enhance as df_enhance

        if hasattr(df_enhance, "init_df") and hasattr(df_enhance, "enhance"):
            model, df_state, _ = df_enhance.init_df()
            audio, _ = df_enhance.load_audio(input_wav, sr=df_state.sr())
            enhanced = df_enhance.enhance(model, df_state, audio)
            df_enhance.save_audio(output_wav, enhanced, sr=df_state.sr())
            return output_wav
    except ImportError:
        pass
    except Exception as e:
        if not fallback_reason:
            fallback_reason = f"DeepFilterNet Python API error: {e}"

    # 3. Fallback: Copy unenhanced file
    if os.path.abspath(input_wav) != os.path.abspath(output_wav):
        shutil.copyfile(input_wav, output_wav)

    if fallback_reason:
        warnings.warn(f"{fallback_reason} - Proceeding with original audio copy.", UserWarning)
    else:
        warnings.warn(
            "DeepFilterNet 3 is not installed or available. Proceeding with original audio copy.",
            UserWarning,
        )

    return output_wav


def denoise_audio_resemble(
    input_wav: str,
    output_wav: str,
    solver: str = "midpoint",
    nfe: int = 64,
) -> str:
    """
    Denoises and enhances speech using Resemble Enhance (deep generative speech enhancement).
    Falls back gracefully to file copy with warning if resemble_enhance is not installed.
    """
    out_dir = os.path.dirname(os.path.abspath(output_wav)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fallback_reason = None

    try:
        import resemble_enhance.enhancer.inference as resemble_inf
        import torch
        import torchaudio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dwav, sr = torchaudio.load(input_wav)

        if dwav.ndim > 1 and dwav.shape[0] > 1:
            dwav_mono = dwav.mean(dim=0)
        else:
            dwav_mono = dwav.squeeze()

        enhanced, new_sr = resemble_inf.enhance(
            dwav_mono,
            sr,
            device,
            nfe=nfe,
            solver=solver,
            lambd=0.9,
            tau=0.5,
        )

        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)

        torchaudio.save(output_wav, enhanced.cpu(), new_sr)
        return output_wav
    except ImportError:
        pass
    except Exception as e:
        fallback_reason = f"resemble_enhance runtime error: {e}"

    # Fallback: Copy unenhanced file
    if os.path.abspath(input_wav) != os.path.abspath(output_wav):
        shutil.copyfile(input_wav, output_wav)

    if fallback_reason:
        warnings.warn(f"{fallback_reason} - Proceeding with original audio copy.", UserWarning)
    else:
        warnings.warn(
            "resemble_enhance is not installed. Proceeding with original audio copy.",
            UserWarning,
        )

    return output_wav


def cut_and_crossfade_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    segments: List[Tuple[float, float]],
    crossfade_ms: int = 30,
) -> np.ndarray:
    """
    Slices audio according to kept intervals and applies non-shortening butt-splice micro
    boundary fading (5-10ms) on segment edges to eliminate cut clicks and DC offset pops.
    Total duration matches sum(end_sec - start_sec) exactly.
    """
    if not segments or audio_data.size == 0:
        if audio_data.ndim == 1:
            return np.zeros(0, dtype=audio_data.dtype)
        return np.zeros((audio_data.shape[0], 0), dtype=audio_data.dtype)

    total_samples = audio_data.shape[-1]
    slices: List[np.ndarray] = []

    for start_sec, end_sec in segments:
        s_sec = max(0.0, float(start_sec))
        e_sec = float(end_sec)
        if e_sec <= s_sec:
            continue

        start_samp = int(round(s_sec * sample_rate))
        end_samp = int(round(e_sec * sample_rate))
        start_samp = min(max(0, start_samp), total_samples)
        end_samp = min(max(start_samp, end_samp), total_samples)

        if end_samp > start_samp:
            slices.append(audio_data[..., start_samp:end_samp].copy())

    if not slices:
        if audio_data.ndim == 1:
            return np.zeros(0, dtype=audio_data.dtype)
        return np.zeros((audio_data.shape[0], 0), dtype=audio_data.dtype)

    if len(slices) == 1:
        return slices[0]

    fade_samples = max(1, int(sample_rate * min(crossfade_ms, 15) / 1000))

    for idx, s in enumerate(slices):
        k = min(fade_samples, s.shape[-1] // 2)
        if k <= 1:
            continue

        t = np.linspace(0.0, 1.0, k, endpoint=True, dtype=np.float64)
        fade_in = 0.5 * (1.0 - np.cos(np.pi * t))
        fade_out = 0.5 * (1.0 + np.cos(np.pi * t))

        if idx > 0:
            head = s[..., :k]
            head_float = head.astype(np.float64) * fade_in
            if np.issubdtype(s.dtype, np.integer):
                s[..., :k] = np.round(head_float).astype(s.dtype)
            else:
                s[..., :k] = head_float.astype(s.dtype)

        if idx < len(slices) - 1:
            tail = s[..., -k:]
            tail_float = tail.astype(np.float64) * fade_out
            if np.issubdtype(s.dtype, np.integer):
                s[..., -k:] = np.round(tail_float).astype(s.dtype)
            else:
                s[..., -k:] = tail_float.astype(s.dtype)

    return np.concatenate(slices, axis=-1)


def normalize_loudness(
    audio_data: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """
    Normalizes audio to EBU R128 standard (default -14.0 LUFS) with true peak limiting (<= 0.95).
    Includes automatic fallback to robust pure-NumPy RMS normalization if pyloudnorm or scipy
    is unavailable or encounters environment version conflicts.
    """
    if audio_data.size == 0:
        return audio_data.copy()

    # Convert to float64 for high-precision measurement
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_float = audio_data.astype(np.float64, copy=True)
    elif np.issubdtype(audio_data.dtype, np.integer):
        max_int = float(np.iinfo(audio_data.dtype).max)
        audio_float = audio_data.astype(np.float64) / max_int
    else:
        audio_float = audio_data.astype(np.float64)

    needs_transpose = False
    if audio_float.ndim == 2:
        # Expects (samples, channels)
        if audio_float.shape[0] <= 8 and audio_float.shape[1] > audio_float.shape[0]:
            data_for_meter = audio_float.T
            needs_transpose = True
        else:
            data_for_meter = audio_float
    else:
        data_for_meter = audio_float

    # 1. Primary path: Attempt EBU R128 loudness normalization via pyloudnorm
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(data_for_meter)

        if not np.isnan(loudness) and not np.isinf(loudness) and loudness >= -70.0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                normalized = pyln.normalize.loudness(data_for_meter, loudness, target_lufs)

            max_peak = float(np.max(np.abs(normalized)))
            if max_peak > 0.95:
                gain = 0.95 / max_peak
                normalized = normalized * gain

            normalized = np.clip(normalized, -0.95, 0.95)

            if needs_transpose:
                normalized = normalized.T

            if np.issubdtype(audio_data.dtype, np.floating):
                return normalized.astype(audio_data.dtype)
            return normalized.astype(np.float32)
    except Exception:
        # Fall through to pure NumPy RMS normalizer
        pass

    # 2. Resilient pure-NumPy RMS normalization fallback
    rms = float(np.sqrt(np.mean(data_for_meter**2)))
    if rms > 1e-5:
        target_rms = (10.0 ** (target_lufs / 20.0)) * 0.9
        gain = min(target_rms / rms, 8.0)
        normalized = data_for_meter * gain
        max_peak = float(np.max(np.abs(normalized)))
        if max_peak > 0.95:
            normalized = normalized * (0.95 / max_peak)
        normalized = np.clip(normalized, -0.95, 0.95)

        if needs_transpose:
            normalized = normalized.T

        if np.issubdtype(audio_data.dtype, np.floating):
            return normalized.astype(audio_data.dtype)
        return normalized.astype(np.float32)

    return audio_data.copy()
