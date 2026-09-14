"""
denoise_engine.py - AI Denoising, Audio Crossfading & Loudness Normalization.
"""
import os
import shutil
import subprocess
import warnings
from typing import List, Tuple

import numpy as np
import pyloudnorm as pyln


def denoise_audio_deepfilter(input_wav: str, output_wav: str) -> str:
    """
    Denoises audio using DeepFilterNet 3 (df.enhance).
    Falls back to `df-enhance` CLI command if available.
    Falls back gracefully to file copy with warning if neither is installed.
    """
    out_dir = os.path.dirname(os.path.abspath(output_wav)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. Attempt using DeepFilterNet Python API (df.enhance)
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
        warnings.warn(f"DeepFilterNet Python API failed ({e}), checking CLI.", UserWarning)

    # 2. Attempt using df-enhance CLI command if available
    cli_path = shutil.which("df-enhance")
    if cli_path:
        try:
            res = subprocess.run([cli_path, input_wav, "-o", out_dir], capture_output=True)
            if res.returncode == 0:
                if os.path.exists(output_wav):
                    return output_wav
                # Check if output file was created under base name
                base = os.path.splitext(os.path.basename(input_wav))[0]
                candidates = [
                    os.path.join(out_dir, f"{base}.wav"),
                    os.path.join(out_dir, f"{base}_enhanced.wav"),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        if cand != output_wav:
                            shutil.move(cand, output_wav)
                        return output_wav
        except Exception as e:
            warnings.warn(f"df-enhance CLI execution failed ({e}).", UserWarning)

    # 3. Fallback gracefully to file copy
    warnings.warn(
        "DeepFilterNet is not installed and 'df-enhance' CLI was not found. "
        "Falling back to unenhanced audio copy.",
        UserWarning,
    )
    shutil.copyfile(input_wav, output_wav)
    return output_wav


def denoise_audio_resemble(
    input_wav: str,
    output_wav: str,
    solver: str = "midpoint",
    nfe: int = 64,
) -> str:
    """
    Denoises and enhances audio using resemble_enhance.
    Falls back gracefully to file copy with warning if not installed.
    """
    out_dir = os.path.dirname(os.path.abspath(output_wav)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. Attempt using resemble_enhance Python API
    try:
        import torch
        import torchaudio
        import resemble_enhance.enhancer.inference as resemble_inf

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dwav, sr = torchaudio.load(input_wav)

        if hasattr(resemble_inf, "enhance"):
            try:
                enhanced, new_sr = resemble_inf.enhance(
                    dwav, sr, device=device, solver=solver, nfe=nfe
                )
            except TypeError:
                enhanced, new_sr = resemble_inf.enhance(dwav, sr, device=device)
        elif hasattr(resemble_inf, "denoise"):
            enhanced, new_sr = resemble_inf.denoise(dwav, sr, device=device)
        else:
            raise AttributeError("resemble_enhance.enhancer.inference has neither enhance nor denoise")

        torchaudio.save(output_wav, enhanced, new_sr)
        return output_wav
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"resemble_enhance runtime error ({e}).", UserWarning)

    # 2. Fallback gracefully to file copy
    warnings.warn(
        "resemble_enhance is not installed. Falling back to unenhanced audio copy.",
        UserWarning,
    )
    shutil.copyfile(input_wav, output_wav)
    return output_wav


def cut_and_crossfade_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    segments: List[Tuple[float, float]],
    crossfade_ms: int = 30,
    curve: str = "cosine",
) -> np.ndarray:
    """
    Slices audio_data (1D mono or 2D stereo shape (channels, samples)) based on kept [start_sec, end_sec] segments.
    Applies smooth cosine or linear crossfade (crossfade_ms) between consecutive slices to eliminate cut clicks/pops.
    Handles empty segments, single segment, and out-of-bound samples safely.
    """
    if audio_data.size == 0 or not segments:
        if audio_data.ndim == 1:
            return np.empty(0, dtype=audio_data.dtype)
        return np.empty((audio_data.shape[0], 0), dtype=audio_data.dtype)

    total_samples = audio_data.shape[-1]
    slices = []

    for start_sec, end_sec in segments:
        start_samp = max(0, min(total_samples, int(round(start_sec * sample_rate))))
        end_samp = max(start_samp, min(total_samples, int(round(end_sec * sample_rate))))
        if end_samp > start_samp:
            slices.append(audio_data[..., start_samp:end_samp])

    if not slices:
        if audio_data.ndim == 1:
            return np.empty(0, dtype=audio_data.dtype)
        return np.empty((audio_data.shape[0], 0), dtype=audio_data.dtype)

    if len(slices) == 1:
        return slices[0].copy()

    fade_len = int(round(sample_rate * max(0, crossfade_ms) / 1000.0))
    if fade_len <= 0:
        return np.concatenate(slices, axis=-1)

    result = slices[0]
    fade_shape = (1,) * (result.ndim - 1)

    for nxt in slices[1:]:
        k = min(fade_len, result.shape[-1], nxt.shape[-1])
        if k <= 0:
            result = np.concatenate([result, nxt], axis=-1)
            continue

        if curve == "linear":
            fade_in = np.linspace(0.0, 1.0, k, endpoint=True)
            fade_out = 1.0 - fade_in
        else:  # raised cosine
            t = np.linspace(0.0, np.pi, k, endpoint=True)
            fade_out = 0.5 * (1.0 + np.cos(t))
            fade_in = 0.5 * (1.0 - np.cos(t))

        fade_out = fade_out.reshape(fade_shape + (k,)).astype(result.dtype)
        fade_in = fade_in.reshape(fade_shape + (k,)).astype(result.dtype)

        head_r = result[..., :-k]
        tail_r = result[..., -k:]
        head_n = nxt[..., :k]
        tail_n = nxt[..., k:]

        overlap = tail_r * fade_out + head_n * fade_in
        result = np.concatenate([head_r, overlap, tail_n], axis=-1)

    return result


def normalize_loudness(
    audio_data: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """
    Normalizes audio to EBU R128 standard (default -14.0 LUFS) using pyloudnorm Meter.
    Includes true peak limiting (<= 0.95 / -0.5 dBFS) to prevent any digital distortion/clipping.
    Preserves 1D mono (samples,) and 2D stereo (channels, samples) array shapes.
    """
    if audio_data.size == 0:
        return audio_data.copy()

    # Convert to float64 for high-precision pyloudnorm measurement
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_float = audio_data.astype(np.float64, copy=True)
    elif np.issubdtype(audio_data.dtype, np.integer):
        max_int = float(np.iinfo(audio_data.dtype).max)
        audio_float = audio_data.astype(np.float64) / max_int
    else:
        audio_float = audio_data.astype(np.float64)

    needs_transpose = False
    if audio_float.ndim == 2:
        # pyloudnorm expects (samples, channels)
        if audio_float.shape[0] <= 8 and audio_float.shape[1] > audio_float.shape[0]:
            data_for_meter = audio_float.T
            needs_transpose = True
        else:
            data_for_meter = audio_float
    else:
        data_for_meter = audio_float

    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(data_for_meter)
    except Exception:
        loudness = float("-inf")

    # If audio is silent or unmeasurable (-inf, nan, or extremely low < -70 LUFS)
    if np.isnan(loudness) or np.isinf(loudness) or loudness < -70.0:
        return audio_data.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        normalized = pyln.normalize.loudness(data_for_meter, loudness, target_lufs)

    # True peak limiting (<= 0.95 / -0.5 dBFS) to prevent clipping
    max_peak = float(np.max(np.abs(normalized)))
    if max_peak > 0.95:
        gain = 0.95 / max_peak
        normalized = normalized * gain

    # Hard ceiling guard
    normalized = np.clip(normalized, -0.95, 0.95)

    if needs_transpose:
        normalized = normalized.T

    if np.issubdtype(audio_data.dtype, np.floating):
        return normalized.astype(audio_data.dtype)
    return normalized.astype(np.float32)
