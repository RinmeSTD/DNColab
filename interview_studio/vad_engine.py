"""
vad_engine.py - Voice Activity Detection and Natural Speech Segmenting using Silero VAD v5.
"""
from typing import List, Tuple, Dict, Any, NamedTuple

try:
    import torch
except ImportError:
    torch = None


class SpeechSegment(NamedTuple):
    start: float
    end: float


def load_vad_model(device: str = "cpu") -> Tuple[Any, Any]:
    """
    Loads Silero VAD v5 from torch hub.
    Raises ImportError if PyTorch is not installed.
    """
    if torch is None:
        raise ImportError("PyTorch is required to load the Silero VAD model.")

    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
    except TypeError:
        # Fallback for older torch versions without trust_repo parameter
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, utils


def get_speech_timestamps(
    audio_tensor: Any,
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
    if utils is None:
        raise ValueError("utils cannot be None")

    if isinstance(utils, (list, tuple)):
        get_speech_ts_fn = utils[0]
    elif hasattr(utils, "get_speech_timestamps"):
        get_speech_ts_fn = getattr(utils, "get_speech_timestamps")
    elif callable(utils):
        get_speech_ts_fn = utils
    else:
        raise TypeError(
            "utils must be a tuple/list containing get_speech_timestamps, an object with that attribute, or callable"
        )

    # silero expects 1D float tensor
    if hasattr(audio_tensor, "ndim") and audio_tensor.ndim > 1:
        if hasattr(audio_tensor, "mean"):
            audio_tensor = audio_tensor.mean(dim=0)
    if hasattr(audio_tensor, "squeeze"):
        audio_tensor = audio_tensor.squeeze()

    speech_timestamps = get_speech_ts_fn(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
    )
    return speech_timestamps


def pad_and_merge_segments(
    segments: List[Dict[str, float]],
    total_duration: float,
    padding_sec: float = 0.25,
    min_silence_sec: float = 0.8,
) -> List[Tuple[float, float]]:
    """
    Expands speech segments with head/tail padding and merges overlapping or closely spaced segments.
    Clamps segments to [0.0, total_duration].
    Merges segments separated by <= min_silence_sec.
    """
    if not segments:
        return []

    # 1. Apply padding and clamp to [0, total_duration]
    padded = []
    for seg in segments:
        s = max(0.0, float(seg["start"]) - padding_sec)
        e = min(total_duration, float(seg["end"]) + padding_sec)
        if e > s:
            padded.append((s, e))

    if not padded:
        return []

    # 2. Sort by start time
    padded.sort(key=lambda x: x[0])

    # 3. Merge overlapping or segments separated by less than or equal to min_silence_sec
    merged: List[Tuple[float, float]] = []
    current_start, current_end = padded[0]

    for next_start, next_end in padded[1:]:
        # If the gap between current segment end and next start is <= min_silence_sec
        if next_start <= current_end + min_silence_sec:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    return merged


def calculate_speech_stats(
    original_duration: float,
    kept_segments: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Calculates summary statistics about removed silence."""
    kept_duration = sum(end - start for start, end in kept_segments)
    silence_removed = max(0.0, original_duration - kept_duration)
    silence_pct = (silence_removed / original_duration * 100.0) if original_duration > 0 else 0.0
    return {
        "original_duration": round(original_duration, 3),
        "kept_duration": round(kept_duration, 3),
        "silence_removed": round(silence_removed, 3),
        "silence_percentage": round(silence_pct, 2),
    }
