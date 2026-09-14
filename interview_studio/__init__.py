"""
Interview AI Studio - Open-Source Video Post-Production & Silence Removal Engine.
"""
from .vad_engine import (
    load_vad_model,
    get_speech_timestamps,
    pad_and_merge_segments,
    calculate_speech_stats,
    SpeechSegment,
)
from .denoise_engine import (
    denoise_audio_deepfilter,
    denoise_audio_resemble,
    cut_and_crossfade_audio,
    normalize_loudness,
)
from .timeline_exporter import (
    seconds_to_timecode,
    generate_fcp7_xml,
    generate_cmx3600_edl,
    export_timeline_files,
)
from .video_engine import (
    probe_video_metadata,
    extract_audio_from_video,
    cut_and_render_video_nvenc,
)
from .processor import (
    process_single_video,
    process_batch,
    SUPPORTED_VIDEO_EXTS,
)

__version__ = "1.0.0"

__all__ = [
    "load_vad_model",
    "get_speech_timestamps",
    "pad_and_merge_segments",
    "calculate_speech_stats",
    "SpeechSegment",
    "denoise_audio_deepfilter",
    "denoise_audio_resemble",
    "cut_and_crossfade_audio",
    "normalize_loudness",
    "seconds_to_timecode",
    "generate_fcp7_xml",
    "generate_cmx3600_edl",
    "export_timeline_files",
    "probe_video_metadata",
    "extract_audio_from_video",
    "cut_and_render_video_nvenc",
    "process_single_video",
    "process_batch",
    "SUPPORTED_VIDEO_EXTS",
]
