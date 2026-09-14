"""
test_package.py - Unit tests for interview_studio package structure and re-exports
"""
import inspect
import pytest
import interview_studio
from interview_studio import (
    load_vad_model,
    get_speech_timestamps,
    pad_and_merge_segments,
    calculate_speech_stats,
    SpeechSegment,
    denoise_audio_deepfilter,
    denoise_audio_resemble,
    cut_and_crossfade_audio,
    normalize_loudness,
    seconds_to_timecode,
    generate_fcp7_xml,
    generate_cmx3600_edl,
    export_timeline_files,
    probe_video_metadata,
    extract_audio_from_video,
    cut_and_render_video_nvenc,
    process_single_video,
    process_batch,
    SUPPORTED_VIDEO_EXTS,
)


def test_package_exports():
    """Verify all top-level symbols are accessible from interview_studio."""
    assert inspect.isfunction(load_vad_model)
    assert inspect.isfunction(get_speech_timestamps)
    assert inspect.isfunction(pad_and_merge_segments)
    assert inspect.isfunction(calculate_speech_stats)
    assert inspect.isclass(SpeechSegment)
    assert inspect.isfunction(denoise_audio_deepfilter)
    assert inspect.isfunction(denoise_audio_resemble)
    assert inspect.isfunction(cut_and_crossfade_audio)
    assert inspect.isfunction(normalize_loudness)
    assert inspect.isfunction(seconds_to_timecode)
    assert inspect.isfunction(generate_fcp7_xml)
    assert inspect.isfunction(generate_cmx3600_edl)
    assert inspect.isfunction(export_timeline_files)
    assert inspect.isfunction(probe_video_metadata)
    assert inspect.isfunction(extract_audio_from_video)
    assert inspect.isfunction(cut_and_render_video_nvenc)
    assert inspect.isfunction(process_single_video)
    assert inspect.isfunction(process_batch)
    assert isinstance(SUPPORTED_VIDEO_EXTS, tuple)
    assert ".mp4" in SUPPORTED_VIDEO_EXTS


def test_package_version():
    """Verify package version is defined."""
    assert hasattr(interview_studio, "__version__")
    assert interview_studio.__version__ == "1.0.0"
