"""
vad_engine.py - Root backward-compatibility alias for interview_studio.vad_engine
"""
from interview_studio.vad_engine import *
from interview_studio.vad_engine import (
    load_vad_model,
    get_speech_timestamps,
    pad_and_merge_segments,
    calculate_speech_stats,
    SpeechSegment,
)
