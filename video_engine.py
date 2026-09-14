"""
video_engine.py - Root backward-compatibility alias for interview_studio.video_engine
"""
from interview_studio.video_engine import *
from interview_studio.video_engine import (
    probe_video_metadata,
    extract_audio_from_video,
    cut_and_render_video_nvenc,
)
