"""
timeline_exporter.py - Root backward-compatibility alias for interview_studio.timeline_exporter
"""
from interview_studio.timeline_exporter import *
from interview_studio.timeline_exporter import (
    seconds_to_timecode,
    generate_fcp7_xml,
    generate_cmx3600_edl,
    export_timeline_files,
)
