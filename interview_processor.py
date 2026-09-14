"""
interview_processor.py - Unified Batch Processor CLI for AI Interview Silence Cutting,
DeepFilterNet Denoising, and NLE Timeline XML/EDL Export.
"""
from interview_studio.processor import (
    SUPPORTED_VIDEO_EXTS,
    process_single_video,
    process_batch,
    build_parser,
    parse_cli_args,
    main,
)

__all__ = [
    "SUPPORTED_VIDEO_EXTS",
    "process_single_video",
    "process_batch",
    "build_parser",
    "parse_cli_args",
    "main",
]

if __name__ == "__main__":
    main()
