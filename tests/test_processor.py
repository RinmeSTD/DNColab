import os
import sys
import json
import shutil
import subprocess
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interview_processor import (
    process_single_video,
    process_batch,
    build_parser,
    parse_cli_args,
    main,
    SUPPORTED_VIDEO_EXTS,
)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Generates a synthetic 3-second test video with 48kHz audio."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in test environment")
    tmp_dir = tmp_path_factory.mktemp("proc_video_test")
    test_file = str(tmp_dir / "test_interview.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        test_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return test_file


@pytest.fixture(scope="module")
def synthetic_no_audio_video(tmp_path_factory):
    """Generates a synthetic 2-second test video without audio."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in test environment")
    tmp_dir = tmp_path_factory.mktemp("proc_no_audio_test")
    test_file = str(tmp_dir / "no_audio.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        test_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return test_file


@pytest.fixture
def mock_vad():
    """Mock Silero VAD model tuple returning speech from 0.5s to 2.0s."""
    mock_model = MagicMock()
    mock_get_ts = MagicMock(return_value=[{"start": 0.5, "end": 2.0}])
    mock_utils = (mock_get_ts,)
    return mock_model, mock_utils


def test_cli_argument_parsing():
    """Tests argparse parsing with default and custom parameters."""
    # 1. Test defaults
    parser = build_parser()
    args = parser.parse_args(["-i", "input_folder", "-o", "output_folder"])
    assert args.input == "input_folder"
    assert args.output == "output_folder"
    assert args.denoise_engine == "DeepFilterNet3"
    assert args.min_silence == 0.8
    assert args.padding == 0.25
    assert args.crossfade_ms == 30
    assert args.no_norm is False
    assert args.no_xml is False
    assert args.cpu is False
    assert args.overwrite is False

    config = parse_cli_args(["-i", "input_folder", "-o", "output_folder"])
    assert config["denoise_engine"] == "DeepFilterNet3"
    assert config["min_silence_sec"] == 0.8
    assert config["padding_sec"] == 0.25
    assert config["crossfade_ms"] == 30
    assert config["normalize_audio"] is True
    assert config["export_timeline"] is True
    assert config["use_gpu"] is True
    assert config["overwrite"] is False

    # 2. Test custom flags
    custom_args = [
        "--input", "custom_in",
        "--output", "custom_out",
        "--denoise-engine", "None",
        "--min-silence", "1.2",
        "--padding", "0.4",
        "--crossfade-ms", "50",
        "--no-norm",
        "--no-xml",
        "--cpu",
        "--overwrite"
    ]
    parsed_config = parse_cli_args(custom_args)
    assert parsed_config["denoise_engine"] == "None"
    assert parsed_config["min_silence_sec"] == 1.2
    assert parsed_config["padding_sec"] == 0.4
    assert parsed_config["crossfade_ms"] == 50
    assert parsed_config["normalize_audio"] is False
    assert parsed_config["export_timeline"] is False
    assert parsed_config["use_gpu"] is False
    assert parsed_config["overwrite"] is True


def test_process_single_video_end_to_end(synthetic_video, mock_vad, tmp_path):
    """Verifies end-to-end processing: rendered MP4, XML, EDL, and summary stats."""
    output_dir = str(tmp_path / "single_out")
    config = {
        "denoise_engine": "None",
        "min_silence_sec": 0.8,
        "padding_sec": 0.25,
        "crossfade_ms": 30,
        "normalize_audio": True,
        "target_lufs": -14.0,
        "export_timeline": True,
        "use_gpu": False,
        "overwrite": True
    }

    result = process_single_video(
        video_path=synthetic_video,
        output_dir=output_dir,
        config=config,
        vad_model_tuple=mock_vad
    )

    assert result["status"] == "success"
    assert "output_video" in result
    assert os.path.exists(result["output_video"])

    # Timelines
    assert "timelines" in result
    assert os.path.exists(result["timelines"]["xml"])
    assert os.path.exists(result["timelines"]["edl"])

    # Stats
    stats = result["stats"]
    assert stats["original_duration"] >= 2.9
    assert stats["kept_duration"] > 0
    assert "silence_removed" in stats
    assert "silence_percentage" in stats
    assert result["processing_time_sec"] >= 0


def test_process_single_video_no_audio(synthetic_no_audio_video, mock_vad, tmp_path):
    """Verifies graceful handling when video has no audio stream."""
    output_dir = str(tmp_path / "no_audio_out")
    config = {"use_gpu": False}

    result = process_single_video(
        video_path=synthetic_no_audio_video,
        output_dir=output_dir,
        config=config,
        vad_model_tuple=mock_vad
    )

    assert result["status"] == "failed"
    assert "No audio stream found" in result["error"]


def test_process_single_video_overwrite_flag(synthetic_video, mock_vad, tmp_path):
    """Verifies overwrite=False skips existing outputs, and overwrite=True reprocesses."""
    output_dir = str(tmp_path / "overwrite_out")
    config = {"denoise_engine": "None", "use_gpu": False, "overwrite": False}

    # First run: should succeed
    res1 = process_single_video(synthetic_video, output_dir, config, vad_model_tuple=mock_vad)
    assert res1["status"] == "success"
    assert os.path.exists(res1["output_video"])

    # Second run without overwrite: should skip
    res2 = process_single_video(synthetic_video, output_dir, config, vad_model_tuple=mock_vad)
    assert res2["status"] == "skipped"
    assert res2["reason"] == "Output already exists"

    # Third run with overwrite=True: should process again
    config["overwrite"] = True
    res3 = process_single_video(synthetic_video, output_dir, config, vad_model_tuple=mock_vad)
    assert res3["status"] == "success"


def test_process_single_video_no_speech_fallback(synthetic_video, tmp_path):
    """Verifies fallback to whole video when no speech is detected by VAD."""
    output_dir = str(tmp_path / "fallback_out")
    mock_model = MagicMock()
    mock_get_ts = MagicMock(return_value=[])  # No speech detected
    mock_vad_empty = (mock_model, (mock_get_ts,))

    config = {"denoise_engine": "None", "use_gpu": False}
    result = process_single_video(synthetic_video, output_dir, config, vad_model_tuple=mock_vad_empty)

    assert result["status"] == "success"
    assert os.path.exists(result["output_video"])
    assert result["stats"]["kept_duration"] == result["stats"]["original_duration"]
    assert result["stats"]["silence_percentage"] == 0.0


def test_process_batch_multiple_files(synthetic_video, mock_vad, tmp_path):
    """Batch processes multiple video files and verifies processing_summary.json."""
    input_dir = str(tmp_path / "batch_in")
    output_dir = str(tmp_path / "batch_out")
    os.makedirs(input_dir, exist_ok=True)

    # Create 2 video files with different case extensions
    vid1 = os.path.join(input_dir, "interview1.mp4")
    vid2 = os.path.join(input_dir, "interview2.MOV")
    shutil.copyfile(synthetic_video, vid1)
    shutil.copyfile(synthetic_video, vid2)

    # Put a non-video file to test filtering
    with open(os.path.join(input_dir, "notes.txt"), "w") as f:
        f.write("Some notes")

    config = {
        "denoise_engine": "None",
        "use_gpu": False,
        "overwrite": True,
        "vad_model_tuple": mock_vad
    }

    results = process_batch(input_dir, output_dir, config)
    assert len(results) == 2

    # Check summary json file
    summary_file = os.path.join(output_dir, "processing_summary.json")
    assert os.path.exists(summary_file)
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert len(summary_data) == 2
    for item in summary_data:
        assert item["status"] == "success"
        assert os.path.exists(item["output_video"])
        assert os.path.exists(item["timelines"]["xml"])
        assert os.path.exists(item["timelines"]["edl"])


def test_process_batch_handles_per_file_error(synthetic_video, synthetic_no_audio_video, mock_vad, tmp_path):
    """Verifies that an error in one video does not abort batch processing."""
    input_dir = str(tmp_path / "err_batch_in")
    output_dir = str(tmp_path / "err_batch_out")
    os.makedirs(input_dir, exist_ok=True)

    vid_good = os.path.join(input_dir, "good.mp4")
    vid_bad = os.path.join(input_dir, "bad.mp4")
    shutil.copyfile(synthetic_video, vid_good)
    shutil.copyfile(synthetic_no_audio_video, vid_bad)

    config = {
        "denoise_engine": "None",
        "use_gpu": False,
        "vad_model_tuple": mock_vad
    }

    results = process_batch(input_dir, output_dir, config)
    assert len(results) == 2

    statuses = [r["status"] for r in results]
    assert "success" in statuses
    assert "failed" in statuses


def test_process_batch_empty_dir(tmp_path):
    """Verifies process_batch on empty input directory returns empty list."""
    empty_in = str(tmp_path / "empty_dir")
    empty_out = str(tmp_path / "empty_out")
    os.makedirs(empty_in, exist_ok=True)

    results = process_batch(empty_in, empty_out, config={})
    assert results == []
    summary_file = os.path.join(empty_out, "processing_summary.json")
    assert os.path.exists(summary_file)
