import os
import sys
import subprocess
import shutil
import json
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video_engine import (
    probe_video_metadata,
    extract_audio_from_video,
    cut_and_render_video_nvenc,
)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """Generates a synthetic 4-second test video with 48kHz audio using ffmpeg."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in test environment")
    tmp_dir = tmp_path_factory.mktemp("video_test")
    test_file = str(tmp_dir / "synthetic_test.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        test_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return test_file


@pytest.fixture(scope="module")
def video_without_audio(tmp_path_factory):
    """Generates a synthetic test video with no audio track."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in test environment")
    tmp_dir = tmp_path_factory.mktemp("video_no_audio")
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


@pytest.fixture(scope="module")
def clean_audio_fixture(tmp_path_factory):
    """Generates a 2-second clean audio WAV fixture (matching 2x1s cut segments)."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available in test environment")
    tmp_dir = tmp_path_factory.mktemp("clean_audio")
    test_file = str(tmp_dir / "clean_tone.wav")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=2:sample_rate=48000",
        "-c:a", "pcm_s16le",
        test_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return test_file


def test_probe_video_metadata(sample_video):
    meta = probe_video_metadata(sample_video)
    assert meta["width"] == 640
    assert meta["height"] == 360
    assert pytest.approx(meta["fps"], 0.1) == 30.0
    assert meta["duration"] >= 3.9
    assert meta["has_audio"] is True
    assert meta["sample_rate"] == 48000


def test_probe_video_metadata_no_audio(video_without_audio):
    meta = probe_video_metadata(video_without_audio)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert pytest.approx(meta["fps"], 0.1) == 25.0
    assert meta["duration"] >= 1.9
    assert meta["has_audio"] is False


def test_probe_video_metadata_nonexistent():
    with pytest.raises(FileNotFoundError):
        probe_video_metadata("non_existent_file_path_12345.mp4")


def test_extract_audio_from_video(sample_video, tmp_path):
    wav_48k = str(tmp_path / "audio_48k.wav")
    wav_16k = str(tmp_path / "audio_16k.wav")
    out_48k, out_16k = extract_audio_from_video(sample_video, wav_48k, wav_16k)

    assert os.path.exists(out_48k)
    assert os.path.exists(out_16k)
    assert os.path.getsize(out_48k) > 0
    assert os.path.getsize(out_16k) > 0

    # Inspect extracted 48kHz audio with ffprobe
    cmd_probe_48k = [
        "ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
        "-of", "json", out_48k
    ]
    res_48k = json.loads(subprocess.run(cmd_probe_48k, capture_output=True, text=True, check=True).stdout)
    stream_48k = res_48k["streams"][0]
    assert int(stream_48k["sample_rate"]) == 48000

    # Inspect extracted 16kHz audio with ffprobe (should be 16000Hz, mono)
    cmd_probe_16k = [
        "ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
        "-of", "json", out_16k
    ]
    res_16k = json.loads(subprocess.run(cmd_probe_16k, capture_output=True, text=True, check=True).stdout)
    stream_16k = res_16k["streams"][0]
    assert int(stream_16k["sample_rate"]) == 16000
    assert int(stream_16k["channels"]) == 1


def test_extract_audio_nonexistent(tmp_path):
    wav_48k = str(tmp_path / "48k.wav")
    wav_16k = str(tmp_path / "16k.wav")
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video("missing_video.mp4", wav_48k, wav_16k)


def test_cut_and_render_video_cpu(sample_video, clean_audio_fixture, tmp_path):
    segments = [(0.5, 1.5), (2.0, 3.0)]  # Total 2.0s
    out_mp4 = str(tmp_path / "rendered_cpu.mp4")

    rendered = cut_and_render_video_nvenc(
        video_path=sample_video,
        clean_audio_wav=clean_audio_fixture,
        segments=segments,
        output_video_path=out_mp4,
        use_gpu=False
    )
    assert rendered == out_mp4
    assert os.path.exists(out_mp4)
    assert os.path.getsize(out_mp4) > 0

    meta = probe_video_metadata(out_mp4)
    assert meta["width"] == 640
    assert meta["height"] == 360
    assert pytest.approx(meta["duration"], abs=0.2) == 2.0
    assert meta["has_audio"] is True


def test_cut_and_render_video_gpu_fallback(sample_video, clean_audio_fixture, tmp_path):
    segments = [(0.5, 1.5), (2.0, 3.0)]  # Total 2.0s
    out_mp4 = str(tmp_path / "rendered_gpu.mp4")

    # With use_gpu=True, if NVENC is unavailable, should cleanly fallback to libx264
    rendered = cut_and_render_video_nvenc(
        video_path=sample_video,
        clean_audio_wav=clean_audio_fixture,
        segments=segments,
        output_video_path=out_mp4,
        use_gpu=True
    )
    assert rendered == out_mp4
    assert os.path.exists(out_mp4)
    assert os.path.getsize(out_mp4) > 0

    meta = probe_video_metadata(out_mp4)
    assert meta["width"] == 640
    assert meta["height"] == 360
    assert pytest.approx(meta["duration"], abs=0.2) == 2.0
    assert meta["has_audio"] is True


def test_cut_and_render_empty_segments(sample_video, clean_audio_fixture, tmp_path):
    out_mp4 = str(tmp_path / "empty.mp4")
    with pytest.raises(ValueError, match="No speech segments"):
        cut_and_render_video_nvenc(sample_video, clean_audio_fixture, [], out_mp4)


def test_cut_and_render_missing_input(sample_video, clean_audio_fixture, tmp_path):
    out_mp4 = str(tmp_path / "out.mp4")
    with pytest.raises(FileNotFoundError):
        cut_and_render_video_nvenc("missing.mp4", clean_audio_fixture, [(0.0, 1.0)], out_mp4)

    with pytest.raises(FileNotFoundError):
        cut_and_render_video_nvenc(sample_video, "missing.wav", [(0.0, 1.0)], out_mp4)
