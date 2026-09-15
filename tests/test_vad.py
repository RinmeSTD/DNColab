import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure project root is in sys.path when running pytest directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vad_engine import (
    pad_and_merge_segments,
    calculate_speech_stats,
    get_speech_timestamps,
    load_vad_model,
)


def test_pad_and_merge_segments_basic():
    # Raw speech intervals in seconds
    raw = [
        {"start": 1.0, "end": 2.0},
        {"start": 2.5, "end": 4.0},
        {"start": 7.0, "end": 8.0},
    ]
    total_duration = 10.0
    # With 0.25s padding and 0.8s min silence:
    # Segment 1 becomes [0.75, 2.25]
    # Segment 2 becomes [2.25, 4.25] -> overlaps with seg 1, merges into [0.75, 4.25]
    # Gap between 4.25 and 6.75 is 2.5s > 0.8s, kept as silence.
    # Segment 3 becomes [6.75, 8.25]
    merged = pad_and_merge_segments(raw, total_duration, padding_sec=0.25, min_silence_sec=0.8)
    assert len(merged) == 2
    assert pytest.approx(merged[0][0], 0.01) == 0.75
    assert pytest.approx(merged[0][1], 0.01) == 4.25
    assert pytest.approx(merged[1][0], 0.01) == 6.75
    assert pytest.approx(merged[1][1], 0.01) == 8.25


def test_pad_and_merge_clamp_boundaries():
    raw = [{"start": 0.1, "end": 9.9}]
    total_duration = 10.0
    # Padding should not exceed 0.0 on start or total_duration on end
    merged = pad_and_merge_segments(raw, total_duration, padding_sec=0.5, min_silence_sec=0.8)
    assert len(merged) == 1
    assert merged[0][0] == 0.0
    assert merged[0][1] == 10.0


def test_pad_and_merge_gap_merging():
    raw = [
        {"start": 1.0, "end": 2.0},
        {"start": 3.0, "end": 4.0},
    ]
    total_duration = 10.0
    # Padding 0.25s:
    # Seg 1: [0.75, 2.25]
    # Seg 2: [2.75, 4.25]
    # Gap is 2.75 - 2.25 = 0.5s <= 0.8s (min_silence_sec) -> should merge into [0.75, 4.25]
    merged = pad_and_merge_segments(raw, total_duration, padding_sec=0.25, min_silence_sec=0.8)
    assert len(merged) == 1
    assert pytest.approx(merged[0][0], 0.01) == 0.75
    assert pytest.approx(merged[0][1], 0.01) == 4.25


def test_pad_and_merge_empty():
    assert pad_and_merge_segments([], 10.0) == []


def test_calculate_speech_stats():
    segments = [(1.0, 4.0), (6.0, 9.0)]  # total kept = 6.0s
    stats = calculate_speech_stats(original_duration=10.0, kept_segments=segments)
    assert stats["original_duration"] == 10.0
    assert stats["kept_duration"] == 6.0
    assert stats["silence_removed"] == 4.0
    assert pytest.approx(stats["silence_percentage"], 0.1) == 40.0


def test_calculate_speech_stats_zero_duration():
    stats = calculate_speech_stats(original_duration=0.0, kept_segments=[])
    assert stats["original_duration"] == 0.0
    assert stats["kept_duration"] == 0.0
    assert stats["silence_removed"] == 0.0
    assert stats["silence_percentage"] == 0.0


def test_get_speech_timestamps_mock():
    mock_get_ts = MagicMock(return_value=[{"start": 1.0, "end": 2.0}])
    mock_utils = (mock_get_ts,)
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([])
    mock_tensor = MagicMock()
    mock_tensor.ndim = 2
    mock_tensor.mean.return_value = mock_tensor
    mock_tensor.squeeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor

    res = get_speech_timestamps(
        audio_tensor=mock_tensor,
        model=mock_model,
        utils=mock_utils,
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=800,
    )
    assert res == [{"start": 1.0, "end": 2.0}]
    mock_get_ts.assert_called_once_with(
        mock_tensor,
        mock_model,
        threshold=0.5,
        sampling_rate=16000,
        min_speech_duration_ms=250,
        min_silence_duration_ms=800,
        return_seconds=True,
    )


def test_get_speech_timestamps_device_alignment():
    mock_get_ts = MagicMock(return_value=[{"start": 0.5, "end": 1.5}])
    mock_utils = [mock_get_ts]

    # Model with CUDA device parameter
    mock_param = MagicMock()
    mock_param.device = "cuda:0"
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([mock_param])

    mock_tensor = MagicMock()
    mock_tensor.ndim = 1
    mock_tensor_cuda = MagicMock()
    mock_tensor_cuda.ndim = 1
    mock_tensor.to.return_value = mock_tensor_cuda
    mock_tensor_cuda.squeeze.return_value = mock_tensor_cuda

    res = get_speech_timestamps(
        audio_tensor=mock_tensor,
        model=mock_model,
        utils=mock_utils,
    )
    assert res == [{"start": 0.5, "end": 1.5}]
    mock_tensor.to.assert_called_once_with("cuda:0")
    mock_get_ts.assert_called_once_with(
        mock_tensor_cuda,
        mock_model,
        threshold=0.5,
        sampling_rate=16000,
        min_speech_duration_ms=250,
        min_silence_duration_ms=800,
        return_seconds=True,
    )


def test_load_vad_model_missing_torch():
    import vad_engine
    if vad_engine.torch is None:
        with pytest.raises(ImportError, match="PyTorch is required"):
            vad_engine.load_vad_model()
    else:
        vad_engine.torch.hub.load = MagicMock(return_value=(MagicMock(), MagicMock()))
        model, utils = vad_engine.load_vad_model()
        assert model is not None
        assert utils is not None

