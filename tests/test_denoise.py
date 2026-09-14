"""
tests/test_denoise.py - Tests for AI Denoise, Audio Crossfading & Loudness Normalization.
"""
import os
import sys
import tempfile
import warnings
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when running pytest directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from denoise_engine import (
    cut_and_crossfade_audio,
    denoise_audio_deepfilter,
    denoise_audio_resemble,
    normalize_loudness,
)


def test_cut_and_crossfade_audio_shape():
    """Verify accurate output sample length, shape, crossfade calculation, and no NaN/Inf."""
    sr = 16000
    # 5 seconds of test audio
    t = np.linspace(0, 5, 5 * sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # 2 segments: 1.0s each (16000 samples each)
    segments = [(1.0, 2.0), (3.0, 4.0)]
    crossfade_ms = 30  # 480 samples
    fade_samples = int(round(sr * crossfade_ms / 1000.0))

    out = cut_and_crossfade_audio(audio, sr, segments, crossfade_ms=crossfade_ms)

    expected_len = (16000 + 16000) - fade_samples
    assert out.ndim == 1
    assert out.shape[0] == expected_len
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_cut_and_crossfade_stereo_shape():
    """Verify 2D stereo (channels, samples) shape handling and crossfading."""
    sr = 16000
    channels = 2
    audio = np.random.uniform(-0.5, 0.5, (channels, 5 * sr)).astype(np.float32)

    # 3 segments of 1.0s each
    segments = [(0.5, 1.5), (2.0, 3.0), (3.5, 4.5)]
    crossfade_ms = 20  # 320 samples
    fade_samples = int(round(sr * crossfade_ms / 1000.0))

    out = cut_and_crossfade_audio(audio, sr, segments, crossfade_ms=crossfade_ms)

    # 3 slices of 16000 samples, 2 crossfades of 320 samples
    expected_len = (3 * 16000) - (2 * fade_samples)
    assert out.ndim == 2
    assert out.shape == (channels, expected_len)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_cut_and_crossfade_integer_audio():
    """Verify integer audio (int16 and int32) crossfades without zero-dropout truncation."""
    sr = 16000
    crossfade_ms = 30
    fade_samples = int(round(sr * crossfade_ms / 1000.0))  # 480 samples

    for dtype in [np.int16, np.int32]:
        # Constant non-zero signal (e.g., 10000)
        constant_val = 10000
        audio_const = np.full(5 * sr, constant_val, dtype=dtype)
        segments = [(1.0, 2.0), (3.0, 4.0)]

        out_const = cut_and_crossfade_audio(audio_const, sr, segments, crossfade_ms=crossfade_ms)

        # Output dtype must remain identical
        assert out_const.dtype == dtype
        # Output length verified
        expected_len = (16000 + 16000) - fade_samples
        assert out_const.shape[0] == expected_len

        # The crossfade region is at index [16000 - fade_samples : 16000]
        overlap_region = out_const[16000 - fade_samples : 16000]
        # Under integer truncation bug, overlap would be zero!
        # With float computation, overlap must stay equal to constant_val
        assert np.all(overlap_region == constant_val), (
            f"Expected constant {constant_val} in crossfade overlap for {dtype}, got dropouts/zeros"
        )

        # Also test alternating non-zero tone signal to verify no zero dropouts
        t = np.linspace(0, 5, 5 * sr, endpoint=False)
        audio_tone = (15000 * np.sin(2 * np.pi * 440 * t)).astype(dtype)
        out_tone = cut_and_crossfade_audio(audio_tone, sr, segments, crossfade_ms=crossfade_ms)
        assert out_tone.dtype == dtype
        overlap_tone = out_tone[16000 - fade_samples : 16000]
        # Should not be all zeros
        assert np.count_nonzero(overlap_tone) > 0.8 * fade_samples


def test_cut_and_crossfade_empty_and_single():
    """Verify edge cases: empty segments, single segment, and out-of-bound segments."""
    sr = 16000
    audio = np.ones(5 * sr, dtype=np.float32)

    # Empty segments list
    empty_out = cut_and_crossfade_audio(audio, sr, [])
    assert empty_out.ndim == 1
    assert empty_out.shape[0] == 0

    # Empty segments on 2D audio
    empty_stereo = cut_and_crossfade_audio(np.ones((2, 5 * sr), dtype=np.float32), sr, [])
    assert empty_stereo.shape == (2, 0)

    # Single segment
    single_out = cut_and_crossfade_audio(audio, sr, [(1.0, 2.5)], crossfade_ms=30)
    expected_single_len = int(round(1.5 * sr))
    assert single_out.shape[0] == expected_single_len

    # Out of bounds clamping (start < 0, end > duration)
    oob_out = cut_and_crossfade_audio(audio, sr, [(-1.0, 0.5), (4.5, 6.0)], crossfade_ms=20)
    # (-1.0, 0.5) clamps to (0.0, 0.5) -> 8000 samples
    # (4.5, 6.0) clamps to (4.5, 5.0) -> 8000 samples
    fade_samples = int(round(sr * 20 / 1000.0))
    assert oob_out.shape[0] == (8000 + 8000) - fade_samples

    # Invalid segments (end <= start or outside audio)
    invalid_out = cut_and_crossfade_audio(audio, sr, [(3.0, 2.0), (10.0, 12.0)])
    assert invalid_out.shape[0] == 0


def test_normalize_loudness_peak():
    """Verify quiet signal is amplified towards target LUFS and peak amplitude stays <= 1.0 (<= 0.95)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Quiet 440 Hz tone (-44 LUFS approx)
    quiet_signal = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    norm_audio = normalize_loudness(quiet_signal, sr, target_lufs=-14.0)

    # Amplified quiet signal
    assert np.max(np.abs(norm_audio)) > np.max(np.abs(quiet_signal))
    # True peak limiting ensures peak stays <= 0.95 (<= 1.0)
    assert np.max(np.abs(norm_audio)) <= 0.95
    assert not np.isnan(norm_audio).any()
    assert not np.isinf(norm_audio).any()


def test_normalize_loudness_transient_limiting():
    """Verify high-amplitude transient signal is limited to <= 0.95 true peak."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Signal with sudden large burst
    signal = (0.005 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    signal[400:500] = 0.9  # extreme transient

    norm_audio = normalize_loudness(signal, sr, target_lufs=-10.0)

    assert np.max(np.abs(norm_audio)) <= 0.95
    assert not np.isnan(norm_audio).any()


def test_normalize_loudness_stereo():
    """Verify stereo 2D audio normalization preserves shape."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    ch1 = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ch2 = (0.02 * np.cos(2 * np.pi * 880 * t)).astype(np.float32)
    stereo = np.stack([ch1, ch2], axis=0)  # (2, 16000)

    norm = normalize_loudness(stereo, sr, target_lufs=-14.0)
    assert norm.shape == (2, sr)
    assert np.max(np.abs(norm)) <= 0.95
    assert not np.isnan(norm).any()


def test_normalize_loudness_silence():
    """Verify silent audio does not result in NaN or Inf."""
    sr = 16000
    silent = np.zeros(sr, dtype=np.float32)

    norm = normalize_loudness(silent, sr, target_lufs=-14.0)
    assert norm.shape == (sr,)
    assert not np.isnan(norm).any()
    assert np.all(norm == 0.0)


def test_denoise_audio_deepfilter_fallback():
    """Verify deepfilter falls back to file copy with warning when df is unavailable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")

        with open(input_wav, "wb") as f:
            f.write(b"RIFFdummywavdata")

        # Mock shutil.which so df-enhance CLI is also not found
        with patch("shutil.which", return_value=None):
            with pytest.warns(UserWarning, match="DeepFilterNet"):
                res = denoise_audio_deepfilter(input_wav, output_wav)

        assert res == output_wav
        assert os.path.exists(output_wav)
        with open(output_wav, "rb") as f:
            assert f.read() == b"RIFFdummywavdata"


def test_denoise_audio_deepfilter_mock_python():
    """Verify deepfilter uses df.enhance when module is available."""
    mock_df_enhance = MagicMock()
    mock_df_pkg = MagicMock()
    mock_df_pkg.enhance = mock_df_enhance

    mock_model = MagicMock()
    mock_df_state = MagicMock()
    mock_df_state.sr.return_value = 48000
    mock_df_enhance.init_df.return_value = (mock_model, mock_df_state, None)
    mock_df_enhance.load_audio.return_value = (np.zeros((1, 48000)), 48000)
    mock_df_enhance.enhance.return_value = np.zeros((1, 48000))

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")
        with open(input_wav, "wb") as f:
            f.write(b"RIFFdata")

        with patch.dict(sys.modules, {"df": mock_df_pkg, "df.enhance": mock_df_enhance}):
            res = denoise_audio_deepfilter(input_wav, output_wav)

        assert res == output_wav
        mock_df_enhance.init_df.assert_called_once()
        mock_df_enhance.enhance.assert_called_once()
        mock_df_enhance.save_audio.assert_called_once()


def test_denoise_audio_deepfilter_cli_fallback():
    """Verify deepfilter falls back to df-enhance CLI when CLI tool is found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")
        with open(input_wav, "wb") as f:
            f.write(b"RIFFdata")

        def fake_run(cmd, *args, **kwargs):
            # simulate CLI creating output
            with open(output_wav, "wb") as f:
                f.write(b"ENHANCED_CLI_DATA")
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res

        # df not installed, but df-enhance in PATH
        with patch.dict(sys.modules, {"df": None, "df.enhance": None}):
            with patch("shutil.which", return_value="/usr/bin/df-enhance"):
                with patch("subprocess.run", side_effect=fake_run) as mock_run:
                    res = denoise_audio_deepfilter(input_wav, output_wav)

        assert res == output_wav
        assert os.path.exists(output_wav)
        with open(output_wav, "rb") as f:
            assert f.read() == b"ENHANCED_CLI_DATA"


def test_denoise_audio_resemble_fallback():
    """Verify resemble falls back to file copy with warning when resemble_enhance is unavailable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")

        with open(input_wav, "wb") as f:
            f.write(b"RIFFdummywavdata")

        with pytest.warns(UserWarning, match="resemble_enhance"):
            res = denoise_audio_resemble(input_wav, output_wav)

        assert res == output_wav
        assert os.path.exists(output_wav)
        with open(output_wav, "rb") as f:
            assert f.read() == b"RIFFdummywavdata"


def test_denoise_audio_resemble_mock():
    """Verify resemble handles 1D mono conversion and 2D tensor output for torchaudio.save."""
    mock_inference = MagicMock()
    mock_resemble = MagicMock()
    mock_resemble.enhancer.inference = mock_inference

    # Simulate 1D enhanced tensor output from resemble_enhance
    enhanced_1d = MagicMock()
    enhanced_1d.ndim = 1
    unsqueezed_2d = MagicMock()
    unsqueezed_2d.ndim = 2
    enhanced_1d.unsqueeze.return_value = unsqueezed_2d

    mock_inference.enhance.return_value = (enhanced_1d, 44100)

    # Simulate torchaudio.load returning stereo 2D waveform (2, 44100)
    stereo_wave = MagicMock()
    stereo_wave.ndim = 2
    stereo_wave.shape = (2, 44100)
    mono_wave = MagicMock()
    mono_wave.ndim = 1
    stereo_wave.mean.return_value = mono_wave

    mock_torchaudio = MagicMock()
    mock_torchaudio.load.return_value = (stereo_wave, 44100)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")
        with open(input_wav, "wb") as f:
            f.write(b"RIFFdata")

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "torchaudio": mock_torchaudio,
                "resemble_enhance": mock_resemble,
                "resemble_enhance.enhancer": mock_resemble.enhancer,
                "resemble_enhance.enhancer.inference": mock_inference,
            },
        ):
            res = denoise_audio_resemble(input_wav, output_wav)

        assert res == output_wav
        # Verify stereo was averaged to mono
        stereo_wave.mean.assert_called_once_with(dim=0)
        # Verify enhance was called with mono wave
        mock_inference.enhance.assert_called_once()
        assert mock_inference.enhance.call_args[0][0] == mono_wave
        # Verify torchaudio.save received the unsqueezed 2D tensor
        mock_torchaudio.save.assert_called_once_with(output_wav, unsqueezed_2d, 44100)


def test_denoise_in_place_copy_protection():
    """Verify in-place copy (input_wav == output_wav) avoids SameFileError on fallback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_wav = os.path.join(tmp_dir, "audio.wav")
        with open(target_wav, "wb") as f:
            f.write(b"SAMPLEAUDIO")

        with patch("shutil.which", return_value=None):
            with pytest.warns(UserWarning):
                res_df = denoise_audio_deepfilter(target_wav, target_wav)
            assert res_df == target_wav

        with pytest.warns(UserWarning):
            res_resemble = denoise_audio_resemble(target_wav, target_wav)
        assert res_resemble == target_wav
