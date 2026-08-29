"""
tests/test_audio_pipeline.py — VoiceVault Phase 1 Tests
========================================================
SIH26104

Tests for src/audio.py, src/preprocessing.py, src/features.py.

Run with:
  pytest tests/test_audio_pipeline.py -v

These tests use synthetic audio ONLY — no real dataset required.
All assertions test actual computed values, not hardcoded expected results.
"""

import io
import logging
from pathlib import Path

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

SR_16K = 16000
SR_44K = 44100


def make_sine_wave(freq_hz: float, duration_sec: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave as float32."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_stereo_wave(sr: int = SR_44K, duration: float = 2.0) -> np.ndarray:
    """Generate a 2-channel stereo waveform."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    ch1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ch2 = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    return np.stack([ch1, ch2], axis=1)  # (samples, 2)


@pytest.fixture
def mono_1s_16k() -> np.ndarray:
    """1-second mono sine wave at 16 kHz."""
    return make_sine_wave(440.0, 1.0, SR_16K)


@pytest.fixture
def stereo_2s_44k() -> np.ndarray:
    """2-second stereo waveform at 44.1 kHz."""
    return make_stereo_wave(SR_44K, 2.0)


@pytest.fixture
def tmp_wav(tmp_path) -> Path:
    """A real WAV file written to a temp directory for I/O tests."""
    import soundfile as sf
    wav_path = tmp_path / "test_sine.wav"
    waveform = make_sine_wave(440.0, 1.0, SR_16K)
    sf.write(str(wav_path), waveform, SR_16K, subtype="PCM_16")
    return wav_path


# ══════════════════════════════════════════════════════════
# Tests: src/audio.py
# ══════════════════════════════════════════════════════════

class TestAudio:
    def test_load_audio_returns_float32(self, tmp_wav):
        from src.audio import load_audio
        wv, sr = load_audio(tmp_wav)
        assert wv.dtype == np.float32
        assert sr == SR_16K

    def test_load_audio_shape_is_1d(self, tmp_wav):
        from src.audio import load_audio
        wv, _ = load_audio(tmp_wav)
        assert wv.ndim == 1, f"Expected 1D array, got shape {wv.shape}"

    def test_load_audio_amplitude_range(self, tmp_wav):
        from src.audio import load_audio
        wv, _ = load_audio(tmp_wav)
        assert wv.min() >= -1.01, "Amplitude below -1.0 (unexpected clipping)"
        assert wv.max() <= 1.01, "Amplitude above +1.0 (unexpected clipping)"

    def test_load_audio_missing_file_raises(self):
        from src.audio import load_audio
        with pytest.raises(FileNotFoundError):
            load_audio("does_not_exist.wav")

    def test_load_audio_unsupported_format_raises(self, tmp_path):
        from src.audio import load_audio
        bad_file = tmp_path / "audio.xyz"
        bad_file.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported"):
            load_audio(bad_file)

    def test_save_and_reload_roundtrip(self, tmp_path):
        from src.audio import save_audio, load_audio
        original = make_sine_wave(440.0, 0.5, SR_16K)
        out_path = tmp_path / "roundtrip.wav"
        save_audio(original, out_path, SR_16K)
        loaded, sr = load_audio(out_path)
        assert sr == SR_16K
        # Allow tiny floating-point error from PCM_16 quantization.
        assert np.allclose(original, loaded, atol=1e-3), "Roundtrip waveform mismatch"

    def test_load_from_bytes(self, tmp_wav):
        from src.audio import load_audio_from_bytes
        audio_bytes = tmp_wav.read_bytes()
        wv, sr = load_audio_from_bytes(audio_bytes, file_extension=".wav")
        assert wv.dtype == np.float32
        assert sr == SR_16K

    def test_get_audio_info(self, tmp_wav):
        from src.audio import get_audio_info
        info = get_audio_info(tmp_wav)
        assert info["sample_rate"] == SR_16K
        assert info["channels"] == 1
        assert abs(info["duration_sec"] - 1.0) < 0.01


# ══════════════════════════════════════════════════════════
# Tests: src/preprocessing.py
# ══════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_to_mono_from_stereo(self, stereo_2s_44k):
        from src.preprocessing import to_mono
        mono = to_mono(stereo_2s_44k)
        assert mono.ndim == 1, "Output must be 1D"
        assert mono.dtype == np.float32

    def test_to_mono_from_mono_is_noop(self, mono_1s_16k):
        from src.preprocessing import to_mono
        out = to_mono(mono_1s_16k)
        assert out.ndim == 1
        np.testing.assert_array_equal(out, mono_1s_16k)

    def test_resample_length_correct(self, mono_1s_16k):
        from src.preprocessing import resample
        # Upsample 16000 → 44100: expect ~2.756× more samples.
        upsampled = resample(mono_1s_16k, 16000, 44100)
        expected = int(len(mono_1s_16k) * 44100 / 16000)
        assert abs(len(upsampled) - expected) <= 5, (
            f"Resample length error: got {len(upsampled)}, expected ~{expected}"
        )

    def test_resample_noop_same_sr(self, mono_1s_16k):
        from src.preprocessing import resample
        out = resample(mono_1s_16k, 16000, 16000)
        np.testing.assert_array_almost_equal(out, mono_1s_16k, decimal=5)

    def test_normalize_peak_max_is_one(self, mono_1s_16k):
        from src.preprocessing import normalize
        # Scale down the waveform.
        quiet = (mono_1s_16k * 0.1).astype(np.float32)
        normalized = normalize(quiet, method="peak")
        assert abs(np.abs(normalized).max() - 1.0) < 1e-5, "Peak not at 1.0 after normalization"

    def test_normalize_silent_signal_no_crash(self):
        from src.preprocessing import normalize
        silent = np.zeros(16000, dtype=np.float32)
        out = normalize(silent, method="peak")
        assert out is not None

    def test_normalize_rms_clips_to_one(self, mono_1s_16k):
        from src.preprocessing import normalize
        loud = (mono_1s_16k * 5.0).astype(np.float32)
        normalized = normalize(loud, method="rms")
        assert normalized.max() <= 1.0 + 1e-6

    def test_chunk_audio_count_correct(self, mono_1s_16k):
        from src.preprocessing import chunk_audio
        # 1-second audio, 0.5s chunks, 0.25s hop → many overlapping chunks.
        chunks = chunk_audio(mono_1s_16k, 16000, chunk_sec=0.5, hop_sec=0.25)
        assert len(chunks) >= 3

    def test_chunk_audio_each_chunk_correct_length(self, mono_1s_16k):
        from src.preprocessing import chunk_audio
        chunks = chunk_audio(mono_1s_16k, 16000, chunk_sec=0.5, hop_sec=0.25)
        expected_len = int(0.5 * 16000)
        for i, ch in enumerate(chunks):
            assert len(ch) == expected_len, f"Chunk {i} wrong length: {len(ch)}"

    def test_chunk_audio_hop_gt_chunk_raises(self, mono_1s_16k):
        from src.preprocessing import chunk_audio
        with pytest.raises(ValueError, match="hop_sec"):
            chunk_audio(mono_1s_16k, 16000, chunk_sec=1.0, hop_sec=2.0)

    def test_vad_fallback_returns_waveform(self):
        """Energy VAD fallback should always return a waveform."""
        from src.preprocessing import _apply_energy_vad
        waveform = make_sine_wave(440, 2.0, 16000, amplitude=0.5)
        out, segs = _apply_energy_vad(waveform, 16000, 20, return_segments=True)
        assert out is not None
        assert isinstance(out, np.ndarray)

    def test_full_preprocess_pipeline_output_shape(self, stereo_2s_44k):
        from src.preprocessing import preprocess
        # Stereo 44100 Hz → mono 16000 Hz.
        out, sr = preprocess(
            stereo_2s_44k,
            sample_rate=44100,
            target_sr=16000,
            apply_vad_flag=False,  # Skip VAD (no webrtcvad needed)
        )
        assert out.ndim == 1
        assert sr == 16000
        assert abs(np.abs(out).max() - 1.0) < 1e-4  # peak normalized


# ══════════════════════════════════════════════════════════
# Tests: src/features.py
# ══════════════════════════════════════════════════════════

class TestFeatures:

    def test_fft_detects_correct_frequency(self):
        """FFT peak should be at the input sine wave frequency."""
        from src.features import compute_fft
        freq = 1000.0
        waveform = make_sine_wave(freq, 1.0, SR_16K)
        freqs, mags = compute_fft(waveform, SR_16K)
        peak_freq = freqs[np.argmax(mags)]
        assert abs(peak_freq - freq) < 50, (
            f"FFT peak at {peak_freq:.1f} Hz, expected ~{freq} Hz"
        )

    def test_fft_output_shapes_consistent(self, mono_1s_16k):
        from src.features import compute_fft
        freqs, mags = compute_fft(mono_1s_16k, SR_16K)
        assert len(freqs) == len(mags), "Frequencies and magnitudes must have same length"
        assert freqs[0] == 0.0, "First frequency bin should be 0 Hz (DC)"
        assert freqs[-1] <= SR_16K / 2.0 + 1.0, "Last freq must be ≤ Nyquist"

    def test_stft_output_shape(self, mono_1s_16k):
        from src.features import compute_stft
        stft = compute_stft(mono_1s_16k, n_fft=1024, hop_length=256)
        n_bins = 1024 // 2 + 1  # = 513
        assert stft.shape[0] == n_bins, f"Expected {n_bins} freq bins, got {stft.shape[0]}"
        assert stft.ndim == 2, "STFT must be 2D (bins × frames)"

    def test_power_spectrogram_non_negative(self, mono_1s_16k):
        from src.features import compute_stft, compute_power_spectrogram
        stft = compute_stft(mono_1s_16k, n_fft=1024, hop_length=256)
        power = compute_power_spectrogram(stft)
        assert (power >= 0).all(), "Power spectrogram must be non-negative"

    def test_mel_filterbank_shape(self):
        from src.features import build_mel_filterbank
        fb = build_mel_filterbank(n_mels=80, n_fft=1024, sample_rate=SR_16K)
        assert fb.shape == (80, 513), f"Wrong filterbank shape: {fb.shape}"

    def test_mel_filterbank_non_negative(self):
        from src.features import build_mel_filterbank
        fb = build_mel_filterbank(n_mels=80, n_fft=1024, sample_rate=SR_16K)
        assert (fb >= 0).all(), "Mel filter bank must be non-negative"

    def test_mel_spectrogram_shape(self, mono_1s_16k):
        from src.features import compute_mel_spectrogram
        mel = compute_mel_spectrogram(mono_1s_16k, SR_16K, n_mels=80)
        assert mel.shape[0] == 80, f"Wrong n_mels: {mel.shape[0]}"
        assert mel.ndim == 2, "Mel spectrogram must be 2D"
        assert mel.dtype == np.float32

    def test_mel_spectrogram_range_is_db(self, mono_1s_16k):
        """Log-Mel spectrogram should have values in a reasonable dB range."""
        from src.features import compute_mel_spectrogram
        mel = compute_mel_spectrogram(mono_1s_16k, SR_16K, n_mels=80, top_db=80.0)
        assert mel.max() <= 5.0, "Max dB unexpectedly high"
        assert mel.min() >= -90.0, "Min dB unexpectedly low"

    def test_mel_for_cnn_shape(self, mono_1s_16k):
        from src.features import mel_spectrogram_for_cnn
        mel_cnn = mel_spectrogram_for_cnn(mono_1s_16k, SR_16K, fixed_time_frames=188)
        assert mel_cnn.shape == (1, 80, 188), (
            f"CNN input shape mismatch: {mel_cnn.shape} (expected (1, 80, 188))"
        )

    def test_mel_for_cnn_pads_short_audio(self):
        """Short audio (< 3s) should be zero-padded to fixed_time_frames."""
        from src.features import mel_spectrogram_for_cnn
        short_wave = make_sine_wave(440, 0.5, SR_16K)
        mel_cnn = mel_spectrogram_for_cnn(short_wave, SR_16K, fixed_time_frames=188)
        assert mel_cnn.shape == (1, 80, 188)

    def test_mel_for_cnn_truncates_long_audio(self):
        """Long audio (> 3s) should be truncated to fixed_time_frames."""
        from src.features import mel_spectrogram_for_cnn
        long_wave = make_sine_wave(440, 10.0, SR_16K)
        mel_cnn = mel_spectrogram_for_cnn(long_wave, SR_16K, fixed_time_frames=188)
        assert mel_cnn.shape == (1, 80, 188)

    def test_mfcc_shape_no_delta(self, mono_1s_16k):
        from src.features import compute_mfcc
        mfcc = compute_mfcc(mono_1s_16k, SR_16K, n_mfcc=40, delta=False, delta2=False)
        assert mfcc.shape[0] == 40, f"Expected 40 MFCCs, got {mfcc.shape[0]}"

    def test_mfcc_shape_with_delta(self, mono_1s_16k):
        from src.features import compute_mfcc
        mfcc = compute_mfcc(mono_1s_16k, SR_16K, n_mfcc=40, delta=True, delta2=True)
        assert mfcc.shape[0] == 120, f"Expected 120 (3×40) MFCCs, got {mfcc.shape[0]}"

    def test_hz_to_mel_and_back(self):
        """Mel ↔ Hz conversion must be invertible."""
        from src.features import hz_to_mel, mel_to_hz
        for f in [100, 500, 1000, 4000, 8000]:
            mel = hz_to_mel(f)
            f_back = mel_to_hz(mel)
            assert abs(f - f_back) < 0.01, f"Hz→Mel→Hz roundtrip failed at {f} Hz"
