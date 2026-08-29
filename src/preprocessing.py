"""
src/preprocessing.py — VoiceVault Audio Preprocessing Module
=============================================================
SIH26104 | Phase 1

PURPOSE
-------
Transforms raw audio (from src/audio.py) into a clean, standardized
signal ready for feature extraction (src/features.py) and ML inference.

PIPELINE ORDER (match the spec exactly)
----------------------------------------
  Raw waveform
      ↓  1. to_mono()         — collapse stereo/multichannel → mono
      ↓  2. resample()        — standardize to target_sr (16 kHz)
      ↓  3. normalize()       — peak-normalize to [-1.0, +1.0]
      ↓  4. apply_vad()       — detect speech regions, remove silence
      ↓  5. chunk_audio()     — split into fixed-length chunks (streaming)

WHY EACH STEP EXISTS (SIH judge explanation)
--------------------------------------------
1. MONO: ML models expect a single channel. Stereo audio has 2 channels;
   mixing them down avoids information asymmetry and halves memory.

2. RESAMPLE: ASVspoof 2019 LA is at 16 kHz. All our models are trained
   at 16 kHz. Audio from phones (8 kHz narrowband), voice memos (44.1 kHz),
   or VoIP must be standardized before feature extraction.

3. NORMALIZE: Raw recordings vary widely in loudness. A whispered sentence
   and a shouted sentence should yield the same spectral shape relative to
   loudness. Peak normalization ensures the waveform fills [-1, 1] without
   clipping.

4. VAD (Voice Activity Detection): In a real call, there are periods of
   silence, background noise, and breathing. VAD lets the system focus
   analysis on frames that actually contain speech, reducing false
   positives from silence and improving model accuracy.

5. CHUNKING: Near-real-time processing requires splitting a long stream
   into fixed windows. The chunk size and hop size determine the update
   frequency of the risk score.

SIH JUDGE QUESTIONS (anticipate these)
---------------------------------------
Q: Why 16 kHz specifically?
A: Human speech energy is concentrated below 8 kHz. At 16 kHz, Nyquist
   = 8 kHz, capturing all phonetically relevant information. Higher rates
   waste compute; lower rates (e.g., 8 kHz) lose important consonant
   information (fricatives, sibilants) that anti-spoofing models rely on.

Q: What is VAD?
A: Voice Activity Detection is the process of classifying each short frame
   of audio as either speech or non-speech. WebRTC VAD uses a GMM-based
   statistical model trained on thousands of hours of speech and noise.

Q: Why use WebRTC VAD instead of energy thresholding?
A: Energy thresholding fails in noisy environments — loud background noise
   can exceed the threshold. WebRTC VAD is robust to stationary noise
   because it models the spectral shape of speech, not just energy.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Type aliases (consistent with audio.py)
# ──────────────────────────────────────────────────────────
Waveform = np.ndarray
SampleRate = int


# ══════════════════════════════════════════════════════════
# 1. MONO CONVERSION
# ══════════════════════════════════════════════════════════

def to_mono(waveform: Waveform) -> Waveform:
    """
    Convert a multi-channel waveform to mono by averaging channels.

    WHAT IT DOES
    ------------
    If waveform has shape (samples, channels), average across channels
    to produce shape (samples,).  If already mono (shape (samples,)),
    return as-is.

    WHY NOT TAKE ONLY THE LEFT CHANNEL?
    ------------------------------------
    Averaging preserves signal energy from both channels. Taking only
    the left channel can miss speaker energy panned to the right (e.g.,
    in a recorded phone call where the caller is on the right channel).

    INPUT  : (samples,) or (samples, channels) float32
    OUTPUT : (samples,) float32 mono waveform
    """
    if waveform.ndim == 1:
        # Already mono.
        return waveform.copy()

    if waveform.ndim == 2:
        # Shape: (samples, channels)
        mono = waveform.mean(axis=1)
        logger.debug("to_mono: %s → %s", waveform.shape, mono.shape)
        return mono.astype(np.float32)

    raise ValueError(
        f"Unexpected waveform shape: {waveform.shape}. "
        "Expected 1D (samples,) or 2D (samples, channels)."
    )


# ══════════════════════════════════════════════════════════
# 2. RESAMPLING
# ══════════════════════════════════════════════════════════

def resample(
    waveform: Waveform,
    orig_sr: SampleRate,
    target_sr: SampleRate,
) -> Waveform:
    """
    Resample a mono waveform from orig_sr to target_sr.

    WHAT IT DOES
    ------------
    Uses torchaudio.functional.resample (sinc interpolation) for high
    quality resampling. Falls back to scipy.signal.resample_poly if
    torchaudio is not yet installed (useful during environment setup).

    WHY SINC INTERPOLATION?
    -----------------------
    Sinc (ideal low-pass) resampling in the frequency domain is the
    theoretically correct method (Shannon-Nyquist). It avoids aliasing
    (false frequencies introduced by naive decimation) and avoids
    spectral leakage artifacts.

    WHY NOT librosa.resample()?
    ---------------------------
    torchaudio.functional.resample is significantly faster on GPU/CPU
    with optimized kernels. librosa wraps scipy, which is slower for
    large batches.

    INPUT
    -----
    waveform  : mono float32 numpy array, shape (samples,)
    orig_sr   : original sample rate in Hz
    target_sr : target sample rate in Hz

    OUTPUT
    ------
    resampled float32 numpy array, shape (new_samples,)
    where new_samples = samples * target_sr / orig_sr
    """
    if orig_sr == target_sr:
        logger.debug("resample: already at %d Hz, skipping.", target_sr)
        return waveform.copy()

    logger.debug("resample: %d Hz → %d Hz", orig_sr, target_sr)

    try:
        import torch
        import torchaudio.functional as F  # type: ignore

        # torchaudio expects a float32 tensor of shape (num_channels, num_samples).
        # Our mono waveform is (samples,), so we unsqueeze to (1, samples).
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampled_tensor = F.resample(tensor, orig_sr, target_sr)
        return resampled_tensor.squeeze(0).numpy()

    except ImportError:
        # torchaudio not yet installed — use scipy fallback.
        logger.warning(
            "torchaudio not found. Using scipy.signal.resample_poly fallback."
        )
        from scipy.signal import resample_poly  # type: ignore
        from math import gcd

        g = gcd(target_sr, orig_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = resample_poly(waveform, up, down)
        return resampled.astype(np.float32)


# ══════════════════════════════════════════════════════════
# 3. NORMALIZATION
# ══════════════════════════════════════════════════════════

def normalize(waveform: Waveform, method: str = "peak") -> Waveform:
    """
    Normalize waveform amplitude.

    WHAT IT DOES
    ------------
    Scales the waveform so that the loudest sample has magnitude 1.0.
    This is called peak normalization.

    WHY NORMALIZE?
    --------------
    ML models are sensitive to input scale. A waveform recorded at low
    volume will produce a Mel spectrogram with very small values, which
    shifts the distribution away from what the model was trained on.
    Normalizing removes this recording-level variation.

    WHY NOT RMS NORMALIZATION?
    --------------------------
    Peak normalization is simpler and always produces a [-1.0, 1.0]
    range, which matches soundfile's float32 convention.  RMS
    normalization (based on average energy) can sometimes clip if the
    audio contains transient peaks. We support both; peak is the default.

    INPUT
    -----
    waveform : mono float32 numpy array
    method   : 'peak' (default) or 'rms'

    OUTPUT
    ------
    normalized float32 numpy array, values in [-1.0, 1.0]
    """
    if method == "peak":
        peak = np.abs(waveform).max()
        if peak < 1e-8:
            # Silent signal — avoid division by zero.
            logger.warning("normalize: near-silent waveform (peak < 1e-8). Returning as-is.")
            return waveform.copy()
        return (waveform / peak).astype(np.float32)

    elif method == "rms":
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms < 1e-8:
            logger.warning("normalize: near-silent waveform (RMS < 1e-8). Returning as-is.")
            return waveform.copy()
        normalized = waveform / rms
        # Clip to avoid clipping after RMS normalization.
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'peak' or 'rms'.")


# ══════════════════════════════════════════════════════════
# 4. VOICE ACTIVITY DETECTION (VAD)
# ══════════════════════════════════════════════════════════

def apply_vad(
    waveform: Waveform,
    sample_rate: SampleRate,
    aggressiveness: int = 2,
    frame_duration_ms: int = 20,
    return_segments: bool = False,
) -> Tuple[Waveform, Optional[List[Tuple[float, float]]]]:
    """
    Remove non-speech regions using WebRTC VAD.

    WHAT IT DOES
    ------------
    Splits the audio into frames (10/20/30 ms), classifies each frame
    as speech or non-speech using a GMM-based model, and concatenates
    only the speech frames. Optionally returns the time segments of
    detected speech.

    WHY VOICEVAULT NEEDS VAD
    ------------------------
    1. Anti-spoofing models perform better on pure speech frames.
      Silence + noise confuse spectral analysis.
    2. Attacker dossiers often contain silence padding at the start/end
      of cloned audio.  VAD helps detect unusual silence distributions.
    3. For near-real-time streaming, processing only speech frames
      reduces inference load.

    INPUT
    -----
    waveform          : mono float32 numpy array at `sample_rate`
    sample_rate       : must be 8000, 16000, 32000, or 48000 Hz
                        (WebRTC VAD constraint)
    aggressiveness    : 0 (keep more audio) to 3 (keep less audio)
    frame_duration_ms : 10, 20, or 30 ms (WebRTC constraint)
    return_segments   : if True, also return [(start_sec, end_sec), ...]

    OUTPUT
    ------
    speech_waveform    : float32 array containing only detected speech
    segments           : list of (start_sec, end_sec) tuples if
                         return_segments=True, else None

    FALLBACK BEHAVIOUR
    ------------------
    If webrtcvad is not installed, falls back to simple energy-based
    VAD (threshold on RMS per frame). This is less robust but prevents
    the pipeline from breaking during environment setup.
    """
    # WebRTC VAD only supports specific sample rates.
    vad_supported_rates = {8000, 16000, 32000, 48000}
    if sample_rate not in vad_supported_rates:
        raise ValueError(
            f"WebRTC VAD requires sample rate in {vad_supported_rates}. "
            f"Got: {sample_rate}. Resample first."
        )

    if aggressiveness not in {0, 1, 2, 3}:
        raise ValueError(f"VAD aggressiveness must be 0–3. Got: {aggressiveness}.")

    if frame_duration_ms not in {10, 20, 30}:
        raise ValueError(
            f"VAD frame_duration_ms must be 10, 20, or 30. Got: {frame_duration_ms}."
        )

    try:
        import webrtcvad  # type: ignore
        return _apply_webrtcvad(
            waveform, sample_rate, aggressiveness, frame_duration_ms, return_segments
        )

    except ImportError:
        logger.warning(
            "webrtcvad not installed. Falling back to energy-based VAD. "
            "On Windows (no C++ Build Tools): pip install webrtcvad-wheels\n"
            "On Linux/macOS: pip install webrtcvad"
        )
        return _apply_energy_vad(waveform, sample_rate, frame_duration_ms, return_segments)


def _apply_webrtcvad(
    waveform: Waveform,
    sample_rate: int,
    aggressiveness: int,
    frame_duration_ms: int,
    return_segments: bool,
) -> Tuple[Waveform, Optional[List[Tuple[float, float]]]]:
    """Internal: WebRTC VAD implementation."""
    import webrtcvad  # type: ignore

    vad = webrtcvad.Vad(aggressiveness)
    frame_length = int(sample_rate * frame_duration_ms / 1000)  # samples per frame

    # Convert float32 → int16 PCM (WebRTC VAD requires int16 bytes).
    waveform_int16 = (waveform * 32767).clip(-32768, 32767).astype(np.int16)

    speech_frames: List[np.ndarray] = []
    segments: List[Tuple[float, float]] = []
    in_speech = False
    speech_start = 0.0

    num_frames = len(waveform_int16) // frame_length

    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frame_bytes = waveform_int16[start:end].tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if is_speech:
            speech_frames.append(waveform[start:end])
            if not in_speech:
                in_speech = True
                speech_start = start / sample_rate
        else:
            if in_speech:
                in_speech = False
                segments.append((speech_start, start / sample_rate))

    if in_speech:
        # Close the final segment.
        segments.append((speech_start, len(waveform) / sample_rate))

    if not speech_frames:
        logger.warning("VAD found no speech frames. Returning original waveform.")
        return waveform.copy(), (segments if return_segments else None)

    speech_waveform = np.concatenate(speech_frames, axis=0).astype(np.float32)
    logger.debug(
        "VAD: kept %.1f%% of audio (%d speech frames of %d total frames)",
        100 * len(speech_frames) / max(num_frames, 1),
        len(speech_frames),
        num_frames,
    )
    return speech_waveform, (segments if return_segments else None)


def _apply_energy_vad(
    waveform: Waveform,
    sample_rate: int,
    frame_duration_ms: int,
    return_segments: bool,
) -> Tuple[Waveform, Optional[List[Tuple[float, float]]]]:
    """
    Fallback energy-based VAD.

    Classifies a frame as speech if its RMS energy exceeds a threshold
    derived from the overall signal RMS. Simple but reasonably effective
    in quiet environments.
    """
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    overall_rms = np.sqrt(np.mean(waveform ** 2))
    # Threshold: 10% of overall RMS. Empirically reasonable default.
    threshold = 0.1 * overall_rms

    speech_frames: List[np.ndarray] = []
    segments: List[Tuple[float, float]] = []
    in_speech = False
    speech_start = 0.0

    num_frames = len(waveform) // frame_length

    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frame = waveform[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        is_speech = rms > threshold

        if is_speech:
            speech_frames.append(frame)
            if not in_speech:
                in_speech = True
                speech_start = start / sample_rate
        else:
            if in_speech:
                in_speech = False
                segments.append((speech_start, start / sample_rate))

    if in_speech:
        segments.append((speech_start, len(waveform) / sample_rate))

    if not speech_frames:
        logger.warning("Energy VAD found no speech. Returning original waveform.")
        return waveform.copy(), (segments if return_segments else None)

    speech_waveform = np.concatenate(speech_frames).astype(np.float32)
    return speech_waveform, (segments if return_segments else None)


# ══════════════════════════════════════════════════════════
# 5. AUDIO CHUNKING (for streaming / near-real-time)
# ══════════════════════════════════════════════════════════

def chunk_audio(
    waveform: Waveform,
    sample_rate: SampleRate,
    chunk_sec: float = 3.0,
    hop_sec: float = 1.0,
    pad_last: bool = True,
) -> List[Waveform]:
    """
    Split waveform into overlapping fixed-length chunks.

    WHAT IT DOES
    ------------
    Divides a long waveform into a sequence of fixed-length windows.
    Consecutive windows overlap by (chunk_sec - hop_sec) seconds.
    This overlapping is essential for streaming: a chunk at time T
    includes context from time T - (chunk_sec - hop_sec).

    WHY OVERLAPPING CHUNKS?
    -----------------------
    Audio events (e.g., an artifact in synthetic speech) may straddle
    a chunk boundary. Without overlap, we might miss boundary events.
    Overlap guarantees every sample is analyzed in at least one complete
    chunk. The tradeoff is redundant computation (acceptable at our scale).

    WHY CHUNKING FOR VOICEVAULT?
    ----------------------------
    The dynamic risk score (Section 25 of spec) updates every few seconds.
    Chunking is the mechanism that drives this update cadence:
      chunk 0: risk 10
      chunk 1: risk 43
      chunk 2: risk 89 → ALERT

    INPUT
    -----
    waveform   : mono float32 numpy array at `sample_rate`
    sample_rate: integer Hz
    chunk_sec  : length of each chunk in seconds (default: 3.0)
    hop_sec    : step between chunk starts in seconds (default: 1.0)
    pad_last   : if True, zero-pad the final chunk to full length

    OUTPUT
    ------
    List of float32 arrays, each of length chunk_length = chunk_sec * sample_rate

    EXAMPLE
    -------
    10-second audio, chunk_sec=3.0, hop_sec=1.0:
      Chunk 0: 0–3 s
      Chunk 1: 1–4 s
      Chunk 2: 2–5 s
      ...
      Chunk 7: 7–10 s (padded if < 3 s)
    """
    chunk_length = int(chunk_sec * sample_rate)
    hop_length = int(hop_sec * sample_rate)

    if chunk_length <= 0 or hop_length <= 0:
        raise ValueError(
            f"chunk_sec ({chunk_sec}) and hop_sec ({hop_sec}) must be > 0."
        )
    if hop_length > chunk_length:
        raise ValueError(
            f"hop_sec ({hop_sec}) must be ≤ chunk_sec ({chunk_sec})."
        )

    chunks: List[Waveform] = []
    start = 0

    while start < len(waveform):
        end = start + chunk_length
        chunk = waveform[start:end]

        if len(chunk) < chunk_length:
            if pad_last:
                # Zero-pad to full chunk length.
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode="constant")
            else:
                # Drop incomplete last chunk.
                break

        chunks.append(chunk.astype(np.float32))
        start += hop_length

    logger.debug(
        "chunk_audio: %d chunks | chunk_sec=%.1f | hop_sec=%.1f",
        len(chunks),
        chunk_sec,
        hop_sec,
    )
    return chunks


# ══════════════════════════════════════════════════════════
# CONVENIENCE: FULL PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════

def preprocess(
    waveform: Waveform,
    sample_rate: SampleRate,
    target_sr: int = 16000,
    normalize_method: str = "peak",
    vad_aggressiveness: int = 2,
    apply_vad_flag: bool = True,
) -> Tuple[Waveform, SampleRate]:
    """
    Run the full preprocessing pipeline on a raw waveform.

    Pipeline:
      to_mono → resample → normalize → [optional VAD]

    INPUT
    -----
    waveform            : raw numpy float32 array (any channels, any sr)
    sample_rate         : original sample rate
    target_sr           : target sample rate (default: 16000 Hz)
    normalize_method    : 'peak' or 'rms'
    vad_aggressiveness  : WebRTC VAD aggressiveness 0–3
    apply_vad_flag      : set False to skip VAD (e.g., for training data
                          that is already segmented like ASVspoof 2019)

    OUTPUT
    ------
    preprocessed waveform, target_sr
    """
    # Step 1: Mono
    waveform = to_mono(waveform)

    # Step 2: Resample
    waveform = resample(waveform, sample_rate, target_sr)

    # Step 3: Normalize
    waveform = normalize(waveform, method=normalize_method)

    # Step 4: VAD (optional — skip for pre-segmented training data)
    if apply_vad_flag:
        waveform, _ = apply_vad(
            waveform, target_sr, aggressiveness=vad_aggressiveness
        )

    return waveform, target_sr


# ──────────────────────────────────────────────────────────
# Smoke test — run directly: python -m src.preprocessing
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("VoiceVault -- preprocessing.py self-test")
    print("=" * 60)

    # Create a synthetic test waveform:
    # 3 seconds at 44100 Hz, stereo (simulating a common recording format).
    sr_orig = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)

    # Two channels: 440 Hz sine (L) + 880 Hz sine (R)
    ch1 = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch2 = 0.3 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo = np.stack([ch1, ch2], axis=1)  # shape: (samples, 2)

    print(f"\n[INPUT]")
    print(f"  Shape     : {stereo.shape}  (stereo, 44100 Hz)")
    print(f"  Dtype     : {stereo.dtype}")

    # Step 1: Mono
    mono = to_mono(stereo)
    print(f"[1] to_mono -> shape: {mono.shape}")
    assert mono.ndim == 1

    # Step 2: Resample
    resampled = resample(mono, sr_orig, 16000)
    expected_len = int(len(mono) * 16000 / sr_orig)
    print(f"[2] resample (44100->16000) -> shape: {resampled.shape} | expected ~{expected_len}")
    assert abs(len(resampled) - expected_len) <= 5, "Resample length mismatch"

    # Step 3: Normalize
    normalized = normalize(resampled, method="peak")
    print(f"[3] normalize (peak) -> min={normalized.min():.4f}, max={normalized.max():.4f}")
    assert abs(np.abs(normalized).max() - 1.0) < 1e-5, "Peak not normalized to 1.0"

    # Step 4: Chunk
    chunks = chunk_audio(normalized, 16000, chunk_sec=1.0, hop_sec=0.5)
    print(f"[4] chunk_audio (1s/0.5s hop) -> {len(chunks)} chunks")
    assert all(len(c) == 16000 for c in chunks), "All chunks must be 1s = 16000 samples"

    print("\n✓ preprocessing.py self-test PASSED")
