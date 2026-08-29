"""
src/features.py — VoiceVault Audio Feature Extraction Module
=============================================================
SIH26104 | Phase 1

═══════════════════════════════════════════════════════════════
SIGNAL PROCESSING THEORY (read before the code)
SIH judges WILL ask about every concept below.
═══════════════════════════════════════════════════════════════

WHY NOT USE RAW AUDIO DIRECTLY?
--------------------------------
A raw waveform is a 1D time-domain signal: amplitude vs. time.
While some models (WaveLM, Wav2Vec2) operate on raw audio, most
anti-spoofing systems — including our baseline CNN — need 2D
input (like an image) to use convolutional neural networks.

The conversion path is:
  Raw audio (time domain)
      → FFT (frequency domain)
      → STFT (time-frequency domain)
      → Spectrogram
      → Mel Spectrogram      ← INPUT TO OUR BASELINE CNN
      → MFCC (optional)

══════════════════════════════════════════════════════════════
CONCEPT 1: SAMPLING & SAMPLING RATE
══════════════════════════════════════════════════════════════
Sound is a continuous pressure wave.  A microphone converts it to
a continuous electrical signal.  An ADC (analog-to-digital converter)
measures (samples) this signal at regular intervals.

  Sampling Rate (sr) = number of samples per second (Hz)
  Sample = amplitude value at one instant in time

  VoiceVault uses sr = 16000 Hz = 16,000 samples per second.

Nyquist-Shannon Theorem: To faithfully represent a frequency f,
you need sr ≥ 2f.  At 16000 Hz, we can represent up to 8000 Hz —
covering ALL phonetically relevant speech information.

══════════════════════════════════════════════════════════════
CONCEPT 2: FREQUENCY DOMAIN — WHY FFT?
══════════════════════════════════════════════════════════════
A speech waveform is a mixture of many different sinusoidal
frequencies oscillating simultaneously.  The time-domain waveform
looks like a complicated wiggle — hard to interpret.

Fourier Transform decomposes any signal into its constituent
sinusoidal components, telling us: which frequencies are present,
and how much energy each frequency has.

  FFT = Fast Fourier Transform
      = an efficient O(N log N) algorithm to compute the Discrete
        Fourier Transform (DFT) of a finite digital signal.

WHY FFT AND NOT DFT?
--------------------
DFT on N samples requires O(N²) multiplications.
FFT achieves the same result in O(N log N) by exploiting symmetry
in the DFT matrix (Cooley-Tukey algorithm). For N=1024, this is
~100× faster.

OUTPUT OF FFT:
  Complex numbers: real + imaginary components.
  Magnitude = sqrt(real² + imag²) → tells us energy at each frequency.
  Phase = arctan(imag/real) → tells us the alignment of the sinusoid.

  For anti-spoofing, we primarily use MAGNITUDE (|FFT|).

LIMITATION OF FFT:
  FFT gives a single global frequency spectrum for the ENTIRE signal.
  It loses temporal information — we cannot tell WHEN a frequency
  occurred. Speech is time-varying (different phonemes have different
  spectral profiles). This is why we need STFT.

══════════════════════════════════════════════════════════════
CONCEPT 3: STFT — SHORT-TIME FOURIER TRANSFORM
══════════════════════════════════════════════════════════════
STFT solves FFT's temporal blindness by computing FFT on short
overlapping windows of the signal.

Algorithm:
  1. Cut the waveform into short overlapping windows.
  2. Apply a window function (e.g., Hann) to each window.
  3. Compute FFT on each window.
  4. Stack the spectra horizontally → a 2D matrix: (freq_bins × time_frames)

  n_fft     = window size in samples (1024 → ~64 ms at 16 kHz)
  hop_length = step between windows (256 → ~16 ms at 16 kHz)
  overlap    = (n_fft - hop_length) / n_fft = 75%

WHY A WINDOW FUNCTION (HANN WINDOW)?
--------------------------------------
Without windowing, cutting a waveform at a sharp boundary (rectangular
window) introduces spectral leakage — artificial frequencies that don't
actually exist in the signal. The Hann window tapers smoothly to zero
at both edges, minimizing this leakage.

OUTPUT OF STFT:
  Complex matrix of shape (n_fft//2 + 1, num_frames) = (513, T) for n_fft=1024.
  |STFT|² = power spectrogram
  |STFT|   = amplitude (linear) spectrogram

══════════════════════════════════════════════════════════════
CONCEPT 4: SPECTROGRAM
══════════════════════════════════════════════════════════════
  Spectrogram = visual/numerical representation of |STFT|² or |STFT|

  X-axis: time (frames)
  Y-axis: frequency (linear Hz scale, 0 to sr/2)
  Color/value: energy at that (time, frequency) point

WHY LOG SCALE?
--------------
Raw power values span many orders of magnitude (quiet speech is 10^-6,
loud speech is 10^-1). Converting to dB (log scale) compresses this
range and makes the spectrogram perceptually meaningful.
  dB = 10 * log10(power)  or  20 * log10(amplitude)

══════════════════════════════════════════════════════════════
CONCEPT 5: MEL SCALE & MEL SPECTROGRAM
══════════════════════════════════════════════════════════════
The linear Hz frequency axis of a spectrogram does not match human
auditory perception.  Human hearing:
  - Is sensitive to RELATIVE frequency differences (e.g., 100 vs 200 Hz
    sounds like a big jump; 5000 vs 5100 Hz is barely perceptible).
  - Has ~3500 inner hair cells distributed non-linearly across the cochlea.

The Mel scale approximates this perceptual non-linearity:
  mel(f) ≈ 2595 * log10(1 + f / 700)

Mel Filter Banks: Instead of one FFT bin per frequency, we group
FFT bins into N_MEL overlapping triangular filters spaced on the Mel
scale.  Each filter outputs ONE number (weighted sum of energy in that
band).

  Output shape: (n_mels × time_frames) = (80 × T)

WHY MEL SPECTROGRAM FOR ANTI-SPOOFING?
---------------------------------------
1. Perceptual relevance: matches how we hear speech.
2. Compact: 80 mel bins instead of 513 FFT bins → smaller model.
3. Proven: most anti-spoofing, speaker verification, and ASR models
   use Mel spectrograms as input.
4. Synthetic voices often have subtle artifacts in their spectral
   envelope that are visible in the Mel spectrogram.

LOG MEL SPECTROGRAM:
  log_mel = log(mel_spec + eps)
  eps prevents log(0). We use eps = 1e-9.

══════════════════════════════════════════════════════════════
CONCEPT 6: MFCC — MEL FREQUENCY CEPSTRAL COEFFICIENTS
══════════════════════════════════════════════════════════════
MFCC further compresses the Mel spectrogram using a Discrete Cosine
Transform (DCT):

  MFCC = DCT(log(mel_filter_bank(|STFT|²)))

The DCT decorrelates the Mel filter bank outputs (adjacent Mel bins
are correlated because the triangular filters overlap).  The first
few DCT coefficients capture the overall spectral shape; higher
coefficients capture fine detail.

We typically keep n_mfcc = 40 coefficients.

WHY USE MFCC IN VOICEVAULT?
----------------------------
MFCCs are an extremely compact feature (40 numbers per frame) that
capture the spectral envelope of speech.  They are particularly useful
for:
  - Speaker identity (the vocal tract shape determines the spectral
    envelope → different speakers have different MFCCs)
  - Prosody analysis combined with delta features
  - Lightweight classification tasks

LIMITATION: MFCCs discard phase and fine spectral detail. They are less
powerful than raw Mel spectrograms for modern deep learning models.
That's why our baseline CNN uses Mel spectrograms, not MFCCs.

══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────
Waveform = np.ndarray  # 1D float32, (samples,)
FeatureMatrix = np.ndarray  # 2D float32, (n_features, n_frames)


# ══════════════════════════════════════════════════════════
# 1. FFT — Fast Fourier Transform
# ══════════════════════════════════════════════════════════

def compute_fft(
    waveform: Waveform,
    sample_rate: int,
    n_fft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided magnitude spectrum of a waveform.

    WHAT IT DOES
    ------------
    Applies NumPy's FFT to the entire waveform to produce a global
    frequency spectrum.  Returns only the positive-frequency half
    (single-sided spectrum) because the spectrum of a real signal is
    symmetric.

    WHY VOICEVAULT NEEDS IT
    -----------------------
    The FFT gives an immediate, global view of which frequencies are
    dominant in a speech sample. Synthetic voices often show:
      - Unusual energy concentrations at specific harmonic frequencies.
      - Missing high-frequency content (vocoders often band-limit at 8 kHz).
      - Artifacts at frequencies that natural speech doesn't produce.

    INPUT
    -----
    waveform    : mono float32 array, shape (samples,)
    sample_rate : integer Hz
    n_fft       : FFT window size (default: len(waveform))

    OUTPUT
    ------
    frequencies : array of frequency values in Hz, shape (n_fft//2 + 1,)
    magnitudes  : magnitude spectrum, shape (n_fft//2 + 1,)
    """
    n = n_fft if n_fft is not None else len(waveform)

    # Apply Hann window to reduce spectral leakage (see theory above).
    window = np.hanning(len(waveform))
    windowed = waveform * window[:len(waveform)]

    # np.fft.rfft computes only the non-redundant half of the spectrum.
    # Output length: n//2 + 1.  This is the single-sided spectrum.
    fft_result = np.fft.rfft(windowed, n=n)

    magnitudes = np.abs(fft_result)  # Complex → magnitude
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)  # Hz

    logger.debug(
        "compute_fft: n=%d | freq range=%.1f–%.1f Hz | n_bins=%d",
        n,
        frequencies[0],
        frequencies[-1],
        len(frequencies),
    )
    return frequencies, magnitudes


def compute_fft_db(
    waveform: Waveform,
    sample_rate: int,
    n_fft: Optional[int] = None,
    ref: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude spectrum in decibels (dB).

    WHY DB SCALE?
    -------------
    The human auditory system perceives loudness on a logarithmic scale.
    dB = 20 * log10(magnitude / ref)  (amplitude dB convention)
    At ref=1.0, 0 dB = full scale; negative dB = quieter.

    OUTPUT
    ------
    frequencies : Hz array
    magnitudes_db : magnitude in dB
    """
    freqs, mags = compute_fft(waveform, sample_rate, n_fft)
    mags_db = 20 * np.log10(np.maximum(mags / ref, 1e-10))
    return freqs, mags_db


# ══════════════════════════════════════════════════════════
# 2. STFT — Short-Time Fourier Transform
# ══════════════════════════════════════════════════════════

def compute_stft(
    waveform: Waveform,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute the STFT of a waveform.

    WHAT IT DOES
    ------------
    Slides a window across the waveform and computes the FFT of each
    windowed segment. Returns the complex STFT matrix.

    WHY VOICEVAULT NEEDS STFT
    --------------------------
    Unlike global FFT, STFT captures HOW the frequency content changes
    over time. This is critical for anti-spoofing because:
      1. Natural prosody creates predictable temporal patterns in the spectrum.
      2. TTS/voice conversion systems often introduce temporal artifacts
         (glitches, unnatural transitions) visible in the STFT.
      3. Speaker identity is partially encoded in spectral DYNAMICS,
         not just static spectra.

    INPUT
    -----
    waveform   : mono float32 array, shape (samples,)
    n_fft      : FFT window size in samples (default: 1024 ≈ 64 ms at 16 kHz)
    hop_length : step between frames in samples (default: 256 ≈ 16 ms)
    win_length : window length (default: same as n_fft)
    window     : window function type ('hann', 'hamming', etc.)

    OUTPUT
    ------
    stft_matrix : complex128 array, shape (n_fft//2 + 1, n_frames)
                  n_frames ≈ (len(waveform) - win_length) / hop_length + 1

    PARAMETERS EXPLAINED
    --------------------
    n_fft=1024 at sr=16000:
      - Window duration = 1024/16000 = 64 ms
      - Frequency resolution = 16000/1024 ≈ 15.6 Hz per bin (fine enough to
        distinguish speech formants, which are spaced ~500–1000 Hz apart)

    hop_length=256 at sr=16000:
      - Frame step = 256/16000 = 16 ms
      - Overlap = (1024-256)/1024 = 75% — high overlap for smooth spectrograms
    """
    if win_length is None:
        win_length = n_fft

    # Build the window function.
    if window == "hann":
        win = np.hanning(win_length).astype(np.float32)
    elif window == "hamming":
        win = np.hamming(win_length).astype(np.float32)
    else:
        raise ValueError(f"Unknown window type: '{window}'. Use 'hann' or 'hamming'.")

    # Pad the beginning and end so that the first/last frame is centered
    # on sample 0/N-1 (center padding — matches librosa default).
    pad_length = n_fft // 2
    waveform_padded = np.pad(waveform, pad_length, mode="reflect")

    # Pad the window to n_fft if win_length < n_fft.
    if win_length < n_fft:
        pad_w = (n_fft - win_length) // 2
        win = np.pad(win, (pad_w, n_fft - win_length - pad_w))

    # Number of frames.
    n_frames = 1 + (len(waveform_padded) - n_fft) // hop_length
    n_bins = n_fft // 2 + 1

    stft_matrix = np.zeros((n_bins, n_frames), dtype=np.complex128)

    for t in range(n_frames):
        start = t * hop_length
        frame = waveform_padded[start: start + n_fft]
        windowed_frame = frame * win
        stft_matrix[:, t] = np.fft.rfft(windowed_frame)

    logger.debug(
        "compute_stft: n_fft=%d | hop=%d | shape=%s",
        n_fft,
        hop_length,
        stft_matrix.shape,
    )
    return stft_matrix


def compute_power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Compute power spectrogram from STFT.

    Power Spectrogram = |STFT|²
    Represents energy at each (frequency, time) point.

    INPUT  : complex STFT matrix, shape (n_bins, n_frames)
    OUTPUT : float32 power spectrogram, shape (n_bins, n_frames)
    """
    return (np.abs(stft_matrix) ** 2).astype(np.float32)


def compute_amplitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Compute amplitude spectrogram from STFT.

    Amplitude Spectrogram = |STFT|  (linear amplitude, not squared)

    INPUT  : complex STFT matrix, shape (n_bins, n_frames)
    OUTPUT : float32 amplitude spectrogram, shape (n_bins, n_frames)
    """
    return np.abs(stft_matrix).astype(np.float32)


# ══════════════════════════════════════════════════════════
# 3. MEL FILTER BANKS
# ══════════════════════════════════════════════════════════

def hz_to_mel(hz: float) -> float:
    """
    Convert frequency in Hz to Mel scale.

    Formula (O'Shaughnessy, 1987):
      mel = 2595 * log10(1 + hz / 700)

    WHY THIS FORMULA?
    -----------------
    Empirically derived from psychoacoustic experiments.  Equal intervals
    on the Mel scale correspond to perceptually equal pitch differences.
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """
    Convert Mel scale back to Hz.

    Inverse of hz_to_mel:
      hz = 700 * (10^(mel / 2595) - 1)
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Build a matrix of triangular Mel filter banks.

    WHAT IT DOES
    ------------
    Creates n_mels triangular filters spaced evenly on the Mel scale.
    Multiplying a power spectrogram by this matrix maps FFT bins to
    Mel bins.

    ALGORITHM
    ---------
    1. Convert fmin and fmax to Mel scale.
    2. Create n_mels + 2 equally spaced Mel center frequencies.
    3. Convert back to Hz (these are the filter center frequencies).
    4. Map Hz frequencies to FFT bin indices.
    5. Create triangular filters: each filter is 0 everywhere except
       between its neighboring center frequencies, where it rises/falls
       linearly (triangle shape).
    6. Normalize each filter by its bandwidth (area normalization).

    INPUT
    -----
    n_mels      : number of Mel bands (default: 80)
    n_fft       : FFT size (determines number of frequency bins = n_fft//2 + 1)
    sample_rate : integer Hz
    fmin        : minimum frequency in Hz (default: 0)
    fmax        : maximum frequency in Hz (default: sample_rate/2 = Nyquist)

    OUTPUT
    ------
    filterbank : float32 array, shape (n_mels, n_fft//2 + 1)
    Multiply this by a power spectrogram to get the Mel spectrogram.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    # Step 1: Mel-scale equally spaced center frequencies.
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Step 2: Convert Mel center frequencies back to Hz.
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    # Step 3: Map Hz frequencies to FFT bin indices.
    # The FFT bin frequencies are: k * sample_rate / n_fft for k = 0..n_fft//2.
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Step 4: Build the triangular filters.
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_left = bin_indices[m - 1]   # left edge
        f_center = bin_indices[m]     # center (peak)
        f_right = bin_indices[m + 1]  # right edge

        # Rising slope: left edge → center
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

        # Falling slope: center → right edge
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    # Step 5: Area-normalize each filter (Slaney normalization).
    # This ensures that each filter contributes equally regardless of bandwidth.
    enorm = 2.0 / (hz_points[2: n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]

    return filterbank


# ══════════════════════════════════════════════════════════
# 4. MEL SPECTROGRAM (via librosa for production quality)
# ══════════════════════════════════════════════════════════

def compute_mel_spectrogram(
    waveform: Waveform,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram.

    WHAT IT DOES
    ------------
    1. Compute STFT → power spectrogram
    2. Apply Mel filter bank → Mel spectrogram
    3. Convert to log scale (dB)

    WHY THIS IS OUR CNN INPUT
    --------------------------
    Shape: (n_mels, n_frames) = (80, T)
    This is a 2D matrix — structurally similar to a grayscale image.
    CNNs (convolutional neural networks) excel at detecting local spatial
    patterns in 2D matrices, just like they detect edges in images.

    The LOCAL PATTERNS in a Mel spectrogram represent:
      - Formants (speaker vocal tract resonances) → speaker identity
      - Spectral artifacts of TTS/vocoder systems → deepfake signals
      - Prosodic features (energy, pitch variation) → speaking style

    INPUT
    -----
    waveform   : mono float32 array at `sample_rate`
    sample_rate: integer Hz (should be 16000 for VoiceVault)
    n_fft      : FFT window size (default: 1024)
    hop_length : STFT frame step (default: 256)
    n_mels     : number of Mel filter bands (default: 80)
    fmin       : minimum frequency for Mel filters (Hz)
    fmax       : maximum frequency for Mel filters (Hz, default: sr/2)
    power      : 2.0 = power spectrogram, 1.0 = amplitude spectrogram
    top_db     : clip log-Mel values below (max - top_db) dB. None = no clip.

    OUTPUT
    ------
    log_mel_spec : float32 array, shape (n_mels, n_frames)
                   Values in dB.  Range approximately [-top_db, 0].
    """
    try:
        # Use librosa for production-quality Mel spectrogram computation.
        # librosa.feature.melspectrogram is well-tested and numerically stable.
        import librosa  # type: ignore

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax if fmax is not None else sample_rate / 2.0,
            power=power,
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)

    except ImportError:
        # Pure NumPy fallback (uses our hand-built filter bank above).
        logger.warning(
            "librosa not installed. Using NumPy fallback for Mel spectrogram. "
            "Results may differ slightly from librosa."
        )
        log_mel_spec = _compute_mel_spectrogram_numpy(
            waveform, sample_rate, n_fft, hop_length, n_mels, fmin, fmax, power, top_db
        )

    logger.debug("compute_mel_spectrogram: shape=%s", log_mel_spec.shape)
    return log_mel_spec.astype(np.float32)


def _compute_mel_spectrogram_numpy(
    waveform, sample_rate, n_fft, hop_length, n_mels, fmin, fmax, power, top_db
) -> np.ndarray:
    """Pure NumPy Mel spectrogram (fallback when librosa is unavailable)."""
    stft = compute_stft(waveform, n_fft=n_fft, hop_length=hop_length)

    if power == 2.0:
        spec = compute_power_spectrogram(stft)
    else:
        spec = compute_amplitude_spectrogram(stft)

    filterbank = build_mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
    mel_spec = filterbank @ spec  # (n_mels, n_fft//2+1) × (n_fft//2+1, T) → (n_mels, T)

    # Log conversion.
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))

    if top_db is not None:
        log_mel = np.maximum(log_mel, log_mel.max() - top_db)

    return log_mel.astype(np.float32)


# ══════════════════════════════════════════════════════════
# 5. MFCC — Mel Frequency Cepstral Coefficients
# ══════════════════════════════════════════════════════════

def compute_mfcc(
    waveform: Waveform,
    sample_rate: int,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    delta: bool = True,
    delta2: bool = True,
) -> np.ndarray:
    """
    Compute MFCC features (optionally with delta and delta-delta).

    WHAT IT DOES
    ------------
    1. Compute log-Mel spectrogram
    2. Apply DCT → MFCC coefficients (decorrelated compact features)
    3. Optionally compute delta (velocity) and delta-delta (acceleration)

    WHY DELTAS?
    -----------
    MFCCs alone are static features per frame.  Delta MFCCs (first-order
    temporal derivative) capture HOW the spectrum is changing over time —
    this encodes information about speaking rate and transitions between
    phonemes.  Delta-delta (second-order) captures the rate of change of
    change, similar to acceleration.

    Together, [MFCC | Δ | ΔΔ] tripled the feature set to 3 × n_mfcc.
    This combination is the classical feature set for speaker verification
    and speech recognition.

    INPUT
    -----
    waveform   : mono float32 array
    sample_rate: Hz
    n_mfcc     : number of MFCC coefficients to keep (default: 40)
    n_fft, hop_length, n_mels: passed to Mel spectrogram computation
    delta      : if True, append delta coefficients
    delta2     : if True, append delta-delta coefficients

    OUTPUT
    ------
    mfcc_features : float32 array, shape:
                    (n_mfcc, n_frames)       if not delta, not delta2
                    (2*n_mfcc, n_frames)     if delta only
                    (3*n_mfcc, n_frames)     if delta and delta2
    """
    try:
        import librosa  # type: ignore

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        features = [mfcc]
        if delta:
            d1 = librosa.feature.delta(mfcc)
            features.append(d1)
        if delta2:
            d2 = librosa.feature.delta(mfcc, order=2)
            features.append(d2)

        result = np.concatenate(features, axis=0).astype(np.float32)

    except ImportError:
        logger.warning("librosa not installed. Computing basic MFCC with NumPy (no deltas).")
        from scipy.fft import dct  # type: ignore

        # Get log-Mel spectrogram.
        log_mel = _compute_mel_spectrogram_numpy(
            waveform, sample_rate, n_fft, hop_length, n_mels, 0.0, None, 2.0, 80.0
        )
        # DCT type-II (standard MFCC) along the Mel-frequency axis.
        mfcc_raw = dct(log_mel, type=2, axis=0, norm="ortho")
        result = mfcc_raw[:n_mfcc, :].astype(np.float32)

    logger.debug("compute_mfcc: shape=%s", result.shape)
    return result


# ══════════════════════════════════════════════════════════
# 6. FIXED-SIZE MEL SPECTROGRAM FOR CNN INPUT
# ══════════════════════════════════════════════════════════

def mel_spectrogram_for_cnn(
    waveform: Waveform,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fixed_time_frames: int = 188,
) -> np.ndarray:
    """
    Compute a fixed-size log-Mel spectrogram for CNN input.

    WHY FIXED SIZE?
    ---------------
    CNNs require fixed-size input tensors (unlike RNNs).  Different audio
    clips have different lengths → different numbers of time frames.
    We standardize by:
      - TRUNCATING if the spectrogram is too long.
      - ZERO-PADDING on the right if it's too short.

    WHY 188 FRAMES?
    ---------------
    At hop_length=256 and sr=16000:
      1 frame = 256/16000 = 16 ms
      3 seconds = 3000 ms / 16 ms = ~187.5 frames → 188 frames

    3 seconds captures 2–4 phonemes with context, enough for both
    deepfake artifact detection and speaker identity assessment.

    INPUT
    -----
    waveform         : mono float32 array (any length)
    sample_rate      : integer Hz
    fixed_time_frames: target number of time frames (default: 188 = 3s)

    OUTPUT
    ------
    mel_spec : float32 array, shape (1, n_mels, fixed_time_frames)
               The leading 1 is the "channel" dimension for CNN (like
               a grayscale image has 1 channel).
    """
    mel = compute_mel_spectrogram(
        waveform, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # mel.shape = (n_mels, n_time_frames)
    n_time = mel.shape[1]

    if n_time >= fixed_time_frames:
        # Truncate from the right.
        mel = mel[:, :fixed_time_frames]
    else:
        # Zero-pad on the right.
        pad_width = fixed_time_frames - n_time
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    # Add channel dimension: (n_mels, T) → (1, n_mels, T)
    return mel[np.newaxis, :, :].astype(np.float32)


# ══════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ══════════════════════════════════════════════════════════

def visualize_waveform(
    waveform: Waveform,
    sample_rate: int,
    title: str = "Waveform",
    ax=None,
):
    """
    Plot waveform amplitude vs. time.

    INPUT  : mono float32 waveform, sample rate
    OUTPUT : matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    times = np.linspace(0, len(waveform) / sample_rate, len(waveform))
    ax.plot(times, waveform, linewidth=0.5, color="#2196F3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim([times[0], times[-1]])
    ax.set_ylim([-1.05, 1.05])
    ax.grid(True, alpha=0.3)

    return fig


def visualize_fft(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "Frequency Spectrum (FFT)",
    ax=None,
    db_scale: bool = True,
):
    """Plot FFT magnitude spectrum."""
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    if db_scale:
        mags_plot = 20 * np.log10(np.maximum(magnitudes, 1e-10))
        ylabel = "Magnitude (dB)"
    else:
        mags_plot = magnitudes
        ylabel = "Magnitude (linear)"

    ax.plot(frequencies, mags_plot, linewidth=0.8, color="#E91E63")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim([0, frequencies[-1]])
    ax.grid(True, alpha=0.3)

    return fig


def visualize_spectrogram(
    spectrogram: np.ndarray,
    hop_length: int = 256,
    sample_rate: int = 16000,
    title: str = "Spectrogram",
    ax=None,
    cmap: str = "viridis",
):
    """
    Plot a spectrogram or Mel spectrogram as a 2D heatmap.

    INPUT
    -----
    spectrogram : 2D array, shape (n_freq_bins or n_mels, n_frames)
    hop_length  : used to compute time axis labels
    sample_rate : used to compute time axis labels
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    n_frames = spectrogram.shape[1]
    duration = n_frames * hop_length / sample_rate

    img = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[0, duration, 0, spectrogram.shape[0]],
    )
    plt.colorbar(img, ax=ax, label="Value (dB or linear)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency bins")
    ax.set_title(title)

    return fig


# ══════════════════════════════════════════════════════════
# Smoke test — run directly: python -m src.features
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import logging
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for CI
    import matplotlib.pyplot as plt
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=" * 65)
    print("VoiceVault -- features.py self-test")
    print("=" * 65)

    # Synthesize a test signal: mixture of 300 Hz + 1000 Hz + 3000 Hz sine waves
    # Simulates rough speech-like content at different frequencies.
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = (
        0.5 * np.sin(2 * np.pi * 300 * t) +
        0.3 * np.sin(2 * np.pi * 1000 * t) +
        0.2 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)
    # Peak normalize.
    waveform /= np.abs(waveform).max()

    print(f"\n[INPUT] Synthetic 3-frequency sine signal")
    print(f"  Duration    : {duration:.1f} s")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Frequencies : 300 Hz, 1000 Hz, 3000 Hz")

    # ── Test 1: FFT ──────────────────────────────────────────
    freqs, mags = compute_fft(waveform, sr)
    print(f"\n[1] FFT")
    print(f"  Output shape : {mags.shape}  (frequency bins)")
    # The three peaks should appear at 300, 1000, 3000 Hz.
    top3_indices = np.argsort(mags)[-3:]
    top3_freqs = freqs[top3_indices]
    top3_freqs_sorted = sorted(top3_freqs)
    print(f"  Top 3 freq peaks : {[f'{f:.0f}' for f in top3_freqs_sorted]} Hz")

    # ── Test 2: STFT ──────────────────────────────────────────
    stft = compute_stft(waveform, n_fft=1024, hop_length=256)
    power_spec = compute_power_spectrogram(stft)
    print(f"\n[2] STFT")
    print(f"  Complex STFT shape  : {stft.shape}  (freq_bins x time_frames)")
    print(f"  Power spec shape    : {power_spec.shape}")
    assert stft.shape[0] == 1024 // 2 + 1, "Wrong number of freq bins"

    # ── Test 3: Mel Spectrogram ───────────────────────────────
    mel = compute_mel_spectrogram(waveform, sr, n_fft=1024, hop_length=256, n_mels=80)
    print(f"\n[3] Mel Spectrogram")
    print(f"  Shape  : {mel.shape}  (n_mels=80 x n_frames)")
    print(f"  Min dB : {mel.min():.1f}")
    print(f"  Max dB : {mel.max():.1f}")
    assert mel.shape[0] == 80, "Wrong n_mels"

    # ── Test 4: CNN-ready Mel ─────────────────────────────────
    mel_cnn = mel_spectrogram_for_cnn(waveform, sr, fixed_time_frames=188)
    print(f"\n[4] CNN-ready Mel Spectrogram")
    print(f"  Shape  : {mel_cnn.shape}  (1 channel × 80 mels × 188 frames)")
    assert mel_cnn.shape == (1, 80, 188), f"Wrong CNN input shape: {mel_cnn.shape}"

    # ── Test 5: MFCC ──────────────────────────────────────────
    mfcc = compute_mfcc(waveform, sr, n_mfcc=40, delta=True, delta2=True)
    print(f"\n[5] MFCC (with delta and delta-delta)")
    print(f"  Shape  : {mfcc.shape}  (3x40=120 features x n_frames)")
    assert mfcc.shape[0] == 120, f"Wrong MFCC shape: {mfcc.shape}"

    # ── Test 6: Mel filter bank (manual) ─────────────────────
    filterbank = build_mel_filterbank(n_mels=80, n_fft=1024, sample_rate=sr)
    print(f"\n[6] Mel Filter Bank")
    print(f"  Shape  : {filterbank.shape}  (80 filters × 513 freq bins)")
    print(f"  Min    : {filterbank.min():.6f}")
    print(f"  Max    : {filterbank.max():.6f}")
    # Each row should be non-negative (triangular filter).
    assert (filterbank >= 0).all(), "Filter bank has negative values"

    # ── Save visualizations ───────────────────────────────────
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    visualize_waveform(waveform, sr, title="Waveform (300+1000+3000 Hz sine)", ax=axes[0])
    visualize_fft(freqs, mags, title="FFT Magnitude Spectrum", ax=axes[1])
    visualize_spectrogram(
        power_spec,
        hop_length=256,
        sample_rate=sr,
        title="Power Spectrogram (STFT)",
        ax=axes[2],
    )
    visualize_spectrogram(
        mel,
        hop_length=256,
        sample_rate=sr,
        title="Log-Mel Spectrogram (80 Mel bins)",
        ax=axes[3],
    )

    plt.tight_layout()
    out_path = out_dir / "features_selftest.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\n  Visualizations saved -> {out_path}")

    print("\n* features.py self-test PASSED (all assertions passed)")
    print("\nSIH JUDGE QUESTIONS - Quick Reference:")
    print("  Q: Why Mel spectrogram and not raw FFT for the CNN?")
    print("     A: Mel is perceptually meaningful + compact (80 vs 513 bins)")
    print("        CNN sees it like a 2D image -> detects spectral artifacts.")
    print("  Q: Why 16 kHz?")
    print("     A: Nyquist=8kHz covers all speech phonemes. ASVspoof 2019 LA standard.")
    print("  Q: Why n_fft=1024?")
    print("     A: 64ms window = good frequency resolution (15.6 Hz/bin)")
    print("        fine enough to distinguish formants (~500-1000 Hz spacing).")
    print("  Q: Why hop_length=256?")
    print("     A: 16ms frame step = standard speech analysis rate.")
    print("        75% overlap ensures smooth, continuous spectrograms.")
