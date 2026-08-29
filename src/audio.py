"""
src/audio.py — VoiceVault Audio I/O Module
===========================================
SIH26104 | Phase 1

PURPOSE
-------
This is the entry-point for all audio data into the VoiceVault pipeline.
Every piece of audio — whether uploaded via the API, streamed from a mic,
or loaded from a dataset — passes through this module first.

WHY A DEDICATED I/O MODULE?
----------------------------
Separating I/O from processing/features keeps the codebase modular:
  - API code calls audio.load_audio(), not librosa directly.
  - If we swap the audio backend (e.g., soundfile → torchaudio), we
    change only THIS file.
  - Unit tests can mock this module without touching ML code.

SUPPORTED FORMATS
-----------------
soundfile handles: WAV, FLAC, OGG (Vorbis), AIFF, RF64.
For MP3/M4A we fall back to librosa (which uses audioread/FFmpeg).
ASVspoof 2019 LA is distributed as FLAC — soundfile reads it natively
without FFmpeg.

SIH JUDGE QUESTIONS (anticipate these)
--------------------------------------
Q: Why 16 kHz?
A: ASVspoof 2019 LA is recorded at 16 kHz. Most anti-spoofing and
   speaker-verification models are trained at 16 kHz. Human speech
   occupies frequencies primarily below 8 kHz (Nyquist = sr/2 = 8 kHz
   at 16 kHz), so no meaningful information is lost. We resample in
   preprocessing.py, not here — I/O stays format-agnostic.

Q: Why soundfile over scipy.io.wavfile?
A: soundfile supports float32 output directly, multi-channel, and FLAC.
   scipy.io.wavfile only handles WAV and returns integer arrays that
   require manual normalization.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────
# Waveform convention throughout VoiceVault:
#   - numpy float32 array
#   - shape: (num_samples,) for mono  OR  (num_samples, num_channels) for stereo
#   - amplitude range: [-1.0, +1.0]  (normalized in preprocessing.py)
Waveform = np.ndarray
SampleRate = int


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".ogg", ".aiff", ".aif", ".rf64", ".mp3", ".m4a", ".opus"}
)


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def load_audio(
    path: Union[str, Path],
    always_2d: bool = False,
) -> Tuple[Waveform, SampleRate]:
    """
    Load an audio file from disk.

    WHAT IT DOES
    ------------
    Reads any supported audio file and returns a float32 numpy array
    plus the native sample rate. Does NOT resample — resampling is done
    in preprocessing.py so that this function stays format-agnostic.

    WHY VOICEVAULT NEEDS THIS
    -------------------------
    Every audio path — file upload, dataset loading, enrollment recording —
    must produce a consistent (waveform, sample_rate) pair before the
    preprocessing pipeline can run.

    INPUT
    -----
    path        : Path to audio file (str or Path object)
    always_2d   : If True, always returns shape (samples, channels).
                  If False (default), mono files return shape (samples,).

    OUTPUT
    ------
    waveform    : numpy float32 array, amplitude in [-1.0, 1.0]
    sample_rate : integer sample rate of the file (e.g. 16000, 44100)

    IMPORTANT PARAMETERS
    --------------------
    dtype='float32'  → soundfile converts integer PCM to float32 automatically,
                       normalizing 16-bit [-32768, 32767] → [-1.0, 1.0].

    RAISES
    ------
    FileNotFoundError  : if the path does not exist
    ValueError         : if the file extension is unsupported
    RuntimeError       : if the file cannot be decoded
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.debug("Loading audio: %s", path)

    try:
        # soundfile handles WAV, FLAC, OGG natively — no FFmpeg required.
        # For MP3/M4A, soundfile will raise; we fall back to librosa.
        waveform, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=always_2d
        )
        logger.debug(
            "Loaded '%s' | sr=%d Hz | shape=%s | duration=%.2f s",
            path.name,
            sample_rate,
            waveform.shape,
            len(waveform) / sample_rate,
        )
        return waveform, sample_rate

    except sf.SoundFileError:
        # Fall back to librosa for MP3, M4A, Opus (requires audioread/FFmpeg).
        logger.warning(
            "soundfile could not read '%s'. Falling back to librosa "
            "(requires audioread/FFmpeg for compressed formats).",
            path.name,
        )
        return _load_with_librosa(path)


def load_audio_from_bytes(
    audio_bytes: bytes,
    file_extension: str = ".wav",
) -> Tuple[Waveform, SampleRate]:
    """
    Decode audio from an in-memory byte buffer.

    WHAT IT DOES
    ------------
    Used by the FastAPI endpoint (Phase 10) when audio arrives as an
    HTTP multipart upload. The API receives raw bytes; we decode them
    here without writing a temp file to disk.

    WHY VOICEVAULT NEEDS THIS
    -------------------------
    Avoiding temp files improves privacy (no audio persists on disk unless
    explicitly saved) and reduces I/O latency.

    INPUT
    -----
    audio_bytes    : raw bytes of an audio file
    file_extension : hint for the decoder (e.g. '.wav', '.flac')

    OUTPUT
    ------
    waveform    : numpy float32 array
    sample_rate : integer
    """
    file_extension = file_extension.lower()
    if not file_extension.startswith("."):
        file_extension = "." + file_extension

    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: '{file_extension}'")

    logger.debug("Decoding audio from bytes (%d bytes)", len(audio_bytes))

    try:
        buf = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(buf, dtype="float32")
        logger.debug(
            "Decoded from bytes | sr=%d | shape=%s | duration=%.2f s",
            sample_rate,
            waveform.shape,
            len(waveform) / sample_rate,
        )
        return waveform, sample_rate
    except sf.SoundFileError as exc:
        raise RuntimeError(
            f"Failed to decode audio bytes ({file_extension}): {exc}"
        ) from exc


def save_audio(
    waveform: Waveform,
    path: Union[str, Path],
    sample_rate: int,
    subtype: str = "PCM_16",
) -> None:
    """
    Save a waveform to disk as a WAV file.

    WHAT IT DOES
    ------------
    Used for saving preprocessed chunks, enrollment recordings, or
    synthetic test signals.

    INPUT
    -----
    waveform    : numpy float32 array, shape (samples,) or (samples, channels)
    path        : output file path (must end in .wav or .flac)
    sample_rate : integer
    subtype     : soundfile subtype — 'PCM_16' (default), 'PCM_24', 'FLOAT'

    OUTPUT
    ------
    None. File written to `path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clamp to [-1, 1] to prevent clipping artifacts when saved as PCM.
    waveform_clamped = np.clip(waveform, -1.0, 1.0)

    sf.write(str(path), waveform_clamped, sample_rate, subtype=subtype)
    logger.debug("Saved audio: %s | sr=%d | shape=%s", path, sample_rate, waveform.shape)


def get_audio_info(path: Union[str, Path]) -> dict:
    """
    Return metadata about an audio file without fully decoding it.

    OUTPUT
    ------
    dict with keys: sample_rate, frames, channels, duration_sec,
                    format, subtype, path
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "frames": info.frames,
        "channels": info.channels,
        "duration_sec": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def _load_with_librosa(path: Path) -> Tuple[Waveform, SampleRate]:
    """
    Librosa fallback for compressed formats (MP3, M4A, Opus).
    librosa.load() uses audioread which requires FFmpeg on Windows.
    Returns float32 mono waveform at the file's native sample rate.
    """
    try:
        import librosa  # type: ignore

        # sr=None → preserve native sample rate (do not resample here).
        # mono=False → preserve channels; mono conversion is in preprocessing.py.
        waveform, sample_rate = librosa.load(str(path), sr=None, mono=False)

        # librosa returns (channels, samples) for multi-channel; transpose to
        # (samples, channels) to match soundfile convention.
        if waveform.ndim == 2:
            waveform = waveform.T

        return waveform.astype(np.float32), int(sample_rate)

    except ImportError as exc:
        raise RuntimeError(
            "librosa is not installed. Install it via: pip install librosa"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"librosa could not decode '{path}'. "
            "Ensure FFmpeg is installed for compressed audio formats. "
            f"Original error: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────
# Smoke test — run directly: python -m src.audio
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python -m src.audio <path_to_audio_file>")
        print("\nRunning self-test with a synthetic signal...")

        # Generate a 1-second 440 Hz sine wave as a test signal.
        # This lets us verify the module works WITHOUT needing a real file.
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        test_path = Path("data/demo/genuine/_test_sine_440hz.wav")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(sine_wave, test_path, sr)
        print(f"Saved synthetic sine wave to: {test_path}")

        loaded, loaded_sr = load_audio(test_path)
        print(f"Loaded back: shape={loaded.shape}, sr={loaded_sr}, "
              f"min={loaded.min():.4f}, max={loaded.max():.4f}")
        print("src/audio.py self-test PASSED.")
    else:
        audio_path = sys.argv[1]
        wv, sr = load_audio(audio_path)
        info = get_audio_info(audio_path)
        print("\n── Audio Info ──────────────────────────────────")
        for k, v in info.items():
            print(f"  {k:<15}: {v}")
        print(f"\n── Waveform Array ─────────────────────────────")
        print(f"  dtype          : {wv.dtype}")
        print(f"  shape          : {wv.shape}")
        print(f"  min amplitude  : {wv.min():.6f}")
        print(f"  max amplitude  : {wv.max():.6f}")
        print(f"  mean amplitude : {wv.mean():.6f}")
        print(f"  RMS            : {np.sqrt(np.mean(wv**2)):.6f}")
