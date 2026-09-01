# VoiceVault 🔐
### AI-Powered Voice Integrity Verification & Impersonation-Risk Detection Platform
**SIH Problem ID: SIH26104**

---

## Problem Statement

AI-generated and cloned voices are being used to impersonate executives, officials, and trusted individuals to authorize fraudulent transactions, bypass verification, and manipulate employees. Conventional mechanisms (caller ID, manual callback, voice familiarity) are insufficient against high-fidelity voice cloning.

## VoiceVault's Solution

An AI-driven voice integrity verification framework that:

```
INCOMING VOICE
    ↓
AUDIO PREPROCESSING (resample → mono → normalize → VAD → chunk)
    ↓
MULTI-LAYER ANALYSIS
  ├── Acoustic/Spectral Analysis (FFT, STFT, Mel Spectrogram)
  ├── Deepfake Detection (CNN on Mel spectrogram)
  ├── Speaker Verification (embedding similarity)
  └── Prosody Analysis (pitch, energy, speaking rate)
    ↓
CONTEXT ENGINE (transaction amount, caller identity, history)
    ↓
RISK ENGINE → Dynamic Risk Score (0–100)
    ↓
LOW/MEDIUM → Continue  |  HIGH/CRITICAL → ALERT + HOLD
    ↓
SECONDARY VERIFICATION (callback, MFA, supervisor approval)
```

---

## Current Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Project scaffold + environment | ✅ COMPLETE |
| 1 | `src/audio.py` — Audio I/O | ✅ COMPLETE |
| 1 | `src/preprocessing.py` — Pipeline | ✅ COMPLETE |
| 1 | `src/features.py` — FFT/STFT/Mel/MFCC | ✅ COMPLETE |
| 1 | `configs/config.yaml` — Configuration | ✅ COMPLETE |
| 1 | `tests/test_audio_pipeline.py` — Unit tests | ✅ COMPLETE |
| 2 | Dataset loader (ASVspoof 2019 LA) | 🟡 DOWNLOADED & EXTRACTED — Loader pending |
| 3 | Jupyter notebooks (01–03) | ⏳ Next |
| 4 | PyTorch Dataset/DataLoader | ⏳ Phase 4 |
| 5 | Baseline CNN deepfake detector | ⏳ Phase 5 |
| 6 | Speaker verification | ⏳ Phase 6 |
| 7 | Prosody extraction | ⏳ Phase 7 |
| 8 | Risk + Context engine | ⏳ Phase 8 |
| 9 | Alert + Mock transaction | ⏳ Phase 9 |
| 10 | FastAPI backend | ⏳ Phase 10 |
| 11 | Dashboard | ⏳ Phase 11 |

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install PyTorch with CUDA 12.8 support (RTX 3050)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Verify GPU detection
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Run Phase 1 module self-tests (no dataset required)
python -m src.audio
python -m src.preprocessing
python -m src.features        # generates reports/features_selftest.png

# Run unit tests
pytest tests/test_audio_pipeline.py -v
```

### 3. Dataset (ASVspoof 2019 LA)

```bash
# 1. Register at: https://datasharing.ed.ac.uk/
# 2. Download LA.zip
# 3. Extract to: data/asvspoof2019/LA/
# 4. Run dataset verification: python scripts/verify_dataset.py  [Phase 4]
```

---

## Project Structure

```
VoiceVault/
├── src/
│   ├── audio.py              # Audio I/O (load/save/decode)
│   ├── preprocessing.py      # Mono, resample, normalize, VAD, chunk
│   ├── features.py           # FFT, STFT, Mel spectrogram, MFCC
│   ├── deepfake_detector.py  # Baseline CNN (Phase 5)
│   ├── speaker_verification.py  # (Phase 6)
│   ├── prosody.py            # (Phase 7)
│   ├── context_engine.py     # (Phase 8)
│   ├── risk_engine.py        # (Phase 8)
│   └── decision_engine.py    # (Phase 9)
├── api/
│   └── main.py               # FastAPI backend (Phase 10)
├── frontend/                 # Dashboard (Phase 11)
├── notebooks/                # Jupyter exploration (Phase 3)
├── models/
│   ├── baseline/             # CNN checkpoints
│   ├── speaker/              # Speaker encoder checkpoints
│   └── advanced/             # WavLM/AASIST (Phase 14)
├── data/
│   ├── asvspoof2019/LA/      # Primary dataset (not committed)
│   ├── demo/genuine/         # Demo genuine audio
│   ├── demo/spoof/           # Demo spoof audio
│   └── processed/            # Cached numpy features
├── tests/                    # pytest unit tests
├── configs/config.yaml       # All configurable parameters
└── reports/
    ├── dataset_registry.md   # Dataset facts and licensing
    └── experiment_log.md     # All experimental results
```

---

## Signal Processing Pipeline

| Step | What | Why |
|------|------|-----|
| Load | Read WAV/FLAC | Entry point for all audio |
| Mono | Average channels | Models expect single channel |
| Resample | → 16 kHz | ASVspoof standard; Nyquist = 8 kHz covers all speech |
| Normalize | Peak → [-1, 1] | Remove recording-level variation |
| VAD | Remove silence | Focus on speech; reduce false positives |
| FFT | Time → frequency | Reveal which frequencies are present |
| STFT | Windowed FFT | Time-varying frequency content |
| Mel Spectrogram | Perceptual freq scale | Compact, CNN-compatible 2D feature |
| MFCC | DCT of log-Mel | Compact speaker/prosody feature |

---

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 3050 (Laptop) |
| VRAM | 4 GiB |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Python | 3.12.10 |

---

## Novelty

VoiceVault's value is not in any single detection algorithm, but in the
**integration of multiple evidence streams into a security decision workflow**:

- Multi-layer analysis (acoustic + deepfake + speaker + prosody)
- Context-aware risk scoring
- Dynamic risk updating (near-real-time)
- Actionable alerts with evidence explanations
- Transaction prevention (not just detection)

---

## Limitations (Honest Disclosure)

1. **ASVspoof 2019 LA evaluation**: EER measures in-domain performance. Generalisation to novel TTS/VC systems is not guaranteed.
2. **RTX 3050 (4 GiB VRAM)**: Limits batch size and model size. Advanced models (WavLM, AASIST) may require gradient checkpointing.
3. **Multilingual**: Not evaluated yet. Architecture is designed to be language-agnostic, but claims of Indian-language support require actual testing.
4. **Real-time**: Near-real-time prototype. True streaming latency not yet measured.
5. **No production integration**: Banking, telecom, and enterprise integrations are mocked.

---

## References

1. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech", *Computer Speech & Language*, 2020.
2. Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks", *ICASSP*, 2022.
3. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", *IEEE JSTSP*, 2022.
4. Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification", *Interspeech*, 2020.
