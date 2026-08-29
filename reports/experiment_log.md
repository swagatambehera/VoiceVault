# VoiceVault — Experiment Log
# ============================
# SIH26104
#
# ANTI-HALLUCINATION RULE:
# Every row in this table must correspond to an ACTUAL experiment.
# Do not enter results that have not been measured.
# Do not copy results from papers or other implementations.
# If an experiment has not been run, leave the cell blank or write NOT RUN.

---

## Experiment Template

```
## EXP-{id}: {short description}

| Field | Value |
|-------|-------|
| Date | YYYY-MM-DD |
| Dataset | Dataset name, split (train/dev/eval) |
| Model | Model name + version |
| Input features | e.g., Log-Mel (80×188), MFCC (40×T) |
| Sampling rate | Hz |
| n_fft | |
| hop_length | |
| n_mels | (if applicable) |
| Batch size | |
| Learning rate | |
| Optimizer | |
| Scheduler | |
| Epochs | |
| Random seed | |
| Hardware | |
| Training time | |

### Results
| Metric | Value |
|--------|-------|
| Accuracy | NOT RUN |
| Precision | NOT RUN |
| Recall | NOT RUN |
| F1 | NOT RUN |
| ROC-AUC | NOT RUN |
| EER | NOT RUN |
| Inference latency (ms/sample) | NOT RUN |

### Notes
- 
```

---

## Experiments

*No experiments run yet. Results will be populated after dataset download
and baseline CNN training (Phase 5).*

---

## Phase 1 Verification Results (Pipeline Smoke Tests)

Date: 2026-08-29  
Hardware: NVIDIA RTX 3050 (4 GiB VRAM) | Python 3.12.10 | Windows  
PyTorch: NOT YET INSTALLED (pending manual download — 2.75 GB wheel)  
All tests run with scipy/soundfile/librosa fallbacks (no torchaudio).

| Test | Status | Actual Output |
|------|--------|---------------|
| `python -m src.audio` | PASSED | Sine wave saved + reloaded: shape=(16000,), sr=16000, min=-0.5000, max=0.5000 |
| `python -m src.preprocessing` | PASSED | Stereo 44100Hz -> mono 16000Hz: shape=(48000,), peak norm: min=-1.0000, max=1.0000, 6 chunks |
| `python -m src.features` | PASSED | FFT shape=(24001,), STFT=(513,188), Mel=(80,188), CNN=(1,80,188), MFCC=(120,188), Filterbank=(80,513) |
| `pytest tests/test_audio_pipeline.py -v` | PASSED | **34/34 tests passed in 2.56s** |

### features.py verified outputs
| Feature | Shape | Value range |
|---------|-------|-------------|
| FFT magnitude spectrum | (24001,) | Peaks at 300 Hz, 1000 Hz (confirmed) |
| Power Spectrogram (STFT) | (513, 188) | freq_bins=513, time_frames=188 |
| Log-Mel Spectrogram | (80, 188) | Min=-80.0 dB, Max=0.0 dB |
| CNN-ready Mel | (1, 80, 188) | 1 channel, 80 mels, 188 frames |
| MFCC + delta + delta-delta | (120, 188) | 3x40=120 features per frame |
| Mel Filter Bank (manual) | (80, 513) | Min=0.0, Max=0.044505 (non-negative) |

Visualization: `reports/features_selftest.png` (generated)
