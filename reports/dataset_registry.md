# VoiceVault — Dataset Registry
# ==============================
# SIH26104 | Phase 1
# 
# ANTI-HALLUCINATION RULE: Never copy published results into this document.
# Only record actual facts about dataset structure and licensing.
# All experimental results go in experiment_log.md — NOT here.

---

## Primary Anti-Spoofing Dataset

### ASVspoof 2019 — Logical Access (LA)

| Field | Detail |
|-------|--------|
| **Full name** | ASVspoof 2019 — Logical Access subset |
| **Status** | ⏳ NOT DOWNLOADED — download pending |
| **Purpose** | Primary training/evaluation dataset for baseline deepfake detector |
| **Source** | https://datasharing.ed.ac.uk/ (Edinburgh DataShare) |
| **License** | CC BY 4.0 — free for research with attribution |
| **Paper** | Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech", Computer Speech & Language, 2020 |
| **Official protocol** | train / dev / eval splits (speaker-disjoint) |

#### Structure
```
ASVspoof2019_LA_/
├── ASVspoof2019_LA_train/
│   ├── flac/         # audio files (.flac, 16 kHz mono)
│   └── ASVspoof2019.LA.cm.train.trn.txt  # labels: file_id, speaker, env, attack_type, label
├── ASVspoof2019_LA_dev/
│   ├── flac/
│   └── ASVspoof2019.LA.cm.dev.trl.txt
└── ASVspoof2019_LA_eval/
    ├── flac/
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

#### Label format
Each line: `SPEAKER_ID FILE_ID - ATTACK_TYPE LABEL`  
- LABEL: `bonafide` or `spoof`
- ATTACK_TYPE for bonafide: `-`
- ATTACK_TYPE for spoof: `A01` through `A19`

#### Attack Types (Logical Access)
| Code | Type | Description |
|------|------|-------------|
| A01 | TTS | Neural waveform model |
| A02 | TTS | Statistical parametric |
| A03 | TTS | Neural |
| A04 | TTS | Waveform concatenation |
| A05 | TTS | Neural waveform |
| A06 | VC | Voice conversion |
| (A07–A19) | TTS/VC | Various synthesis and conversion systems |

> [!IMPORTANT]
> **Data Leakage Policy**: ASVspoof 2019 LA uses SPEAKER-DISJOINT splits.
> Speakers in train are NOT in dev or eval.  
> Attack generators seen in train MAY appear in eval (same attack types).  
> Generalisation to UNSEEN attack types is a known limitation.

#### Known Limitations
1. All attacks are Logical Access (TTS/VC) — no physical access (replay) attacks in LA subset.
2. Training and eval sets contain the same attack types → the eval EER measures intra-domain performance, not generalisation to novel attacks.
3. Speaker population is mostly English — limited multilingual coverage.
4. Does not include telephone-quality (8 kHz narrowband) samples.

#### Download Instructions (TODO)
```bash
# 1. Register at: https://datasharing.ed.ac.uk/
# 2. Accept the license agreement for ASVspoof2019
# 3. Download:
#    - LA_train.zip
#    - LA_dev.zip  
#    - LA_eval.zip
# 4. Extract to: data/asvspoof2019/LA/
# 5. Verify with: python scripts/verify_dataset.py
```

---

## Candidate Supplementary Datasets

### WaveFake
| Field | Detail |
|-------|--------|
| **Status** | 🔜 CANDIDATE — not downloaded |
| **Source** | https://github.com/RUB-SysSec/WaveFake |
| **License** | MIT |
| **Purpose** | Additional spoof samples; different generators from ASVspoof 2019 |
| **Samples** | ~117,985 generated audio samples |
| **Generators** | MelGAN, Parallel WaveGAN, HiFi-GAN, WaveGlow, WaveNet, Full Band MelGAN |
| **Bonafide source** | LJSpeech (English, single speaker) |

> [!WARNING]
> WaveFake is single-speaker (LJSpeech). Speaker identity is not disjoint from
> the generator. Cross-evaluation must account for this.

### In-the-Wild Audio Deepfake Dataset
| Field | Detail |
|-------|--------|
| **Status** | 🔜 CANDIDATE — not downloaded |
| **Source** | https://deepfake-demo.aisec.fraunhofer.de/ |
| **License** | Research use |
| **Purpose** | Real-world deepfake samples (not lab-synthesized) |

---

## Data Usage Policy

1. **Raw audio is NEVER committed to git** (see .gitignore).
2. **Training data** → `data/asvspoof2019/LA/`
3. **Processed features** → `data/processed/` (numpy arrays)
4. **Demo audio** → `data/demo/genuine/` and `data/demo/spoof/`
   - Demo audio must come from a separately sourced, licensed set.
   - Demo audio must NOT come from ASVspoof 2019 train set.
   - Demo audio must NOT be used for training or evaluation.
5. **Metadata/labels** → `data/metadata/`

---

## Anti-Hallucination Reminder
All statistics in this file are sourced from the official dataset papers.  
VoiceVault experimental results (accuracy, EER, F1, etc.) are recorded in  
`reports/experiment_log.md` and are only entered AFTER actual experiments.
