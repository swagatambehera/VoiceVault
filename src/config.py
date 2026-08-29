"""
src/config.py — VoiceVault Configuration Loader
================================================
SIH26104 | Phase 1

Loads configs/config.yaml and provides typed access to all parameters.
Every module that needs a configurable value imports from here.

WHY A CENTRAL CONFIG LOADER?
-----------------------------
Without this, every module would either:
  1. Hardcode values (violates Section 39 principle #5), or
  2. Re-parse the YAML independently (wastes I/O, risks inconsistency).

All thresholds, hyperparameters, and file paths are defined ONCE in
configs/config.yaml and accessed via this module.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default path relative to repository root.
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and cache the VoiceVault configuration from YAML.

    The config is loaded once and cached — subsequent calls return the
    same object without re-reading the file.  To reload (e.g., in tests),
    call load_config.cache_clear() first.

    INPUT
    -----
    config_path : path to config YAML (default: configs/config.yaml)

    OUTPUT
    ------
    dict containing all configuration values
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected at: {DEFAULT_CONFIG_PATH}"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Loaded config from: %s", path)
    return config


def get_audio_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the 'audio' section of the config."""
    return load_config(config_path)["audio"]


def get_features_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the 'features' section of the config."""
    return load_config(config_path)["features"]


def get_vad_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the 'vad' section of the config."""
    return load_config(config_path)["vad"]


def get_model_config(model_name: str = "baseline_cnn", config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return a model-specific config section."""
    return load_config(config_path)["model"][model_name]


def get_risk_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the 'risk' section of the config."""
    return load_config(config_path)["risk"]


def get_speaker_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the 'speaker' section of the config."""
    return load_config(config_path)["speaker"]


def get_data_paths(config_path: Optional[str] = None) -> Dict[str, Path]:
    """Return data paths as Path objects (resolved relative to repo root)."""
    repo_root = Path(__file__).parent.parent
    raw_paths = load_config(config_path)["data"]
    return {k: repo_root / v for k, v in raw_paths.items()}


# ──────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)
    cfg = load_config()
    print("Configuration loaded successfully:")
    print(json.dumps(cfg, indent=2, default=str))
