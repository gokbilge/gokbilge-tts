"""Path helpers for datasets, checkpoints, and outputs."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def get_data_dir() -> Path:
    """Return data directory, respecting GOKBILGE_DATA_DIR env var."""
    env = os.environ.get("GOKBILGE_DATA_DIR")
    if env:
        return Path(env)
    return get_project_root() / "data"


def get_checkpoint_dir() -> Path:
    """Return checkpoint directory, respecting GOKBILGE_CHECKPOINT_DIR env var."""
    env = os.environ.get("GOKBILGE_CHECKPOINT_DIR")
    if env:
        return Path(env)
    return get_project_root() / "checkpoints"


def get_output_dir() -> Path:
    """Return output directory for generated audio."""
    return get_project_root() / "output"


def ensure_dir(path: Path) -> Path:
    """Create directory and parents if they don't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
