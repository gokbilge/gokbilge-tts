"""VITS PyTorch inference.

TODO:
    - Load VITS checkpoint
    - Run G2P + symbol lookup
    - Run VITS generator
    - Return or save audio
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def synthesize(
    text: str,
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    speaker_id: Optional[int] = None,
) -> np.ndarray:
    """Synthesize speech using a VITS PyTorch model.

    Args:
        text: Normalized input text.
        checkpoint_path: Path to VITS .pth checkpoint.
        config_path: Path to VITS YAML config.
        output_path: Output WAV path (or None to return array only).
        speaker_id: Optional speaker ID.

    Returns:
        Audio as numpy array (float32, mono).

    TODO:
        - Load VITS model from checkpoint
        - Tokenize via symbols.py
        - Run generator network
        - Save WAV via soundfile
    """
    raise NotImplementedError("VITS inference not yet implemented")
