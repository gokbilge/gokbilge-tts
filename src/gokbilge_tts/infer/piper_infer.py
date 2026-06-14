"""Piper-compatible ONNX inference.

Piper is a fast, local neural TTS system that uses VITS exported to ONNX.
See: https://github.com/rhasspy/piper

TODO:
    - Load .onnx model + .json config
    - Run text normalization + G2P
    - Run ONNX inference
    - Write output WAV
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def synthesize(
    text: str,
    model_path: Path,
    config_path: Path,
    output_path: Path,
    speaker_id: Optional[int] = None,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
) -> None:
    """Synthesize speech using a Piper-compatible ONNX model.

    Args:
        text: Normalized input text.
        model_path: Path to .onnx model file.
        config_path: Path to Piper JSON config file.
        output_path: Output WAV path.
        speaker_id: Optional speaker ID for multi-speaker models.
        length_scale: Speech rate control (>1 = slower).
        noise_scale: Variation in phoneme duration.
        noise_scale_w: Variation in prosody.

    TODO:
        - Load onnxruntime session
        - Parse phonemes from config phoneme_type
        - Run inference
        - Save WAV at config sample_rate
    """
    raise NotImplementedError("Piper ONNX inference not yet implemented")
