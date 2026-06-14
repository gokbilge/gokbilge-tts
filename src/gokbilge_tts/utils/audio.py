"""Audio utility functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wav(path: Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load a WAV file and optionally resample.

    Returns:
        (audio_array, sample_rate)
    """
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    # TODO: resample if sr != target_sr using torchaudio or librosa
    return audio, sr


def save_wav(audio: np.ndarray, path: Path, sr: int = 22050) -> None:
    """Save audio array as WAV."""
    import soundfile as sf
    sf.write(str(path), audio, sr)


def get_duration(path: Path) -> float:
    """Return audio duration in seconds."""
    import soundfile as sf
    info = sf.info(str(path))
    return info.duration
