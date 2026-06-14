"""Prepare ISSAI Turkish Speech Corpus for TTS training.

Dataset: issai/Turkish_Speech_Corpus
HuggingFace: https://huggingface.co/datasets/issai/Turkish_Speech_Corpus

Expected manifest format (one JSON object per line, JSONL):

    {
        "audio_filepath": "/path/to/audio.wav",
        "text": "Bugün hava çok güzel.",
        "normalized_text": "Bugün hava çok güzel.",
        "phonemes": "b u g y n _ h a v a ~ t ʃ o k _ g y z e l ~",
        "duration": 2.34,
        "speaker_id": "tr_f_001"
    }

Usage:
    from gokbilge_tts.datasets.prepare_issai import prepare

    prepare(
        dataset_dir=Path("./data/issai"),
        output_dir=Path("./data/manifests"),
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare(dataset_dir: Path, output_dir: Path) -> None:
    """Prepare ISSAI dataset manifests.

    TODO:
        - Load dataset from local path or HuggingFace hub:
              from datasets import load_dataset
              ds = load_dataset("issai/Turkish_Speech_Corpus")
        - Validate audio files exist and are readable
        - Validate audio sample rate (target: 22050 Hz or 16000 Hz)
        - Apply normalize_text() to each transcript
        - Apply text_to_phonemes() to each normalized transcript
        - Compute audio duration with soundfile or librosa
        - Write train/val/test JSONL manifests
        - Log statistics: total hours, speaker counts, duration histogram
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement actual preparation
    print(f"[prepare_issai] TODO: load from {dataset_dir}")
    print(f"[prepare_issai] TODO: write manifests to {output_dir}")

    # Placeholder: write an empty manifest with schema comment
    example: dict[str, Any] = {
        "_schema": "gokbilge-tts manifest v1",
        "audio_filepath": "path/to/audio.wav",
        "text": "Bugün hava çok güzel.",
        "normalized_text": "Bugün hava çok güzel.",
        "phonemes": "b u g y n _ h a v a ~ t ʃ o k _ g y z e l ~",
        "duration": 2.34,
        "speaker_id": "tr_f_001",
    }
    out_file = output_dir / "train_manifest.jsonl.example"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"[prepare_issai] Example manifest schema written to {out_file}")


def validate_audio(path: Path, target_sr: int = 22050) -> bool:
    """Check that audio file exists and has the expected sample rate.

    TODO: implement with soundfile.
    """
    # TODO
    return path.exists()
