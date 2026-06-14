"""Manifest validation utilities."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = {"audio_filepath", "text", "normalized_text", "duration"}


def validate_manifest(manifest_path: Path) -> tuple[int, list[str]]:
    """Validate a JSONL manifest file.

    Returns:
        (valid_count, list_of_errors)

    TODO:
        - Check all required fields present
        - Check audio files exist
        - Check duration > 0 and < max_duration
        - Check text is non-empty
        - Check phonemes field if present
        - Report total hours, missing files, invalid entries
    """
    errors: list[str] = []
    valid = 0

    with open(manifest_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: JSON error: {e}")
                continue

            missing = REQUIRED_FIELDS - set(entry.keys())
            if missing:
                errors.append(f"line {i}: missing fields: {missing}")
                continue

            # TODO: check audio file existence
            valid += 1

    return valid, errors
