"""Manifest validation utilities."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = {
    "audio_filepath",
    "text",
    "normalized_text",
    "phonemes",
    "duration",
    "speaker_id",
}

_MIN_DURATION = 0.5
_MAX_DURATION = 20.0


def validate_manifest(
    manifest_path: Path,
    check_audio: bool = True,
    min_duration: float = _MIN_DURATION,
    max_duration: float = _MAX_DURATION,
) -> tuple[int, list[str]]:
    """Validate a JSONL manifest file.

    Checks performed per record:
    - Valid JSON
    - All required fields present
    - audio_filepath exists on disk (when check_audio=True)
    - duration within [min_duration, max_duration]
    - text and phonemes are non-empty

    Args:
        manifest_path:  Path to the JSONL file.
        check_audio:    Whether to verify audio_filepath existence.
        min_duration:   Minimum valid duration in seconds.
        max_duration:   Maximum valid duration in seconds.

    Returns:
        (valid_count, list_of_error_strings)
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
                errors.append(f"line {i}: missing fields: {sorted(missing)}")
                continue

            if check_audio:
                audio_path = entry["audio_filepath"]
                if not Path(audio_path).exists():
                    errors.append(f"line {i}: audio not found: {audio_path}")
                    continue

            dur = entry["duration"]
            if not (min_duration <= dur <= max_duration):
                errors.append(
                    f"line {i}: duration {dur:.3f}s outside [{min_duration}, {max_duration}]"
                )
                continue

            if not str(entry["text"]).strip():
                errors.append(f"line {i}: empty text")
                continue

            if not str(entry["phonemes"]).strip():
                errors.append(f"line {i}: empty phonemes")
                continue

            valid += 1

    return valid, errors
