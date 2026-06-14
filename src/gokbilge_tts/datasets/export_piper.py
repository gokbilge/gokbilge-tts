"""Convert canonical Gokbilge JSONL manifests to Piper/LJSpeech training format.

Input:  manifest_dir/ containing train.jsonl, val.jsonl, (test.jsonl), symbols.txt
Output: out_dir/ containing:
    wavs/           — symlinks (or copies on Windows) to source audio
    metadata.csv    — LJSpeech: stem|text  (or stem|speaker|text for multi-speaker)
    config.json     — Piper training config skeleton (phoneme_id_map filled by preprocess)
    train.txt       — stems for training split
    val.txt         — stems for val split
    test.txt        — stems for test split

Smoke mode (--limit N):
    train: first N records
    val:   first min(10, available) records
    test:  first min(10, available) records
    This keeps val/test representative while capping the total dataset size.

Next step after this tool:
    python3 -m piper_train.preprocess \
        --language tr --input-dir <out_dir> --output-dir <training_dir> \
        --dataset-format ljspeech --single-speaker --sample-rate 22050
    Note: preprocess writes the final config.json (with phoneme_id_map) to <training_dir>.
    Use <training_dir>/config.json — not <out_dir>/config.json — when exporting ONNX.

Usage:
    gokbilge-tts export-piper --manifest-dir ./data/manifests --out ./data/piper
    gokbilge-tts export-piper --manifest-dir ./data/manifests --out ./data/piper --limit 100
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from gokbilge_tts.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_SAMPLE_RATE = 22050
_DEFAULT_LANGUAGE = "tr"
_PIPER_NUM_SYMBOLS = 256
_SMOKE_VAL_LIMIT = 10   # max val/test records when --limit is active
_PIPER_INFERENCE_DEFAULTS = {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8,
}


def _read_jsonl(path: Path, limit: Optional[int] = None) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping invalid JSON line in %s", path.name)
            if limit is not None and len(records) >= limit:
                break
    return records


def _link_or_copy(src: str, dst: Path) -> None:
    """Symlink dst → src; fall back to copy if symlinks are unavailable (Windows)."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            log.warning("Cannot link or copy %s: %s", dst.name, exc)


def _unique_stem(audio_filepath: str, seen: dict[str, str]) -> str:
    """Return a filesystem-safe stem unique within this export."""
    stem = Path(audio_filepath).stem
    if stem not in seen or seen[stem] == audio_filepath:
        seen[stem] = audio_filepath
        return stem
    i = 2
    while f"{stem}_{i}" in seen:
        i += 1
    new_stem = f"{stem}_{i}"
    seen[new_stem] = audio_filepath
    return new_stem


def export_piper(
    manifest_dir: Path,
    out_dir: Path,
    limit: Optional[int] = None,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    language: str = _DEFAULT_LANGUAGE,
) -> None:
    """Convert Gokbilge JSONL manifests to Piper LJSpeech training format.

    Args:
        manifest_dir: Directory with train.jsonl, val.jsonl, symbols.txt.
        out_dir:      Destination directory (created if absent).
        limit:        Cap training records at this count (smoke mode).
        sample_rate:  Target audio sample rate Hz.
        language:     espeak-ng language code written into config.json.
    """
    manifest_dir = Path(manifest_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wavs").mkdir(exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    # In smoke mode (limit set): cap val/test at SMOKE_VAL_LIMIT so they stay
    # proportional but don't balloon when the full corpus has thousands of records.
    side_limit = _SMOKE_VAL_LIMIT if limit is not None else None
    split_recs: dict[str, list[dict]] = {
        "train": _read_jsonl(manifest_dir / "train.jsonl", limit=limit),
        "val":   _read_jsonl(manifest_dir / "val.jsonl",   limit=side_limit),
        "test":  _read_jsonl(manifest_dir / "test.jsonl",  limit=side_limit),
    }

    all_recs = [r for recs in split_recs.values() for r in recs]
    if not all_recs:
        log.error("No records loaded from %s", manifest_dir)
        return

    log.info(
        "Loaded splits: train=%d, val=%d, test=%d",
        len(split_recs["train"]), len(split_recs["val"]), len(split_recs["test"]),
    )

    # ── Speaker map ──────────────────────────────────────────────────────────
    speakers = sorted({r.get("speaker_id", "default") for r in all_recs})
    speaker_id_map = {s: i for i, s in enumerate(speakers)}
    is_multispeaker = len(speakers) > 1

    # ── metadata.csv + wavs/ links + split stem lists ────────────────────────
    csv_rows: list[str] = []
    stem_to_split: dict[str, str] = {}
    seen_stems: dict[str, str] = {}

    for split_name, recs in split_recs.items():
        for rec in recs:
            audio_path = rec.get("audio_filepath", "")
            text = rec.get("normalized_text") or rec.get("text", "")
            speaker = rec.get("speaker_id", "default")
            stem = _unique_stem(audio_path, seen_stems)

            if audio_path:
                _link_or_copy(audio_path, out_dir / "wavs" / f"{stem}.wav")

            if is_multispeaker:
                csv_rows.append(f"{stem}|{speaker}|{text}")
            else:
                csv_rows.append(f"{stem}|{text}")

            stem_to_split[stem] = split_name

    (out_dir / "metadata.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    log.info("Wrote metadata.csv (%d rows)", len(csv_rows))

    for split_name in ("train", "val", "test"):
        stems = [s for s, sp in stem_to_split.items() if sp == split_name]
        (out_dir / f"{split_name}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    # ── config.json ──────────────────────────────────────────────────────────
    config: dict = {
        "audio": {
            "sample_rate": sample_rate,
        },
        "espeak": {
            "language": language,
            "voice": language,
        },
        "inference": _PIPER_INFERENCE_DEFAULTS.copy(),
        "phoneme_type": "espeak",
        "num_speakers": len(speakers),
        "num_symbols": _PIPER_NUM_SYMBOLS,
        "phoneme_id_map": {},
        "speaker_id_map": speaker_id_map if is_multispeaker else {},
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Wrote config.json")

    log.info(
        "Piper export done: %d utterances | %d speaker(s) | sr=%d | lang=%s → %s",
        len(all_recs), len(speakers), sample_rate, language, out_dir,
    )


def validate_piper_export(piper_dir: Path) -> tuple[int, list[str]]:
    """Validate the output directory of export_piper.

    Checks:
    - metadata.csv and config.json exist
    - wavs/ directory exists
    - each row in metadata.csv has ≥2 columns (stem + text)
    - each stem's wav file is present under wavs/
    - no row has an empty text field

    Args:
        piper_dir: Directory written by export_piper().

    Returns:
        (valid_count, list_of_error_strings)
    """
    piper_dir = Path(piper_dir)
    errors: list[str] = []
    valid = 0

    for required in ("metadata.csv", "config.json"):
        if not (piper_dir / required).exists():
            errors.append(f"missing: {required}")

    wavs_dir = piper_dir / "wavs"
    if not wavs_dir.is_dir():
        errors.append("missing: wavs/ directory")

    if errors:
        return 0, errors

    metadata_path = piper_dir / "metadata.csv"
    for i, raw in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            errors.append(f"metadata.csv line {i}: too few columns")
            continue

        stem = parts[0].strip()
        text = parts[-1].strip()

        if not stem:
            errors.append(f"metadata.csv line {i}: empty stem")
            continue

        if not text:
            errors.append(f"metadata.csv line {i}: empty text for stem {stem!r}")
            continue

        wav_path = wavs_dir / f"{stem}.wav"
        if not wav_path.exists():
            errors.append(f"metadata.csv line {i}: wavs/{stem}.wav not found")
            continue

        valid += 1

    return valid, errors
