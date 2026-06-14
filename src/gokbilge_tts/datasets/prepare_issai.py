"""Prepare ISSAI Turkish Speech Corpus for TTS training.

Supports two loading modes:
  - Local ISSAI TSC directory with Train/Dev/Test subdirs of N.wav + N.txt pairs
  - HuggingFace dataset (issai/Turkish_Speech_Corpus) as fallback

Outputs per split:
  {out}/train.jsonl  val.jsonl  test.jsonl  stats.json  symbols.txt

Usage:
    gokbilge-tts prepare-issai --dataset-dir /path/to/ISSAI --out ./data/manifests

    from gokbilge_tts.datasets.prepare_issai import prepare
    prepare(Path("/path/to/ISSAI"), Path("./data/manifests"))
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from gokbilge_tts.g2p.turkish import text_to_phonemes
from gokbilge_tts.normalize.text import normalize_text
from gokbilge_tts.utils.logging import get_logger

log = get_logger(__name__)

_HF_NAME = "issai/Turkish_Speech_Corpus"
_MIN_DUR = 0.5
_MAX_DUR = 20.0

# ISSAI TSC on-disk split directory names (multiple candidates, case varies)
_SPLIT_DIRS: dict[str, tuple[str, ...]] = {
    "train": ("Train", "train"),
    "val":   ("Dev",   "dev", "Val", "val"),
    "test":  ("Test",  "test"),
}


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _get_duration(wav_path: str) -> float:
    """Return WAV duration in seconds. Tries soundfile first, then stdlib wave."""
    try:
        import soundfile as sf  # type: ignore
        return sf.info(wav_path).duration
    except ImportError:
        pass
    except Exception as exc:
        log.debug("soundfile could not read %s: %s", wav_path, exc)
        return 0.0

    import wave as wave_mod
    try:
        with wave_mod.open(wav_path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception as exc:
        log.debug("wave could not read %s: %s", wav_path, exc)
        return 0.0


def validate_audio(path: Path, target_sr: int = 22050) -> bool:
    """Return True if the WAV exists at the expected sample rate."""
    try:
        import soundfile as sf  # type: ignore
        return sf.info(str(path)).samplerate == target_sr
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local ISSAI TSC loader
# ---------------------------------------------------------------------------

def _find_issai_root(dataset_dir: Path | None) -> Path | None:
    """Return the directory that has Train/Dev/Test subdirs, or None."""
    if dataset_dir is None or not dataset_dir.exists():
        return None

    split_names = {"Train", "Dev", "Test", "train", "dev", "test"}

    # Check the directory itself
    if any((dataset_dir / d).exists() for d in split_names):
        return dataset_dir

    # Check one level of subdirectories (e.g. ISSAI/ISSAI_TSC_218/)
    for sub in dataset_dir.iterdir():
        if sub.is_dir() and any((sub / d).exists() for d in split_names):
            return sub

    return None


def _resolve_split_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        d = root / name
        if d.is_dir():
            return d
    return None


def _process_wav_txt_dir(
    split_dir: Path,
    min_dur: float,
    max_dur: float,
) -> tuple[list[dict[str, Any]], int]:
    """Scan N.wav + N.txt pairs; return (records, filtered_count)."""
    records: list[dict[str, Any]] = []
    filtered = 0

    stems: dict[str, str] = {}
    with os.scandir(split_dir) as it:
        for entry in it:
            if entry.name.endswith(".wav"):
                stems[entry.name[:-4]] = entry.path

    for stem in sorted(stems):
        wav_path = stems[stem]
        txt_path = Path(wav_path).with_suffix(".txt")

        if not txt_path.exists():
            filtered += 1
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            filtered += 1
            continue

        duration = _get_duration(wav_path)
        if not (min_dur <= duration <= max_dur):
            filtered += 1
            continue

        normalized = normalize_text(text)
        phonemes = text_to_phonemes(normalized)

        if not normalized.strip() or not phonemes.strip():
            filtered += 1
            continue

        records.append({
            "audio_filepath": wav_path,
            "text": text,
            "normalized_text": normalized,
            "phonemes": phonemes,
            "duration": round(duration, 4),
            "speaker_id": "issai",
        })

    return records, filtered


# ---------------------------------------------------------------------------
# HuggingFace fallback loader
# ---------------------------------------------------------------------------

def _extract_audio_info(item: dict[str, Any]) -> tuple[str, float]:
    audio = item.get("audio", "")
    if isinstance(audio, dict):
        path = str(audio.get("path") or "")
        array = audio.get("array")
        sr = audio.get("sampling_rate") or 16000
        if path and Path(path).exists():
            return path, _get_duration(path)
        if array is not None:
            return path, len(array) / sr
        return path, 0.0
    path = str(audio)
    return path, _get_duration(path) if path else 0.0


def _extract_text(item: dict[str, Any]) -> str:
    for field in ("transcription", "text", "sentence", "transcript"):
        v = item.get(field, "")
        if v:
            return str(v).strip()
    return ""


def _extract_speaker(item: dict[str, Any]) -> str:
    for field in ("speaker_id", "speaker", "client_id"):
        v = item.get(field)
        if v:
            return str(v).strip()
    return "unknown"


def _iter_hf(hf_name: str, cache_dir: Path | None) -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc
    return load_dataset(
        hf_name,
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("  %-12s  %d records", path.name, len(records))


def prepare(
    dataset_dir: Path | None,
    output_dir: Path,
    hf_dataset: str = _HF_NAME,
    split_ratios: tuple[float, float, float] = (0.95, 0.025, 0.025),
    min_duration: float = _MIN_DUR,
    max_duration: float = _MAX_DUR,
    seed: int = 42,
) -> None:
    """Prepare ISSAI manifests for TTS training.

    Args:
        dataset_dir:  Path to local ISSAI TSC directory, or None to load from HuggingFace.
        output_dir:   Directory to write train/val/test.jsonl, stats.json, symbols.txt.
        hf_dataset:   HuggingFace dataset name used when dataset_dir is None.
        split_ratios: (train, val, test) fractions; only used for HuggingFace mode.
        min_duration: Drop utterances shorter than this (seconds).
        max_duration: Drop utterances longer than this (seconds).
        seed:         Random seed for HuggingFace-mode shuffle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {}
    total_filtered = 0

    issai_root = _find_issai_root(dataset_dir)

    # ---- Local ISSAI TSC ----
    if issai_root is not None:
        log.info("ISSAI TSC root: %s", issai_root)
        for split_name, candidates in _SPLIT_DIRS.items():
            split_dir = _resolve_split_dir(issai_root, candidates)
            if split_dir is None:
                log.warning("No %s split dir in %s — skipping", split_name, issai_root)
                continue
            log.info("Processing %s …", split_dir)
            recs, filt = _process_wav_txt_dir(split_dir, min_duration, max_duration)
            total_filtered += filt
            splits[split_name] = recs
            log.info("  %d records, %d filtered", len(recs), filt)

    # ---- HuggingFace fallback ----
    else:
        log.info("Loading from HuggingFace: %s", hf_dataset)
        all_recs: list[dict[str, Any]] = []
        filt = 0
        for item in _iter_hf(hf_dataset, dataset_dir):
            audio_path, duration = _extract_audio_info(item)
            text = _extract_text(item)
            speaker = _extract_speaker(item)
            if not text or not audio_path:
                filt += 1
                continue
            if not (min_duration <= duration <= max_duration):
                filt += 1
                continue
            normalized = normalize_text(text)
            phonemes = text_to_phonemes(normalized)
            if not normalized.strip() or not phonemes.strip():
                filt += 1
                continue
            all_recs.append({
                "audio_filepath": audio_path,
                "text": text,
                "normalized_text": normalized,
                "phonemes": phonemes,
                "duration": round(duration, 4),
                "speaker_id": speaker,
            })
        total_filtered = filt
        log.info("%d records loaded, %d filtered", len(all_recs), filt)

        rng = random.Random(seed)
        rng.shuffle(all_recs)
        n = len(all_recs)
        n_val = max(1, int(n * split_ratios[1]))
        n_test = max(1, int(n * split_ratios[2]))
        splits["train"] = all_recs[: n - n_val - n_test]
        splits["val"] = all_recs[n - n_val - n_test : n - n_test]
        splits["test"] = all_recs[n - n_test :]

    if not splits:
        log.error("No data produced. Check --dataset-dir path or HuggingFace access.")
        return

    # ---- Write manifests ----
    for name, recs in splits.items():
        _write_jsonl(recs, output_dir / f"{name}.jsonl")

    # ---- Symbol inventory ----
    all_flat = [r for recs in splits.values() for r in recs]
    symbols: set[str] = set()
    for rec in all_flat:
        symbols.update(rec["phonemes"].split())
    (output_dir / "symbols.txt").write_text(
        "\n".join(sorted(symbols)) + "\n", encoding="utf-8"
    )
    log.info("Symbols: %d → %s/symbols.txt", len(symbols), output_dir)

    # ---- Stats ----
    total_dur = sum(r["duration"] for r in all_flat)
    speakers = sorted({r["speaker_id"] for r in all_flat})
    stats: dict[str, Any] = {
        "total_hours": round(total_dur / 3600, 3),
        "total_count": len(all_flat),
        "filtered_count": total_filtered,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "duration_filter": {"min_s": min_duration, "max_s": max_duration},
        "splits": {name: len(recs) for name, recs in splits.items()},
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info(
        "Done — %.1fh | %d total | %d filtered | %d speaker(s)",
        stats["total_hours"], len(all_flat), total_filtered, len(speakers),
    )
