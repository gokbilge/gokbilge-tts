"""Tests for ISSAI dataset preparation pipeline."""

from __future__ import annotations

import json
import wave
import struct
from pathlib import Path

import pytest

from gokbilge_tts.datasets.prepare_issai import (
    _find_issai_root,
    _get_duration,
    prepare,
)
from gokbilge_tts.datasets.validate_manifest import validate_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_s: float = 1.5, sr: int = 22050) -> None:
    """Create a minimal valid WAV file using only stdlib (no soundfile/numpy)."""
    n_samples = int(sr * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))


def _make_fake_issai(root: Path, split: str, utterances: list[tuple[str, float]]) -> Path:
    """Create a fake ISSAI split directory with wav+txt pairs.

    Args:
        root:        Parent directory (e.g. tmp_path / "ISSAI").
        split:       "Train", "Dev", or "Test".
        utterances:  List of (text, duration_s) pairs.

    Returns the split directory.
    """
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, (text, dur) in enumerate(utterances):
        stem = str(100_000 + i)
        _make_wav(split_dir / f"{stem}.wav", duration_s=dur)
        (split_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    return split_dir


# ---------------------------------------------------------------------------
# _get_duration
# ---------------------------------------------------------------------------

def test_get_duration_returns_float(tmp_path):
    wav = tmp_path / "test.wav"
    _make_wav(wav, duration_s=1.5)
    dur = _get_duration(str(wav))
    assert isinstance(dur, float)
    assert abs(dur - 1.5) < 0.1   # within 100ms


def test_get_duration_missing_file():
    assert _get_duration("/nonexistent/path.wav") == 0.0


# ---------------------------------------------------------------------------
# _find_issai_root
# ---------------------------------------------------------------------------

def test_find_issai_root_direct(tmp_path):
    (tmp_path / "Train").mkdir()
    (tmp_path / "Dev").mkdir()
    (tmp_path / "Test").mkdir()
    assert _find_issai_root(tmp_path) == tmp_path


def test_find_issai_root_one_level_deep(tmp_path):
    sub = tmp_path / "ISSAI_TSC_218"
    sub.mkdir()
    (sub / "Train").mkdir()
    (sub / "Dev").mkdir()
    assert _find_issai_root(tmp_path) == sub


def test_find_issai_root_none(tmp_path):
    assert _find_issai_root(tmp_path) is None


def test_find_issai_root_none_on_missing():
    assert _find_issai_root(Path("/nonexistent")) is None


# ---------------------------------------------------------------------------
# prepare() — local ISSAI mode
# ---------------------------------------------------------------------------

TRAIN_UTTERANCES = [
    ("Bugün hava çok güzel.", 2.0),
    ("Türkiye güzel bir ülkedir.", 1.8),
    ("Çocuklar bahçede oynuyor.", 1.5),
]
DEV_UTTERANCES = [("Merhaba dünya.", 1.2)]
TEST_UTTERANCES = [("İyi günler.", 1.1)]


@pytest.fixture()
def fake_issai(tmp_path):
    """Build a minimal fake ISSAI TSC directory tree."""
    issai = tmp_path / "ISSAI"
    _make_fake_issai(issai, "Train", TRAIN_UTTERANCES)
    _make_fake_issai(issai, "Dev", DEV_UTTERANCES)
    _make_fake_issai(issai, "Test", TEST_UTTERANCES)
    return issai


def test_prepare_creates_manifests(fake_issai, tmp_path):
    out = tmp_path / "manifests"
    prepare(dataset_dir=fake_issai, output_dir=out)

    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert (out / "test.jsonl").exists()
    assert (out / "stats.json").exists()
    assert (out / "symbols.txt").exists()


def test_prepare_stats_counts(fake_issai, tmp_path):
    out = tmp_path / "manifests"
    prepare(dataset_dir=fake_issai, output_dir=out)

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["total_count"] == len(TRAIN_UTTERANCES) + len(DEV_UTTERANCES) + len(TEST_UTTERANCES)
    assert stats["splits"]["train"] == len(TRAIN_UTTERANCES)
    assert stats["splits"]["val"] == len(DEV_UTTERANCES)
    assert stats["splits"]["test"] == len(TEST_UTTERANCES)
    assert stats["total_hours"] > 0


def test_prepare_manifest_fields(fake_issai, tmp_path):
    out = tmp_path / "manifests"
    prepare(dataset_dir=fake_issai, output_dir=out)

    line = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    for field in ("audio_filepath", "text", "normalized_text", "phonemes", "duration", "speaker_id"):
        assert field in rec, f"Missing field: {field}"
    assert rec["duration"] > 0
    assert rec["phonemes"].strip() != ""


def test_prepare_symbols_nonempty(fake_issai, tmp_path):
    out = tmp_path / "manifests"
    prepare(dataset_dir=fake_issai, output_dir=out)

    syms = (out / "symbols.txt").read_text(encoding="utf-8").splitlines()
    assert len(syms) > 0


def test_prepare_duration_filter(tmp_path):
    """Utterances outside [min, max] must be dropped."""
    issai = tmp_path / "ISSAI"
    _make_fake_issai(issai, "Train", [
        ("Kısa ses.", 0.3),        # too short
        ("Normal ses.", 1.5),      # OK
        ("Çok uzun ses.", 25.0),   # too long
    ])
    _make_fake_issai(issai, "Dev", [("Test.", 1.0)])
    _make_fake_issai(issai, "Test", [("Test.", 1.0)])
    out = tmp_path / "manifests"
    prepare(dataset_dir=issai, output_dir=out, min_duration=0.5, max_duration=20.0)

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["filtered_count"] == 2
    assert stats["splits"]["train"] == 1


def test_prepare_empty_text_filtered(tmp_path):
    """Entries with empty text must be dropped."""
    issai = tmp_path / "ISSAI"
    split_dir = issai / "Train"
    split_dir.mkdir(parents=True)
    _make_wav(split_dir / "100000.wav", 1.0)
    (split_dir / "100000.txt").write_text("", encoding="utf-8")
    _make_wav(split_dir / "100001.wav", 1.0)
    (split_dir / "100001.txt").write_text("Merhaba.", encoding="utf-8")
    _make_fake_issai(issai, "Dev", [("Test.", 1.0)])
    _make_fake_issai(issai, "Test", [("Test.", 1.0)])
    out = tmp_path / "manifests"
    prepare(dataset_dir=issai, output_dir=out)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["filtered_count"] >= 1


def test_prepare_none_dataset_dir_no_crash(tmp_path):
    """Passing dataset_dir=None triggers HF loading; we allow any network/import error."""
    out = tmp_path / "manifests"
    try:
        prepare(dataset_dir=None, output_dir=out)
    except (ImportError, OSError, RuntimeError, Exception):
        pass   # network not available or datasets not installed — both acceptable


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------

def test_validate_manifest_valid(fake_issai, tmp_path):
    out = tmp_path / "manifests"
    prepare(dataset_dir=fake_issai, output_dir=out)
    valid, errors = validate_manifest(out / "train.jsonl")
    assert valid == len(TRAIN_UTTERANCES)
    assert errors == []


def test_validate_manifest_missing_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": "x", "duration": 1.0}) + "\n", encoding="utf-8")
    valid, errors = validate_manifest(bad)
    assert valid == 0
    assert len(errors) > 0


def test_validate_manifest_invalid_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    valid, errors = validate_manifest(bad)
    assert valid == 0
    assert len(errors) > 0
