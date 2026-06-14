"""Tests for export_piper and hardened validate_manifest."""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from gokbilge_tts.datasets.export_piper import export_piper
from gokbilge_tts.datasets.validate_manifest import validate_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_s: float = 1.5, sr: int = 22050) -> None:
    n = int(sr * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture()
def manifest_dir(tmp_path):
    """5 fake records split 3/1/1 across train/val/test."""
    wav_src = tmp_path / "wav_src"
    wav_src.mkdir()

    records = []
    for i in range(5):
        wav = wav_src / f"utt_{i:04d}.wav"
        _make_wav(wav, 1.5)
        records.append({
            "audio_filepath": str(wav),
            "text": f"Örnek metin {i}.",
            "normalized_text": f"örnek metin {i}",
            "phonemes": "ø r n ɛ k",
            "duration": 1.5,
            "speaker_id": "issai",
        })

    mdir = tmp_path / "manifests"
    mdir.mkdir()
    _write_jsonl(mdir / "train.jsonl", records[:3])
    _write_jsonl(mdir / "val.jsonl", records[3:4])
    _write_jsonl(mdir / "test.jsonl", records[4:5])
    (mdir / "symbols.txt").write_text("ø\nr\nn\nɛ\nk\n", encoding="utf-8")
    return mdir


# ---------------------------------------------------------------------------
# export_piper — file creation
# ---------------------------------------------------------------------------

def test_export_creates_metadata_csv(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out)
    assert (out / "metadata.csv").exists()
    lines = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # 3 train + 1 val + 1 test


def test_export_creates_config_json(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["audio"]["sample_rate"] == 22050
    assert cfg["espeak"]["language"] == "tr"
    assert cfg["num_speakers"] == 1
    assert "inference" in cfg
    assert "phoneme_id_map" in cfg


def test_export_creates_split_txt_files(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out)
    assert (out / "train.txt").exists()
    assert (out / "val.txt").exists()
    train_stems = (out / "train.txt").read_text(encoding="utf-8").splitlines()
    val_stems = (out / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_stems) == 3
    assert len(val_stems) == 1


def test_export_creates_wavs_dir(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out)
    assert (out / "wavs").is_dir()
    wav_files = list((out / "wavs").iterdir())
    assert len(wav_files) == 5


def test_export_metadata_csv_format_single_speaker(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out)
    first_line = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()[0]
    parts = first_line.split("|")
    assert len(parts) == 2, "Single-speaker: stem|text (2 columns)"
    stem, text = parts
    assert stem != ""
    assert text != ""


def test_export_multispeaker_uses_3_columns(tmp_path):
    wav_src = tmp_path / "wav_src"
    wav_src.mkdir()
    records = []
    for i, spk in enumerate(["spk_a", "spk_b"]):
        wav = wav_src / f"utt_{i}.wav"
        _make_wav(wav, 1.5)
        records.append({
            "audio_filepath": str(wav),
            "text": f"Metin {i}.",
            "normalized_text": f"metin {i}",
            "phonemes": "m ɛ t i n",
            "duration": 1.5,
            "speaker_id": spk,
        })
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    _write_jsonl(mdir / "train.jsonl", records)
    _write_jsonl(mdir / "val.jsonl", [records[0]])

    out = tmp_path / "piper"
    export_piper(mdir, out)
    first_line = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()[0]
    parts = first_line.split("|")
    assert len(parts) == 3, "Multi-speaker: stem|speaker|text (3 columns)"


# ---------------------------------------------------------------------------
# export_piper — --limit (smoke mode)
# ---------------------------------------------------------------------------

def test_export_limit_caps_training_records(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out, limit=2)
    train_stems = (out / "train.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_stems) == 2  # limit=2, down from 3


def test_export_limit_does_not_affect_val(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out, limit=1)
    val_stems = (out / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(val_stems) == 1  # val is never limited


def test_export_limit_total_csv_rows(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out, limit=2)
    lines = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()
    # 2 train + 1 val + 1 test = 4 rows
    assert len(lines) == 4


# ---------------------------------------------------------------------------
# validate_manifest — hardened checks (Sprint 3 additions)
# ---------------------------------------------------------------------------

def _full_record(wav_path: str) -> dict:
    return {
        "audio_filepath": wav_path,
        "text": "Merhaba.",
        "normalized_text": "merhaba",
        "phonemes": "m ɛ r h a b a",
        "duration": 1.5,
        "speaker_id": "issai",
    }


def test_validate_missing_phonemes_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    rec = {
        "audio_filepath": "/x.wav",
        "text": "x",
        "normalized_text": "x",
        "duration": 1.5,
        "speaker_id": "s",
        # phonemes missing
    }
    _write_jsonl(bad, [rec])
    valid, errors = validate_manifest(bad, check_audio=False)
    assert valid == 0
    assert any("phonemes" in e for e in errors)


def test_validate_missing_speaker_id_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    rec = {
        "audio_filepath": "/x.wav",
        "text": "x",
        "normalized_text": "x",
        "phonemes": "x",
        "duration": 1.5,
        # speaker_id missing
    }
    _write_jsonl(bad, [rec])
    valid, errors = validate_manifest(bad, check_audio=False)
    assert valid == 0
    assert any("speaker_id" in e for e in errors)


def test_validate_missing_audio_file(tmp_path):
    bad = tmp_path / "bad.jsonl"
    rec = _full_record("/nonexistent/path/audio.wav")
    _write_jsonl(bad, [rec])
    valid, errors = validate_manifest(bad, check_audio=True)
    assert valid == 0
    assert any("not found" in e for e in errors)


def test_validate_skip_audio_check(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rec = _full_record("/nonexistent/path/audio.wav")
    _write_jsonl(manifest, [rec])
    valid, errors = validate_manifest(manifest, check_audio=False)
    assert valid == 1
    assert errors == []


def test_validate_duration_too_short(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rec = _full_record("/x.wav")
    rec["duration"] = 0.1
    _write_jsonl(manifest, [rec])
    valid, errors = validate_manifest(manifest, check_audio=False)
    assert valid == 0
    assert any("duration" in e for e in errors)


def test_validate_duration_too_long(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rec = _full_record("/x.wav")
    rec["duration"] = 25.0
    _write_jsonl(manifest, [rec])
    valid, errors = validate_manifest(manifest, check_audio=False)
    assert valid == 0
    assert any("duration" in e for e in errors)


def test_validate_empty_phonemes(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rec = _full_record("/x.wav")
    rec["phonemes"] = "   "
    _write_jsonl(manifest, [rec])
    valid, errors = validate_manifest(manifest, check_audio=False)
    assert valid == 0
    assert any("phonemes" in e for e in errors)


def test_validate_empty_text(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rec = _full_record("/x.wav")
    rec["text"] = ""
    _write_jsonl(manifest, [rec])
    valid, errors = validate_manifest(manifest, check_audio=False)
    assert valid == 0
    assert any("text" in e for e in errors)


def test_validate_valid_record_with_real_audio(tmp_path):
    wav = tmp_path / "test.wav"
    _make_wav(wav, 1.5)
    manifest = tmp_path / "m.jsonl"
    rec = _full_record(str(wav))
    _write_jsonl(manifest, [rec])
    valid, errors = validate_manifest(manifest, check_audio=True)
    assert valid == 1
    assert errors == []
