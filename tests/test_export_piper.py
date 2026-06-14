"""Tests for export_piper, validate_piper_export, and hardened validate_manifest."""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from gokbilge_tts.datasets.export_piper import export_piper, validate_piper_export
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


def test_export_limit_val_capped_at_10(tmp_path):
    """In smoke mode, val is capped at min(10, available) even if more exist."""
    wav_src = tmp_path / "wav_src"
    wav_src.mkdir()
    records = []
    for i in range(35):
        wav = wav_src / f"utt_{i:04d}.wav"
        _make_wav(wav, 1.0)
        records.append({
            "audio_filepath": str(wav),
            "text": f"Metin {i}.",
            "normalized_text": f"metin {i}",
            "phonemes": "m ɛ t i n",
            "duration": 1.0,
            "speaker_id": "issai",
        })
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    _write_jsonl(mdir / "train.jsonl", records[:5])
    _write_jsonl(mdir / "val.jsonl", records[5:])   # 30 val records available
    out = tmp_path / "piper"
    export_piper(mdir, out, limit=3)
    val_stems = (out / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(val_stems) == 10  # capped at SMOKE_VAL_LIMIT


def test_export_limit_val_not_padded_when_fewer_than_10(manifest_dir, tmp_path):
    """Smoke limit of 10 is a max, not a minimum — don't require exactly 10."""
    out = tmp_path / "piper"
    export_piper(manifest_dir, out, limit=2)
    val_stems = (out / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(val_stems) == 1  # only 1 val record available; not padded


def test_export_limit_total_csv_rows(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out, limit=2)
    lines = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()
    # 2 train + 1 val (capped, but only 1 available) + 1 test = 4 rows
    assert len(lines) == 4


def test_export_no_limit_val_uncapped(tmp_path):
    """Without --limit, all val records are included regardless of count."""
    wav_src = tmp_path / "wav_src"
    wav_src.mkdir()
    records = []
    for i in range(25):
        wav = wav_src / f"utt_{i:04d}.wav"
        _make_wav(wav, 1.0)
        records.append({
            "audio_filepath": str(wav),
            "text": f"Metin {i}.",
            "normalized_text": f"metin {i}",
            "phonemes": "m ɛ t i n",
            "duration": 1.0,
            "speaker_id": "issai",
        })
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    _write_jsonl(mdir / "train.jsonl", records[:5])
    _write_jsonl(mdir / "val.jsonl", records[5:])   # 20 val records
    out = tmp_path / "piper"
    export_piper(mdir, out, limit=None)  # no limit
    val_stems = (out / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(val_stems) == 20  # all val records included


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


# ---------------------------------------------------------------------------
# validate_piper_export
# ---------------------------------------------------------------------------

def test_validate_piper_export_valid(manifest_dir, tmp_path):
    out = tmp_path / "piper"
    export_piper(manifest_dir, out)
    valid, errors = validate_piper_export(out)
    assert errors == []
    assert valid == 5  # 3 train + 1 val + 1 test


def test_validate_piper_export_missing_metadata(tmp_path):
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    (piper_dir / "wavs").mkdir()
    (piper_dir / "config.json").write_text("{}", encoding="utf-8")
    # metadata.csv is absent
    valid, errors = validate_piper_export(piper_dir)
    assert valid == 0
    assert any("metadata.csv" in e for e in errors)


def test_validate_piper_export_missing_wav(tmp_path):
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    wavs_dir = piper_dir / "wavs"
    wavs_dir.mkdir()
    (piper_dir / "config.json").write_text("{}", encoding="utf-8")
    # metadata.csv references a stem with no matching wav
    (piper_dir / "metadata.csv").write_text("ghost_stem|metin\n", encoding="utf-8")
    valid, errors = validate_piper_export(piper_dir)
    assert valid == 0
    assert any("ghost_stem.wav" in e for e in errors)


def test_validate_piper_export_empty_text(tmp_path):
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    wavs_dir = piper_dir / "wavs"
    wavs_dir.mkdir()
    (piper_dir / "config.json").write_text("{}", encoding="utf-8")
    # write metadata with an empty text column and matching wav
    wav = wavs_dir / "utt_0000.wav"
    _make_wav(wav, 1.0)
    (piper_dir / "metadata.csv").write_text("utt_0000|\n", encoding="utf-8")
    valid, errors = validate_piper_export(piper_dir)
    assert valid == 0
    assert any("empty text" in e for e in errors)


def test_validate_piper_export_missing_wavs_dir(tmp_path):
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    (piper_dir / "config.json").write_text("{}", encoding="utf-8")
    (piper_dir / "metadata.csv").write_text("utt_0|metin\n", encoding="utf-8")
    # wavs/ directory is absent
    valid, errors = validate_piper_export(piper_dir)
    assert valid == 0
    assert any("wavs" in e for e in errors)


# ---------------------------------------------------------------------------
# smoke.sh existence
# ---------------------------------------------------------------------------

def test_smoke_sh_exists():
    smoke = Path(__file__).parent.parent / "recipes" / "issai_piper" / "smoke.sh"
    assert smoke.exists(), "recipes/issai_piper/smoke.sh must exist"


def test_smoke_sh_is_executable_bash(tmp_path):
    smoke = Path(__file__).parent.parent / "recipes" / "issai_piper" / "smoke.sh"
    content = smoke.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash"), "smoke.sh must have bash shebang"
    assert "export-piper" in content and "--limit" in content, \
        "smoke.sh must call export-piper with --limit"
    assert "piper_train.preprocess" in content, \
        "smoke.sh must call piper_train.preprocess"
    assert "piper_train" in content, \
        "smoke.sh must call piper_train for training"
