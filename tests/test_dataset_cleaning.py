"""Tests for v0.2 dataset cleaning audit and manifest filtering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from gokbilge_tts.datasets.dataset_cleaning import (
    AuditThresholds,
    audit_manifest,
    build_clean_manifests,
    filter_manifest_with_audit,
    load_audit_csv,
    write_audit_csv,
)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 22050) -> None:
    sf.write(str(path), audio.astype(np.float32), sr)


def _tone_with_silence(
    duration_sec: float,
    sr: int = 22050,
    leading_sec: float = 0.0,
    trailing_sec: float = 0.0,
    amplitude: float = 0.2,
    frequency_hz: float = 220.0,
) -> np.ndarray:
    total_samples = int(duration_sec * sr)
    audio = np.zeros(total_samples, dtype=np.float32)
    start = int(leading_sec * sr)
    end = total_samples - int(trailing_sec * sr)
    if end > start:
        t = np.arange(end - start, dtype=np.float32) / sr
        audio[start:end] = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    return audio


def _write_manifest(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_audit_computes_duration_and_metadata(tmp_path):
    wav_path = tmp_path / "keep.wav"
    _write_wav(wav_path, _tone_with_silence(duration_sec=2.0))
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [{"audio_filepath": str(wav_path), "text": "Bugün hava çok güzel."}],
    )

    rows = audit_manifest(manifest)

    assert len(rows) == 1
    row = rows[0]
    assert row.status == "keep"
    assert abs(row.audio_duration_sec - 2.0) < 0.05
    assert row.sample_rate == 22050
    assert row.channels == 1
    assert row.text_length == len("Bugün hava çok güzel.")
    assert row.word_count == 4
    assert row.chars_per_sec > 6.0
    assert row.quality_score >= 75


def test_unreadable_audio_becomes_reject(tmp_path):
    bad_wav = tmp_path / "bad.wav"
    bad_wav.write_text("not a wav", encoding="utf-8")
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [{"audio_filepath": str(bad_wav), "text": "Merhaba dünya."}])

    rows = audit_manifest(manifest)

    assert rows[0].status == "reject"
    assert rows[0].reasons == ["audio_unreadable"]
    assert rows[0].quality_score == 0


def test_filter_manifest_modes(tmp_path):
    keep_wav = tmp_path / "keep.wav"
    suspicious_wav = tmp_path / "suspicious.wav"
    reject_wav = tmp_path / "reject.wav"

    _write_wav(keep_wav, _tone_with_silence(duration_sec=2.0))
    _write_wav(suspicious_wav, _tone_with_silence(duration_sec=2.0, leading_sec=0.6))
    _write_wav(reject_wav, _tone_with_silence(duration_sec=0.4))

    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            {"audio_filepath": str(keep_wav), "text": "Bugün hava çok güzel."},
            {"audio_filepath": str(suspicious_wav), "text": "Türkiye Cumhuriyeti bugün burada."},
            {"audio_filepath": str(reject_wav), "text": "Kısa."},
        ],
    )

    rows = audit_manifest(manifest)
    statuses = [row.status for row in rows]
    assert statuses == ["keep", "suspicious", "reject"]

    strict_manifest = tmp_path / "strict.jsonl"
    balanced_manifest = tmp_path / "balanced.jsonl"
    rejects_manifest = tmp_path / "rejects.jsonl"
    suspicious_manifest = tmp_path / "suspicious.jsonl"

    strict_result = filter_manifest_with_audit(manifest, rows, strict_manifest, "strict")
    balanced_result = filter_manifest_with_audit(manifest, rows, balanced_manifest, "balanced")
    rejects_result = filter_manifest_with_audit(manifest, rows, rejects_manifest, "rejects")
    suspicious_result = filter_manifest_with_audit(manifest, rows, suspicious_manifest, "suspicious")

    assert strict_result["kept"] == 1
    assert balanced_result["kept"] == 2
    assert rejects_result["kept"] == 1
    assert suspicious_result["kept"] == 1

    assert len(_read_jsonl(strict_manifest)) == 1
    assert len(_read_jsonl(balanced_manifest)) == 2
    assert len(_read_jsonl(rejects_manifest)) == 1
    assert len(_read_jsonl(suspicious_manifest)) == 1


def test_load_audit_csv_roundtrip(tmp_path):
    wav_path = tmp_path / "keep.wav"
    _write_wav(wav_path, _tone_with_silence(duration_sec=2.0))
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [{"audio_filepath": str(wav_path), "text": "Öğrenciler ölçüm yaptı."}])

    rows = audit_manifest(manifest)
    audit_csv = tmp_path / "audit.csv"
    write_audit_csv(rows, audit_csv)
    loaded = load_audit_csv(audit_csv)

    assert len(loaded) == 1
    assert loaded[0].audio_filepath == rows[0].audio_filepath
    assert loaded[0].status == rows[0].status
    assert loaded[0].reasons == rows[0].reasons


def test_build_clean_manifests_creates_expected_files(tmp_path):
    keep_wav = tmp_path / "keep.wav"
    suspicious_wav = tmp_path / "suspicious.wav"
    reject_wav = tmp_path / "reject.wav"

    _write_wav(keep_wav, _tone_with_silence(duration_sec=2.0))
    _write_wav(suspicious_wav, _tone_with_silence(duration_sec=2.0, trailing_sec=0.8))
    _write_wav(reject_wav, _tone_with_silence(duration_sec=0.4))

    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            {"audio_filepath": str(keep_wav), "text": "Bugün hava çok güzel."},
            {"audio_filepath": str(suspicious_wav), "text": "Şirket yüzde otuz beş büyüme açıkladı."},
            {"audio_filepath": str(reject_wav), "text": "Kısa."},
        ],
    )

    reports_dir = tmp_path / "reports"
    manifests_dir = tmp_path / "data" / "manifests"
    results = build_clean_manifests(
        manifest_path=manifest,
        audit_csv_path=reports_dir / "dataset_audit.csv",
        summary_path=reports_dir / "dataset_quality_summary.md",
        manifests_dir=manifests_dir,
        reports_dir=reports_dir,
        thresholds=AuditThresholds(),
    )

    assert results["strict"]["kept"] == 1
    assert results["balanced"]["kept"] == 2
    assert (reports_dir / "dataset_audit.csv").exists()
    assert (reports_dir / "dataset_quality_summary.md").exists()
    assert (reports_dir / "rejected_samples.csv").exists()
    assert (reports_dir / "suspicious_samples.csv").exists()
    assert (manifests_dir / "train_clean_strict.jsonl").exists()
    assert (manifests_dir / "train_clean_balanced.jsonl").exists()
    assert (manifests_dir / "train_rejected.jsonl").exists()
    assert (manifests_dir / "train_suspicious.jsonl").exists()
