"""Dataset audit and clean-manifest utilities for v0.2 preparation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class AuditThresholds:
    """Configurable thresholds for dataset auditing."""

    min_duration_sec: float = 0.8
    max_duration_sec: float = 15.0
    min_text_chars: int = 3
    max_text_chars: int = 250
    min_chars_per_sec: float = 6.0
    max_chars_per_sec: float = 22.0
    max_leading_silence_sec: float = 0.5
    max_trailing_silence_sec: float = 0.7
    max_internal_silence_ratio: float = 0.40
    max_longest_internal_gap_sec: float = 0.45
    max_clipping_ratio: float = 0.01
    silence_threshold_db: float = -40.0
    silence_frame_ms: float = 20.0
    silence_hop_ms: float = 10.0
    min_rms_db: float = -30.0
    max_peak_db: float = -0.5


SEVERE_REASONS = {
    "audio_unreadable",
    "duration_too_short",
    "duration_too_long",
    "text_too_short",
    "text_too_long",
    "chars_per_sec_too_low",
    "chars_per_sec_too_high",
    "clipping_too_high",
    "internal_silence_too_high",
}

BALANCED_EXCLUDED_REASONS = {
    "audio_unreadable",
    "duration_too_short",
    "duration_too_long",
    "text_too_short",
    "text_too_long",
    "chars_per_sec_too_low",
    "chars_per_sec_too_high",
    "clipping_too_high",
    "internal_silence_too_high",
}

AUDIT_FIELDNAMES = [
    "manifest_line",
    "audio_filepath",
    "text",
    "text_length",
    "word_count",
    "audio_duration_sec",
    "sample_rate",
    "channels",
    "rms_level_db",
    "peak_level_db",
    "clipping_ratio",
    "leading_silence_sec",
    "trailing_silence_sec",
    "internal_silence_ratio",
    "longest_internal_gap_sec",
    "chars_per_sec",
    "words_per_sec",
    "quality_score",
    "status",
    "reasons",
]


@dataclass
class AuditRow:
    """Computed metrics for one manifest row."""

    manifest_line: int
    audio_filepath: str
    text: str
    text_length: int
    word_count: int
    audio_duration_sec: float
    sample_rate: int
    channels: int
    rms_level_db: float
    peak_level_db: float
    clipping_ratio: float
    leading_silence_sec: float
    trailing_silence_sec: float
    internal_silence_ratio: float
    longest_internal_gap_sec: float
    chars_per_sec: float
    words_per_sec: float
    quality_score: int
    status: str
    reasons: list[str]

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = ";".join(self.reasons)
        return row


def _safe_db(value: float) -> float:
    if value <= 0.0:
        return -100.0
    return 20.0 * math.log10(value)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _score_range(value: float, low: float, high: float, weight: float) -> float:
    if low <= value <= high:
        return weight
    if value < low:
        span = max(low, 1e-6)
        return weight * _clamp(value / span)
    span = max(high, 1e-6)
    penalty = (value - high) / span
    return weight * _clamp(1.0 - penalty)


def _score_ceiling(value: float, maximum: float, weight: float) -> float:
    if value <= maximum:
        return weight
    if maximum <= 0:
        return 0.0
    penalty = (value - maximum) / maximum
    return weight * _clamp(1.0 - penalty)


def iter_manifest_records(manifest_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield manifest rows with 1-based line numbers."""
    with open(manifest_path, encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _read_audio_metrics(audio_path: Path, thresholds: AuditThresholds) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(audio_path))
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    channels = int(audio.shape[1])
    mono = audio.mean(axis=1)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    clipping_ratio = float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0

    frame_length = max(1, int(sample_rate * thresholds.silence_frame_ms / 1000.0))
    hop_length = max(1, int(sample_rate * thresholds.silence_hop_ms / 1000.0))
    silence_amp = 10 ** (thresholds.silence_threshold_db / 20.0)

    frame_starts = list(range(0, len(mono), hop_length))
    frame_is_silent: list[bool] = []
    for start in frame_starts:
        frame = mono[start : start + frame_length]
        frame_peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        frame_is_silent.append(frame_peak <= silence_amp)

    first_sound_idx = next((i for i, silent in enumerate(frame_is_silent) if not silent), None)
    last_sound_idx = next(
        (i for i in range(len(frame_is_silent) - 1, -1, -1) if not frame_is_silent[i]),
        None,
    )

    if first_sound_idx is None or last_sound_idx is None:
        leading_silence_sec = info.duration
        trailing_silence_sec = info.duration
        internal_silence_ratio = 1.0
        longest_internal_gap_sec = info.duration
    else:
        leading_silence_sec = first_sound_idx * hop_length / sample_rate
        tail_start = last_sound_idx * hop_length + frame_length
        trailing_silence_sec = max(0.0, (len(mono) - tail_start) / sample_rate)

        core = frame_is_silent[first_sound_idx : last_sound_idx + 1]
        silent_frames = 0
        longest_gap_frames = 0
        current_gap = 0
        for silent in core:
            if silent:
                silent_frames += 1
                current_gap += 1
                longest_gap_frames = max(longest_gap_frames, current_gap)
            else:
                current_gap = 0

        internal_silence_ratio = silent_frames / len(core) if core else 0.0
        longest_internal_gap_sec = longest_gap_frames * hop_length / sample_rate

    return {
        "audio_duration_sec": float(info.duration),
        "sample_rate": int(info.samplerate),
        "channels": channels,
        "rms_level_db": _safe_db(rms),
        "peak_level_db": _safe_db(peak),
        "clipping_ratio": clipping_ratio,
        "leading_silence_sec": float(leading_silence_sec),
        "trailing_silence_sec": float(trailing_silence_sec),
        "internal_silence_ratio": float(internal_silence_ratio),
        "longest_internal_gap_sec": float(longest_internal_gap_sec),
    }


def _compute_reasons(
    metrics: dict[str, Any],
    text_length: int,
    chars_per_sec: float,
    thresholds: AuditThresholds,
) -> list[str]:
    reasons: list[str] = []

    duration = metrics["audio_duration_sec"]
    if duration < thresholds.min_duration_sec:
        reasons.append("duration_too_short")
    if duration > thresholds.max_duration_sec:
        reasons.append("duration_too_long")
    if text_length < thresholds.min_text_chars:
        reasons.append("text_too_short")
    if text_length > thresholds.max_text_chars:
        reasons.append("text_too_long")
    if chars_per_sec < thresholds.min_chars_per_sec:
        reasons.append("chars_per_sec_too_low")
    if chars_per_sec > thresholds.max_chars_per_sec:
        reasons.append("chars_per_sec_too_high")
    if metrics["leading_silence_sec"] > thresholds.max_leading_silence_sec:
        reasons.append("leading_silence_too_high")
    if metrics["trailing_silence_sec"] > thresholds.max_trailing_silence_sec:
        reasons.append("trailing_silence_too_high")
    if metrics["internal_silence_ratio"] > thresholds.max_internal_silence_ratio:
        reasons.append("internal_silence_too_high")
    if metrics["longest_internal_gap_sec"] > thresholds.max_longest_internal_gap_sec:
        reasons.append("longest_internal_gap_too_high")
    if metrics["clipping_ratio"] > thresholds.max_clipping_ratio:
        reasons.append("clipping_too_high")
    if metrics["rms_level_db"] < thresholds.min_rms_db:
        reasons.append("rms_too_low")
    if metrics["peak_level_db"] > thresholds.max_peak_db:
        reasons.append("peak_too_high")

    return reasons


def compute_quality_score(
    metrics: dict[str, Any],
    text_length: int,
    word_count: int,
    thresholds: AuditThresholds,
) -> int:
    """Compute a deterministic 0-100 quality score."""
    duration = metrics["audio_duration_sec"]
    chars_per_sec = text_length / duration if duration > 0 else 0.0
    words_per_sec = word_count / duration if duration > 0 else 0.0

    duration_score = _score_range(duration, 1.0, 12.0, 20.0)
    speed_score = _score_range(chars_per_sec, 8.0, 18.0, 20.0)

    silence_penalty_value = max(
        metrics["leading_silence_sec"] / max(thresholds.max_leading_silence_sec, 1e-6),
        metrics["trailing_silence_sec"] / max(thresholds.max_trailing_silence_sec, 1e-6),
        metrics["internal_silence_ratio"] / max(thresholds.max_internal_silence_ratio, 1e-6),
        metrics["longest_internal_gap_sec"]
        / max(thresholds.max_longest_internal_gap_sec, 1e-6),
    )
    silence_score = 20.0 * _clamp(1.0 - max(0.0, silence_penalty_value - 1.0))

    rms_score = _score_range(metrics["rms_level_db"], -26.0, -14.0, 12.0)
    peak_score = _score_range(-metrics["peak_level_db"], 0.7, 12.0, 8.0)
    volume_score = rms_score + peak_score

    clipping_score = _score_ceiling(metrics["clipping_ratio"], thresholds.max_clipping_ratio, 20.0)

    score = duration_score + speed_score + silence_score + volume_score + clipping_score

    if words_per_sec > 4.5:
        score -= 5.0
    if words_per_sec < 1.0 and duration > 0:
        score -= 5.0

    return int(round(_clamp(score, 0.0, 100.0)))


def status_from_reasons_and_score(reasons: list[str], quality_score: int) -> str:
    has_severe = any(reason in SEVERE_REASONS for reason in reasons)
    if has_severe or quality_score < 50:
        return "reject"
    if quality_score >= 75 and not reasons:
        return "keep"
    return "suspicious"


def audit_manifest(
    manifest_path: Path,
    thresholds: AuditThresholds | None = None,
) -> list[AuditRow]:
    """Audit a manifest and return computed rows."""
    thresholds = thresholds or AuditThresholds()
    rows: list[AuditRow] = []

    for line_no, entry in iter_manifest_records(manifest_path):
        audio_path = Path(str(entry.get("audio_filepath", "")))
        text = str(entry.get("text", "")).strip()
        text_length = len(text)
        word_count = len(text.split()) if text else 0

        try:
            metrics = _read_audio_metrics(audio_path, thresholds)
            duration = metrics["audio_duration_sec"]
            chars_per_sec = text_length / duration if duration > 0 else 0.0
            words_per_sec = word_count / duration if duration > 0 else 0.0
            reasons = _compute_reasons(metrics, text_length, chars_per_sec, thresholds)
            quality_score = compute_quality_score(metrics, text_length, word_count, thresholds)
            status = status_from_reasons_and_score(reasons, quality_score)
        except Exception:
            metrics = {
                "audio_duration_sec": 0.0,
                "sample_rate": 0,
                "channels": 0,
                "rms_level_db": -100.0,
                "peak_level_db": -100.0,
                "clipping_ratio": 0.0,
                "leading_silence_sec": 0.0,
                "trailing_silence_sec": 0.0,
                "internal_silence_ratio": 0.0,
                "longest_internal_gap_sec": 0.0,
            }
            chars_per_sec = 0.0
            words_per_sec = 0.0
            reasons = ["audio_unreadable"]
            quality_score = 0
            status = "reject"

        rows.append(
            AuditRow(
                manifest_line=line_no,
                audio_filepath=str(audio_path),
                text=text,
                text_length=text_length,
                word_count=word_count,
                audio_duration_sec=round(metrics["audio_duration_sec"], 4),
                sample_rate=metrics["sample_rate"],
                channels=metrics["channels"],
                rms_level_db=round(metrics["rms_level_db"], 2),
                peak_level_db=round(metrics["peak_level_db"], 2),
                clipping_ratio=round(metrics["clipping_ratio"], 6),
                leading_silence_sec=round(metrics["leading_silence_sec"], 4),
                trailing_silence_sec=round(metrics["trailing_silence_sec"], 4),
                internal_silence_ratio=round(metrics["internal_silence_ratio"], 4),
                longest_internal_gap_sec=round(metrics["longest_internal_gap_sec"], 4),
                chars_per_sec=round(chars_per_sec, 4),
                words_per_sec=round(words_per_sec, 4),
                quality_score=quality_score,
                status=status,
                reasons=reasons,
            )
        )

    return rows


def write_audit_csv(rows: Iterable[AuditRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def write_audit_subset_csv(rows: Iterable[AuditRow], output_path: Path, status: str) -> None:
    write_audit_csv((row for row in rows if row.status == status), output_path)


def _mean(rows: list[AuditRow], attr: str) -> float:
    if not rows:
        return 0.0
    return sum(getattr(row, attr) for row in rows) / len(rows)


def write_summary_markdown(
    rows: list[AuditRow],
    output_path: Path,
    manifest_path: Path,
    thresholds: AuditThresholds,
) -> None:
    counts = {"keep": 0, "suspicious": 0, "reject": 0}
    reasons: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        for reason in row.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1

    top_reasons = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:10]

    lines = [
        "# Dataset Cleaning v0.2 Audit Summary",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Total rows audited: {len(rows)}",
        f"- Keep: {counts.get('keep', 0)}",
        f"- Suspicious: {counts.get('suspicious', 0)}",
        f"- Reject: {counts.get('reject', 0)}",
        f"- Mean quality score: {_mean(rows, 'quality_score'):.1f}",
        f"- Mean duration: {_mean(rows, 'audio_duration_sec'):.2f} s",
        f"- Mean chars/sec: {_mean(rows, 'chars_per_sec'):.2f}",
        "",
        "## Default Thresholds",
        "",
    ]

    for key, value in asdict(thresholds).items():
        lines.append(f"- `{key}` = `{value}`")

    lines.extend(["", "## Top Reasons", ""])
    if top_reasons:
        for reason, count in top_reasons:
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- No audit flags recorded.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_audit_csv(audit_csv_path: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    with open(audit_csv_path, encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                AuditRow(
                    manifest_line=int(record["manifest_line"]),
                    audio_filepath=record["audio_filepath"],
                    text=record["text"],
                    text_length=int(record["text_length"]),
                    word_count=int(record["word_count"]),
                    audio_duration_sec=float(record["audio_duration_sec"]),
                    sample_rate=int(record["sample_rate"]),
                    channels=int(record["channels"]),
                    rms_level_db=float(record["rms_level_db"]),
                    peak_level_db=float(record["peak_level_db"]),
                    clipping_ratio=float(record["clipping_ratio"]),
                    leading_silence_sec=float(record["leading_silence_sec"]),
                    trailing_silence_sec=float(record["trailing_silence_sec"]),
                    internal_silence_ratio=float(record["internal_silence_ratio"]),
                    longest_internal_gap_sec=float(record["longest_internal_gap_sec"]),
                    chars_per_sec=float(record["chars_per_sec"]),
                    words_per_sec=float(record["words_per_sec"]),
                    quality_score=int(record["quality_score"]),
                    status=record["status"],
                    reasons=[value for value in record["reasons"].split(";") if value],
                )
            )
    return rows


def should_keep_row(row: AuditRow, mode: str) -> bool:
    if mode == "strict":
        return row.status == "keep"
    if mode == "balanced":
        if row.status == "keep":
            return True
        if row.status != "suspicious":
            return False
        return not any(reason in BALANCED_EXCLUDED_REASONS for reason in row.reasons)
    if mode == "rejects":
        return row.status == "reject"
    if mode == "suspicious":
        return row.status == "suspicious"
    raise ValueError(f"Unknown filter mode: {mode}")


def filter_manifest_with_audit(
    manifest_path: Path,
    audit_rows: list[AuditRow],
    output_manifest_path: Path,
    mode: str,
) -> dict[str, int]:
    selected_lines = {row.manifest_line for row in audit_rows if should_keep_row(row, mode)}

    kept = 0
    total = 0
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, encoding="utf-8") as source, open(
        output_manifest_path, "w", encoding="utf-8"
    ) as target:
        for line_no, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            total += 1
            if line_no in selected_lines:
                target.write(raw_line)
                kept += 1

    return {"mode": mode, "kept": kept, "total": total, "dropped": total - kept}


def build_clean_manifests(
    manifest_path: Path,
    audit_csv_path: Path,
    summary_path: Path,
    manifests_dir: Path,
    reports_dir: Path,
    thresholds: AuditThresholds | None = None,
) -> dict[str, dict[str, int]]:
    thresholds = thresholds or AuditThresholds()
    audit_rows = audit_manifest(manifest_path, thresholds)
    write_audit_csv(audit_rows, audit_csv_path)
    write_summary_markdown(audit_rows, summary_path, manifest_path, thresholds)
    write_audit_subset_csv(audit_rows, reports_dir / "rejected_samples.csv", "reject")
    write_audit_subset_csv(audit_rows, reports_dir / "suspicious_samples.csv", "suspicious")

    manifests_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "strict": filter_manifest_with_audit(
            manifest_path,
            audit_rows,
            manifests_dir / "train_clean_strict.jsonl",
            "strict",
        ),
        "balanced": filter_manifest_with_audit(
            manifest_path,
            audit_rows,
            manifests_dir / "train_clean_balanced.jsonl",
            "balanced",
        ),
        "rejects": filter_manifest_with_audit(
            manifest_path,
            audit_rows,
            manifests_dir / "train_rejected.jsonl",
            "rejects",
        ),
        "suspicious": filter_manifest_with_audit(
            manifest_path,
            audit_rows,
            manifests_dir / "train_suspicious.jsonl",
            "suspicious",
        ),
    }
    return results
