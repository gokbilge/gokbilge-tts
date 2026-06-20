#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
import wave

try:
    import audioop
except ImportError:  # pragma: no cover
    audioop = None

TEXT_KEYS = ("text", "sentence", "normalized_text")
AUDIO_KEYS = ("audio_filepath", "audio", "path", "wav")
TARGET_CHARS = ("ç", "ş", "ğ", "ü", "ö", "ı")
CSV_FIELDS = (
    "line_no",
    "text",
    "audio_filepath",
    "matched_terms",
    "matched_chars",
    "duration",
    "chars_per_sec",
    "rms",
    "peak",
    "sample_rate",
    "channels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Turkish-heavy target terms in a JSONL manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to input JSONL manifest")
    parser.add_argument("--terms", required=True, help="Comma-separated UTF-8 target terms")
    parser.add_argument("--output", required=True, help="CSV output path for matching rows")
    parser.add_argument("--summary", required=True, help="Markdown summary output path")
    parser.add_argument(
        "--jsonl-output",
        help="Optional JSONL output path for matching rows",
    )
    return parser.parse_args()


def parse_terms(raw_terms: str) -> list[str]:
    terms = [term.strip() for term in raw_terms.split(",") if term.strip()]
    if not terms:
        raise ValueError("At least one non-empty term is required")
    return terms


def extract_text(record: dict) -> str:
    for key in TEXT_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def extract_audio_value(record: dict) -> str:
    for key in AUDIO_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def resolve_audio_path(manifest_path: Path, audio_value: str) -> Path:
    audio_path = Path(audio_value)
    if audio_path.is_absolute():
        return audio_path
    return (manifest_path.parent / audio_path).resolve()


def round_or_blank(value: float | int | None) -> str | float | int:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 4)
    return value


def audio_stats(audio_path: Path) -> dict[str, float | int | None]:
    stats = {
        "duration": None,
        "chars_per_sec": None,
        "rms": None,
        "peak": None,
        "sample_rate": None,
        "channels": None,
    }
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(frame_count)
            duration = frame_count / sample_rate if sample_rate else None
            stats["duration"] = duration
            stats["sample_rate"] = sample_rate
            stats["channels"] = channels
            if frames and audioop is not None:
                stats["rms"] = audioop.rms(frames, sample_width)
                stats["peak"] = audioop.max(frames, sample_width)
    except Exception:
        return stats
    return stats


def compute_numeric_stats(rows: list[dict], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if row[key] != ""]
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def audit_manifest(manifest_path: Path, terms: list[str]) -> tuple[dict, list[dict]]:
    term_counts = {term: 0 for term in terms}
    char_counts = {char: 0 for char in TARGET_CHARS}
    term_needles = {term: term.casefold() for term in terms}

    summary = {
        "manifest": str(manifest_path),
        "total_rows": 0,
        "rows_with_any_target_term": 0,
        "rows_with_any_target_char": 0,
        "extracted_rows": 0,
        "term_counts": term_counts,
        "char_counts": char_counts,
    }
    extracted_rows: list[dict] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            summary["total_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = extract_text(record)
            text_casefold = text.casefold()
            matched_terms = [term for term in terms if term_needles[term] in text_casefold]
            matched_chars = [char for char in TARGET_CHARS if char in text_casefold]

            if matched_terms:
                summary["rows_with_any_target_term"] += 1
            if matched_chars:
                summary["rows_with_any_target_char"] += 1

            for term in matched_terms:
                term_counts[term] += 1
            for char in matched_chars:
                char_counts[char] += 1

            if not matched_terms:
                continue

            audio_value = extract_audio_value(record)
            resolved_audio = resolve_audio_path(manifest_path, audio_value) if audio_value else None
            stats = audio_stats(resolved_audio) if resolved_audio else {
                "duration": None,
                "chars_per_sec": None,
                "rms": None,
                "peak": None,
                "sample_rate": None,
                "channels": None,
            }

            if stats["duration"] and stats["duration"] > 0:
                stats["chars_per_sec"] = len(text) / stats["duration"]

            row = {
                "line_no": line_no,
                "text": text,
                "audio_filepath": str(resolved_audio) if resolved_audio else "",
                "matched_terms": "|".join(matched_terms),
                "matched_chars": "|".join(matched_chars),
                "duration": round_or_blank(stats["duration"]),
                "chars_per_sec": round_or_blank(stats["chars_per_sec"]),
                "rms": round_or_blank(stats["rms"]),
                "peak": round_or_blank(stats["peak"]),
                "sample_rate": round_or_blank(stats["sample_rate"]),
                "channels": round_or_blank(stats["channels"]),
            }
            extracted_rows.append(row)

    summary["extracted_rows"] = len(extracted_rows)
    return summary, extracted_rows


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(output_path: Path, summary: dict, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_stats = compute_numeric_stats(rows, "duration")
    cps_stats = compute_numeric_stats(rows, "chars_per_sec")
    rms_stats = compute_numeric_stats(rows, "rms")
    peak_stats = compute_numeric_stats(rows, "peak")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Turkish-heavy target audit\n\n")
        handle.write(f"- manifest: `{summary['manifest']}`\n")
        handle.write(f"- total rows: {summary['total_rows']}\n")
        handle.write(f"- rows with any target term: {summary['rows_with_any_target_term']}\n")
        handle.write(f"- rows with any Turkish target char: {summary['rows_with_any_target_char']}\n")
        handle.write(f"- extracted rows written: {summary['extracted_rows']}\n\n")

        handle.write("## Term counts\n\n")
        for term, count in sorted(summary["term_counts"].items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"- {term}: {count}\n")

        handle.write("\n## Character counts\n\n")
        for char, count in sorted(summary["char_counts"].items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"- {char}: {count}\n")

        handle.write("\n## Extracted-row stats\n\n")
        write_stats_block(handle, "duration", duration_stats)
        write_stats_block(handle, "chars_per_sec", cps_stats)
        write_stats_block(handle, "rms", rms_stats)
        write_stats_block(handle, "peak", peak_stats)


def write_stats_block(handle, label: str, stats: dict[str, float] | None) -> None:
    if stats is None:
        handle.write(f"- {label}: no readable values\n")
        return
    handle.write(
        f"- {label}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}\n"
    )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    terms = parse_terms(args.terms)
    summary, rows = audit_manifest(manifest_path, terms)
    write_csv(Path(args.output), rows)
    if args.jsonl_output:
        write_jsonl(Path(args.jsonl_output), rows)
    write_summary(Path(args.summary), summary, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
