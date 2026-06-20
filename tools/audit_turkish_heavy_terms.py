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
TARGET_CHARS = ("\u00e7", "\u015f", "\u011f", "\u00fc", "\u00f6", "\u0131")
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
    "leading_silence_sec",
    "trailing_silence_sec",
    "internal_silence_ratio",
    "longest_internal_gap_sec",
    "internal_gap_count",
    "low_energy_ratio",
)
WINDOW_MS = 20
SILENCE_THRESHOLD_SCALE = 0.03
ABSOLUTE_RMS_FLOOR = 16
MIN_INTERNAL_GAP_SEC = 0.08


def normalize_cli_text(raw_text: str) -> str:
    if not any("\udc80" <= char <= "\udcff" for char in raw_text):
        return raw_text

    raw_bytes = raw_text.encode("utf-8", "surrogateescape")
    for encoding in ("utf-8", "cp1254", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_text


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
    normalized_terms = normalize_cli_text(raw_terms)
    terms = [term.strip() for term in normalized_terms.split(",") if term.strip()]
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


def convert_frames_to_mono(frames: bytes, sample_width: int, channels: int) -> bytes:
    if channels <= 1:
        return frames

    sample_count = len(frames) // sample_width
    if sample_count == 0:
        return b""

    mono_samples = bytearray()
    frame_width = sample_width * channels
    max_value = (1 << (sample_width * 8 - 1)) - 1
    min_value = -(1 << (sample_width * 8 - 1))

    for offset in range(0, len(frames), frame_width):
        channel_values = []
        for channel in range(channels):
            start = offset + channel * sample_width
            end = start + sample_width
            if end > len(frames):
                break
            value = int.from_bytes(frames[start:end], byteorder="little", signed=True)
            channel_values.append(value)
        if not channel_values:
            continue
        mono_value = int(sum(channel_values) / len(channel_values))
        mono_value = max(min_value, min(max_value, mono_value))
        mono_samples.extend(mono_value.to_bytes(sample_width, byteorder="little", signed=True))
    return bytes(mono_samples)


def rms_for_chunk(chunk: bytes, sample_width: int) -> float:
    if not chunk:
        return 0.0
    if audioop is not None:
        return float(audioop.rms(chunk, sample_width))

    sample_count = len(chunk) // sample_width
    if sample_count == 0:
        return 0.0

    total = 0.0
    for offset in range(0, sample_count * sample_width, sample_width):
        value = int.from_bytes(
            chunk[offset : offset + sample_width],
            byteorder="little",
            signed=True,
        )
        total += float(value * value)
    return math.sqrt(total / sample_count)


def compute_window_rms_values(
    frames: bytes,
    sample_width: int,
    channels: int,
    sample_rate: int,
    window_ms: int = WINDOW_MS,
) -> tuple[list[float], float]:
    if not frames or sample_width <= 0 or sample_rate <= 0 or channels <= 0:
        return [], window_ms / 1000.0

    mono_frames = convert_frames_to_mono(frames, sample_width, channels)
    samples_per_window = max(1, int(sample_rate * window_ms / 1000.0))
    bytes_per_window = samples_per_window * sample_width
    window_sec = samples_per_window / sample_rate

    window_rms_values: list[float] = []
    for offset in range(0, len(mono_frames), bytes_per_window):
        chunk = mono_frames[offset : offset + bytes_per_window]
        if not chunk:
            continue
        window_rms_values.append(rms_for_chunk(chunk, sample_width))
    return window_rms_values, window_sec


def find_leading_silent_windows(silent_flags: list[bool]) -> int:
    count = 0
    for flag in silent_flags:
        if not flag:
            break
        count += 1
    return count


def find_trailing_silent_windows(silent_flags: list[bool]) -> int:
    count = 0
    for flag in reversed(silent_flags):
        if not flag:
            break
        count += 1
    return count


def compute_silence_metrics(
    frames: bytes,
    sample_width: int,
    channels: int,
    sample_rate: int,
) -> dict[str, float | int | None]:
    metrics = {
        "leading_silence_sec": None,
        "trailing_silence_sec": None,
        "internal_silence_ratio": None,
        "longest_internal_gap_sec": None,
        "internal_gap_count": None,
        "low_energy_ratio": None,
    }
    window_rms_values, window_sec = compute_window_rms_values(
        frames,
        sample_width,
        channels,
        sample_rate,
    )
    if not window_rms_values:
        return metrics

    max_window_rms = max(window_rms_values)
    silence_threshold = max(ABSOLUTE_RMS_FLOOR, SILENCE_THRESHOLD_SCALE * max_window_rms)
    silent_flags = [value <= silence_threshold for value in window_rms_values]
    total_windows = len(silent_flags)
    silent_windows = sum(1 for flag in silent_flags if flag)

    leading = find_leading_silent_windows(silent_flags)
    trailing = find_trailing_silent_windows(silent_flags)

    core_start = leading
    core_end = total_windows - trailing
    if core_end < core_start:
        core_end = core_start
    core_flags = silent_flags[core_start:core_end]
    internal_silent_windows = sum(1 for flag in core_flags if flag)

    min_gap_windows = max(1, int(math.ceil(MIN_INTERNAL_GAP_SEC / window_sec)))
    longest_gap_windows = 0
    gap_count = 0
    run_length = 0
    for flag in core_flags:
        if flag:
            run_length += 1
            continue
        if run_length:
            longest_gap_windows = max(longest_gap_windows, run_length)
            if run_length >= min_gap_windows:
                gap_count += 1
            run_length = 0
    if run_length:
        longest_gap_windows = max(longest_gap_windows, run_length)
        if run_length >= min_gap_windows:
            gap_count += 1

    core_window_count = len(core_flags)
    metrics["leading_silence_sec"] = leading * window_sec
    metrics["trailing_silence_sec"] = trailing * window_sec
    metrics["internal_silence_ratio"] = (
        internal_silent_windows / core_window_count if core_window_count else 0.0
    )
    metrics["longest_internal_gap_sec"] = longest_gap_windows * window_sec
    metrics["internal_gap_count"] = gap_count
    metrics["low_energy_ratio"] = silent_windows / total_windows if total_windows else 0.0
    return metrics


def audio_stats(audio_path: Path) -> dict[str, float | int | None]:
    stats = {
        "duration": None,
        "chars_per_sec": None,
        "rms": None,
        "peak": None,
        "sample_rate": None,
        "channels": None,
        "leading_silence_sec": None,
        "trailing_silence_sec": None,
        "internal_silence_ratio": None,
        "longest_internal_gap_sec": None,
        "internal_gap_count": None,
        "low_energy_ratio": None,
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
            if frames:
                if audioop is not None:
                    stats["rms"] = audioop.rms(frames, sample_width)
                    stats["peak"] = audioop.max(frames, sample_width)
                silence_metrics = compute_silence_metrics(frames, sample_width, channels, sample_rate)
                stats.update(silence_metrics)
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
                "leading_silence_sec": None,
                "trailing_silence_sec": None,
                "internal_silence_ratio": None,
                "longest_internal_gap_sec": None,
                "internal_gap_count": None,
                "low_energy_ratio": None,
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
                "leading_silence_sec": round_or_blank(stats["leading_silence_sec"]),
                "trailing_silence_sec": round_or_blank(stats["trailing_silence_sec"]),
                "internal_silence_ratio": round_or_blank(stats["internal_silence_ratio"]),
                "longest_internal_gap_sec": round_or_blank(stats["longest_internal_gap_sec"]),
                "internal_gap_count": round_or_blank(stats["internal_gap_count"]),
                "low_energy_ratio": round_or_blank(stats["low_energy_ratio"]),
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
    stat_labels = [
        "duration",
        "chars_per_sec",
        "rms",
        "peak",
        "leading_silence_sec",
        "trailing_silence_sec",
        "internal_silence_ratio",
        "longest_internal_gap_sec",
        "internal_gap_count",
        "low_energy_ratio",
    ]
    stats_by_label = {label: compute_numeric_stats(rows, label) for label in stat_labels}

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
        for label in stat_labels:
            write_stats_block(handle, label, stats_by_label[label])


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
