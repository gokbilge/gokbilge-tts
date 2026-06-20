#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

TEXT_KEYS = ("text", "sentence", "normalized_text")
AUDIO_KEYS = ("audio_filepath", "audio", "path", "wav")
DEFAULT_EXCLUDE_BUCKETS = ("bad_or_reject_like",)
SPARSE_TERM_THRESHOLD = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v0.4 Turkish-heavy fine-tune manifests from a base manifest and bucket CSV."
    )
    parser.add_argument("--base-manifest", required=True, help="Input base manifest JSONL path")
    parser.add_argument("--bucket-csv", required=True, help="Bucket CSV path")
    parser.add_argument("--output-manifest", required=True, help="Output manifest JSONL path")
    parser.add_argument("--summary", required=True, help="Summary markdown path")
    parser.add_argument(
        "--mode",
        choices=("conservative", "weighted", "target_only"),
        default="conservative",
        help="Manifest construction mode",
    )
    parser.add_argument("--target-terms", required=True, help="Comma-separated target terms")
    parser.add_argument(
        "--exclude-buckets",
        default=",".join(DEFAULT_EXCLUDE_BUCKETS),
        help="Comma-separated bucket values to exclude from the base manifest",
    )
    parser.add_argument(
        "--max-target-repeat",
        type=int,
        help="Optional maximum number of appended repeats per selected target row",
    )
    parser.add_argument(
        "--min-target-quality",
        choices=("good", "good_or_suspicious"),
        default="good",
        help="Minimum bucket quality eligible for target oversampling or target-only output",
    )
    parser.add_argument("--max-output-rows", type=int, help="Optional output row cap")
    return parser.parse_args()


def parse_terms(raw_terms: str) -> list[str]:
    return [term.strip() for term in raw_terms.split(",") if term.strip()]


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


def resolve_audio_path(manifest_path: Path, audio_value: str) -> str:
    if not audio_value:
        return ""
    path = Path(audio_value)
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent / path).resolve())


def count_terms_in_text(text: str, terms: list[str]) -> dict[str, int]:
    text_cf = text.casefold()
    return {term: (1 if term.casefold() in text_cf else 0) for term in terms}


def parse_line_no(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def load_bucket_rows(bucket_csv_path: Path) -> list[dict]:
    with bucket_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def allowed_target_buckets(min_target_quality: str) -> set[str]:
    if min_target_quality == "good_or_suspicious":
        return {"good", "suspicious"}
    return {"good"}


def load_bucket_indexes(
    bucket_csv_path: Path,
    exclude_buckets: set[str],
    min_target_quality: str,
) -> tuple[set[str], set[int], set[str], set[int], dict[str, int], dict[str, int]]:
    excluded_audio: set[str] = set()
    excluded_line_nos: set[int] = set()
    target_audio: set[str] = set()
    target_line_nos: set[int] = set()
    bucket_counts: dict[str, int] = {}
    target_bucket_counts: dict[str, int] = {}
    eligible_target_buckets = allowed_target_buckets(min_target_quality)

    for row in load_bucket_rows(bucket_csv_path):
        bucket = (row.get("bucket") or "").strip()
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        audio = (row.get("audio_filepath") or "").strip()
        line_no = parse_line_no(row.get("line_no"))

        if bucket in exclude_buckets:
            if audio:
                excluded_audio.add(audio)
            if line_no is not None:
                excluded_line_nos.add(line_no)

        if bucket in eligible_target_buckets:
            if audio:
                target_audio.add(audio)
            if line_no is not None:
                target_line_nos.add(line_no)
            target_bucket_counts[bucket] = target_bucket_counts.get(bucket, 0) + 1

    return excluded_audio, excluded_line_nos, target_audio, target_line_nos, bucket_counts, target_bucket_counts


def repeat_count_for_mode(mode: str, max_target_repeat: int | None) -> int:
    default = {
        "conservative": 1,
        "weighted": 2,
        "target_only": 0,
    }[mode]
    if max_target_repeat is None:
        return default
    return max(0, min(default, max_target_repeat))


def exposure_multiplier(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return after / before


def build_manifest(
    base_manifest_path: Path,
    bucket_csv_path: Path,
    output_manifest_path: Path,
    summary_path: Path,
    mode: str,
    target_terms: list[str],
    exclude_buckets: set[str],
    max_target_repeat: int | None,
    min_target_quality: str,
    max_output_rows: int | None,
) -> dict:
    (
        excluded_audio,
        excluded_line_nos,
        target_audio,
        target_line_nos,
        bucket_counts,
        target_bucket_counts,
    ) = load_bucket_indexes(bucket_csv_path, exclude_buckets, min_target_quality)

    input_row_count = 0
    excluded_row_count = 0
    output_row_count = 0
    appended_target_rows = 0
    before_counts = {term: 0 for term in target_terms}
    after_counts = {term: 0 for term in target_terms}
    selected_target_rows: list[tuple[str, dict, dict[str, int]]] = []
    base_output_lines: list[str] = []
    warnings: list[str] = []
    matched_excluded = 0
    matched_target = 0

    with base_manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            input_row_count += 1
            record = json.loads(line)
            text = extract_text(record)
            audio_key = resolve_audio_path(base_manifest_path, extract_audio_value(record))
            term_hits = count_terms_in_text(text, target_terms)
            for term, hit in term_hits.items():
                before_counts[term] += hit

            is_excluded = False
            if audio_key:
                is_excluded = audio_key in excluded_audio
            else:
                is_excluded = line_no in excluded_line_nos
            if is_excluded:
                excluded_row_count += 1
                matched_excluded += 1
                continue

            is_target_candidate = False
            if audio_key:
                is_target_candidate = audio_key in target_audio
            else:
                is_target_candidate = line_no in target_line_nos
            if is_target_candidate and any(term_hits.values()):
                matched_target += 1
                selected_target_rows.append((line, record, term_hits))

            if mode != "target_only":
                base_output_lines.append(line)
                output_row_count += 1
                for term, hit in term_hits.items():
                    after_counts[term] += hit

    if mode == "target_only":
        base_output_lines = []
        output_row_count = 0
        after_counts = {term: 0 for term in target_terms}

    append_repeats = repeat_count_for_mode(mode, max_target_repeat)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        written_rows = 0
        for line in base_output_lines:
            if max_output_rows is not None and written_rows >= max_output_rows:
                warnings.append(f"max_output_rows reached while writing base rows: {max_output_rows}")
                break
            handle.write(line)
            written_rows += 1

        if mode == "target_only":
            selected_copies = 1
        else:
            selected_copies = append_repeats

        stop_appends = False
        for line, _record, term_hits in selected_target_rows:
            for _ in range(selected_copies):
                if max_output_rows is not None and written_rows >= max_output_rows:
                    warnings.append(f"max_output_rows reached while appending target rows: {max_output_rows}")
                    stop_appends = True
                    break
                handle.write(line)
                written_rows += 1
                appended_target_rows += 1
                for term, hit in term_hits.items():
                    after_counts[term] += hit
            if stop_appends:
                break

    output_row_count = written_rows

    for term in target_terms:
        if before_counts[term] == 0:
            warnings.append(f"target term absent from base manifest: {term}")
        elif after_counts[term] <= SPARSE_TERM_THRESHOLD:
            warnings.append(f"target term remains sparse after build: {term} ({after_counts[term]})")

    if output_row_count > input_row_count * 1.5:
        warnings.append(
            f"output manifest is substantially larger than base manifest: {output_row_count} vs {input_row_count}"
        )

    summary = {
        "base_manifest": str(base_manifest_path),
        "bucket_csv": str(bucket_csv_path),
        "output_manifest": str(output_manifest_path),
        "mode": mode,
        "exclude_buckets": sorted(exclude_buckets),
        "min_target_quality": min_target_quality,
        "input_row_count": input_row_count,
        "excluded_row_count": excluded_row_count,
        "output_row_count": output_row_count,
        "appended_target_rows": appended_target_rows,
        "selected_target_rows": len(selected_target_rows),
        "matched_excluded_rows": matched_excluded,
        "matched_target_rows": matched_target,
        "bucket_counts": bucket_counts,
        "target_bucket_counts": target_bucket_counts,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "target_terms": target_terms,
        "warnings": warnings,
        "append_repeats": append_repeats,
        "max_output_rows": max_output_rows,
        "exposure_multiplier_estimate": {
            term: exposure_multiplier(before_counts[term], after_counts[term]) for term in target_terms
        },
    }
    write_summary(summary_path, summary)
    return summary


def write_summary(summary_path: Path, summary: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# v0.4 fine-tune manifest summary\n\n")
        handle.write(f"- base_manifest: `{summary['base_manifest']}`\n")
        handle.write(f"- bucket_csv: `{summary['bucket_csv']}`\n")
        handle.write(f"- output_manifest: `{summary['output_manifest']}`\n")
        handle.write(f"- mode: `{summary['mode']}`\n")
        handle.write(f"- exclude_buckets: `{','.join(summary['exclude_buckets'])}`\n")
        handle.write(f"- min_target_quality: `{summary['min_target_quality']}`\n")
        handle.write(f"- input_row_count: {summary['input_row_count']}\n")
        handle.write(f"- excluded_row_count: {summary['excluded_row_count']}\n")
        handle.write(f"- appended_target_rows: {summary['appended_target_rows']}\n")
        handle.write(f"- output_row_count: {summary['output_row_count']}\n")
        handle.write(f"- selected_target_rows: {summary['selected_target_rows']}\n")
        handle.write(f"- append_repeats: {summary['append_repeats']}\n")

        handle.write("\n## Bucket counts\n\n")
        for bucket, count in sorted(summary["bucket_counts"].items()):
            handle.write(f"- {bucket}: {count}\n")

        handle.write("\n## Eligible target bucket counts\n\n")
        if summary["target_bucket_counts"]:
            for bucket, count in sorted(summary["target_bucket_counts"].items()):
                handle.write(f"- {bucket}: {count}\n")
        else:
            handle.write("- none\n")

        handle.write("\n## Target term counts\n\n")
        handle.write("| term | before | after | exposure_multiplier_estimate |\n")
        handle.write("|---|---:|---:|---:|\n")
        for term in summary["target_terms"]:
            handle.write(
                f"| {term} | {summary['before_counts'][term]} | {summary['after_counts'][term]} | {summary['exposure_multiplier_estimate'][term]:.2f} |\n"
            )

        handle.write("\n## Warnings\n\n")
        if summary["warnings"]:
            for warning in summary["warnings"]:
                handle.write(f"- {warning}\n")
        else:
            handle.write("- none\n")


def main() -> int:
    args = parse_args()
    summary = build_manifest(
        base_manifest_path=Path(args.base_manifest).expanduser().resolve(),
        bucket_csv_path=Path(args.bucket_csv).expanduser().resolve(),
        output_manifest_path=Path(args.output_manifest).expanduser().resolve(),
        summary_path=Path(args.summary).expanduser().resolve(),
        mode=args.mode,
        target_terms=parse_terms(args.target_terms),
        exclude_buckets=set(parse_terms(args.exclude_buckets)),
        max_target_repeat=args.max_target_repeat,
        min_target_quality=args.min_target_quality,
        max_output_rows=args.max_output_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
