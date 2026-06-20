#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

TEXT_KEYS = ("text", "sentence", "normalized_text")
AUDIO_KEYS = ("audio_filepath", "audio", "path", "wav")
DEFAULT_EXCLUDE_BUCKET = "bad_or_reject_like"
DEFAULT_MIN_PROTECTED_RETENTION = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a v0.3 Turkish-heavy clean manifest from a balanced manifest and bucket CSV."
    )
    parser.add_argument("--balanced-manifest", required=True, help="Input balanced manifest JSONL path")
    parser.add_argument("--bucket-csv", required=True, help="Bucket CSV path")
    parser.add_argument("--output-manifest", required=True, help="Output manifest JSONL path")
    parser.add_argument("--summary", required=True, help="Summary markdown path")
    parser.add_argument(
        "--exclude-bucket",
        default=DEFAULT_EXCLUDE_BUCKET,
        help=f"Bucket value to exclude (default: {DEFAULT_EXCLUDE_BUCKET})",
    )
    parser.add_argument(
        "--protect-terms",
        help="Optional comma-separated protected terms for retention reporting",
    )
    parser.add_argument(
        "--min-protected-retention",
        type=float,
        default=DEFAULT_MIN_PROTECTED_RETENTION,
        help=(
            "Minimum acceptable protected-term retention fraction "
            f"(default: {DEFAULT_MIN_PROTECTED_RETENTION})"
        ),
    )
    return parser.parse_args()


def parse_terms(raw_terms: str | None) -> list[str]:
    if not raw_terms:
        return []
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
    audio_path = Path(audio_value)
    if audio_path.is_absolute():
        return str(audio_path)
    return str((manifest_path.parent / audio_path).resolve())


def count_terms_in_text(text: str, terms: list[str]) -> dict[str, int]:
    text_casefold = text.casefold()
    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = 1 if term.casefold() in text_casefold else 0
    return counts


def load_bucket_exclusions(bucket_csv_path: Path, exclude_bucket: str) -> tuple[set[str], set[int], dict[str, int]]:
    excluded_audio_paths: set[str] = set()
    excluded_line_numbers: set[int] = set()
    stats = {
        "bucket_rows_total": 0,
        "bucket_rows_matching_exclude": 0,
        "bucket_rows_with_audio_key": 0,
        "bucket_rows_with_line_key": 0,
    }

    with bucket_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["bucket_rows_total"] += 1
            if row.get("bucket") != exclude_bucket:
                continue
            stats["bucket_rows_matching_exclude"] += 1

            audio_path = (row.get("audio_filepath") or "").strip()
            if audio_path:
                excluded_audio_paths.add(audio_path)
                stats["bucket_rows_with_audio_key"] += 1

            line_no_text = (row.get("line_no") or "").strip()
            if line_no_text:
                try:
                    excluded_line_numbers.add(int(float(line_no_text)))
                    stats["bucket_rows_with_line_key"] += 1
                except ValueError:
                    pass

    return excluded_audio_paths, excluded_line_numbers, stats


def compute_retention(before: int, after: int) -> float:
    if before <= 0:
        return 1.0
    return after / before


def build_manifest(
    balanced_manifest_path: Path,
    bucket_csv_path: Path,
    output_manifest_path: Path,
    summary_path: Path,
    exclude_bucket: str,
    protect_terms: list[str],
    min_protected_retention: float,
) -> dict:
    excluded_audio_paths, excluded_line_numbers, bucket_stats = load_bucket_exclusions(
        bucket_csv_path,
        exclude_bucket,
    )

    before_counts = {term: 0 for term in protect_terms}
    after_counts = {term: 0 for term in protect_terms}
    matched_excluded_audio_paths: set[str] = set()
    matched_excluded_line_numbers: set[int] = set()

    input_row_count = 0
    excluded_row_count = 0
    output_row_count = 0

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with balanced_manifest_path.open("r", encoding="utf-8") as src, output_manifest_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            input_row_count += 1
            record = json.loads(line)
            text = extract_text(record)
            audio_key = resolve_audio_path(balanced_manifest_path, extract_audio_value(record))

            row_term_hits = count_terms_in_text(text, protect_terms)
            for term, hit in row_term_hits.items():
                before_counts[term] += hit

            exclude_row = False
            if audio_key and audio_key in excluded_audio_paths:
                exclude_row = True
                matched_excluded_audio_paths.add(audio_key)
            if line_no in excluded_line_numbers:
                exclude_row = True
                matched_excluded_line_numbers.add(line_no)

            if exclude_row:
                excluded_row_count += 1
                continue

            dst.write(line)
            output_row_count += 1
            for term, hit in row_term_hits.items():
                after_counts[term] += hit

    summary = {
        "balanced_manifest": str(balanced_manifest_path),
        "bucket_csv": str(bucket_csv_path),
        "output_manifest": str(output_manifest_path),
        "input_row_count": input_row_count,
        "excluded_row_count": excluded_row_count,
        "output_row_count": output_row_count,
        "exclude_bucket": exclude_bucket,
        "bucket_stats": bucket_stats,
        "matched_excluded_audio_path_count": len(matched_excluded_audio_paths),
        "matched_excluded_line_number_count": len(matched_excluded_line_numbers),
        "unmatched_excluded_audio_path_count": len(excluded_audio_paths - matched_excluded_audio_paths),
        "unmatched_excluded_line_number_count": len(excluded_line_numbers - matched_excluded_line_numbers),
        "protected_terms": protect_terms,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "retention": {
            term: compute_retention(before_counts[term], after_counts[term]) for term in protect_terms
        },
        "min_protected_retention": min_protected_retention,
    }
    write_summary(summary_path, summary)
    return summary


def write_summary(summary_path: Path, summary: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# v0.3 Turkish-heavy clean manifest summary\n\n")
        handle.write(f"- balanced_manifest: `{summary['balanced_manifest']}`\n")
        handle.write(f"- bucket_csv: `{summary['bucket_csv']}`\n")
        handle.write(f"- output_manifest: `{summary['output_manifest']}`\n")
        handle.write(f"- exclude_bucket: `{summary['exclude_bucket']}`\n")
        handle.write(f"- input_row_count: {summary['input_row_count']}\n")
        handle.write(f"- excluded_row_count: {summary['excluded_row_count']}\n")
        handle.write(f"- output_row_count: {summary['output_row_count']}\n")
        handle.write("\n## Bucket match stats\n\n")
        handle.write(
            f"- bucket_rows_total: {summary['bucket_stats']['bucket_rows_total']}\n"
        )
        handle.write(
            f"- bucket_rows_matching_exclude: {summary['bucket_stats']['bucket_rows_matching_exclude']}\n"
        )
        handle.write(
            f"- bucket_rows_with_audio_key: {summary['bucket_stats']['bucket_rows_with_audio_key']}\n"
        )
        handle.write(
            f"- bucket_rows_with_line_key: {summary['bucket_stats']['bucket_rows_with_line_key']}\n"
        )
        handle.write(
            f"- matched_excluded_audio_path_count: {summary['matched_excluded_audio_path_count']}\n"
        )
        handle.write(
            f"- matched_excluded_line_number_count: {summary['matched_excluded_line_number_count']}\n"
        )
        handle.write(
            f"- unmatched_excluded_audio_path_count: {summary['unmatched_excluded_audio_path_count']}\n"
        )
        handle.write(
            f"- unmatched_excluded_line_number_count: {summary['unmatched_excluded_line_number_count']}\n"
        )

        if summary["protected_terms"]:
            handle.write("\n## Protected term retention\n\n")
            handle.write("| term | before | after | retention_pct | status |\n")
            handle.write("|---|---:|---:|---:|---|\n")
            for term in summary["protected_terms"]:
                before = summary["before_counts"][term]
                after = summary["after_counts"][term]
                retention = summary["retention"][term]
                status = "OK"
                if retention < summary["min_protected_retention"]:
                    status = "WARNING"
                handle.write(
                    f"| {term} | {before} | {after} | {retention * 100:.2f} | {status} |\n"
                )

            warnings = [
                term
                for term in summary["protected_terms"]
                if summary["retention"][term] < summary["min_protected_retention"]
            ]
            handle.write("\n## Warnings\n\n")
            if warnings:
                for term in warnings:
                    retention = summary["retention"][term] * 100.0
                    handle.write(
                        f"- protected term retention below threshold: {term} ({retention:.2f}% < {summary['min_protected_retention'] * 100:.2f}%)\n"
                    )
            else:
                handle.write("- none\n")


def main() -> int:
    args = parse_args()
    balanced_manifest_path = Path(args.balanced_manifest).expanduser().resolve()
    bucket_csv_path = Path(args.bucket_csv).expanduser().resolve()
    output_manifest_path = Path(args.output_manifest).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()

    if not balanced_manifest_path.is_file():
        raise SystemExit(f"Balanced manifest not found: {balanced_manifest_path}")
    if not bucket_csv_path.is_file():
        raise SystemExit(f"Bucket CSV not found: {bucket_csv_path}")

    protect_terms = parse_terms(args.protect_terms)
    build_manifest(
        balanced_manifest_path=balanced_manifest_path,
        bucket_csv_path=bucket_csv_path,
        output_manifest_path=output_manifest_path,
        summary_path=summary_path,
        exclude_bucket=args.exclude_bucket,
        protect_terms=protect_terms,
        min_protected_retention=args.min_protected_retention,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
