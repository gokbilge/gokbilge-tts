"""Run audit + filtered-manifest generation for v0.2 dataset cleaning."""

from __future__ import annotations

import argparse
from pathlib import Path

from gokbilge_tts.datasets.dataset_cleaning import AuditThresholds, build_clean_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest JSONL")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for audit CSVs and Markdown summary",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory for filtered manifest candidates",
    )
    parser.add_argument("--min-duration-sec", type=float, default=0.8)
    parser.add_argument("--max-duration-sec", type=float, default=15.0)
    parser.add_argument("--min-text-chars", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=250)
    parser.add_argument("--min-chars-per-sec", type=float, default=6.0)
    parser.add_argument("--max-chars-per-sec", type=float, default=22.0)
    parser.add_argument("--max-leading-silence-sec", type=float, default=0.5)
    parser.add_argument("--max-trailing-silence-sec", type=float, default=0.7)
    parser.add_argument("--max-internal-silence-ratio", type=float, default=0.40)
    parser.add_argument("--max-longest-internal-gap-sec", type=float, default=0.45)
    parser.add_argument("--max-clipping-ratio", type=float, default=0.01)
    parser.add_argument("--silence-threshold-db", type=float, default=-40.0)
    parser.add_argument("--silence-frame-ms", type=float, default=20.0)
    parser.add_argument("--silence-hop-ms", type=float, default=10.0)
    parser.add_argument("--min-rms-db", type=float, default=-30.0)
    parser.add_argument("--max-peak-db", type=float, default=-0.5)
    return parser


def thresholds_from_args(args: argparse.Namespace) -> AuditThresholds:
    return AuditThresholds(
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        min_text_chars=args.min_text_chars,
        max_text_chars=args.max_text_chars,
        min_chars_per_sec=args.min_chars_per_sec,
        max_chars_per_sec=args.max_chars_per_sec,
        max_leading_silence_sec=args.max_leading_silence_sec,
        max_trailing_silence_sec=args.max_trailing_silence_sec,
        max_internal_silence_ratio=args.max_internal_silence_ratio,
        max_longest_internal_gap_sec=args.max_longest_internal_gap_sec,
        max_clipping_ratio=args.max_clipping_ratio,
        silence_threshold_db=args.silence_threshold_db,
        silence_frame_ms=args.silence_frame_ms,
        silence_hop_ms=args.silence_hop_ms,
        min_rms_db=args.min_rms_db,
        max_peak_db=args.max_peak_db,
    )


def main() -> None:
    args = build_parser().parse_args()
    thresholds = thresholds_from_args(args)
    results = build_clean_manifests(
        manifest_path=args.manifest,
        audit_csv_path=args.reports_dir / "dataset_audit.csv",
        summary_path=args.reports_dir / "dataset_quality_summary.md",
        manifests_dir=args.manifests_dir,
        reports_dir=args.reports_dir,
        thresholds=thresholds,
    )

    for mode, result in results.items():
        print(
            f"{mode}: kept={result['kept']} dropped={result['dropped']} "
            f"total={result['total']}"
        )

    print(f"audit_csv={args.reports_dir / 'dataset_audit.csv'}")
    print(f"summary={args.reports_dir / 'dataset_quality_summary.md'}")


if __name__ == "__main__":
    main()
