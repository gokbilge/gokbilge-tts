"""Generate filtered manifest candidates from an audit CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from gokbilge_tts.datasets.dataset_cleaning import filter_manifest_with_audit, load_audit_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Original manifest JSONL")
    parser.add_argument("--audit-csv", type=Path, required=True, help="Audit CSV from audit_dataset")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Filtered manifest JSONL")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["strict", "balanced", "rejects", "suspicious"],
        help="Filtering strategy",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_rows = load_audit_csv(args.audit_csv)
    result = filter_manifest_with_audit(
        manifest_path=args.manifest,
        audit_rows=audit_rows,
        output_manifest_path=args.output_manifest,
        mode=args.mode,
    )
    print(
        f"mode={result['mode']} kept={result['kept']} dropped={result['dropped']} "
        f"total={result['total']} output={args.output_manifest}"
    )


if __name__ == "__main__":
    main()
