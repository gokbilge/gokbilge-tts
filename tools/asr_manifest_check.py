"""Planned scaffold for a future ASR/text mismatch audit."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument(
        "--output",
        required=False,
        help="Reserved for future ASR mismatch report output",
    )
    parser.parse_args()

    print("ASR manifest checking is not implemented in this first pass.")
    print("Planned direction: faster-whisper based transcript mismatch audit as a later step.")


if __name__ == "__main__":
    main()
