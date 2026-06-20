import importlib.util
import csv
import json
from pathlib import Path

TERM_COCUKLAR = "\u00e7ocuklar"
TERM_CICEK = "\u00e7i\u00e7ek"
TERM_OGR = "\u00f6\u011frenciler"


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "build_v0_3_turkish_heavy_manifest.py"
    spec = importlib.util.spec_from_file_location("build_v0_3_turkish_heavy_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_bucket_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["line_no", "audio_filepath", "bucket", "matched_terms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest_texts(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


def test_builder_excludes_only_bad_rows_and_preserves_order(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "train_clean_balanced.jsonl"
    output_manifest = tmp_path / "train_v0_3.jsonl"
    summary_path = tmp_path / "summary.md"

    rows = [
        {"text": "alpha non target", "audio_filepath": "audio/001.wav"},
        {"text": TERM_COCUKLAR + " iyi satir", "audio_filepath": "audio/002.wav"},
        {"text": TERM_CICEK + " kotu satir", "audio_filepath": "audio/003.wav"},
        {"text": TERM_OGR + " supheli satir", "audio_filepath": "audio/004.wav"},
        {"text": "beta non target", "audio_filepath": "audio/005.wav"},
    ]
    write_manifest(manifest_path, rows)

    write_bucket_csv(
        tmp_path / "buckets.csv",
        [
            {
                "line_no": "2",
                "audio_filepath": str((tmp_path / "audio/002.wav").resolve()),
                "bucket": "good",
                "matched_terms": TERM_COCUKLAR,
            },
            {
                "line_no": "3",
                "audio_filepath": str((tmp_path / "audio/003.wav").resolve()),
                "bucket": "bad_or_reject_like",
                "matched_terms": TERM_CICEK,
            },
            {
                "line_no": "4",
                "audio_filepath": str((tmp_path / "audio/004.wav").resolve()),
                "bucket": "suspicious",
                "matched_terms": TERM_OGR,
            },
        ],
    )

    summary = module.build_manifest(
        balanced_manifest_path=manifest_path,
        bucket_csv_path=tmp_path / "buckets.csv",
        output_manifest_path=output_manifest,
        summary_path=summary_path,
        exclude_bucket="bad_or_reject_like",
        protect_terms=[TERM_COCUKLAR, TERM_CICEK, TERM_OGR],
        min_protected_retention=0.80,
    )

    assert summary["input_row_count"] == 5
    assert summary["excluded_row_count"] == 1
    assert summary["output_row_count"] == 4
    assert read_manifest_texts(output_manifest) == [
        "alpha non target",
        TERM_COCUKLAR + " iyi satir",
        TERM_OGR + " supheli satir",
        "beta non target",
    ]


def test_builder_preserves_non_target_rows_and_keeps_suspicious_rows(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "manifest.jsonl"
    output_manifest = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.md"

    rows = [
        {"text": TERM_COCUKLAR + " satir", "audio_filepath": "audio/001.wav"},
        {"text": "non target satir", "audio_filepath": "audio/002.wav"},
        {"text": TERM_OGR + " satir", "audio_filepath": "audio/003.wav"},
    ]
    write_manifest(manifest_path, rows)
    write_bucket_csv(
        tmp_path / "buckets.csv",
        [
            {
                "line_no": "1",
                "audio_filepath": str((tmp_path / "audio/001.wav").resolve()),
                "bucket": "good",
                "matched_terms": TERM_COCUKLAR,
            },
            {
                "line_no": "3",
                "audio_filepath": str((tmp_path / "audio/003.wav").resolve()),
                "bucket": "suspicious",
                "matched_terms": TERM_OGR,
            },
        ],
    )

    module.build_manifest(
        balanced_manifest_path=manifest_path,
        bucket_csv_path=tmp_path / "buckets.csv",
        output_manifest_path=output_manifest,
        summary_path=summary_path,
        exclude_bucket="bad_or_reject_like",
        protect_terms=[TERM_COCUKLAR, TERM_OGR],
        min_protected_retention=0.80,
    )

    assert read_manifest_texts(output_manifest) == [
        TERM_COCUKLAR + " satir",
        "non target satir",
        TERM_OGR + " satir",
    ]


def test_builder_supports_line_number_matching_and_retention_warning(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "manifest.jsonl"
    output_manifest = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.md"

    rows = [
        {"text": TERM_COCUKLAR + " bir", "audio_filepath": "audio/001.wav"},
        {"text": TERM_COCUKLAR + " iki", "audio_filepath": "audio/002.wav"},
        {"text": TERM_COCUKLAR + " uc", "audio_filepath": "audio/003.wav"},
        {"text": TERM_CICEK + " dort", "audio_filepath": "audio/004.wav"},
    ]
    write_manifest(manifest_path, rows)
    write_bucket_csv(
        tmp_path / "buckets.csv",
        [
            {
                "line_no": "2",
                "audio_filepath": "",
                "bucket": "bad_or_reject_like",
                "matched_terms": TERM_COCUKLAR,
            },
            {
                "line_no": "3",
                "audio_filepath": "",
                "bucket": "bad_or_reject_like",
                "matched_terms": TERM_COCUKLAR,
            },
        ],
    )

    summary = module.build_manifest(
        balanced_manifest_path=manifest_path,
        bucket_csv_path=tmp_path / "buckets.csv",
        output_manifest_path=output_manifest,
        summary_path=summary_path,
        exclude_bucket="bad_or_reject_like",
        protect_terms=[TERM_COCUKLAR, TERM_CICEK],
        min_protected_retention=0.80,
    )

    assert summary["before_counts"][TERM_COCUKLAR] == 3
    assert summary["after_counts"][TERM_COCUKLAR] == 1
    assert round(summary["retention"][TERM_COCUKLAR], 4) == round(1 / 3, 4)

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "WARNING" in summary_text
    assert "protected term retention below threshold" in summary_text
    assert TERM_COCUKLAR in summary_text
