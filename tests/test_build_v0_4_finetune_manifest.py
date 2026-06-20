import csv
import importlib.util
import json
from pathlib import Path

TERM_COCUKLAR = "\u00e7ocuklar"
TERM_CICEK = "\u00e7i\u00e7ek"
TERM_OGR = "\u00f6\u011frenciler"
TERM_SEKER = "\u015feker"
ALL_TERMS = [TERM_COCUKLAR, TERM_CICEK, TERM_OGR, TERM_SEKER]


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "build_v0_4_finetune_manifest.py"
    spec = importlib.util.spec_from_file_location("build_v0_4_finetune_manifest", module_path)
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


def manifest_rows():
    return [
        {"text": "alpha non target", "audio_filepath": "audio/001.wav"},
        {"text": f"{TERM_COCUKLAR} iyi satir", "audio_filepath": "audio/002.wav"},
        {"text": f"{TERM_CICEK} kotu satir", "audio_filepath": "audio/003.wav"},
        {"text": f"{TERM_OGR} supheli satir", "audio_filepath": "audio/004.wav"},
        {"text": f"{TERM_SEKER} iyi satir", "audio_filepath": "audio/005.wav"},
        {"text": "beta non target", "audio_filepath": "audio/006.wav"},
    ]


def bucket_rows(tmp_path: Path):
    return [
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
        {
            "line_no": "5",
            "audio_filepath": str((tmp_path / "audio/005.wav").resolve()),
            "bucket": "good",
            "matched_terms": TERM_SEKER,
        },
    ]


def run_builder(tmp_path: Path, mode: str, min_target_quality: str = "good"):
    module = load_module()
    manifest_path = tmp_path / "manifest.jsonl"
    bucket_path = tmp_path / "buckets.csv"
    output_path = tmp_path / f"{mode}.jsonl"
    summary_path = tmp_path / f"{mode}.md"
    write_manifest(manifest_path, manifest_rows())
    write_bucket_csv(bucket_path, bucket_rows(tmp_path))
    summary = module.build_manifest(
        base_manifest_path=manifest_path,
        bucket_csv_path=bucket_path,
        output_manifest_path=output_path,
        summary_path=summary_path,
        mode=mode,
        target_terms=ALL_TERMS,
        exclude_buckets={"bad_or_reject_like"},
        max_target_repeat=None,
        min_target_quality=min_target_quality,
        max_output_rows=None,
    )
    return summary, output_path, summary_path


def test_excludes_bad_target_rows_and_keeps_non_target_rows(tmp_path):
    summary, output_path, _ = run_builder(tmp_path, "conservative")
    texts = read_manifest_texts(output_path)
    assert summary["excluded_row_count"] == 1
    assert "alpha non target" in texts
    assert "beta non target" in texts
    assert f"{TERM_CICEK} kotu satir" not in texts


def test_conservative_mode_appends_good_target_rows_once(tmp_path):
    summary, output_path, _ = run_builder(tmp_path, "conservative")
    texts = read_manifest_texts(output_path)
    assert summary["appended_target_rows"] == 2
    assert texts.count(f"{TERM_COCUKLAR} iyi satir") == 2
    assert texts.count(f"{TERM_SEKER} iyi satir") == 2


def test_weighted_mode_appends_more_than_conservative(tmp_path):
    conservative_summary, _, _ = run_builder(tmp_path, "conservative")
    weighted_summary, output_path, _ = run_builder(tmp_path, "weighted")
    texts = read_manifest_texts(output_path)
    assert weighted_summary["appended_target_rows"] > conservative_summary["appended_target_rows"]
    assert texts.count(f"{TERM_COCUKLAR} iyi satir") == 3


def test_target_only_includes_only_good_target_rows(tmp_path):
    summary, output_path, _ = run_builder(tmp_path, "target_only")
    texts = read_manifest_texts(output_path)
    assert summary["output_row_count"] == 2
    assert texts == [f"{TERM_COCUKLAR} iyi satir", f"{TERM_SEKER} iyi satir"]


def test_deterministic_order_and_good_or_suspicious_utf8_matching(tmp_path):
    summary, output_path, _ = run_builder(tmp_path, "conservative", min_target_quality="good_or_suspicious")
    texts = read_manifest_texts(output_path)
    assert texts[:5] == [
        "alpha non target",
        f"{TERM_COCUKLAR} iyi satir",
        f"{TERM_OGR} supheli satir",
        f"{TERM_SEKER} iyi satir",
        "beta non target",
    ]
    assert texts[5:] == [
        f"{TERM_COCUKLAR} iyi satir",
        f"{TERM_OGR} supheli satir",
        f"{TERM_SEKER} iyi satir",
    ]
    assert summary["after_counts"][TERM_OGR] == 2



def test_audio_key_takes_priority_over_line_number_fallback(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "manifest.jsonl"
    bucket_path = tmp_path / "buckets.csv"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "out.md"
    write_manifest(
        manifest_path,
        [
            {"text": "alpha non target", "audio_filepath": "audio/001.wav"},
            {"text": f"{TERM_COCUKLAR} iyi satir", "audio_filepath": "audio/002.wav"},
        ],
    )
    write_bucket_csv(
        bucket_path,
        [
            {
                "line_no": "1",
                "audio_filepath": str((tmp_path / "audio/999.wav").resolve()),
                "bucket": "bad_or_reject_like",
                "matched_terms": TERM_CICEK,
            },
            {
                "line_no": "2",
                "audio_filepath": str((tmp_path / "audio/002.wav").resolve()),
                "bucket": "good",
                "matched_terms": TERM_COCUKLAR,
            },
        ],
    )
    summary = module.build_manifest(
        base_manifest_path=manifest_path,
        bucket_csv_path=bucket_path,
        output_manifest_path=output_path,
        summary_path=summary_path,
        mode="conservative",
        target_terms=ALL_TERMS,
        exclude_buckets={"bad_or_reject_like"},
        max_target_repeat=None,
        min_target_quality="good",
        max_output_rows=None,
    )
    texts = read_manifest_texts(output_path)
    assert summary["excluded_row_count"] == 0
    assert texts.count(f"{TERM_COCUKLAR} iyi satir") == 2
