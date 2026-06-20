import importlib.util
import csv
import json
from pathlib import Path

TERM_COCUKLAR = "\u00e7ocuklar"
TERM_CICEK = "\u00e7i\u00e7ek"
TERM_OGR = "\u00f6\u011frenciler"
TERM_SEKER = "\u015feker"


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
    fieldnames = [
        "line_no",
        "audio_filepath",
        "bucket",
        "matched_terms",
        "bucket_reasons",
        "longest_internal_gap_sec",
        "internal_gap_count",
        "internal_silence_ratio",
        "low_energy_ratio",
    ]
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


def build_rows(tmp_path: Path) -> list[dict]:
    return [
        {"text": "alpha non target", "audio_filepath": "audio/001.wav"},
        {"text": TERM_COCUKLAR + " iyi satir", "audio_filepath": "audio/002.wav"},
        {"text": TERM_CICEK + " hafif gap satir", "audio_filepath": "audio/003.wav"},
        {"text": TERM_OGR + " siddetli gap satir", "audio_filepath": "audio/004.wav"},
        {"text": TERM_SEKER + " clip satir", "audio_filepath": "audio/005.wav"},
        {"text": "beta non target", "audio_filepath": "audio/006.wav"},
    ]


def build_bucket_rows(tmp_path: Path) -> list[dict]:
    return [
        {
            "line_no": "2",
            "audio_filepath": str((tmp_path / "audio/002.wav").resolve()),
            "bucket": "good",
            "matched_terms": TERM_COCUKLAR,
            "bucket_reasons": "",
            "longest_internal_gap_sec": "0.00",
            "internal_gap_count": "0",
            "internal_silence_ratio": "0.00",
            "low_energy_ratio": "0.00",
        },
        {
            "line_no": "3",
            "audio_filepath": str((tmp_path / "audio/003.wav").resolve()),
            "bucket": "bad_or_reject_like",
            "matched_terms": TERM_CICEK,
            "bucket_reasons": "internal_gap_count_too_high|longest_internal_gap_too_high",
            "longest_internal_gap_sec": "0.46",
            "internal_gap_count": "3",
            "internal_silence_ratio": "0.20",
            "low_energy_ratio": "0.20",
        },
        {
            "line_no": "4",
            "audio_filepath": str((tmp_path / "audio/004.wav").resolve()),
            "bucket": "bad_or_reject_like",
            "matched_terms": TERM_OGR,
            "bucket_reasons": "longest_internal_gap_too_high|internal_gap_count_too_high",
            "longest_internal_gap_sec": "0.90",
            "internal_gap_count": "6",
            "internal_silence_ratio": "0.40",
            "low_energy_ratio": "0.40",
        },
        {
            "line_no": "5",
            "audio_filepath": str((tmp_path / "audio/005.wav").resolve()),
            "bucket": "bad_or_reject_like",
            "matched_terms": TERM_SEKER,
            "bucket_reasons": "peak_near_clip",
            "longest_internal_gap_sec": "0.00",
            "internal_gap_count": "0",
            "internal_silence_ratio": "0.00",
            "low_energy_ratio": "0.00",
        },
    ]


def run_builder(tmp_path: Path, exclusion_mode: str, min_retention: float = 0.80):
    module = load_module()
    manifest_path = tmp_path / "manifest.jsonl"
    output_manifest = tmp_path / f"output_{exclusion_mode}.jsonl"
    summary_path = tmp_path / f"summary_{exclusion_mode}.md"
    write_manifest(manifest_path, build_rows(tmp_path))
    write_bucket_csv(tmp_path / "buckets.csv", build_bucket_rows(tmp_path))
    summary = module.build_manifest(
        balanced_manifest_path=manifest_path,
        bucket_csv_path=tmp_path / "buckets.csv",
        output_manifest_path=output_manifest,
        summary_path=summary_path,
        exclude_bucket="bad_or_reject_like",
        exclusion_mode=exclusion_mode,
        protect_terms=[TERM_COCUKLAR, TERM_CICEK, TERM_OGR, TERM_SEKER],
        min_protected_retention=min_retention,
    )
    return summary, output_manifest, summary_path


def test_default_aggressive_mode_keeps_existing_behavior(tmp_path):
    summary, output_manifest, _ = run_builder(tmp_path, "aggressive")
    assert summary["exclusion_mode"] == "aggressive"
    assert summary["excluded_row_count"] == 3
    assert read_manifest_texts(output_manifest) == [
        "alpha non target",
        TERM_COCUKLAR + " iyi satir",
        "beta non target",
    ]


def test_relaxed_mode_keeps_mild_gap_only_row(tmp_path):
    summary, output_manifest, _ = run_builder(tmp_path, "relaxed")
    assert summary["excluded_row_count"] == 2
    texts = read_manifest_texts(output_manifest)
    assert TERM_CICEK + " hafif gap satir" in texts
    assert TERM_OGR + " siddetli gap satir" not in texts
    assert TERM_SEKER + " clip satir" not in texts


def test_relaxed_mode_excludes_severe_gap_and_high_gap_count(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "manifest.jsonl"
    output_manifest = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.md"
    rows = [
        {"text": TERM_OGR + " bir", "audio_filepath": "audio/001.wav"},
        {"text": TERM_OGR + " iki", "audio_filepath": "audio/002.wav"},
    ]
    write_manifest(manifest_path, rows)
    write_bucket_csv(
        tmp_path / "buckets.csv",
        [
            {
                "line_no": "1",
                "audio_filepath": str((tmp_path / "audio/001.wav").resolve()),
                "bucket": "bad_or_reject_like",
                "matched_terms": TERM_OGR,
                "bucket_reasons": "longest_internal_gap_too_high",
                "longest_internal_gap_sec": "0.90",
                "internal_gap_count": "1",
                "internal_silence_ratio": "0.20",
                "low_energy_ratio": "0.20",
            },
            {
                "line_no": "2",
                "audio_filepath": str((tmp_path / "audio/002.wav").resolve()),
                "bucket": "bad_or_reject_like",
                "matched_terms": TERM_OGR,
                "bucket_reasons": "internal_gap_count_too_high",
                "longest_internal_gap_sec": "0.20",
                "internal_gap_count": "6",
                "internal_silence_ratio": "0.20",
                "low_energy_ratio": "0.20",
            },
        ],
    )

    summary = module.build_manifest(
        balanced_manifest_path=manifest_path,
        bucket_csv_path=tmp_path / "buckets.csv",
        output_manifest_path=output_manifest,
        summary_path=summary_path,
        exclude_bucket="bad_or_reject_like",
        exclusion_mode="relaxed",
        protect_terms=[TERM_OGR],
        min_protected_retention=0.80,
    )
    assert summary["excluded_row_count"] == 2


def test_hard_mode_keeps_gap_only_row_but_excludes_clip_or_rms(tmp_path):
    summary, output_manifest, _ = run_builder(tmp_path, "hard")
    texts = read_manifest_texts(output_manifest)
    assert TERM_CICEK + " hafif gap satir" in texts
    assert TERM_OGR + " siddetli gap satir" in texts
    assert TERM_SEKER + " clip satir" not in texts
    assert summary["excluded_row_count"] == 1


def test_retention_summary_and_warnings_still_work(tmp_path):
    summary, _, summary_path = run_builder(tmp_path, "aggressive", min_retention=0.90)
    assert summary["before_counts"][TERM_CICEK] == 1
    assert summary["after_counts"][TERM_CICEK] == 0
    assert summary["retention"][TERM_CICEK] == 0.0
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Protected term retention" in summary_text
    assert "protected term retention below threshold" in summary_text
    assert TERM_CICEK in summary_text
