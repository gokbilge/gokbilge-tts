import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "audit_turkish_heavy_terms.py"
    spec = importlib.util.spec_from_file_location("audit_turkish_heavy_terms", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, frames: int = 800):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x01\x00" * frames)


def test_audit_manifest_matches_turkish_terms_and_resolves_relative_audio(tmp_path):
    module = load_module()
    manifest = tmp_path / "manifest.jsonl"
    audio_rel = Path("audio") / "sample.wav"
    audio_abs = tmp_path / audio_rel
    write_wav(audio_abs)
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": "Çocuklar çiçek ve üzüm aldı.",
                        "audio_filepath": str(audio_rel).replace('\\', '/'),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "normalized_text": "Bugün hava çok güzel.",
                        "audio": "missing.wav",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary, rows = module.audit_manifest(manifest, ["çocuklar", "çiçek", "üzüm"])

    assert summary["total_rows"] == 2
    assert summary["rows_with_any_target_term"] == 1
    assert summary["rows_with_any_target_char"] == 2
    assert summary["term_counts"]["çocuklar"] == 1
    assert len(rows) == 1
    assert rows[0]["matched_terms"] == "çocuklar|çiçek|üzüm"
    assert rows[0]["audio_filepath"] == str(audio_abs.resolve())
    assert rows[0]["sample_rate"] == 16000
    assert rows[0]["channels"] == 1


def test_outputs_are_created_and_missing_audio_does_not_crash(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sentence": "Öğrenciler ölçüm sonuçlarını değerlendirdi.",
                "wav": "missing.wav",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "out" / "rows.csv"
    output_md = tmp_path / "out" / "summary.md"
    output_jsonl = tmp_path / "out" / "rows.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "audit_turkish_heavy_terms.py"),
            "--manifest",
            str(manifest),
            "--terms",
            "ölçüm,öğrenciler",
            "--output",
            str(output_csv),
            "--summary",
            str(output_md),
            "--jsonl-output",
            str(output_jsonl),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert output_csv.is_file()
    assert output_md.is_file()
    assert output_jsonl.is_file()

    csv_text = output_csv.read_text(encoding="utf-8")
    summary_text = output_md.read_text(encoding="utf-8")
    jsonl_text = output_jsonl.read_text(encoding="utf-8")

    assert "Öğrenciler ölçüm sonuçlarını değerlendirdi." in csv_text
    assert "rows with any target term: 1" in summary_text
    assert "ölçüm: 1" in summary_text
    assert "Öğrenciler ölçüm sonuçlarını değerlendirdi." in jsonl_text
