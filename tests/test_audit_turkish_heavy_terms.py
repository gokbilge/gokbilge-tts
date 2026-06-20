import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
TEXT_TARGET = "\u00c7ocuklar \u00e7i\u00e7ek ve \u00fcz\u00fcm ald\u0131."
WEATHER_TEXT = "Bug\u00fcn hava \u00e7ok g\u00fczel."
METRIC_TEXT = "\u00d6\u011frenciler \u00f6l\u00e7\u00fcm sonu\u00e7lar\u0131n\u0131 de\u011ferlendirdi."
TERM_LIST = ["\u00e7ocuklar", "\u00e7i\u00e7ek", "\u00fcz\u00fcm"]
CLI_TERMS = "\u00f6l\u00e7\u00fcm,\u00f6\u011frenciler"


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "audit_turkish_heavy_terms.py"
    spec = importlib.util.spec_from_file_location("audit_turkish_heavy_terms", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pcm_sample_bytes(value: int) -> bytes:
    return int(value).to_bytes(SAMPLE_WIDTH, byteorder="little", signed=True)


def write_pattern_wav(path: Path, amplitudes: list[int], window_ms: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_per_window = SAMPLE_RATE * window_ms // 1000
    frame_bytes = bytearray()
    for amplitude in amplitudes:
        frame_bytes.extend(pcm_sample_bytes(amplitude) * frames_per_window)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(bytes(frame_bytes))


def test_audit_manifest_matches_utf8_terms_and_resolves_relative_audio(tmp_path):
    module = load_module()
    manifest = tmp_path / "manifest.jsonl"
    audio_rel = Path("audio") / "sample.wav"
    audio_abs = tmp_path / audio_rel
    write_pattern_wav(audio_abs, [2000] * 5)
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": TEXT_TARGET,
                        "audio_filepath": str(audio_rel).replace('\\', '/'),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "normalized_text": WEATHER_TEXT,
                        "audio": "missing.wav",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary, rows = module.audit_manifest(manifest, TERM_LIST)

    assert summary["total_rows"] == 2
    assert summary["rows_with_any_target_term"] == 1
    assert summary["rows_with_any_target_char"] == 2
    assert summary["term_counts"]["\u00e7ocuklar"] == 1
    assert len(rows) == 1
    assert rows[0]["matched_terms"] == "\u00e7ocuklar|\u00e7i\u00e7ek|\u00fcz\u00fcm"
    assert rows[0]["audio_filepath"] == str(audio_abs.resolve())
    assert rows[0]["sample_rate"] == 16000
    assert rows[0]["channels"] == 1
    assert rows[0]["leading_silence_sec"] == 0.0
    assert rows[0]["trailing_silence_sec"] == 0.0


def test_leading_and_trailing_silence_metrics_are_non_zero(tmp_path):
    module = load_module()
    audio_path = tmp_path / "silence_edges.wav"
    write_pattern_wav(audio_path, [0, 0, 3000, 3000, 0, 0])

    stats = module.audio_stats(audio_path)

    assert stats["leading_silence_sec"] and stats["leading_silence_sec"] > 0
    assert stats["trailing_silence_sec"] and stats["trailing_silence_sec"] > 0
    assert stats["low_energy_ratio"] and stats["low_energy_ratio"] > 0


def test_internal_gap_metrics_are_non_zero_for_internal_silence(tmp_path):
    module = load_module()
    audio_path = tmp_path / "internal_gap.wav"
    write_pattern_wav(audio_path, [3200, 3200, 0, 0, 0, 0, 0, 3200, 3200])

    stats = module.audio_stats(audio_path)

    assert stats["leading_silence_sec"] == 0.0
    assert stats["trailing_silence_sec"] == 0.0
    assert stats["internal_gap_count"] and stats["internal_gap_count"] > 0
    assert stats["longest_internal_gap_sec"] and stats["longest_internal_gap_sec"] > 0
    assert stats["internal_silence_ratio"] and stats["internal_silence_ratio"] > 0


def test_outputs_are_created_and_missing_audio_does_not_crash(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sentence": METRIC_TEXT,
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
            CLI_TERMS,
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

    assert "leading_silence_sec" in csv_text.splitlines()[0]
    assert "internal_gap_count" in csv_text.splitlines()[0]
    assert METRIC_TEXT in csv_text
    assert "rows with any target term: 1" in summary_text
    assert "\u00f6l\u00e7\u00fcm: 1" in summary_text
    assert "leading_silence_sec: no readable values" in summary_text
    assert METRIC_TEXT in jsonl_text
