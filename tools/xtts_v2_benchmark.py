#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SENTENCES = [
    ("s1_bugun_hava.wav", "Bugün hava çok güzel."),
    ("s2_turkiye_cumh.wav", "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."),
    ("s3_cocuklar.wav", "Çocuklar çiçek, şeker ve üzüm yedi."),
    ("s4_ogrenciler.wav", "Öğrenciler ölçüm sonuçlarını değerlendirdi."),
    ("s5_sirket.wav", "Şirket yüzde otuz beş büyüme açıkladı."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XTTS v2 benchmark samples.")
    parser.add_argument("--out-dir", required=True, help="Output directory for WAVs and metadata.")
    parser.add_argument("--speaker-wav", help="Optional reference speaker WAV path.")
    parser.add_argument("--language", default="tr", help="XTTS language code. Default: tr")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, ... Default: auto",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tts(device: str) -> TTS:
    patch_torch_load_defaults()
    tts = TTS(model_name=MODEL_NAME, progress_bar=True)
    if hasattr(tts, "to"):
        tts = tts.to(device)
    return tts


def patch_torch_load_defaults() -> None:
    original_torch_load = torch.load
    if getattr(original_torch_load, "_gokbilge_xtts_patched", False):
        return

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    setattr(patched_torch_load, "_gokbilge_xtts_patched", True)
    torch.load = patched_torch_load


def choose_default_speaker(tts: TTS) -> str | None:
    speakers = getattr(tts, "speakers", None) or []
    if isinstance(speakers, (list, tuple)) and speakers:
        return str(speakers[0])
    return None


def synthesize(
    tts: TTS,
    out_dir: Path,
    language: str,
    speaker_wav: str | None,
    default_speaker: str | None,
) -> str:
    speaker_mode = "speaker_wav" if speaker_wav else "default"
    for filename, text in SENTENCES:
        out_path = out_dir / filename
        kwargs = {
            "text": text,
            "file_path": str(out_path),
            "language": language,
        }
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif default_speaker:
            kwargs["speaker"] = default_speaker
            speaker_mode = f"default:{default_speaker}"

        try:
            tts.tts_to_file(**kwargs)
        except Exception as exc:
            if speaker_wav or default_speaker:
                raise
            print(
                "XTTS default/no-speaker mode failed. XTTS likely requires a speaker reference WAV.",
                file=sys.stderr,
            )
            print(f"Underlying error: {exc!r}", file=sys.stderr)
            print("BLOCKED_XTTS_NEEDS_SPEAKER_WAV", file=sys.stderr)
            raise SystemExit(2) from exc

        print(f"generated: {out_path}")

    return speaker_mode


def write_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
    speaker_mode: str,
) -> None:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "language": args.language,
        "device": device,
        "speaker_wav": str(Path(args.speaker_wav).resolve()) if args.speaker_wav else None,
        "speaker_mode": speaker_mode,
        "sentences": [{"file": name, "text": text} for name, text in SENTENCES],
        "command": command,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# XTTS v2 Benchmark Output",
        "",
        f"- model: `{MODEL_NAME}`",
        f"- language: `{args.language}`",
        f"- device: `{device}`",
        f"- speaker mode: `{speaker_mode}`",
        f"- speaker wav: `{args.speaker_wav or 'none'}`",
        f"- command: `{command}`",
        "",
        "## Files",
    ]
    for filename, text in SENTENCES:
        readme.append(f"- `{filename}` — {text}")
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_wav = None
    if args.speaker_wav:
        speaker_wav = str(Path(args.speaker_wav).expanduser().resolve())
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(f"speaker wav not found: {speaker_wav}")

    device = resolve_device(args.device)
    print(f"loading model: {MODEL_NAME}")
    print(f"device: {device}")
    tts = load_tts(device)
    default_speaker = None if speaker_wav else choose_default_speaker(tts)
    if default_speaker:
        print(f"default speaker: {default_speaker}")

    speaker_mode = synthesize(tts, out_dir, args.language, speaker_wav, default_speaker)
    write_metadata(out_dir, args, device, speaker_mode)
    print(f"metadata: {out_dir / 'metadata.json'}")
    print(f"readme: {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
