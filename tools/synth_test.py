# -*- coding: utf-8 -*-
"""Test synthesis using piper_train.infer_onnx + piper_phonemize stub."""
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, "/home/hcfk/.local/lib/python3.12/site-packages")

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

MODEL = "/home/hcfk/gokbilge-tts/runs/smoke/gokbilge_tr_smoke.onnx"
OUTPUT_DIR = "/tmp/synth_out"
OUTPUT_WAV = "/tmp/output.wav"
TEXT = "Bugün hava çok güzel."

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Text: {TEXT}")
sentences = phonemize_espeak(TEXT, "tr")
print(f"Phonemes: {sentences}")

phonemes = [ph for sent in sentences for ph in sent]
missing: Counter = Counter()
ids = phoneme_ids_espeak(phonemes, missing_phonemes=missing)
if missing:
    print(f"WARNING missing: {dict(missing)}")
print(f"Phoneme IDs: {ids[:10]}... (len={len(ids)})")

utt = json.dumps({"id": "0", "phoneme_ids": ids})

proc = subprocess.run(
    [
        sys.executable, "-m", "piper_train.infer_onnx",
        "--model", MODEL,
        "--output-dir", OUTPUT_DIR,
        "--sample-rate", "22050",
    ],
    input=utt + "\n",
    capture_output=True,
    text=True,
    encoding="utf-8",
    cwd="/home/hcfk/piper-src/src/python",
)
print("Return code:", proc.returncode)
print("STDOUT:", proc.stdout[-300:])
print("STDERR:", proc.stderr[-300:])

wav_out = os.path.join(OUTPUT_DIR, "0.wav")
if os.path.exists(wav_out):
    size = os.path.getsize(wav_out)
    print(f"Output WAV: {wav_out} ({size} bytes)")
    import shutil
    shutil.copy(wav_out, OUTPUT_WAV)
    print(f"Copied to: {OUTPUT_WAV}")
else:
    print("ERROR: output WAV not found", file=sys.stderr)
    sys.exit(1)

print("SUCCESS")
