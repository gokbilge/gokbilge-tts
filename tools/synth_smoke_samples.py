# -*- coding: utf-8 -*-
"""Generate the 5 fixed benchmark sentences from the smoke model.

infer_onnx.py names outputs 0.wav, 1.wav, ... (ignores the 'id' field).
Send all 5 as one batch so they get indices 0-4, then rename.

Run on server:
  python3 /tmp/synth_smoke_samples.py
"""
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

MODEL = "/home/hcfk/gokbilge-tts/runs/smoke/gokbilge_tr_smoke.onnx"
OUT_DIR = "/tmp/smoke_samples"

SENTENCES = [
    ("s1_bugun_hava",    "Bugün hava çok güzel."),
    ("s2_turkiye_cumh",  "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."),
    ("s3_cocuklar",      "Çocuklar çiçek, şeker ve üzüm yedi."),
    ("s4_ogrenciler",    "Öğrenciler ölçüm sonuçlarını değerlendirdi."),
    ("s5_sirket",        "Şirket yüzde otuz beş büyüme açıkladı."),
]

sys.path.insert(0, "/home/hcfk/piper-src/src/python")
sys.path.insert(0, "/home/hcfk/.local/lib/python3.12/site-packages")

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

os.makedirs(OUT_DIR, exist_ok=True)

# Build JSONL for all 5 sentences
lines = []
for stem, text in SENTENCES:
    sentences = phonemize_espeak(text, "tr")
    phonemes = [ph for sent in sentences for ph in sent]
    missing: Counter = Counter()
    ids = phoneme_ids_espeak(phonemes, missing_phonemes=missing)
    if missing:
        print(f"WARNING [{stem}] missing IDs: {dict(missing)}")
    lines.append(json.dumps({"id": stem, "phoneme_ids": ids}))
    print(f"Phonemized [{stem}]: {len(ids)} IDs")

stdin_data = "\n".join(lines) + "\n"

proc = subprocess.run(
    [
        sys.executable, "-m", "piper_train.infer_onnx",
        "--model", MODEL,
        "--output-dir", OUT_DIR,
        "--sample-rate", "22050",
    ],
    input=stdin_data,
    capture_output=True,
    text=True,
    encoding="utf-8",
    cwd="/home/hcfk/piper-src/src/python",
)
if proc.returncode != 0:
    print("ERROR:", proc.stderr)
    sys.exit(1)

# infer_onnx writes 0.wav, 1.wav, ... — rename to stem names
for i, (stem, _) in enumerate(SENTENCES):
    src = os.path.join(OUT_DIR, f"{i}.wav")
    dst = os.path.join(OUT_DIR, f"{stem}.wav")
    if os.path.exists(src):
        shutil.move(src, dst)
        size = os.path.getsize(dst)
        print(f"  {dst} ({size:,} bytes)")
    else:
        print(f"  MISSING: {src}", file=sys.stderr)

print(proc.stderr)
print("Done.")
