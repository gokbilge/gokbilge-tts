# -*- coding: utf-8 -*-
"""Text → WAV via piper_train.infer_onnx + piper_phonemize stub.

Used by recipes/issai_piper/infer.sh when the piper CLI binary is not installed.

Usage:
  python3 tools/piper_infer.py <model.onnx> <output.wav> <text>

The script:
  1. Phonemizes <text> with espeak-ng (voice=tr) via piper_phonemize.
  2. Encodes phoneme IDs in piper format (BOS/PAD-interspersed/EOS).
  3. Runs OnnxRuntime inference via piper_train.infer_onnx.
  4. Writes the result to <output.wav>.

Requirements on the server:
  - piper_phonemize stub in site-packages (tools/piper_phonemize_stub.py)
  - piper_train installed from /home/hcfk/piper-src/src/python
  - espeak-ng on PATH
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

# Ensure piper_train is importable when running from the repo root
_PIPER_SRC = os.environ.get("PIPER_SRC", "/home/hcfk/piper-src/src/python")
if _PIPER_SRC not in sys.path:
    sys.path.insert(0, _PIPER_SRC)

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: piper_infer.py <model.onnx> <output.wav> <text>", file=sys.stderr)
        sys.exit(1)

    model_onnx, output_wav, text = sys.argv[1], sys.argv[2], sys.argv[3]

    if not os.path.exists(model_onnx):
        print(f"ERROR: model not found: {model_onnx}", file=sys.stderr)
        sys.exit(1)

    # Phonemize
    sentences = phonemize_espeak(text, "tr")
    phonemes = [ph for sent in sentences for ph in sent]
    missing: Counter = Counter()
    ids = phoneme_ids_espeak(phonemes, missing_phonemes=missing)
    if missing:
        print(f"WARNING: unmapped phonemes (will be skipped): {dict(missing)}", file=sys.stderr)

    utt = json.dumps({"id": "0", "phoneme_ids": ids})

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [
                sys.executable, "-m", "piper_train.infer_onnx",
                "--model", model_onnx,
                "--output-dir", tmp,
                "--sample-rate", "22050",
            ],
            input=utt + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=_PIPER_SRC,
        )
        if proc.returncode != 0:
            print(f"ERROR: piper_train.infer_onnx failed:\n{proc.stderr}", file=sys.stderr)
            sys.exit(1)

        wav_tmp = os.path.join(tmp, "0.wav")
        if not os.path.exists(wav_tmp):
            print(f"ERROR: inference produced no output. stderr:\n{proc.stderr}", file=sys.stderr)
            sys.exit(1)

        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        shutil.move(wav_tmp, output_wav)

    # Print RTF line from stderr for visibility
    for line in proc.stderr.splitlines():
        if "Real-time factor" in line or "ERROR" in line.upper():
            print(line, file=sys.stderr)

    print(f"[infer] Saved: {output_wav}")


if __name__ == "__main__":
    main()
