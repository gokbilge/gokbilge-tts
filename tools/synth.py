"""Synthesize speech using piper_train.infer_onnx + piper_phonemize stub.

Usage: python3 synth.py <model.onnx> <output.wav> "text to synthesize"
"""
import json
import subprocess
import sys
from collections import Counter

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

def synthesize(model_onnx: str, output_wav: str, text: str) -> None:
    sentences = phonemize_espeak(text, "tr")
    phonemes = [ph for sent in sentences for ph in sent]
    missing: Counter = Counter()
    ids = phoneme_ids_espeak(phonemes, missing_phonemes=missing)
    if missing:
        print(f"WARNING: missing phoneme IDs for: {dict(missing)}", file=sys.stderr)

    utt = json.dumps({"id": "0", "phoneme_ids": ids})

    proc = subprocess.run(
        [
            "python3", "-m", "piper_train.infer_onnx",
            "--model", model_onnx,
            "--output-dir", "/tmp/synth_out",
            "--sample-rate", "22050",
        ],
        input=utt + "\n",
        capture_output=True,
        text=True,
        cwd="/home/hcfk/piper-src/src/python",
    )
    if proc.returncode != 0:
        print("STDERR:", proc.stderr, file=sys.stderr)
        sys.exit(1)

    # Move output file
    import shutil
    shutil.move("/tmp/synth_out/0.wav", output_wav)
    print(f"Saved: {output_wav}")
    print("STDERR:", proc.stderr[-500:] if proc.stderr else "(none)", file=sys.stderr)

if __name__ == "__main__":
    model, out, text = sys.argv[1], sys.argv[2], sys.argv[3]
    synthesize(model, out, text)
