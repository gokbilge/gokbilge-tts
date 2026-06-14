# Piper Training Guide

How to train a Turkish TTS model with Piper/VITS on the ISSAI corpus.

---

## Selected Toolchain

| Component | Package / Command |
|-----------|-------------------|
| Training framework | `piper-train` (PyPI) — `python3 -m piper_train` |
| Preprocessing | `python3 -m piper_train.preprocess` |
| ONNX export | `python3 -m piper_train.export_onnx` |
| Inference | `piper` CLI (rhasspy/piper) |
| Dataset format | LJSpeech (pipe-separated metadata.csv + wavs/) |
| Phonemization | espeak-ng language `tr` (via piper_train.preprocess) |

Source: [rhasspy/piper TRAINING.md](https://github.com/rhasspy/piper/blob/master/TRAINING.md).  
Development has moved to [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl).

---

## Installation

```bash
# Install Gokbilge TTS with training dependencies
pip install -e ".[train]"

# Install espeak-ng (required by piper_train.preprocess)
# Ubuntu/Debian:
sudo apt-get install espeak-ng

# Install Piper inference binary (for infer.sh)
# See https://github.com/rhasspy/piper/releases
```

---

## Pipeline Overview

```
ISSAI corpus (WAV + TXT)
  │
  ▼  gokbilge-tts prepare-issai
JSONL manifests (train/val/test.jsonl, symbols.txt)
  │
  ▼  gokbilge-tts validate-manifest
validated manifests
  │
  ▼  gokbilge-tts export-piper
Piper LJSpeech format (wavs/, metadata.csv, config.json)
  │
  ▼  python3 -m piper_train.preprocess
phoneme-processed training dataset
  │
  ▼  python3 -m piper_train
checkpoints (.ckpt)
  │
  ▼  python3 -m piper_train.export_onnx
model.onnx + model.onnx.json
  │
  ▼  piper CLI
synthesized speech
```

---

## Step-by-Step Commands

### 1. Prepare manifests

```bash
gokbilge-tts prepare-issai \
    --dataset-dir /home/hcfk/datasets/ISSAI \
    --out ./data/manifests
```

Outputs: `data/manifests/{train,val,test}.jsonl`, `stats.json`, `symbols.txt`.

### 2. Validate manifests

```bash
gokbilge-tts validate-manifest ./data/manifests/train.jsonl
```

Checks: required fields, audio existence, duration 0.5–20 s, non-empty text and phonemes. Exits with code 1 on any error.

### 3. Export to Piper format

```bash
gokbilge-tts export-piper \
    --manifest-dir ./data/manifests \
    --out ./data/piper
```

Outputs: `data/piper/wavs/` (symlinks), `metadata.csv`, `config.json`, `train.txt`, `val.txt`.

### 4. Preprocess (espeak phonemization)

```bash
python3 -m piper_train.preprocess \
    --language tr \
    --input-dir ./data/piper \
    --output-dir ./data/training \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050
```

### 5. Train

```bash
python3 -m piper_train \
    --dataset-dir ./data/training \
    --accelerator gpu \
    --devices 1 \
    --batch-size 16 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint latest \
    --checkpoint-epochs 100 \
    --precision 32 \
    2>&1 | tee ./checkpoints/train.log
```

### 6. Export ONNX

```bash
python3 -m piper_train.export_onnx \
    ./checkpoints/last.ckpt \
    ./models/gokbilge_tr_v0_1.onnx

cp ./data/piper/config.json ./models/gokbilge_tr_v0_1.onnx.json
```

### 7. Synthesize

```bash
echo "Bugün hava çok güzel." | piper \
    --model ./models/gokbilge_tr_v0_1.onnx \
    --config ./models/gokbilge_tr_v0_1.onnx.json \
    --output_file output.wav
```

---

## Recipe Scripts

The `recipes/issai_piper/` directory provides wrapper scripts for steps 1–7:

| Script | Steps covered |
|--------|---------------|
| `prepare.sh <corpus> <manifests> <piper>` | 1–3 |
| `train.sh <piper> <training> <checkpoints>` | 4–5 |
| `export_onnx.sh <ckpt> <piper> <model_name>` | 6 |
| `infer.sh <model_base> <text> <output.wav>` | 7 |

---

## Smoke Test (fast correctness check)

To verify the full pipeline on a small subset before committing to a full training run:

```bash
# Export only 100 training utterances
gokbilge-tts export-piper \
    --manifest-dir ./data/manifests \
    --out ./data/piper_smoke \
    --limit 100

# Preprocess the 100-utterance dataset
python3 -m piper_train.preprocess \
    --language tr \
    --input-dir ./data/piper_smoke \
    --output-dir ./data/training_smoke \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050

# Short smoke training run (5 epochs, CPU or GPU)
python3 -m piper_train \
    --dataset-dir ./data/training_smoke \
    --accelerator cpu \
    --devices 1 \
    --batch-size 4 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 5 \
    --checkpoint-epochs 5 \
    --precision 32
```

A successful smoke run confirms: espeak-ng is installed and processes Turkish, the dataset pipeline is wired correctly, and piper_train can train without crashing.

---

## Input Format Details

### metadata.csv (LJSpeech)

Single-speaker (ISSAI default):
```
stem|normalized_text
100000|bugün hava çok güzel
100001|türkiye cumhuriyeti...
```

Multi-speaker:
```
stem|speaker_id|normalized_text
100000|issai|bugün hava çok güzel
```

### config.json skeleton (generated by export-piper)

```json
{
  "audio": { "sample_rate": 22050 },
  "espeak": { "language": "tr", "voice": "tr" },
  "inference": { "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8 },
  "phoneme_type": "espeak",
  "num_speakers": 1,
  "num_symbols": 256,
  "phoneme_id_map": {},
  "speaker_id_map": {}
}
```

`phoneme_id_map` is intentionally empty — `piper_train.preprocess` fills it based on espeak output.

---

## Known Limitations

- **Phonemization**: `piper_train.preprocess` uses espeak-ng for Turkish. Our custom Turkish G2P (`gokbilge_tts.g2p.turkish`) is used for the JSONL manifest phonemes field but is not yet wired into the Piper training path. Integrating our G2P as the phonemizer is a Sprint 4/5 goal.
- **espeak-ng Turkish quality**: espeak-ng tr voice handles most Turkish correctly but may mis-stress compound words and loanwords.
- **Sample rate**: ISSAI corpus is 16 kHz; Piper default is 22050 Hz. Resampling is handled by `piper_train.preprocess`. The config.json `sample_rate` should match the preprocessor's `--sample-rate`.
