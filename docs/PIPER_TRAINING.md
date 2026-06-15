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
# 1. System dependencies
sudo apt-get install espeak-ng espeak-ng-data python3-dev build-essential

# 2. piper_phonemize stub (aarch64 only — no wheel on PyPI, C++ build requires OnnxRuntime headers)
#    Deploy the pure-Python stub from tools/piper_phonemize_stub.py:
cp tools/piper_phonemize_stub.py \
    $(python3 -c "import site; print(site.getusersitepackages())")/piper_phonemize.py

# 3. Install pytorch-lightning and other piper_train deps (without piper-phonemize or torch version pin)
pip3 install 'pytorch-lightning>=2.0,<2.4' Cython librosa onnxscript --break-system-packages

# 4. Install piper_train from source (not on PyPI)
git clone https://github.com/rhasspy/piper /home/hcfk/piper-src
cd /home/hcfk/piper-src/src/python
pip3 install -e . --no-deps --break-system-packages

# Build the Cython monotonic_align extension (required for training)
export PATH=$PATH:~/.local/bin
cd piper_train/vits/monotonic_align && mkdir -p monotonic_align && cythonize -i core.pyx && mv core*.so monotonic_align/

# 5. Apply compatibility patches for pytorch-lightning 2.x + PyTorch 2.6+ (see TRAINING_LOG.md 2026-06-15)
#    Run these from the piper-src repo root. Patches are in tools/:
python3 tools/piper_main_patch.py          # overwrites piper_train/__main__.py
python3 tools/patch_lightning.py /home/hcfk/piper-src/src/python/piper_train/vits/lightning.py
python3 tools/patch_export_onnx.py  /home/hcfk/piper-src/src/python/piper_train/export_onnx.py
python3 tools/patch_export_onnx2.py /home/hcfk/piper-src/src/python/piper_train/export_onnx.py
python3 tools/patch_export_onnx3.py /home/hcfk/piper-src/src/python/piper_train/export_onnx.py

# 6. Install Gokbilge TTS dev tools
cd /path/to/gokbilge-tts
pip3 install -e ".[dev]" --break-system-packages

# 7. Inference: use piper_train.infer_onnx (piper binary not required)
#    See tools/synth_test.py for a working example.
```

> **Note:** `pip install -e ".[train]"` does not install piper_train — the `[train]`
> extra is intentionally empty because piper_train is not published on PyPI.
> Always follow steps 3–5 above for the training toolchain.

### Compatibility patches summary (for PyTorch 2.12 / pytorch-lightning 2.3)

| File patched | Why |
|---|---|
| `piper_train/__main__.py` | `Trainer.add_argparse_args` / `from_argparse_args` removed in lightning 2.0 |
| `piper_train/vits/lightning.py` | `training_step(optimizer_idx)` removed in lightning 2.0; requires manual optimization |
| `piper_train/export_onnx.py` (3 patches) | `torch.load` `weights_only=True` default; dynamo exporter fails on VITS; model must be moved to CPU before export |

All patches are idempotent Python scripts in `tools/`. See `TRAINING_LOG.md` § 2026-06-15 for full root-cause analysis.

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
Piper LJSpeech format (wavs/, metadata.csv, config.json skeleton)
  │
  ▼  gokbilge-tts validate-piper
validated Piper export
  │
  ▼  python3 -m piper_train.preprocess
phoneme-processed training dataset + config.json with phoneme_id_map
  │
  ▼  python3 -m piper_train
checkpoints (.ckpt)
  │
  ▼  python3 -m piper_train.export_onnx
model.onnx + model.onnx.json (from training_dir/config.json)
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

Outputs: `data/piper/wavs/` (symlinks), `metadata.csv`, `config.json` (skeleton), `train.txt`, `val.txt`.

The `config.json` written here is a **skeleton** — `phoneme_id_map` is empty and will be filled by `piper_train.preprocess`.

### 3b. Validate Piper export

```bash
gokbilge-tts validate-piper ./data/piper
```

Checks: `metadata.csv` format, every stem has a matching `wavs/<stem>.wav`, no empty text rows.

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

Outputs: `data/training/*.pt` (phoneme tensors), `data/training/config.json` (**final config with phoneme_id_map populated**).

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

# Use training_dir/config.json — it has the filled phoneme_id_map.
# Do NOT use data/piper/config.json (the skeleton).
cp ./data/training/config.json ./models/gokbilge_tr_v0_1.onnx.json
```

### 7. Synthesize

Use `infer.sh`, which defaults to `piper_train.infer_onnx` (no external binary required):

```bash
bash recipes/issai_piper/infer.sh \
    ./models/gokbilge_tr_v0_1 \
    "Bugün hava çok güzel." \
    output.wav
```

To use the external `piper` CLI binary instead (must be on PATH):

```bash
PIPER_BACKEND=piper bash recipes/issai_piper/infer.sh \
    ./models/gokbilge_tr_v0_1 \
    "Bugün hava çok güzel." \
    output.wav
```

> **`piper` binary:** Not installed on the current server. The `infer_onnx` backend
> (via `tools/piper_infer.py`) produces identical output using OnnxRuntime directly.
> Install the piper binary from https://github.com/rhasspy/piper/releases if
> the CLI interface is needed (e.g. streaming or real-time use).

---

## Recipe Scripts

The `recipes/issai_piper/` directory provides wrapper scripts:

| Script | Steps covered | Usage |
|--------|---------------|-------|
| `prepare.sh` | 1–3 | `bash prepare.sh <corpus> <manifests> <piper>` |
| `train.sh` | 4–5 | `bash train.sh <piper> <training> <checkpoints>` |
| `export_onnx.sh` | 6 | `bash export_onnx.sh <ckpt> <training_dir> <model_name>` |
| `infer.sh` | 7 | `bash infer.sh <model_base> <text> <output.wav>` (default: `infer_onnx` backend; set `PIPER_BACKEND=piper` for CLI) |
| `smoke.sh` | 1–5 (100 utts) | `bash smoke.sh <issai_dir> <run_dir>` |

---

## Training Stop Policy

`--max_epochs 10000` in `train.sh` is a safety ceiling, not a training target. With 178k utterances and batch_size=16, each epoch takes ~3 hours; 10,000 epochs would take ~3.4 years.

**Real stop criterion:** perceptual quality plateau, evaluated at step milestones using `tools/eval_step.sh`.

### Step milestone evaluation

| Step | Sample dir | What to check |
|---|---|---|
| ~12k (epoch 1) | `02_v0_1_full_step012k` | Pipeline alive; `last.ckpt` exists |
| ~50k | `03_v0_1_full_step050k` | Noise → speech-like structure? |
| ~100k | `04_v0_1_full_step100k` | First speech-like output expected |
| ~200k | `05_v0_1_full_step200k` | Turkish phonemes distinct? |
| ~300k | `06_v0_1_full_step300k` | Continue / adjust / stop decision |
| ~500k | `07_v0_1_full_step500k` | v0.1 candidate check |
| ~800k | `08_v0_1_full_step800k` | Stop / retrain / architecture decision |

### Quality thresholds

- 0–50k: noise is normal
- 50k–100k: murmur / rhythm may emerge
- 100k–200k: speech-like output expected
- 200k–500k: intelligibility should arrive
- 500k–800k: meaningful quality evaluation

### Stop conditions

Stop training when any of the following is true:
- Benchmark samples plateau (no improvement between consecutive milestones)
- Audio quality is sufficient for v0.1
- 800k steps still fails to produce usable speech (architecture/data decision needed)

### Evaluation command

```bash
# Run from repo root on the server:
bash tools/eval_step.sh <sample_dir> [run_dir]

# Examples:
bash tools/eval_step.sh 02_v0_1_full_step012k runs/v0_1_full_001   # epoch 1 check
bash tools/eval_step.sh 04_v0_1_full_step100k runs/v0_1_full_001   # 100k step check
```

Exports ONNX from `last.ckpt` and generates all 5 benchmark WAVs in one command.
`last.ckpt` is updated after every epoch — safe to export without stopping training.

---

## Smoke Run

To verify the full pipeline on 100 utterances before a full training run:

```bash
bash recipes/issai_piper/smoke.sh \
    /home/hcfk/datasets/ISSAI/ISSAI_TSC_218 \
    ./runs/smoke_001
```

Use the run naming convention: `runs/smoke_NNN/` (see CLAUDE.md § Run Naming Convention).

This runs steps 1–5 with `--limit 100` (100 train + up to 10 val + up to 10 test utterances) and 5 training epochs. It does not run ONNX export automatically.

**Do not start full training until the smoke run report is committed** (`docs/SMOKE_RUN_REPORT.md`).
The smoke run for this project is documented and passed — see `docs/SMOKE_RUN_REPORT.md`.

### Smoke mode limits

When `--limit N` is passed to `export-piper`:
- **train**: first N records
- **val**: first `min(10, available)` records
- **test**: first `min(10, available)` records

Val and test are capped at 10 so they stay proportional and don't dominate the smoke dataset.

---

## Smoke Run Checklist

### After `prepare-issai`

Expected files in `<run>/manifests/`:
- `train.jsonl` — training records (one JSON object per line)
- `val.jsonl` — validation records
- `test.jsonl` — test records
- `stats.json` — counts, hours, speaker list, filter stats
- `symbols.txt` — IPA phoneme inventory (one symbol per line)

Each JSONL record has fields: `audio_filepath`, `text`, `normalized_text`, `phonemes`, `duration`, `speaker_id`.

### After `export-piper`

Expected files in `<run>/piper/`:
- `metadata.csv` — LJSpeech format: `stem|text` (single-speaker) or `stem|speaker|text`
- `wavs/` — symlinks (Linux) or copies (Windows) of source WAV files
- `config.json` — Piper training config skeleton (`phoneme_id_map` is empty at this stage)
- `train.txt`, `val.txt`, `test.txt` — one stem per line

### After `piper_train.preprocess`

Expected files in `<run>/training/`:
- `*.pt` — phoneme ID tensors for each utterance
- `config.json` — **final config with `phoneme_id_map` populated by espeak-ng**

This `config.json` is the one that must be copied next to the ONNX model.

### After smoke training (5 epochs)

Expected files in `<run>/checkpoints/`:
- `*.ckpt` — checkpoint files (at minimum `last.ckpt` after epoch 5)
- `train.log` — captured stdout/stderr from `piper_train`

A successful smoke run means the pipeline is wired correctly. Audio quality at 5 epochs will be poor (noise) — that is expected.

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `preprocess` fails immediately | `espeak-ng` not installed | `sudo apt-get install espeak-ng espeak-ng-data` |
| `preprocess` produces no output | `metadata.csv` has wrong column count or missing `wavs/` | Run `gokbilge-tts validate-piper <piper_dir>` |
| `piper_train` crashes at start | `config.json` mismatch or missing from training_dir | Confirm preprocess completed; check `training_dir/config.json` |
| `wavs/<stem>.wav` not found | Symlinks broken (moved corpus after export) or Windows privilege issue | Re-run `export-piper` with corpus in place; use `--limit` to test |
| Audio resampling warning | ISSAI is 16 kHz but config says 22050 Hz | This is expected; preprocess handles resampling |
| `CUDA error` / `device not found` | GPU unavailable | Change `--accelerator gpu` to `--accelerator cpu` in smoke.sh or train.sh |
| `piper_train` not found | `piper-train` not installed from source | Follow Installation § 3–4 above |
| Empty `phoneme_id_map` in ONNX config | Copied `piper_dir/config.json` instead of `training_dir/config.json` | Use `training_dir/config.json`; it is filled by preprocess |
| `No module named 'piper_phonemize'` | C extension not available on aarch64 | Deploy `tools/piper_phonemize_stub.py` to site-packages (see Installation § 2) |
| `AttributeError: Trainer has no attribute 'add_argparse_args'` | lightning 2.x removed these methods | Apply `tools/piper_main_patch.py` (see Installation § 5) |
| `RuntimeError: Training with multiple optimizers …` | lightning 2.x requires manual optimization | Apply `tools/patch_lightning.py` (see Installation § 5) |
| `UnpicklingError: GLOBAL pathlib.PosixPath` on export | torch 2.6+ `weights_only=True` default | Apply `tools/patch_export_onnx.py` (see Installation § 5) |
| `TorchExportError: Unhandled FakeTensor Device Propagation` on export | dynamo exporter fails on VITS mixed-device | Apply `tools/patch_export_onnx2.py` (see Installation § 5) |
| `RuntimeError: Expected all tensors on same device` on export | model on CUDA, dummy inputs on CPU | Apply `tools/patch_export_onnx3.py` (see Installation § 5) |

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

### config.json — skeleton vs. final

Two versions of `config.json` exist in the pipeline:

| File | When created | phoneme_id_map |
|------|-------------|----------------|
| `piper_dir/config.json` | By `export-piper` | Empty `{}` |
| `training_dir/config.json` | By `piper_train.preprocess` | Populated by espeak-ng |

Always use `training_dir/config.json` for ONNX export and inference.

Example skeleton (from `export-piper`):
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

---

## Known Limitations

- **Phonemization**: `piper_train.preprocess` uses espeak-ng for Turkish. Our custom Turkish G2P (`gokbilge_tts.g2p.turkish`) produces IPA phonemes stored in JSONL manifests but is not yet wired into Piper training. Integrating our G2P as the Piper phonemizer is a Sprint 4/5 goal.
- **espeak-ng Turkish quality**: espeak-ng tr voice handles most Turkish correctly but may mis-stress compound words and loanwords.
- **Sample rate**: ISSAI corpus is 16 kHz; Piper default target is 22050 Hz. Resampling is handled by `piper_train.preprocess`. The `sample_rate` in `config.json` must match `--sample-rate` passed to preprocess.
