# Training Log

Chronological record of all training runs and decisions.

---

## 2026-06-15 — Smoke Run 1 (Sprint 3 smoke pipeline validation)

**Purpose:** Validate the full training pipeline end-to-end before committing to a full training run.

**Command:**
```bash
bash recipes/issai_piper/smoke.sh \
    /home/hcfk/datasets/ISSAI/ISSAI_TSC_218 \
    ./runs/smoke
```

**Hyperparameters:**
- `--limit 100` (100 train utterances, 10 val, 10 test)
- `--max_epochs 5`
- `--batch-size 4`
- `--accelerator gpu --devices 1`
- `--precision 32`
- Model quality: medium (default — hidden=192, inter=192, filter=768, n_layers=6, n_heads=2)
- Learning rate: 2e-4 (default)

**Why these values:** Smoke run only — smallest possible dataset/epoch count to verify pipeline correctness. Audio quality is expected to be noise at 5 epochs.

**Environment:**
- Server: 10.20.20.9 (aarch64, Ubuntu)
- Python 3.12.3
- PyTorch 2.12.0+cu130, CUDA (NVIDIA GB10)
- pytorch-lightning 2.3.3
- onnxruntime 1.26.0
- espeak-ng 1.51

**Steps and results:**

1. `prepare-issai` — 178,901 train / 3,426 val / 3,483 test records (360 filtered for duration/empty text)
2. `validate-manifest` — PASS (0 errors)
3. `export-piper --limit 100` — 100 train + 10 val + 10 test = 120 utterances exported
4. `validate-piper` — PASS (120 valid)
5. `piper_train.preprocess` — 120 utterances processed, `training/config.json` written with 159 phoneme_id_map entries
6. `piper_train` — 5 epochs, 30 batches/epoch, checkpoint at `lightning_logs/version_1/checkpoints/epoch=4-step=300.ckpt`
7. `export_onnx.sh` — `gokbilge_tr_smoke.onnx` + `gokbilge_tr_smoke.onnx.json` (159 phoneme entries confirmed)
8. Synthesis (`piper_train.infer_onnx`) — "Bugün hava çok güzel." → `/tmp/output.wav` (59,948 bytes, 1.36 sec audio, RTF 0.04)

**Audio quality:** Not evaluated — 5-epoch smoke run produces noise. Correct behavior.

**Issues encountered and fixes applied (piper-src patches):**

### A. piper_phonemize not available on aarch64
- **Root cause:** No aarch64 PyPI wheel; source build requires OnnxRuntime C++ headers.
- **Fix:** Deployed a pure-Python stub to `/home/hcfk/.local/lib/python3.12/site-packages/piper_phonemize.py`.
  Source: `tools/piper_phonemize_stub.py`. Uses `espeak-ng -v tr --ipa -q` subprocess; hardcoded DEFAULT_PHONEME_ID_MAP (159 entries from piper-phonemize/src/phoneme_ids.hpp).

### B. pytorch-lightning 2.x removed `Trainer.add_argparse_args` / `from_argparse_args`
- **Root cause:** piper_train was written for lightning 1.7; these methods were removed in 2.0.
- **Fix:** Patched `/home/hcfk/piper-src/src/python/piper_train/__main__.py` — replaced `Trainer.add_argparse_args(parser)` with explicit argparse args; replaced `Trainer.from_argparse_args(args)` with direct `Trainer(accelerator=..., devices=..., max_epochs=..., ...)` constructor.
- Source: `tools/piper_main_patch.py`

### C. pytorch-lightning 2.x requires manual optimization for multiple optimizers
- **Root cause:** `training_step(batch, batch_idx, optimizer_idx)` API removed in lightning 2.0.
- **Fix:** Patched `/home/hcfk/piper-src/src/python/piper_train/vits/lightning.py`:
  - Added `self.automatic_optimization = False` to VitsModel.__init__
  - Rewrote `training_step` to use `self.optimizers()`, `self.manual_backward()`, `opt.step()`
  - Added `on_train_epoch_end` to step LR schedulers
- Source: `tools/patch_lightning.py`

### D. torch 2.6+ `torch.load` defaults to `weights_only=True`; PosixPath not allowed
- **Root cause:** PyTorch 2.6 changed `torch.load` default to `weights_only=True`. Lightning checkpoint contains `pathlib.PosixPath`.
- **Fix:** Patched `export_onnx.py` to call `torch.serialization.add_safe_globals([pathlib.PosixPath])` before `load_from_checkpoint`.
- Source: `tools/patch_export_onnx.py`

### E. torch 2.6+ ONNX exporter defaults to dynamo — fails on mixed-device VITS model
- **Root cause:** New dynamo-based exporter can't trace models with cuda/cpu device mixing during tracing.
- **Fix:** Added `dynamo=False` to `torch.onnx.export(...)` call in `export_onnx.py` to force legacy TorchScript exporter.
- Source: `tools/patch_export_onnx2.py`

### F. ONNX export fails: model on CUDA, dummy inputs on CPU
- **Root cause:** `VitsModel.load_from_checkpoint` loads model to CUDA (matching training device); dummy inputs created on CPU.
- **Fix:** Added `map_location="cpu"` to `load_from_checkpoint` and `.cpu()` call on `model_g` in `export_onnx.py`.
- Source: `tools/patch_export_onnx3.py`

**Verdict:** Pipeline verified end-to-end. All 6 server-side patches documented above. Audio produced at RTF 0.04 (25× real-time inference on NVIDIA GB10 with OnnxRuntime CPU backend).

**Next step:** Full training run (ISSAI full 178k utterances, ~10k epochs). See `docs/PIPER_TRAINING.md`.

---

## 2026-06-15 — Full Training Run v0_1_full_001

**Purpose:** Full VITS training on the complete ISSAI Turkish corpus (178,901 utterances) to produce the v0.1 Turkish TTS model. First checkpoint at epoch 100 will be used for audio quality assessment.

**Commands:**
```bash
cd /home/hcfk/gokbilge-tts
git pull && git status

mkdir -p runs/v0_1_full_001

bash recipes/issai_piper/prepare.sh \
    /home/hcfk/datasets/ISSAI/ISSAI_TSC_218 \
    runs/v0_1_full_001/manifests \
    runs/v0_1_full_001/piper

bash recipes/issai_piper/train.sh \
    runs/v0_1_full_001/piper \
    runs/v0_1_full_001/training \
    runs/v0_1_full_001/checkpoints
```

**Hyperparameters (train.sh defaults):**

| Parameter | Value | Rationale |
|---|---|---|
| Dataset | 178,901 utterances (full ISSAI) | No `--limit`; smoke confirmed full pipeline |
| `--batch-size` | 16 | 4× smoke size; minimum practical for VITS stability |
| `--max_epochs` | 10,000 | Standard piper target; first quality check at epoch 100 |
| `--checkpoint-epochs` | 100 | ~1.12M steps per save; `save_last=True` gives epoch-granularity resume |
| `--precision` | 32 | fp32 for stable first run; reduces debugging surface vs mixed precision |
| `--accelerator` | gpu | NVIDIA GB10 on server (CUDA 13.0) |
| `--devices` | 1 | Single GPU |
| `--validation-split` | 0.0 | All data used for training |
| `--num-test-examples` | 0 | No per-epoch inference during training |
| `--resume_from_checkpoint` | latest | Auto-resume from `last.ckpt` if run is interrupted |
| `--default_root_dir` | `runs/v0_1_full_001/checkpoints` | New (fixed in train.sh): ensures checkpoints land in the expected directory |
| Model quality | medium (default) | hidden=192, inter=192, filter=768, n_layers=6, n_heads=2 |
| Learning rate | 2e-4 (default) | Standard Adam LR for VITS |

**Steps per epoch (estimated):** ceil(178,901 / 16) = 11,182 batches/epoch

**Pre-run changes committed in this session:**
- `tools/piper_main_patch.py`: added `save_last=True` to `ModelCheckpoint`; "latest"/"last" in `--resume_from_checkpoint` now resolves by searching for `last.ckpt` under `default_root_dir` (start fresh on first run instead of crashing)
- `recipes/issai_piper/train.sh`: added `--default_root_dir "$CHECKPOINT_DIR"` so checkpoints go to `runs/v0_1_full_001/checkpoints/`, not inside `training/`
- Server: `piper_train/__main__.py` re-deployed with updated patch (via `tools/piper_main_patch.py`)

**First audit criteria (after epoch 99 checkpoint):**
1. `find runs/v0_1_full_001/checkpoints -name "*.ckpt"` — checkpoint exists
2. `tail -50 runs/v0_1_full_001/checkpoints/train.log` — losses not NaN, both gen/disc decreasing
3. `python3 -c "import json; d=json.load(open('runs/v0_1_full_001/training/config.json')); print(len(d['phoneme_id_map']))"` — must print 159
4. `nvidia-smi` — GPU not OOM at batch_size=16
5. ONNX export from epoch-99 checkpoint + synthesis of all 5 benchmark sentences

**Result:** (to be filled after first checkpoint)

---
