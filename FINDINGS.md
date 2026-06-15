# Findings

Empirical findings from experiments and debugging. Stable facts that should inform future decisions.
Updated whenever something non-obvious is confirmed.

---

## Phonemization

### espeak-ng output format (verified 2026-06-15)
`espeak-ng -v tr --ipa -q` outputs one line per clause/sentence. Each Unicode codepoint in the line is one phoneme. Spaces between words are preserved as the space character `' '`. Example:

```
echo "Bugün hava çok güzel." | espeak-ng -v tr --ipa -q
→ buɡˈøn havˈa tʃˈɔk ɟyzˈæl
```

The piper convention treats each character in this output as a distinct phoneme token.

### DEFAULT_PHONEME_ID_MAP size (verified 2026-06-15)
The map from `piper-phonemize/src/phoneme_ids.hpp` has **159 entries** including the three reserved tokens:
- `'_'` → [0] (PAD)
- `'^'` → [1] (BOS)
- `'$'` → [2] (EOS)

`get_max_phonemes()` returns 256 regardless of map size — this sets `num_symbols` in `config.json`.

### Phoneme ID encoding (verified 2026-06-15)
The piper encoding (PhonemeIdConfig defaults: `addBos=True`, `addEos=True`, `interspersePad=True`) produces:

```
[BOS(1), PAD(0), id₀, PAD(0), id₁, PAD(0), ..., idₙ, PAD(0), EOS(2)]
```

"Bugün hava çok güzel." phonemizes to 25 phoneme characters → 53 IDs (2 + 25×2 + 1).

---

## piper_train Compatibility (PyTorch 2.12 / pytorch-lightning 2.3, aarch64)

Six patches are required to run rhasspy/piper's training stack on a modern server. All patch scripts are in `tools/`. All root causes are in `TRAINING_LOG.md` § 2026-06-15.

| # | File | Problem | Patch |
|---|---|---|---|
| 1 | site-packages | `piper_phonemize` has no aarch64 wheel; C++ build fails (missing OnnxRuntime headers) | Deploy `tools/piper_phonemize_stub.py` |
| 2 | `piper_train/__main__.py` | `Trainer.add_argparse_args` / `from_argparse_args` removed in lightning 2.0 | `tools/piper_main_patch.py` |
| 3 | `piper_train/vits/lightning.py` | `training_step(optimizer_idx)` removed in lightning 2.0; multiple optimizers require `automatic_optimization=False` | `tools/patch_lightning.py` |
| 4 | `piper_train/export_onnx.py` | PyTorch 2.6+ `torch.load` defaults to `weights_only=True`; checkpoint has `pathlib.PosixPath` | `tools/patch_export_onnx.py` |
| 5 | `piper_train/export_onnx.py` | New dynamo ONNX exporter fails on VITS mixed-device tracing | `tools/patch_export_onnx2.py` (`dynamo=False`) |
| 6 | `piper_train/export_onnx.py` | Model loads to CUDA; dummy inputs are on CPU → device mismatch at trace | `tools/patch_export_onnx3.py` (`map_location="cpu"`) |

These patches are **not needed if running on x86_64 with pytorch-lightning 1.7** (the original target environment). They exist solely because the server is aarch64 with a modern PyTorch stack.

---

## ONNX Config Source

Two `config.json` files exist in the pipeline. **Only one is correct for inference:**

| File | `phoneme_id_map` | Use for |
|---|---|---|
| `piper_dir/config.json` | Empty `{}` (skeleton written by `export-piper`) | Nothing — do not use for ONNX |
| `training_dir/config.json` | Populated (159 entries, written by `piper_train.preprocess`) | ONNX export and inference |

`export_onnx.sh` copies `training_dir/config.json` → `model.onnx.json`. Using the wrong one results in silent inference failures (all-pad phoneme sequences).

---

## ISSAI Corpus Statistics (full corpus, verified 2026-06-15)

| Split | Records | Filtered out |
|---|---|---|
| Train | 178,901 | 357 |
| Dev/Val | 3,426 | 2 |
| Test | 3,483 | 1 |
| **Total** | **185,810** | **360** |

Filtered records fail duration or empty-text checks (`validate_manifest` thresholds: 0.5–20 s). Single speaker (`issai`). Total duration: ~218 hours.

---

## Inference Performance (NVIDIA GB10, OnnxRuntime CPU backend, 2026-06-15)

Measured on a 5-epoch smoke checkpoint (noise quality — not a real model):

- Real-time factor: **0.04** (25× real-time)
- "Bugün hava çok güzel." → 1.36 sec of audio in 0.06 sec inference time
- OnnxRuntime session on CPU; GPU not used during inference

RTF will change with a fully trained model (larger activations from longer outputs) but the order of magnitude should hold.

---

## Smoke Run Expectations

At 5 epochs with 100 utterances, the model outputs noise. This is expected and correct — it only verifies that the pipeline is wired, not that audio quality is acceptable. Audio becomes intelligible after hundreds to thousands of epochs on the full corpus.

---

## Samples Directory Convention

`samples/` uses numbered subdirectories (`NN_<label>/`) so milestones sort chronologically and are never overwritten. Each directory holds the 5 fixed benchmark WAVs. Smoke/diagnostic samples are committed even when audio is noise — the point is confirming the pipeline produces audio of the right duration, not evaluating quality.

```
samples/
  01_smoke_test/    — epoch=4, 100 utterances (2026-06-15)
  02_<next>/        — next evaluation milestone
```

When adding a new milestone: create the next `NN_` directory, generate all 5 WAVs, and add a metadata block to `samples/README.md`.
