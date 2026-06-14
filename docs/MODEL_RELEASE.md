# Model Release Guide

## Release checklist

Before publishing a model to HuggingFace as `gokbilge-tts/gokbilge-tr-*`:

- [ ] Benchmark scores recorded in TRAINING_LOG.md (all 5 sentences, full rubric)
- [ ] Known limitations documented in model card
- [ ] ONNX export tested — inference produces audio without errors
- [ ] Audio quality acceptable on all 5 benchmark sentences (mean ≥ 3.5/5.0)
- [ ] License confirmed: ISSAI license allows derivative model release
- [ ] Model card written (see template below)
- [ ] Piper config `model.onnx.json` populated with correct phoneme_id_map

## Model naming convention

```
gokbilge-tr-{quality}-v{major}.{minor}
```

Examples:
- `gokbilge-tr-medium-v0.1` — first release, medium quality
- `gokbilge-tr-high-v0.2` — second release, high quality

Quality tiers follow Piper convention: low (22kHz/16bit), medium (22kHz/22bit), high (22kHz/22bit, larger model).

## HuggingFace upload

```bash
huggingface-cli login
huggingface-cli repo create gokbilge-tr-medium-v0.1 --type model --organization gokbilge-tts
# Upload model files
huggingface-cli upload gokbilge-tts/gokbilge-tr-medium-v0.1 ./models/gokbilge-tr-medium/
```

## Model card template

```markdown
---
language: tr
license: mit
tags:
  - text-to-speech
  - turkish
  - vits
  - piper
datasets:
  - issai/Turkish_Speech_Corpus
---

# Gokbilge TTS — Turkish (gokbilge-tr-medium-v0.1)

Turkish text-to-speech model based on VITS, trained on ISSAI Turkish Speech Corpus.
Compatible with Piper TTS.

## Usage

With Piper CLI:
\`\`\`bash
echo "Bugün hava çok güzel." | piper --model gokbilge-tr-medium-v0.1.onnx --output_file output.wav
\`\`\`

## Benchmark

| Sentence | Intelligibility | Phoneme | Prosody | Quality | Mean |
|----------|----------------|---------|---------|---------|------|
| S1 | | | | | |
| S2 | | | | | |
| S3 | | | | | |
| S4 | | | | | |
| S5 | | | | | |

## Known limitations

- [ ] Fill after evaluation

## Training

- Architecture: VITS
- Dataset: ISSAI Turkish Speech Corpus (~10h)
- Training: X epochs on NVIDIA A100
- G2P: Rule-based (gokbilge_tts.g2p.turkish)
```
