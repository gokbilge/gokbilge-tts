# Benchmark Plan

## Goals

1. Track perceptual quality across training checkpoints
2. Compare models (VITS, VITS2, future architectures) on the same fixed sentence set
3. Identify which Turkish phonemes remain problematic

## Fixed Evaluation Set

See `turkish_eval_sentences.txt` — 5 sentences, never modified between evaluations.

Sentences cover:
- Basic vocabulary + vowel harmony (S1)
- Numbers written as words (S2) — digits cause fallback in most pretrained models
- Consonant clusters: ç, ş, back vowels (S3)
- Vowel rounding ö, ü + compound suffixes (S4)
- ş, back vowels, percentage — mixed difficulty (S5)

## Evaluation Protocol

1. Generate one WAV per sentence using the same TTS command
2. Listen blind (no model label) and score with `SCORING_RUBRIC.md`
3. Record scores in `TRAINING_LOG.md` under the checkpoint entry
4. Flag regressions immediately — any dimension drop >1 point is a regression

## Planned Evaluations

| Milestone | Trigger |
|-----------|---------|
| ISSAI data prep complete | Validate phonemization on all 5 sentences |
| VITS Stage 1 (100 epochs) | First perceptual check |
| VITS Stage 1 (500 epochs) | Mid-training gate |
| VITS Stage 2 (attn-only fine-tune) | Compare with Stage 1 baseline |
| ONNX export | Verify no quality loss from export |
| v0.1 release candidate | Full rubric, publish scores |

## Objective Metrics (future)

When ground-truth references are available from ISSAI:
- MOS-predicted (UTMOS or similar)
- Character Error Rate via ASR (Whisper-large-v3 Turkish)
- Pitch correlation (for prosody)

Do not use objective metrics as primary quality gate until they are calibrated against perceptual scores on this sentence set.
