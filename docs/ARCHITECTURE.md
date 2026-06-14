# Architecture

## Overview

Gokbilge TTS uses a VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture trained from scratch on the ISSAI Turkish Speech Corpus.

```
Text
 │
 ▼
[Text Normalizer]         src/gokbilge_tts/normalize/
 │  numbers, dates, abbreviations, punctuation
 ▼
[Turkish G2P]             src/gokbilge_tts/g2p/
 │  grapheme → IPA phoneme sequence
 ▼
[Phoneme Encoder]         VITS posterior encoder
 │  phoneme IDs → hidden representation
 ▼
[Flow / Prior]            VITS normalizing flow
 │  latent variable z
 ▼
[HiFi-GAN Decoder]        VITS generator
 │  z → mel spectrogram → waveform
 ▼
Audio (22050 Hz WAV)
```

## Text Normalization

Pipeline (applied in order):

1. Unicode NFC normalization
2. Remove non-speech characters
3. Normalize punctuation (full-width → ASCII, whitespace)
4. Expand abbreviations (Dr. → doktor, TL → Türk lirası)
5. Expand dates (15 Mart → on beş Mart)
6. Expand numbers (35 → otuz beş)

Numbers must be fully expanded before G2P because the G2P has no digit handling.
Digits left unreplaced produce incorrect phoneme sequences.

## G2P (Grapheme-to-Phoneme)

Current approach: rule-based lookup table (`g2p/turkish.py`).

Turkish has near-perfect phonemic orthography — one grapheme maps to exactly one phoneme in almost all cases. Exceptions:

- `ğ` (soft g): silent between vowels, lengthens preceding vowel before consonant
- `c` → /dʒ/: the letter C in Turkish represents the affricate, not /k/
- Word boundaries and stress (not yet modeled)

Future: espeak-ng backend (`g2p/phonemizer.py`) for broader coverage including borrowed words.

## VITS Model

Architecture follows Kim et al. (2021) "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech."

Key components:

- **Text encoder**: Transformer-based phoneme encoder
- **Posterior encoder**: Mel-spectrogram to latent z
- **Normalizing flow**: Invertible transforms between latent spaces
- **Generator**: HiFi-GAN v1 adapted for VITS
- **Discriminators**: Multi-Period + Multi-Scale for adversarial training

Target configuration (v0.1, medium quality):
- 22050 Hz sample rate
- 80-bin mel spectrogram
- 256 hop length (~11.6 ms frames)
- Hidden channels: 192

## Deployment (Piper)

For deployment, the trained VITS model is exported to ONNX using the Piper training framework. The ONNX model runs with `onnxruntime` on CPU — no GPU required at inference.

Output: `model.onnx` + `model.onnx.json` (Piper config). Compatible with the Piper CLI and HomeAssistant TTS.
