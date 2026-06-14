# Gokbilge TTS

Open Turkish Text-to-Speech Toolkit.

Gokbilge TTS is an open-source project for building Turkish-native text-to-speech models with reproducible training recipes, Turkish text normalization, Turkish G2P/phonemization, and Piper/VITS-compatible deployment paths.

---

## Project Goals

- Provide a **fully open, reproducible** training pipeline for Turkish TTS
- Support **Turkish-native phonemization** without relying on non-Turkish acoustic priors
- Target **practical deployment**: CPU-friendly Piper/ONNX inference
- Build on the **ISSAI Turkish Speech Corpus** as the primary open dataset
- Maintain clean documentation so the community can replicate and extend results

---

## Why Turkish-native TTS?

Turkish has distinct phonological properties that general multilingual TTS models handle poorly:

- **ğ** (soft g) — a vowel-lengthening silent consonant, not a stop
- **ı** — back unrounded vowel /ɯ/, absent from most European languages
- **c** — always /dʒ/ (as in "jam"), never /k/
- **ç** — always /tʃ/ (as in "church")
- **ş** — always /ʃ/ (as in "shoe")
- **Vowel harmony** — suffix vowels follow the root vowel pattern
- **Agglutinative morphology** — compound words and long suffixed forms are common

A Turkish-native model trained on real Turkish speech, with a proper Turkish G2P frontend, is the correct solution.

---

## Dataset

Primary dataset:
- [`issai/Turkish_Speech_Corpus`](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)

The ISSAI Turkish Speech Corpus (TSC) is an open Turkish speech dataset with clean studio recordings, suitable for training high-quality TTS models.

See [docs/DATASETS.md](docs/DATASETS.md) for details on data preparation, manifest format, and audio validation.

---

## Architecture

Initial architecture path:
- Piper-compatible VITS model for practical CPU-friendly inference
- VITS/VITS2-style training experiments
- ONNX export where possible

Pipeline stages:
1. **Text normalization** — numbers, dates, abbreviations, punctuation
2. **Turkish G2P** — rule-based phonemization with optional espeak-ng comparison
3. **VITS/Piper training** — end-to-end neural TTS
4. **ONNX export** — for Piper-compatible CPU deployment

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline description.

---

## Repository Structure

```
src/gokbilge_tts/       — Python package
  normalize/            — Turkish text normalization
  g2p/                  — Turkish G2P / phonemization
  datasets/             — Dataset preparation (ISSAI)
  infer/                — Inference: Piper/VITS/ONNX
  utils/                — Audio, paths, logging

configs/                — Training configuration files
recipes/                — Shell-based training recipes
  issai_piper/          — Piper-compatible VITS on ISSAI
  issai_vits/           — VITS training on ISSAI
benchmarks/             — Evaluation sentences and scoring rubrics
docs/                   — Extended documentation
tests/                  — Unit tests
samples/                — Audio sample gallery (post-training)
```

---

## Current Status

**Status: repository scaffold / pre-training stage.**
No public model weights are released yet.

- [x] Repository structure
- [x] Python package skeleton
- [x] CLI skeleton
- [x] Text normalization placeholders
- [x] Turkish G2P placeholders
- [x] Dataset preparation placeholders
- [x] Training recipe shells
- [x] Benchmark sentence set
- [ ] Turkish G2P implementation
- [ ] Text normalization implementation
- [ ] ISSAI dataset pipeline
- [ ] First training run (Piper baseline)
- [ ] Model release

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phased plan.

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Repository scaffold | ✅ Done |
| 1 | Dataset preparation | 🔲 Next |
| 2 | Turkish G2P | 🔲 Planned |
| 3 | Piper/VITS baseline | 🔲 Planned |
| 4 | Quality improvements | 🔲 Planned |
| 5 | Release | 🔲 Planned |

---

## Benchmark Sentences

Fixed evaluation set (do not change between model comparisons):

```
Bugün hava çok güzel.
Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu.
Çocuklar çiçek, şeker ve üzüm yedi.
Öğrenciler ölçüm sonuçlarını değerlendirdi.
Şirket yüzde otuz beş büyüme açıkladı.
Cem çocuklara çiçek aldı.
Çağrı çorbayı çok sıcak buldu.
Cihan camdan dışarı baktı.
Çiğdem çantasını çabukça aldı.
Işık ılık ırmaktan yansıdı.
```

See [benchmarks/](benchmarks/) for the scoring rubric and benchmark plan.

---

## Installation

```bash
pip install -e ".[dev]"
```

```bash
gokbilge-tts --help
gokbilge-tts normalize "Bugün hava çok güzel."
gokbilge-tts phonemize "Bugün hava çok güzel."
```

---

## License

MIT — see [LICENSE](LICENSE).

Dataset and model licenses must be verified separately before any model release.
See [docs/LICENSE_STRATEGY.md](docs/LICENSE_STRATEGY.md).

---

## References

- [ISSAI Turkish Speech Corpus](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus)
- [Piper TTS](https://github.com/rhasspy/piper)
- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
- [VITS2](https://arxiv.org/abs/2307.16430)
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) (optional phonemization reference)
