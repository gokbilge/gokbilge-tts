# CLAUDE.md — gokbilge-tts Project Rules

Guidelines for working on this project.

---

## Project Roadmap

| Version | Goal |
|---|---|
| **v0.1** | Piper/VITS + espeak-ng Turkish baseline — current run (`v0_1_full_001`) |
| **v0.2** | Gokbilge custom Turkish G2P integration (replace espeak-ng phonemizer) |
| **v0.3** | Better normalization + prosody + Turkish stress rules |
| **v0.4** | Alternative architecture exploration: StyleTTS2 / VITS2 / Matcha-TTS |
| **v1.0** | Best quality + clean model release |

Each version builds on the previous. v0.1 establishes the training pipeline and quality baseline. v0.2 is the first point where Turkish-specific improvements land. Do not skip to v0.4 architecture experiments before v0.2–v0.3 improvements are evaluated.

---

## Repository Layout

```
train.py              — fine-tuning script (do not rename, server depends on this path)
inference.py          — single inference
run_inference_test.py — 5-sentence eval, always transfer via pscp (UTF-8 safe)
prepare_dataset.py    — dataset tokenization / preprocessing
configs/              — experiment YAML specs
docs/                 — extended notes
samples/              — tracked audio outputs (numbered subdirs: 01_smoke_test/, 02_…)
FINDINGS.md           — empirical findings (key reference)
TRAINING_LOG.md       — chronological run log (always update this)
MODEL_CARD.md         — model description for external readers
```

---

## Commit Rules

- No `Co-Authored-By` lines in commit messages
- Commit messages must explain **why**, not just what changed
- Reference `TRAINING_LOG.md` when a commit fixes something diagnosed there

---

## Training Log Rules

Every training decision must be documented in `TRAINING_LOG.md`:

- **Before a run:** command, hyperparameters, and why those values were chosen
- **After a run:** result, what the losses did, and whether audio quality improved
- **If something is wrong:** root cause analysis with evidence
- **If something is changed:** what changed, why, and expected effect

Do not leave a run undocumented. Even failed/discarded runs get a section.

---

## Evaluation Rule

Always use `run_inference_test.py` — never pass Turkish text inline via plink.
PuTTY plink corrupts Turkish characters. Transfer script via pscp, run via plink.

Test sentences (fixed — do not change between evaluations):
1. "Bugün hava çok güzel."
2. "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."
3. "Çocuklar çiçek, şeker ve üzüm yedi."
4. "Öğrenciler ölçüm sonuçlarını değerlendirdi."
5. "Şirket yüzde otuz beş büyüme açıkladı."

Write numbers as words. Digits may fall back to non-Turkish phoneme behavior.

---

## Remote Server

- Connect: `"C:\Program Files\PuTTY\plink.exe" -batch -ssh -pw "..." hcfk@[server-ip] "..."`
- Transfer: `"C:\Program Files\PuTTY\pscp.exe" -batch -pw "..." src hcfk@[server-ip]:dst`
- Models: `/home/hcfk/models/`
- Dataset: `/home/hcfk/datasets/`
- Checkpoints: `/home/hcfk/checkpoints/`

**Do not commit server IP, password, or username to any tracked file.**

---

## Run Naming Convention

Run directories live under `runs/` (never committed — in `.gitignore`):

```
runs/
  smoke_001/        — first smoke run
  smoke_002/        — second smoke run (if needed)
  v0_1_full_001/    — first full v0.1 training run
  v0_1_full_002/    — resume or retry of v0.1
```

Rules:
- Smoke runs: `runs/smoke_NNN/`
- Full training runs: `runs/<version>_full_NNN/`
- Never reuse a run directory. Increment the suffix.
- Never commit `runs/`, checkpoints, ONNX files, generated WAVs, or dataset outputs.
- The current smoke run is `runs/smoke/` (retroactively `smoke_001`). New runs start at `smoke_002`.

---

## Training Stop Policy

Full training does not finish by reaching 10,000 epochs.
`--max_epochs 10000` is only a safety ceiling.

Training is evaluated by step milestones:
- 50k: early noise/speech-shape check
- 100k: first early intelligibility check
- 200k: Turkish phoneme/prosody check
- 300k: continue/adjust decision
- 500k: v0.1 candidate check
- 800k: stop/retrain/architecture decision

Stop training when:
- benchmark samples plateau,
- audio quality is sufficient for v0.1,
- or 800k steps still fails to produce usable speech.

Do not wait for 10,000 epochs.

---

## Samples Convention

`samples/` holds audio outputs for each training milestone, named `NN_<label>/` so they sort chronologically:

```
samples/
  01_smoke_test/              — 5-epoch smoke run (noise; pipeline check only)
  02_v0_1_full_step012k/      — v0.1 epoch 1 check (~12k steps; last.ckpt exists?)
  03_v0_1_full_step050k/      — v0.1 ~50k steps (noise → speech-like?)
  04_v0_1_full_step100k/      — v0.1 ~100k steps (first speech-like expected)
  05_v0_1_full_step200k/      — v0.1 ~200k steps (Turkish phonemes distinct?)
  06_v0_1_full_step300k/      — v0.1 ~300k steps (continue/stop decision)
  07_v0_1_full_step500k/      — v0.1 candidate check
  08_v0_1_full_step800k/      — stop/retrain/architecture decision
  ...
```

Rules:
- Every evaluation produces 5 WAV files, one per fixed benchmark sentence.
- New milestone = new `NN_` directory. Never overwrite a previous directory.
- Milestones are **step-based** (`stepNNNk`), not epoch-based. Stop criterion is perceptual quality plateau.
- Smoke / diagnostic samples are committed even though audio is noise.
- Add a metadata block to `samples/README.md` for every new directory (model, checkpoint, date, RTF).
- Generate with: `bash tools/eval_step.sh <sample_dir> [run_dir]` — exports ONNX from `last.ckpt` and synthesizes all 5 sentences.
