# Samples

Audio samples generated from the fixed benchmark sentences (CLAUDE.md § Evaluation Rule):

1. "Bugün hava çok güzel."
2. "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."
3. "Çocuklar çiçek, şeker ve üzüm yedi."
4. "Öğrenciler ölçüm sonuçlarını değerlendirdi."
5. "Şirket yüzde otuz beş büyüme açıkladı."

---

## Directory structure

Directories are numbered so they sort chronologically. Each training milestone gets its own directory.

```
samples/
  01_smoke_test/                — 5-epoch smoke run (noise; pipeline only)
  02_v0_1_full_step010k/        — 10k steps
  03_v0_1_full_step020k/        — 20k steps
  04_v0_1_full_step050k/        — 50k  (noise → speech-like?)
  05_v0_1_full_step080k/        — 80k
  06_v0_1_full_step100k/        — 100k (first speech-like expected)
  07_v0_1_full_step120k/        — 120k
  08_v0_1_full_step140k/        — 140k
  09_v0_1_full_step160k/        — 160k
  10_v0_1_full_step200k/        — 200k (Turkish phonemes distinct?)
  11_v0_1_full_step250k/        — 250k
  12_v0_1_full_step300k/        — 300k (continue / stop decision)
  13_v0_1_full_step400k/        — 400k
  14_v0_1_full_step500k/        — 500k (v0.1 candidate check)
  15_v0_1_full_step600k/        — 600k
  16_v0_1_full_step700k/        — 700k
  17_v0_1_full_step800k/        — 800k (stop / retrain / architecture)
```

Each directory contains 5 WAVs: `s1_bugun_hava.wav` … `s5_sirket.wav` (fixed benchmark sentences).

Generate with: `bash tools/eval_step.sh <sample_dir>` — exports ONNX from `last.ckpt` and runs all 5 sentences automatically.

## Convention

- Directories are prefixed `NN_` (01, 02, …) so they sort in training order.
- Each directory gets a metadata block in this README (model, checkpoint, date, RTF).
- **Smoke / diagnostic runs** (`01_smoke_test`, etc.): committed; audio quality not evaluated.
- **Full training milestones**: step-based (`stepNNNk`), not epoch-based. Stop criterion is perceptual quality, not epoch count.

## Quality notes by milestone

| Directory | Steps | Quality |
|---|---|---|
| 01_smoke_test | ~300 | pure noise — pipeline check only |
| 02_v0_1_full_step010k | 10k | noise |
| 03_v0_1_full_step020k | 20k | noise |
| 03b_v0_1_full_step032k | 32k | dirty but some words and sentences audible |
| 04_v0_1_full_step050k | 50k | — |
| 05_v0_1_full_step080k | 80k | — |
| 06_v0_1_full_step100k | 100k | noisy, partial words |
| 07_v0_1_full_step120k | 120k | more stable rhythm, some intelligible phrases |
| 08_v0_1_full_step140k | 140k | sentences increasingly intelligible, Turkish accent improving, still noisy/cut syllables |
| 09_v0_1_full_step160k | 160k | clearer sentence structure, still noisy with unstable articulation |
| 10_v0_1_full_step200k | 200k | better than 160k; more intelligible words and stronger phoneme separation |

---

## 01_smoke_test — metadata (2026-06-15)

| File | Text | Duration | RTF |
|---|---|---|---|
| s1_bugun_hava.wav | Bugün hava çok güzel. | 1.65 s | 0.04 |
| s2_turkiye_cumh.wav | Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu. | 3.53 s | 0.03 |
| s3_cocuklar.wav | Çocuklar çiçek, şeker ve üzüm yedi. | 2.41 s | 0.03 |
| s4_ogrenciler.wav | Öğrenciler ölçüm sonuçlarını değerlendirdi. | 2.80 s | 0.03 |
| s5_sirket.wav | Şirket yüzde otuz beş büyüme açıkladı. | 2.24 s | 0.03 |

Model: `runs/smoke/gokbilge_tr_smoke.onnx` — checkpoint `epoch=4-step=300`, 100 training utterances.
Inference: `piper_train.infer_onnx`, OnnxRuntime CPU, NVIDIA GB10 server.

---

## 09_v0_1_full_step160k — metadata (2026-06-16)

| File | Duration |
|---|---|
| s1_bugun_hava.wav | 1.44 s |
| s2_turkiye_cumh.wav | 3.00 s |
| s3_cocuklar.wav | 1.79 s |
| s4_ogrenciler.wav | 2.67 s |
| s5_sirket.wav | 1.86 s |

Checkpoint label: `step160k` full-corpus evaluation export.
Listening note: clearer sentence structure than 140k, but still noisy with clipped syllables.

---

## 10_v0_1_full_step200k — metadata (2026-06-16)

| File | Duration |
|---|---|
| s1_bugun_hava.wav | 2.08 s |
| s2_turkiye_cumh.wav | 2.87 s |
| s3_cocuklar.wav | 2.72 s |
| s4_ogrenciler.wav | 3.04 s |
| s5_sirket.wav | 2.37 s |

Checkpoint label: `step200k` full-corpus evaluation export.
Listening note: perceptually better than 160k, with more intelligible words and cleaner phoneme separation across the benchmark set.
