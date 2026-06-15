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
  01_smoke_test/                — 5-epoch smoke run (noise; pipeline check only)
  02_v0_1_full_step050k/        — v0.1 full training, ~50k steps  (noise→speech-like?)
  03_v0_1_full_step100k/        — v0.1 full training, ~100k steps (first speech-like)
  04_v0_1_full_step200k/        — v0.1 full training, ~200k steps (phonemes distinct?)
  05_v0_1_full_step300k/        — v0.1 full training, ~300k steps (continue/stop?)
  06_v0_1_full_step500k+/       — v0.1 quality decision
```

Each directory contains 5 WAVs: `s1_bugun_hava.wav` … `s5_sirket.wav` (fixed benchmark sentences).

Generate with: `bash tools/eval_step.sh <sample_dir>` — exports ONNX from `last.ckpt` and runs all 5 sentences automatically.

## Convention

- Directories are prefixed `NN_` (01, 02, …) so they sort in training order.
- Each directory gets a metadata block in this README (model, checkpoint, date, RTF).
- **Smoke / diagnostic runs** (`01_smoke_test`, etc.): committed; audio quality not evaluated.
- **Full training milestones**: step-based (`stepNNNk`), not epoch-based. Stop criterion is perceptual quality, not epoch count.

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
