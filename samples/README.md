# Samples

Audio samples generated from the fixed benchmark sentences (CLAUDE.md § Evaluation Rule):

1. "Bugün hava çok güzel."
2. "Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu."
3. "Çocuklar çiçek, şeker ve üzüm yedi."
4. "Öğrenciler ölçüm sonuçlarını değerlendirdi."
5. "Şirket yüzde otuz beş büyüme açıkladı."

---

## Directory structure

```
samples/
  smoke/                    — 5-epoch smoke run (pipeline correctness only; audio is noise)
    s1_bugun_hava.wav
    s2_turkiye_cumh.wav
    s3_cocuklar.wav
    s4_ogrenciler.wav
    s5_sirket.wav
  v0.1/                     — first real release (not yet trained)
    s1_bugun_hava.wav
    ...
```

## Convention

- **smoke/**: Committed. Generated from the smoke checkpoint (`epoch=4`, 100 utterances). Audio is
  noise — these files exist only to confirm end-to-end pipeline correctness (inference produces
  audio of the right length, phoneme IDs round-trip correctly). Do not evaluate quality here.
- **vX.Y/**: Production release samples. Committed when a model is released to HuggingFace.
  These are also available on the HuggingFace model page.

## Smoke sample metadata (2026-06-15)

| File | Text | Duration | RTF |
|---|---|---|---|
| s1_bugun_hava.wav | Bugün hava çok güzel. | 1.65 s | 0.04 |
| s2_turkiye_cumh.wav | Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu. | 3.53 s | 0.03 |
| s3_cocuklar.wav | Çocuklar çiçek, şeker ve üzüm yedi. | 2.41 s | 0.03 |
| s4_ogrenciler.wav | Öğrenciler ölçüm sonuçlarını değerlendirdi. | 2.80 s | 0.03 |
| s5_sirket.wav | Şirket yüzde otuz beş büyüme açıkladı. | 2.24 s | 0.03 |

Model: `runs/smoke/gokbilge_tr_smoke.onnx` — checkpoint `epoch=4-step=300`, 100 training utterances.
Inference: `piper_train.infer_onnx`, OnnxRuntime CPU, NVIDIA GB10 server.
