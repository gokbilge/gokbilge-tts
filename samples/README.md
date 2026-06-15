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
  01_smoke_test/            — 5-epoch smoke run (pipeline correctness only; audio is noise)
    s1_bugun_hava.wav
    s2_turkiye_cumh.wav
    s3_cocuklar.wav
    s4_ogrenciler.wav
    s5_sirket.wav
  02_<next_milestone>/      — e.g. first full training checkpoint
    ...
```

## Convention

- Directories are prefixed `NN_` (01, 02, …) so they sort in training order.
- Each directory gets a metadata block in this README (model, checkpoint, date, RTF).
- **Smoke / diagnostic runs** (`01_smoke_test`, etc.): committed; audio quality not evaluated.
- **Production releases**: also committed here and mirrored on the HuggingFace model page.

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
