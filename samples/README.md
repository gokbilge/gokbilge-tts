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
| 10_v0_1_full_step200k | 200k | sentences are now structurally recognizable; speech rhythm and Turkish-like accent continue to improve; intermittent noise, stuttering, skipped syllables, and swallowed words remain; some readings are fast or energetic, suggesting prosody is emerging but not yet stable |
| 11_v0_1_full_step250k | 250k | most sentences are more stable and intelligible; short sentence s1 is partially swallowed; residual noise, skipped syllables, fast tempo, and accent remain; prosody is improving but still unstable |
| 12_v0_1_full_step300k | 300k | strong v0.1 candidate signal; s1 no longer fully swallowed but remains segmented with audible gaps; long sentences are more coherent; s2 has some initial noise and energetic tempo; s3 still stutters on Turkish-heavy words; s4 and especially s5 show strong improvement; residual noise, accent, skipped syllables, and unstable prosody remain |
| 14_v0_1_full_step500k | 500k | primary v0.1 release-candidate signal; clearly improves over 300k on most benchmark samples, especially s1, s2, s4, and s5; s3 remains the main unresolved weakness with stuttering/gap behavior on Turkish-heavy words such as “çocuklar,” “çiçek,” “şeker,” and “üzüm”; preserve 500k as the primary RC candidate while keeping 300k as a fallback until wider Turkish-heavy evaluation is completed |

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
Listening note: sentences are now structurally recognizable; speech rhythm and Turkish-like accent continue to improve; intermittent noise, stuttering, skipped syllables, and swallowed words remain; some readings are fast or energetic, suggesting prosody is emerging but not yet stable.

---

## 11_v0_1_full_step250k — metadata

Checkpoint label: `step250k` full-corpus evaluation export.

Listening note: most benchmark sentences are more stable and intelligible than 200k. The short sentence `s1_bugun_hava.wav` remains partially swallowed / unstable. The remaining samples show improved structure, clearer sentence flow, and stronger Turkish-like rhythm, but residual noise, skipped syllables, fast tempo, accent, and unstable prosody remain.

| File | Notes |
|---|---|
| s1_bugun_hava.wav | short sentence remains unstable; partially swallowed / cut |
| s2_turkiye_cumh.wav | more coherent and stable than earlier milestones; long sentence structure is clearer |
| s3_cocuklar.wav | improved structure; still has skipped syllables / articulation instability |
| s4_ogrenciler.wav | comparatively stable; sentence flow is clearer |
| s5_sirket.wav | short and fast; accented but more intelligible |

---

## 12_v0_1_full_step300k — metadata

Checkpoint label: `step300k` full-corpus evaluation export.

Listening note: strong v0.1 candidate signal. The short sentence `s1_bugun_hava.wav` is no longer fully swallowed, but remains segmented with audible gaps between syllables/words. `s2_turkiye_cumh.wav` is more coherent as a long sentence, with some initial noise and energetic tempo. `s3_cocuklar.wav` still stutters on Turkish-heavy words, especially around “çocuklar,” but continues to improve. `s4_ogrenciler.wav` is comparatively stable, and `s5_sirket.wav` is one of the strongest samples so far. Residual noise, accent, skipped syllables, and unstable prosody remain.

Candidate status: `step300k` is a v0.1 candidate checkpoint. Training should continue to 500k before final release-candidate selection. Compare 300k vs 500k using the same 5 benchmark sentences.

| File | Notes |
|---|---|
| s1_bugun_hava.wav | no longer fully swallowed; still segmented with audible gaps |
| s2_turkiye_cumh.wav | coherent long-sentence structure; some initial noise and energetic tempo |
| s3_cocuklar.wav | still stutters on Turkish-heavy words; fast but improving |
| s4_ogrenciler.wav | comparatively stable; sentence flow continues to improve |
| s5_sirket.wav | strongest or near-strongest sample; substantially more intelligible |

---

## 14_v0_1_full_step500k — metadata

Checkpoint label: `step500k` full-corpus release-candidate comparison export.

Listening note: 500k shows a strong v0.1 release-candidate signal and is clearly better than 300k on most benchmark samples. `s1_bugun_hava.wav` is substantially more stable than earlier checkpoints, `s2_turkiye_cumh.wav` is more coherent as a long sentence, and `s4_ogrenciler.wav` plus `s5_sirket.wav` are among the strongest samples so far. The main unresolved weakness is `s3_cocuklar.wav`, which still shows stuttering/gap behavior on Turkish-heavy words such as “çocuklar,” “çiçek,” “şeker,” and “üzüm.” Keep 500k as the primary v0.1 RC candidate while preserving 300k as a fallback candidate until wider Turkish-heavy evaluation is completed.

| File | Notes |
|---|---|
| s1_bugun_hava.wav | substantially more stable than 300k; fewer audible gaps |
| s2_turkiye_cumh.wav | coherent long-sentence structure; improved over 300k |
| s3_cocuklar.wav | main unresolved weakness; stutters/skips on Turkish-heavy words |
| s4_ogrenciler.wav | strong and comparatively stable |
| s5_sirket.wav | strong; supports 500k as primary RC candidate |


## 18_v0_4_finetune_step556k — metadata

Checkpoint label: `step556k` v0.4 fine-tune continuation export.

Listening note: first positive v0.4 continuation signal. `s4_ogrenciler.wav` and `s5_sirket.wav` are very good, while `s1_bugun_hava.wav`, `s2_turkiye_cumh.wav`, and `s3_cocuklar.wav` are improving. Later listening remains mixed-positive: `s2` and `s5` improved further, `s4` stayed good with some foreign-accent character, `s1` showed mild regression risk, and `s3` remained the key unresolved Turkish-heavy weakness with mild stutter / gap behavior. This direction is materially better than the v0.3 scratch pilot and should be preserved locally for comparison against later v0.4 checkpoints.

Candidate status: preserve locally as the first positive v0.4 fine-tune signal. Do not replace `v0.1 step500k` as the primary release candidate until later v0.4 checkpoints confirm consistent quality.
