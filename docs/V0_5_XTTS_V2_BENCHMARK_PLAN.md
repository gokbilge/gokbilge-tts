# v0.5 XTTS v2 Turkish Benchmark Plan

## Purpose

XTTS v2 is evaluated as an external Turkish TTS reference against our internal candidates.

## Internal baselines

- v0.1 step500k: primary RC baseline
- v0.4 checkpoint 20: best fine-tune candidate so far

## Why

v0.4 fine-tuning reached plateau/regression. Further improvement should be compared against an external multilingual Turkish-capable model before choosing the next architecture/adaptation path.

## Fixed benchmark sentences

1. Bugün hava çok güzel.
2. Türkiye Cumhuriyeti bin dokuz yüz yirmi üç yılında kuruldu.
3. Çocuklar çiçek, şeker ve üzüm yedi.
4. Öğrenciler ölçüm sonuçlarını değerlendirdi.
5. Şirket yüzde otuz beş büyüme açıkladı.

## Output convention

XTTS generated samples are local-only and should be placed under:

samples/25_xtts_v2_tr_zero_shot/

or, if a speaker reference is used:

samples/25_xtts_v2_tr_ref_<short_label>/

## Evaluation criteria

- intelligibility
- Turkish-heavy words
- stutter/gap/cut-up behavior
- accent / naturalness
- prosody
- comparison to v0.1 step500k
- comparison to v0.4 candidate 20

## Safety

Do not commit generated WAV files or downloaded model weights.
