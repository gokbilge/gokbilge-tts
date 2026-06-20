# v0.3 Turkish-Heavy Diagnostic Plan

## Goal

Reduce stutter/gap behavior on Turkish-heavy words and phoneme patterns.

## Scope

- not general dataset cleaning
- not another immediate full training run
- diagnostic-first

## Primary Diagnostic Weakness

- `s3_cocuklar.wav`
- sentence: `?ocuklar ?i?ek, ?eker ve ?z?m yedi.`

## Target Words

- `?ocuklar`
- `?i?ek`
- `?eker`
- `?z?m`
- `?l??m`
- `??renciler`
- `b?y?me`
- `T?rkiye`
- `Cumhuriyeti`
- `y?zde`

## Target Turkish Characters

- `?`
- `?`
- `?`
- `?`
- `?`
- `?`

## Required Analyses

1. Count these target words in the training manifest.
2. Extract matching rows for manual/audio audit.
3. Compute duration, chars/sec, internal silence, longest gap, RMS, peak.
4. Compare target-word rows against dataset-wide averages.
5. Identify whether problematic words correlate with bad audio, silence, speed, low volume, clipping, or text/audio mismatch.
6. Build a v0.3 diagnostic eval set before any new training.

## Candidate Future Experiments

- strict manifest pilot
- Turkish-heavy filtered manifest
- phoneme/G2P audit
- text-normalization fixes

## Explicit Non-Goals

- do not continue v0.2 balanced blindly
- do not overwrite v0.1 candidates
- do not release v0.3 until it beats v0.1 `step500k` on Turkish-heavy samples
## Gap-aware audit update

- The v0.3 audit tool now emits gap and silence metrics:
  - `leading_silence_sec`
  - `trailing_silence_sec`
  - `internal_silence_ratio`
  - `longest_internal_gap_sec`
  - `internal_gap_count`
  - `low_energy_ratio`
- Do not choose the next manifest strategy before reviewing:
  - `reports/v0_3_turkish_heavy/balanced_target_quality_buckets_summary.md`
- Generated reports under `reports/v0_3_turkish_heavy/` remain local-only.
- The next decision should be based on which failure modes dominate bad target rows:
  - longest internal gap
  - internal silence
  - low RMS
  - speed / chars-per-sec
  - clipping

## Manifest strategy selected

- Selected strategy: `balanced + target bad-row exclusion`
- Generated local manifest path: `data/manifests/train_v0_3_turkish_heavy_clean.jsonl`
- Generated local summary path: `reports/v0_3_turkish_heavy/train_v0_3_turkish_heavy_clean_summary.md`
- The generated manifest and reports are local-only and must not be committed.
- Do not start training until all of the following are reviewed:
  1. protected term retention
  2. worst excluded rows spot-check
  3. v0.1 `step500k` remains the benchmark to beat

## Relaxed exclusion modes

- Aggressive exclusion was too destructive for protected Turkish-heavy terms.
- The manifest builder now supports `aggressive`, `relaxed`, and `hard` exclusion modes.
- Generated local paths:
  - `data/manifests/train_v0_3_turkish_heavy_relaxed.jsonl`
  - `data/manifests/train_v0_3_turkish_heavy_hard.jsonl`
- Manifest selection must be based on protected retention together with gap-metric improvement.
- Do not train until the selected manifest is reviewed.

## Relaxed pilot review package

- Relaxed strategy is the selected candidate for pilot review.
- Spot-check local report path: `reports/v0_3_turkish_heavy/spotcheck/SPOTCHECK_README.md`
- Excluded top100: `reports/v0_3_turkish_heavy/spotcheck/excluded_relaxed/excluded_relaxed_top100.csv`
- Retained near-threshold top100: `reports/v0_3_turkish_heavy/spotcheck/retained_near_threshold/retained_near_threshold_top100.csv`
- Random reference: `reports/v0_3_turkish_heavy/spotcheck/random_reference/random_reference_50.csv`
- Training script: `recipes/issai_piper/train_v0_3_turkish_heavy_relaxed.sh`
- Run directory: `runs/v0_3_turkish_heavy_relaxed_001`
- Manifest: `data/manifests/train_v0_3_turkish_heavy_relaxed.jsonl`
- Warning: do not start training until the spot-check package is reviewed.
