# Dataset Cleaning for v0.2

This is a side-preparation workflow for future v0.2 training while the current v0.1 run continues unchanged.

## Why v0.2 needs dataset cleaning

The current v0.1 work is focused on baseline training behavior and listening-based checkpoint selection. For v0.2, we want a safer way to audit dataset quality and generate cleaner manifest candidates without touching the raw ISSAI corpus.

The goals are:

- inspect the existing manifest row by row
- compute reproducible quality metrics from each audio/text pair
- flag suspicious or reject-worthy samples
- generate new manifest candidates for future training experiments
- keep all raw audio and all original manifests intact

## Safety rules

- Raw dataset audio is never modified.
- Raw dataset files are never deleted.
- Existing v0.1 manifests are never rewritten in place.
- Current training runs, checkpoints, logs, and hyperparameters are not touched.
- Clean manifests are written as new files for later v0.2 experiments.

## Tools

- `tools/audit_dataset.py`
- `tools/filter_manifest.py`
- `tools/build_clean_manifest.py`
- `tools/asr_manifest_check.py` is a scaffold only in this pass

## What the audit computes

For each manifest row, the audit records:

- `audio_filepath`
- `text`
- `text_length`
- `word_count`
- `audio_duration_sec`
- `sample_rate`
- `channels`
- `rms_level_db`
- `peak_level_db`
- `clipping_ratio`
- `leading_silence_sec`
- `trailing_silence_sec`
- `internal_silence_ratio`
- `longest_internal_gap_sec`
- `chars_per_sec`
- `words_per_sec`
- `quality_score`
- `status`
- `reasons`

Unreadable audio is handled safely:

- the row is marked `reject`
- the reason is `audio_unreadable`
- processing continues for the rest of the manifest

## Default thresholds

```text
min_duration_sec = 0.8
max_duration_sec = 15.0
min_text_chars = 3
max_text_chars = 250
min_chars_per_sec = 6
max_chars_per_sec = 22
max_leading_silence_sec = 0.5
max_trailing_silence_sec = 0.7
max_internal_silence_ratio = 0.40
max_longest_internal_gap_sec = 0.45
max_clipping_ratio = 0.01
```

Additional implementation defaults:

- silence detection threshold: `-40 dB`
- silence frame size: `20 ms`
- silence hop size: `10 ms`
- low-RMS warning threshold: `-30 dB`
- peak warning threshold: `-0.5 dB`

## Quality score and status mapping

The current `quality_score` is a deterministic 0-100 heuristic built from:

- duration score
- speed score
- silence score
- volume score
- clipping score

Status mapping:

- `keep`: `quality_score >= 75` and no audit reasons
- `suspicious`: `50 <= quality_score < 75`, or minor reasons only
- `reject`: any severe reason, or `quality_score < 50`

Severe reasons currently include:

- `audio_unreadable`
- `duration_too_short`
- `duration_too_long`
- `text_too_short`
- `text_too_long`
- `chars_per_sec_too_low`
- `chars_per_sec_too_high`
- `clipping_too_high`
- `internal_silence_too_high`

## Strict vs balanced strategy

`strict` keeps only rows with status `keep`.

`balanced` keeps:

- all `keep` rows
- `suspicious` rows that do not contain severe reasons

This gives two candidate manifests:

- a conservative clean set for higher-confidence training
- a broader set that preserves borderline-but-usable samples

## ASR mismatch check

ASR-based transcript mismatch scoring is intentionally not implemented in this first pass.

`tools/asr_manifest_check.py` is only a scaffold. The planned next step is a faster-whisper based mismatch audit once we are ready to accept the dependency and runtime cost.

## How to run an audit

Example:

```bash
python tools/audit_dataset.py \
  --manifest data/manifests/train.jsonl \
  --output reports/dataset_audit.csv \
  --summary reports/dataset_quality_summary.md
```

You can override thresholds from CLI, for example:

```bash
python tools/audit_dataset.py \
  --manifest data/manifests/train.jsonl \
  --output reports/dataset_audit.csv \
  --summary reports/dataset_quality_summary.md \
  --max-duration-sec 12 \
  --max-leading-silence-sec 0.4
```

## How to generate clean manifests

Run the wrapper:

```bash
python tools/build_clean_manifest.py \
  --manifest data/manifests/train.jsonl \
  --reports-dir reports \
  --manifests-dir data/manifests
```

Expected outputs:

```text
reports/dataset_quality_summary.md
reports/dataset_audit.csv
reports/rejected_samples.csv
reports/suspicious_samples.csv
data/manifests/train_clean_strict.jsonl
data/manifests/train_clean_balanced.jsonl
data/manifests/train_rejected.jsonl
data/manifests/train_suspicious.jsonl
```

## How to inspect suspicious or rejected samples

- read `reports/dataset_audit.csv` for the full audit table
- read `reports/suspicious_samples.csv` for borderline rows
- read `reports/rejected_samples.csv` for likely removals
- use `status`, `reasons`, `quality_score`, and the silence/speed metrics to review patterns before future filtering changes

## Future v0.2 usage

This workflow is preparation only. It does not switch training over to filtered manifests yet.

Recommended future usage:

1. audit the current baseline manifest
2. inspect `suspicious` and `reject` patterns manually
3. refine thresholds if needed
4. compare `train_clean_strict.jsonl` vs `train_clean_balanced.jsonl`
5. choose one of those new manifests for an explicit future v0.2 training run
