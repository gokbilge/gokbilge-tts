# v0.2 Balanced Pilot Negative Closure

Status: `STOPPED_NEGATIVE_PILOT`

## Run

- Run directory: `runs/v0_2_balanced_001`
- Manifest: `data/manifests/train_clean_balanced.jsonl`

## Observed Point

- `global_step ~= 134424`
- `epoch = 5`

## Human Verdict

- not improving
- worse than v0.1 `step500k`

## Decision

- stop and close the v0.2 balanced pilot
- do not continue this run
- preserve artifacts locally for later analysis

## Release Position

- v0.1 `step500k` remains the primary RC
- v0.1 `step300k` remains the fallback

## Lesson

Balanced dataset cleaning alone did not solve the Turkish-heavy / `s3_cocuklar.wav` weakness.

## Next Direction

Move to the v0.3 Turkish-heavy / phoneme-focused diagnostics track.
