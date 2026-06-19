# v0.2 balanced pilot plan

## Pilot scope

- First pilot manifest: `data/manifests/train_clean_balanced.jsonl`
- Source manifest: `runs/v0_1_full_001/manifests/train.jsonl`
- Proposed run name: `v0_2_balanced_001`

## Audit result

- Original: `178901`
- Balanced: `172314`
- Strict: `134036`
- Rejected: `6587`
- Suspicious: `38278`

## Decision

Use the balanced manifest for the first v0.2 pilot.

Reason: the balanced manifest removes clearly bad rows while preserving most of the training data. The strict manifest should be kept for a second comparison run only, after the balanced pilot has been evaluated.

## Evaluation plan

Milestone evaluations to preserve and compare:

- `50k`
- `100k`
- `150k`
- `200k`

Do not compare against the smoke run. Compare directly against:

- v0.1 `step300k` fallback candidate
- v0.1 `step500k` primary RC candidate

## Listening focus

Watch especially:

- `s3_cocuklar.wav`
- Turkish-heavy words: `?ocuklar`, `?i?ek`, `?eker`, `?z?m`, `?l??m`, `??renciler`, `b?y?me`

Primary question: does the balanced-clean manifest reduce stuttering, gaps, or unstable articulation on Turkish-heavy words without regressing the stronger benchmark samples.

## Safety rules

- Do not overwrite any v0.1 run directories.
- Do not delete or modify v0.1 candidate checkpoints.
- Keep generated clean manifests local-only unless release policy changes.
- Do not start v0.2 training until the run command and evaluation checkpoints are explicitly approved.
