# v0.4 Step500k Fine-tune Turkish-heavy Plan

## Decision

v0.4 is a continuation/fine-tune track from the v0.1 step500k checkpoint.

It is not a scratch-training track.

## Base checkpoint

runs/v0_1_full_001/candidates/step500k/step500k.ckpt

## Why

v0.1 step500k remains the best overall model. v0.3 relaxed scratch training improved dataset-level metrics but produced fragmented, cut-up, sometimes unintelligible audio. Therefore, v0.4 must preserve v0.1 quality and only attempt a controlled Turkish-heavy improvement.

## Objective

Improve Turkish-heavy target-word behavior while preserving v0.1 step500k intelligibility and prosody.

## Target words

?ocuklar, ?i?ek, ?eker, ?z?m, ?l??m, ??renciler, b?y?me, T?rkiye, Cumhuriyeti, y?zde

## Default manifest strategy

Use conservative target-good oversampling:
- keep broad clean base data
- exclude known bad target rows
- oversample good target rows in a controlled way
- do not use target-only as default release training

## Candidate manifests

1. conservative - default first pilot
2. weighted - only if conservative under-exposes target terms
3. target_only - diagnostic only, not release training

## Milestones

Fine-tune milestones:
- 5k
- 10k
- 25k
- 50k

## Evaluation

Always compare to:
- v0.1 step500k
- v0.1 step300k fallback if needed

Core sentences:
1. Bug?n hava ?ok g?zel.
2. T?rkiye Cumhuriyeti bin dokuz y?z yirmi ?? y?l?nda kuruldu.
3. ?ocuklar ?i?ek, ?eker ve ?z?m yedi.
4. ??renciler ?l??m sonu?lar?n? de?erlendirdi.
5. ?irket y?zde otuz be? b?y?me a??klad?.

## Stop criteria

Stop early if:
- audio becomes fragmented or cut-up like v0.3
- general intelligibility drops below v0.1 step500k
- s1/s2/s4/s5 regress while s3 does not improve
- any NaN/crash/checkpoint corruption appears

## Generated local manifests

- data/manifests/train_v0_4_finetune_turkish_heavy_conservative.jsonl
- data/manifests/train_v0_4_finetune_turkish_heavy_weighted.jsonl
- data/manifests/train_v0_4_finetune_turkish_heavy_target_only.jsonl

## Generated local reports

- reports/v0_4_step500k_finetune/conservative_summary.md
- reports/v0_4_step500k_finetune/weighted_summary.md
- reports/v0_4_step500k_finetune/target_only_summary.md
- reports/v0_4_step500k_finetune/v0_4_prep_output.log
- reports/v0_4_step500k_finetune/V0_4_TRAINING_STARTED.md

## Selected default manifest

- conservative
- data/manifests/train_v0_4_finetune_turkish_heavy_conservative.jsonl

## Train script

- recipes/issai_piper/train_v0_4_step500k_finetune_turkish_heavy.sh

## Resume support verification result

Underlying resume support is available through the patched `piper_train` CLI and `tools/piper_main_patch.py`, which accepts `--resume_from_checkpoint <path>`. `recipes/issai_piper/train.sh` is wired to pass an explicit base checkpoint path when provided as its fourth argument, while keeping `latest` as the default for existing runs.

## Start decision gate

Do not start v0.4 unless all of the following are true:
- base checkpoint exists and hash is recorded
- selected manifest exists
- prep script prints a train command that includes the base checkpoint path
- early log output confirms checkpoint restore/resume instead of scratch start


## Interim Step556k Result

Decision: V0_4_POSITIVE_SIGNAL_CONTINUE_MONITORING

The first v0.4 step556k benchmark sample set produced a positive listening signal. Human listening found s4 and s5 very good, and the remaining benchmark samples improving. This is a materially better direction than v0.3 relaxed scratch training.

The v0.4 run should continue under close monitoring, but v0.1 step500k remains the primary release candidate until v0.4 proves consistent quality across later checkpoints.
