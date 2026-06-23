# v0.4 Interim Status Report

## Executive Verdict

V0_4_POSITIVE_SIGNAL_CONTINUE_MONITORING

## Current Best Baseline

v0.1 step500k remains the primary release candidate and benchmark to beat.

## Prior Experiments

### v0.2 Balanced

Closed as negative.

### v0.3 Relaxed Scratch

Decision: CLOSED_NEGATIVE_EARLY_PILOT

v0.3 relaxed filtering improved dataset-level target-row metrics, but scratch training produced fragmented, cut-up, sometimes unintelligible synthesis. v0.1 step500k remained far better.

## v0.4 Strategy

v0.4 is a fine-tune / continuation experiment from v0.1 step500k, not scratch training.

Base checkpoint:
runs/v0_1_full_001/candidates/step500k/step500k.ckpt

Base checkpoint SHA256:
2683f1ee0ed6c64acf6e6693b68355803373cb2b7b7a481a28f6060de805de19

Run:
runs/v0_4_step500k_finetune_turkish_heavy_001

Manifest:
data/manifests/train_v0_4_finetune_turkish_heavy_conservative.jsonl

## v0.4 Configuration Summary

- medium VITS
- batch size 16
- precision 32
- GPU, 1 device
- validation split 0.0
- no test examples during training
- shared cache from v0.1:
  runs/v0_1_full_001/training/cache/22050
- checkpoint cadence:
  every 30000 steps
- last.ckpt enabled
- resume_from_checkpoint:
  v0.1 step500k
- trusted checkpoint loader patch active for PyTorch 2.6 weights_only behavior

## Data Strategy

The conservative v0.4 fine-tune manifest keeps broad clean base data, excludes known bad target rows, and gives controlled extra exposure to good Turkish-heavy target rows.

It is not target-only training.

## Step556k Evaluation

Sample set:
samples/18_v0_4_finetune_step556k

Human listening:
- s4 and s5 are very good.
- s1, s2, and s3 are improving.
- v0.4 is much better than the v0.3 scratch pilot.
- v0.4 shows the first positive continuation/fine-tune signal.

Technical interpretation:
- v0.4 confirms that step500k continuation is the correct direction.
- v0.3 scratch training should not be repeated as the next improvement path.
- step556k should be preserved locally as the first positive v0.4 candidate signal.

## Latest v0.4 Benchmark Listening Update

Decision: V0_4_MIXED_POSITIVE_CONTINUE_MONITORING

Human listening summary:
- `s1_bugun_hava.wav`: slightly worse; mild regression risk.
- `s2_turkiye_cumh.wav`: improved and cleaner.
- `s3_cocuklar.wav`: still the main unresolved weakness; mild stutter / Turkish-heavy difficulty remains.
- `s4_ogrenciler.wav`: good, though with some foreign-accent character.
- `s5_sirket.wav`: improved and moving in the right direction.

Technical interpretation:
- `s2` is technically clean, with near-zero internal gap signal.
- `s4` is technically clean, with near-zero internal gap signal.
- `s3` still has the clearest problem signal, with a larger internal gap / stutter pattern.
- `s1` shows some regression risk.
- `s5` is promising but should continue to be monitored for pauses/gaps.
- v0.4 remains a better direction than v0.3 scratch training.
- The fine-tune is not uniformly improving every benchmark sentence.
- v0.1 step500k remains the primary release candidate until v0.4 proves consistent quality over later checkpoints.

Current action:
- Continue monitoring v0.4.
- Preserve this benchmark set locally.
- Compare the next checkpoint against both v0.1 step500k and the first positive v0.4 step556k set.

## v0.4 Checkpoint Comparison Update

Decision: V0_4_CURRENT_BEST_CANDIDATE_PRESERVE

Current best v0.4 candidate so far:

`samples/20_v0_4_finetune_epoch53_step1230000`

Compared sample sets:
- `18_v0_4_finetune_step556k` - first positive v0.4 continuation signal.
- `19_v0_4_finetune_step574k` - mixed result with regression risk.
- `20_v0_4_finetune_epoch53_step1230000` - current best v0.4 candidate so far.
- `21_v0_4_finetune_epoch54_step1260000` - later regression signal, especially on `s1`.

Human listening summary for 20:
- `s1_bugun_hava.wav`: still needs monitoring, but better than the later 21 regression.
- `s2_turkiye_cumh.wav`: improved / good.
- `s3_cocuklar.wav`: still the main unresolved weakness; mild stutter / Turkish-heavy difficulty remains.
- `s4_ogrenciler.wav`: good, with a slight foreign-accent character.
- `s5_sirket.wav`: improved and good.

Interpretation:
- v0.4 remains a better direction than v0.3 scratch training.
- `20_v0_4_finetune_epoch53_step1230000` should be preserved as the current best v0.4 candidate.
- Later fine-tuning may introduce regression or over-fine-tuning; 21 does not clearly beat 20.
- v0.1 step500k remains the primary release candidate until v0.4 is validated more broadly.

Current action:
- Preserve 20 locally.
- Continue using v0.1 step500k as the benchmark to beat.
- Compare later checkpoints against both v0.1 step500k and v0.4 checkpoint 20.

## Current Decision

Continue monitoring v0.4.

Do not replace v0.1 step500k as the primary release candidate yet.

Preserve v0.4 step556k as a positive candidate for comparison.

## Next Milestones

- next checkpoint/sample around the next 30000-step cadence
- compare against:
  - v0.1 step500k
  - v0.4 step556k
- watch for:
  - regression in s1/s2/s4/s5
  - improvement or regression in s3
  - over-fine-tuning
  - fragmented/cut-up audio

## Stop / Preserve Criteria

Preserve a v0.4 checkpoint if:
- s4/s5 remain strong
- s3 improves or at least does not regress
- general intelligibility stays close to or better than v0.1 step500k

Stop or roll back if:
- fragmented/cut-up synthesis appears
- s1/s2/s4/s5 regress
- s3 does not improve and general quality declines

## v0.4 Closure Decision

Decision: V0_4_PLATEAU_REGRESSION_STOP

v0.4 fine-tuning has been stopped because progress plateaued and later checkpoints did not clearly beat the current best candidate.

Current best v0.4 candidate:

`samples/20_v0_4_finetune_epoch53_step1230000`

Summary:
- `18_v0_4_finetune_step556k` was the first positive continuation signal.
- `19_v0_4_finetune_step574k` was mixed with regression risk.
- `20_v0_4_finetune_epoch53_step1230000` is the current best v0.4 candidate.
- Later checkpoints, including 21 and beyond, did not clearly improve over 20 and showed regression/trade-off signals.

Interpretation:
- v0.4 confirmed that fine-tuning from v0.1 step500k is a better direction than v0.3 scratch training.
- Further continuation of this run is not justified because it appears to trade quality across benchmark sentences rather than improve globally.
- v0.1 step500k remains the primary release candidate.
- v0.4 candidate 20 should be preserved locally for broader validation.
