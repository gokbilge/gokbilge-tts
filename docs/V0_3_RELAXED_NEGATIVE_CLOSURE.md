# v0.3 Relaxed Negative Closure

## Purpose

Record the v0.3 relaxed Turkish-heavy pilot as a negative early pilot and preserve the decision path.

## Manifest Used

- data/manifests/train_v0_3_turkish_heavy_relaxed.jsonl

## Why This Trial Was Run

- Aggressive exclusion improved gap metrics but removed too many protected target rows.
- Hard exclusion preserved target rows but barely changed the gap profile.
- Relaxed exclusion was the best provisional balance, so it was advanced to a controlled pilot.

## Mode Summary

- aggressive: too destructive for protected Turkish-heavy terms
- hard: too weak to change the target-row gap profile meaningfully
- relaxed: best retention/quality tradeoff on paper, but not enough in synthesis

## Training Outcome

- Early pilot reached the first evaluation window around 50k.
- Evaluation samples were generated and reviewed by human listening.

## Listening Verdict

- Samples were fragmented, cut up, and sometimes unintelligible.
- v0.1 step500k remained clearly better.

## Decision

- CLOSED_NEGATIVE_EARLY_PILOT
- Do not continue v0.3 relaxed to 100k.
- Do not use v0.3 relaxed as a release candidate.

## Next Recommended Direction

1. Package and document v0.1 step500k as the current primary release candidate.
2. Investigate fine-tuning from v0.1 step500k instead of training from scratch.
3. Run a separate phoneme/G2P/text-normalization diagnostic track for Turkish-heavy words.
