# Detector / Ranker V2.10 - offline monotonic feature search

V2.10 freezes the camera/Vault/clustering/recall-pool pipeline and moves ranking
research offline.

## Why a monotonic rank ensemble?

The first real V2.9 dataset showed a repeated physical pattern. Compared with
the baseline's strongest wrong candidate, the nearest GT hypothesis was very
often LOWER in support, signal, z-score, source diversity and member count,
while compactness was HIGHER. A general pairwise linear model did not exploit
this reliably from only 100 shots.

V8 therefore makes the learned direction explicit:

- a feature can vote `LOW is hole-like`;
- a feature can vote `HIGH is hole-like`;
- values are converted to within-shot percentiles;
- highly correlated features are not allowed to multiply the same evidence
  many times;
- only a small set of stable features is combined.

## Honest model selection

V2.10 uses two levels of protection against overfitting:

1. configuration search uses cross-validation on development shots only;
2. an untouched confirmation split is evaluated only after the winning
   configuration has been chosen.

The final JSON model may be fit on all captured shots for future shadow use,
but the quality recommendation is based on development CV plus confirmation.

## Runtime safety

Ranker V8 is shadow-only. `ranker_v8_extension.py` wraps the already functioning
V2.9/V2.8 pipeline after it has made its real decision. V8 can only write a
separate comparison record; it cannot change the returned candidate list.

## Primary commands

```bash
python3 -m automation.ranker_v210_selftest
python3 -m automation.ranker_v210_optimize
```

No projector, camera or running game is needed for those commands.

For a later unseen camera holdout:

```bash
python3 -m automation.ranker_v210_verify
python3 -m automation.ai_training_loop 1 1 --seed 54321
python3 -m automation.ranker_v210_shadow_analyze
```
