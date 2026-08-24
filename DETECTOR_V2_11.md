# Detector / AI V2.11 — physical-feature listwise rank optimizer

V2.11 does **not** change Detector V2, Shot Vault, V2.7 micro-clustering or the
V2.8 recall pool. Those stages remain authoritative and unchanged.

The goal of V2.11 is narrower:

> When a correct hypothesis already exists in the V2.8 recall pool, rank it
> using physical/signal evidence rather than artifacts of our own pool policy.

## Why V2.10 was rejected

V2.10 selected features such as:

- `reason_core`
- `reason_keep_all`
- `core_member`
- `reason_support`

Those are bookkeeping/policy features describing *why our own algorithm kept a
candidate*. They are not properties of a bullet hole.

V2.11 forbids all of those features from model input.

## Physical feature set only

V2.11 may use signal/shape/history/source evidence such as:

- support / signal
- member count / hit history
- compactness / spread
- V1/V2/tile/source agreement
- patch prior
- z-score / absdiff / DoG / saliency
- persistence / age / current-vs-carried evidence

`baseline_score`, `core_member`, `reason_*` and all `rel_*` policy/duplicate
features are excluded.

## Broad negatives

V2.10 mostly compared GT with baseline hard negatives.

V2.11 samples false candidates from the whole pool:

- baseline hard negatives,
- high and low extremes of every physical feature,
- candidates distributed through baseline rank,
- spatially diverse candidates,
- deterministic spread through the remaining negative population.

Feature direction therefore means approximately:

> P(the correct candidate beats a broad false candidate using this feature)

rather than:

> P(the correct candidate beats the particular false candidate baseline likes).

## Listwise search

V2.11 first tests every physical feature alone using shot-level held-out
cross-validation.

It then searches family-diverse combinations of two and three features.

A rule is forbidden from winning if:

`Top-1 <=20px < baseline Top-1 <=20px`

Median rank cannot rescue a rule that selects fewer correct hits.

## Untouched confirmation set

Approximately 80% of the saved shots are development data.

Approximately 20% are never used to select features, feature directions, rule
weights or the winning configuration. The winning development rule is evaluated
once against this untouched confirmation set.

## V9 shadow model

If the winner passes both:

1. the hard development Top-1 gate, and
2. the confirmation sanity gate,

V2.11 writes:

`content/ai/ranker_v9_offline.json`

This model is still **shadow-only**.

It never changes the game's actual hit selection.

If gates are not passed, only:

`content/ai/ranker_v9_candidate.json`

is written and no V9 camera holdout should be run.

## Recommended workflow

Existing V2.9 dataset:

```bash
python3 -m automation.ranker_v211_selftest
python3 -m automation.ranker_v211_optimize
```

No game, camera or projector is required.

Only if the optimizer prints a `V9 shadow-ready` model:

```bash
python3 main.py
python3 -m automation.ranker_v211_verify
python3 -m automation.ai_training_loop 1 1 --seed 54321
python3 -m automation.ranker_v211_shadow_analyze
```

Do not run 10x100 until a 1x100 unseen-seed shadow test beats or at least
credibly matches the current authoritative ranking.
