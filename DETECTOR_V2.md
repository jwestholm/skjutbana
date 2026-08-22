# Detector / Ranking V2.3

## Purpose

V2.3 is the next experimental camera/AI pipeline for Skjutbana. It is built from the measured V2.2 bottlenecks rather than from blind threshold tuning.

The V2.2 benchmark showed roughly:

- the correct candidate survived into the ranked list in almost half of synthetic shots,
- but the selected candidate was correct only a few percent of the time,
- the median ground-truth candidate rank was about 74,
- hundreds of candidates appeared during a shot and then disappeared before F2 evaluated it,
- many ground-truth locations had measurable camera signal even when no candidate was generated.

V2.3 therefore attacks three separate stages:

1. **candidate generation / strong-signal rescue**,
2. **candidate persistence until the exact F2 evaluation snapshot**,
3. **candidate ranking with a supervised pairwise learner trained directly from synthetic ground truth**.

The existing V1 detector, existing AI memory and normal game pipeline remain in place. V2.3 is additive and fail-open where possible.

---

## Files in this update

```text
src/engine/camera/__init__.py
src/engine/camera/candidate_generator_v2.py
src/engine/ai/runtime.py
src/engine/ai/pairwise_ranker.py
content/ai/detector_v2.json
automation/ai_training_loop.py
automation/detector_v2_analyze.py
DETECTOR_V2.md
INSTALL_DETECTOR_V2.txt
```

No `menu.json` is included. No game/menu scene is removed or replaced.

No learned `ranker_v3.json` is included. It is created by the running game so installing this ZIP cannot overwrite an already learned V3 model.

---

# 1. Pairwise Ranker V3

## Why

V2.2 diagnostics showed that the correct candidate often existed in the final ranked candidate list but was placed far down the ranking. The old heuristic frequently preferred the wrong candidate by a large margin, while old memory similarity contributed very little separation.

Synthetic F2 training gives something unusually valuable: **exact ground truth for every shot**.

V2.3 uses that ground truth directly.

For each labelled shot where a candidate exists within the accepted GT radius:

```text
positive = candidate nearest ground truth

hard negatives = strongest wrong candidates from the same shot
                 that are safely outside the GT area
```

The model learns pairwise statements:

```text
positive candidate > hard negative 1
positive candidate > hard negative 2
...
```

This is a small online logistic ranker implemented in ordinary Python. It does not require PyTorch, TensorFlow or scikit-learn.

## Model file

The learned ranker is stored separately in:

```text
content/ai/ranker_v3.json
```

It contains:

- fixed feature names,
- learned weights,
- positive-shot count,
- pair-update count,
- last training loss,
- update timestamp.

## Position independent

The V3 features deliberately do **not** learn X/Y position. Synthetic holes are randomly placed and the task is to learn what a new hole looks like, not where previous holes happened to occur.

Important feature groups include:

```text
detector strength
area / radius / circularity
center change
local contrast
pre-shot change
patch variation / edge strength
persistence
V1 / V2 provenance
true V1+V2 agreement
candidate-bank confirmation
V2 absdiff / z-score / local contrast
V2 primary/rescue source
peak refinement amount
```

## Conservative ramp-in

The V3 model does not immediately take over ranking.

Default behaviour:

```text
< 20 labelled positive shots  -> V3 ranking weight = 0
20..120 positive shots        -> weight gradually increases
>= 120 positive shots         -> max V3 weight = 0.72
```

The previous ranker remains part of the score while V3 is learning.

## Reset behaviour

The existing AI-memory reset remains authoritative.

When the existing UI/action calls:

```python
runtime.memory.reset()
```

V2.3 also resets `ranker_v3.json` through a reset callback.

This means the user's existing **R / reset AI memory** operation clears:

```text
content/ai/memory.json learned examples
+
content/ai/ranker_v3.json learned pairwise ranker
```

It does **not** clear benchmark/result history.

---

# 2. Candidate-bank V2.3

## Problem in V2.2

The V2.2 bank tried to preserve candidates that appeared earlier in a shot, but it only filled unused slots.

With the normal candidate limit at 200:

```text
current frame = 200 candidates
bank capacity = 200 - 200 = 0
```

The bank was therefore almost useless exactly on the noisy frames where it was needed most.

## V2.3 reserve

The current hybrid frame is still limited normally, but the bank output has a separate ceiling:

```text
current hybrid candidates: normally <= 200
candidate-bank output:     <= 240
carried reserve:            <= 40
```

A confirmed earlier candidate can therefore survive alongside a full current snapshot.

Current candidates are never deleted merely to make room for history.

## Current-candidate annotation

A candidate still visible in the current frame is now annotated with its matching bank history:

```text
candidate_bank_hits
candidate_bank_streak
candidate_bank_confirmed
```

This lets the ranker use temporal consistency even before a candidate has disappeared and been carried.

## Anti-noise rules

The bank still prevents dense one-frame noise from manufacturing persistence:

- one bank entry can match only once per camera frame,
- V2-only rescue candidates normally require repeated frames,
- stale unconfirmed hypotheses expire quickly,
- carried output is capped,
- current-frame evidence remains authoritative.

---

# 3. Strong-signal rescue

V2.2 still had a large `strong_gt_signal_but_peak_missing` class. The camera measured change at ground truth, but no candidate was emitted close enough.

V2.3 keeps the multi-path V2 detector and makes the rescue stage more recall-oriented:

```text
composite saliency local maxima
+
independent temporal-map peaks
+
connected-component weighted centres
```

The rescue candidate pool is broader, while spatial quotas and NMS prevent one image region from consuming every slot.

The long-term design rule remains:

> Candidate generation should optimise recall. Ranking should recover precision.

---

# 4. Deterministic synthetic benchmarks

Random synthetic holes made earlier V2 version comparisons noisy. V2.3 adds an optional deterministic benchmark seed.

Run:

```bash
python3 -m automation.ai_training_loop 1 10 --seed 12345
```

Run N receives:

```text
12345
12346
12347
...
```

Using the same base seed in another code version gives the same synthetic target/hole random stream for static backgrounds.

A manifest is saved with the automation result session:

```text
seed_manifest.json
```

The detector diagnostic ground truth also records `benchmark_seed`.

The temporary control file is:

```text
content/ai/benchmark_control.json
```

The loop disables deterministic mode in a `finally` block after it finishes so later manual F2 runs return to normal randomness.

### Important limitation

Static backgrounds such as `white` are the best choice for strict A/B testing. Animated/random backgrounds can consume random numbers as part of their animation state before a run, so deterministic hole sampling is the primary guarantee.

---

# 5. Benchmark integrity

Earlier 1000-shot runs sometimes produced only 985-994 V2 diagnostic records.

V2.3 registers a diagnostic skeleton as soon as the synthetic hole ground truth is revealed, before a useful V2 detector frame is required.

The desired integrity check is now:

```text
Synthetic shots:              1000
Diagnostics:                  1000
With synthetic ground truth:  1000
With evaluation funnel:       1000
```

If these numbers still differ, treat that as a separate telemetry/state bug rather than silently excluding missing shots.

---

# 6. Detector diagnostics

Per-shot machine-readable diagnostics remain in:

```text
content/ai/detector_v2/shot_diagnostics.jsonl
```

Analyse only the newest detector runtime session:

```bash
python3 -m automation.detector_v2_analyze
```

The analyser reports:

```text
BEST/EVER detector recall
    V1
    V2 frame
    V2 bank
    merged

ACTUAL F2 evaluation funnel
    raw
    filtered
    ranked
    selected

candidate-bank provenance
pipeline loss classification
ground-truth signal statistics
ranking quality
Ranker V3 score separation
Ranker V3 effective weight
Ranker V3 persisted model statistics
benchmark seeds
```

It writes:

```text
content/ai/detector_v2/latest_summary.json
```

---

# 7. How to test V2.3

## First learning test

Start the game, then run:

```bash
python3 -m automation.ai_training_loop 1 10 --seed 12345
```

Then:

```bash
python3 -m automation.detector_v2_analyze
```

The first 20 positive labelled shots deliberately have V3 weight 0. After that, the V3 contribution ramps up. Therefore a 10 x 100 series is much more informative than a single 100-shot run for the new ranker.

### Most important numbers

Compare with V2.2:

```text
ACTUAL F2 raw <=42px
ACTUAL F2 ranked <=42px
ACTUAL F2 selected <=42px
GT median rank
GT rank=1
GT rank<=3
selected-GT Ranker V3 score median
Ranker V3 weight median
candidate_disappeared_before_evaluation
strong_gt_signal_but_peak_missing
```

The ideal direction is:

```text
raw recall       up
ranked recall    up / stable
selected recall  strongly up
GT median rank   strongly down
candidate disappeared count down
```

## Fair detector A/B tests

For a strict detector-only comparison, the AI model should be frozen or reset to the same starting state before each version. The runtime already supports:

```json
"benchmark_mode": true
```

which makes F1/F2 evaluate without updating learned models.

Use the same `--seed` in both code versions.

For a **learning** comparison, leave benchmark mode false and start both compared tests from the same AI/ranker state.

---

# 8. Rollback / switches

## Disable Detector V2

Edit:

```text
content/ai/detector_v2.json
```

and set:

```json
"enabled": false
```

The detector config is hot-reloaded.

## Disable Pairwise Ranker V3

The V3 switch lives in normal AI settings:

```json
"ranking_v3_enabled": false
```

If that key does not exist in an older `content/ai/settings.json`, the runtime default is `true`.

No existing settings file is included in this ZIP, so installation does not overwrite user settings.

---

# 9. Important rules for future AI development

1. Do not delete or bypass V1 merely because V2 exists. Hybrid recall is still useful.
2. Do not use synthetic X/Y position as a learning feature.
3. Never train on the selected candidate as if selection itself proved correctness. Synthetic ground truth determines the positive.
4. Keep hard negatives outside the ambiguous GT radius.
5. Do not measure detector improvement from `selected` alone; inspect raw/merged detector recall first.
6. Do not claim an A/B improvement from different random shot sequences. Use `--seed`.
7. Do not overwrite `memory.json`, `ranker_v3.json`, result history or normal settings in code update ZIPs.
8. If diagnostics count differs from requested/completed synthetic shots, fix telemetry integrity before drawing precise percentages.
9. Candidate generation may be permissive. Precision is primarily the ranker's job once raw recall is high enough.
10. Any experimental failure should fail open to the existing detector/ranking path where possible.
