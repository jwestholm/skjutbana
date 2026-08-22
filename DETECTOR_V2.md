# Detector V2.2 — hybrid candidate recall + ranking repair

## Why V2.2 exists

The V2.1 benchmark gave a much clearer picture than a single Found percentage:

- merged BEST/EVER recall at 42 px was around two thirds of synthetic shots;
- the exact F2 raw snapshot still contained ground truth far more often than the
  final selected candidate was correct;
- filtering removed relatively few true candidates compared with ranking;
- hundreds of shots had a useful candidate earlier in the shot window but not
  at the exact evaluation snapshot;
- many misses had strong temporal signal at ground truth but no emitted peak.

V2.2 therefore changes both sides of the remaining problem:

1. candidate generation/preservation;
2. ranking/selection.

The existing legacy detector remains active and V2 stays hybrid/fail-open.

---

## Files

```text
src/engine/camera/__init__.py
src/engine/camera/candidate_generator_v2.py
src/engine/ai/runtime.py
content/ai/detector_v2.json
automation/detector_v2_analyze.py
DETECTOR_V2.md
INSTALL_DETECTOR_V2.txt
```

`hit_scanner.py`, menus, games and other scenes are not replaced by this
package.

---

## Candidate changes

### 1. Hybrid bank instead of V2-only bank

V2.1 preserved only V2 candidates. V2.2 first creates the current hybrid
V1+V2 snapshot and then banks that snapshot.

Important invariant:

> Current-frame candidates are never deleted to make room for old bank data.

Historical candidates only fill unused slots.

A strict V1 candidate may be carried for a short interval after one observation.
V2 primary/rescue candidates still require repeated evidence before carry.

This directly targets `candidate_disappeared_before_evaluation`.

### 2. Merge provenance bug fixed

Older V2 merge logic searched the complete, growing merged list when matching a
new V2 point. A V2 point could therefore match an earlier V2 point and be
incorrectly labelled as V1+V2 agreement.

V2.2 only tests V2 candidates against the original legacy candidate segment.
Detector agreement now actually means agreement between independent detector
paths.

### 3. Legacy-biased source reservation

V1 had higher measured recall than V2.1 on the latest benchmark, so dense
candidate snapshots now reserve more capacity for V1:

```text
legacy reserved: 140
V2 reserved:      60
```

When the union is below the scanner candidate limit, nothing is removed.

### 4. Component-centre temporal rescue

A tiny synthetic hole can form a ring or plateau where no single pixel is a
stable local maximum. V2.2 adds a connected-component rescue path directly on
strong temporal evidence.

For compact components it emits a weighted centre instead of requiring one
pixel to win local-max competition.

This specifically targets the measured `strong_gt_signal_but_peak_missing`
class.

### 5. More permissive high-recall peak search

V2.2 uses:

```text
primary local-max kernel: 3
NMS radius:               3.5 px
per-tile candidates:      6
global extras:            80
max V2 candidates:        200
```

Candidate generation is intentionally recall-oriented. Precision belongs in the
later filter/ranker stages.

---

## Ranking V2.2

The old ranker could give 65% of the score to the nearest-memory AI when memory
was full. In the measured run, a correct candidate survived filtering much more
often than it was selected, so the memory model was behaving as a harmful
ranking signal.

### Position no longer affects nearest-memory distance

Synthetic shot position is random. `x_norm` and `y_norm` are still stored for
compatibility/diagnostics, but have zero weight in the AI-memory distance.

The learner should learn what a hole looks like, not where old holes happened to
appear.

### Feature normalization is clipped

Candidate generators can change feature ranges. Values outside historical
min/max no longer create arbitrarily large nearest-neighbour distances.
Normalized values are clipped to 0..1.

### Relative within-shot ranking

Absolute V1 and V2 detector score scales are not assumed to be identical.
V2.2 calculates relative ranks inside the current candidate snapshot for:

- detector strength;
- temporal/new-hole evidence;
- V2 temporal evidence when present.

It also adds small independent evidence for:

- genuine V1+V2 detector agreement;
- confirmed candidate-bank persistence.

### AI memory is capped

Default full-memory AI weight is now capped at 30% in Ranking V2.2. Detector and
temporal evidence dominate until the learned model proves itself useful.

The legacy ranker is still available. To disable Ranking V2.2, add/change this
in `content/ai/settings.json`:

```json
"ranking_v22_enabled": false
```

Do not replace the whole settings file just for this switch.

---

## Diagnostics V2.2

`automation.detector_v2_analyze` still reports BEST/EVER detector recall and the
actual F2 funnel, and now additionally reports:

- legacy candidates lost during merge;
- V2 candidates lost during merge;
- hybrid-bank carried candidate counts;
- whether carried bank evidence itself covered ground truth;
- rank of the nearest ground-truth candidate;
- selected-vs-ground-truth combined score margin;
- selected-vs-ground-truth AI-score difference;
- selected-vs-ground-truth heuristic-score difference.

Run:

```bash
python3 -m automation.detector_v2_analyze
```

The key V2.2 questions are:

```text
Did merged BEST/EVER recall rise?
Did ACTUAL F2 raw recall rise?
Did candidate_disappeared_before_evaluation fall?
Did legacy_lost_in_merge fall?
Did selected/top-1 improve strongly relative to filtered/ranked recall?
What is the GT median rank when GT survives ranking?
```

---

## Recommended test

Restart the game after installing the files so the new runtime and detector
wrapper are loaded.

First run a short 10 x 100 benchmark:

```bash
python3 -m automation.ai_training_loop 1 10
```

Then:

```bash
python3 -m automation.detector_v2_analyze
```

Do not start with another 100 x 100 run. Ten complete runs are enough to see
whether V2.2 moved the main bottlenecks before spending more time.

---

## Rollback / isolation

Disable Detector V2 while keeping the code installed:

```json
"enabled": false
```

in `content/ai/detector_v2.json`.

Disable only the new ranker:

```json
"ranking_v22_enabled": false
```

in `content/ai/settings.json`.

These two switches make it possible to test:

```text
V1 detector + legacy ranker
V1/V2 hybrid + legacy ranker
V1/V2 hybrid + Ranking V2.2
```

without reverting the entire branch.
