# Detector V2.4 — local temporal sampling, shot accumulation and patch-shape ranking

## Status

V2.4 is an **additive experimental layer** on top of the Detector V2/V2.3 code
already present in the repository.

It deliberately does **not** replace:

- `src/engine/camera/candidate_generator_v2.py`
- `src/engine/ai/runtime.py`
- `content/menu.json`
- any game/menu scene

The tested V2/V2.3 core remains the fallback/base implementation. V2.4 patches
it at runtime and is fail-open: if an extension cannot load, the older detector
or ranker remains usable.

Expected startup lines:

```text
[DETECTOR-V2.4] local-tile + shot-accumulator extension installed
[DETECTOR-V2] Hybrid candidate generator installed
[RANKER-V4] Patch-descriptor pairwise ranker installed
```

---

## Why V2.4 exists

The deterministic V2.3 benchmark (`12345..12354`) showed approximately:

```text
BEST/EVER merged <=42 px       64.96 %
Actual F2 raw <=42 px          45.59 %
Actual F2 filtered <=42 px     43.44 %
Actual F2 selected <=42 px      1.13 %
GT median rank                    78
GT Top-3                        2.83 %
```

The loss classes also showed three independent bottlenecks:

```text
detector never covered GT              ~342
candidate disappeared before F2 eval   ~337
selected wrong candidate               ~270
```

And roughly 243 labelled shots had strong temporal signal around ground truth
but no useful peak.

V2.4 therefore changes *representation and temporal handling*, not merely one
threshold.

---

## V2.4 architecture

```text
existing V1/V2 candidate generator
              |
              +--------------------+
                                   |
local temporal tile probes         |
(absdiff + z-score per small tile) |
              |                    |
              +------ merge -------+
                        |
               local patch descriptors
                        |
               shot-level accumulator
                        |
           current + carried candidates
                        |
                 existing base ranker
                        |
              Ranker V4 patch model
                        |
                    selected
```

### 1. Local temporal tile probes

The full projector image is divided into small tiles. Each tile contributes a
few of its strongest **local** temporal changes.

A 4-gray-level hole therefore does not have to beat the strongest artifact
somewhere else in the image.

Default V2.4 values use 10 x 8 tiles, up to 3 local probes per tile and a
bounded high-recall reserve.

### 2. Shot-level accumulator

Candidates are clustered spatially across the whole shot window.

The accumulator tracks:

- number of separate observations
- first/last observation time
- positional stability
- V1/V2/tile provenance
- best local patch quality

Repeated candidates can survive disappearance from the final camera frame.
A single very hole-like temporal patch may also be carried for a very short
window. Carried candidates have a strict bounded budget and never replace the
entire current candidate set.

### 3. Local patch descriptor

V2.3 demonstrated that scalar fields such as detector score, area and
persistence do not tell the ranker enough about *what the local image looks
like*.

V2.4 describes a small temporal patch around every candidate using:

- core/ring/outer absdiff
- core/ring z-score
- dark core / bright ring response
- local SNR
- centre-to-outer contrast
- compactness
- centredness
- isotropy / line-likeness
- bipolar response
- ringness
- centroid offset

It also stores three normalized 5 x 5 grids:

- absolute temporal change
- signed dark-vs-bright change
- z-score

This gives the small linear ranker an actual coarse *picture* of the local
change without adding a neural-network dependency.

### 4. Exact-ground-truth training for Ranker V4

This is an important change from V3.

The 42 px radius is useful as a lenient evaluation tolerance, but it is too
loose to label local image appearance. A candidate 35-40 px from GT can be a
nearby projection artifact.

After a synthetic shot has already been ranked/evaluated, V2.4 samples the
patch descriptor **exactly at camera-space ground truth** and uses it only as a
training positive for the *next* shots.

```text
rank current shot WITHOUT GT
        |
        v
measure result
        |
        v
sample exact GT patch for training only
        |
        v
learn: true patch > hard wrong candidates
```

There is therefore no ground-truth leakage into the score being measured for
the current round.

If an exact GT patch is unavailable/too weak, V4 can fall back to a candidate
within a much tighter 16 px training radius rather than 42 px.

### 5. Ranker V4 focuses on patch shape

The previous/base ranker is still present and contributes a bounded score.
The learned V4 component focuses mainly on temporal patch appearance instead
of relearning detector-score/blob-size biases.

Pairwise training uses hard negatives from the same shot:

```text
exact GT patch > high-ranked wrong candidate #1
exact GT patch > high-ranked wrong candidate #2
...
```

Model state:

```text
content/ai/ranker_v4.json
```

Training log:

```text
content/ai/ranker_v4/training_pairs.jsonl
```

Ranker configuration:

```text
content/ai/ranker_v4_config.json
```

The model weight ramps in gradually; a fresh untrained model is not allowed to
immediately dominate the existing ranker.

The existing AI-memory reset action also resets Ranker V4. It does **not**
remove historical benchmark results.

---

## Deterministic benchmark

Use the same seed series when comparing versions:

```bash
python3 -m automation.ai_training_loop 1 10 --seed 12345
```

Ten runs currently mean 1000 synthetic shots. Run 1 uses seed 12345, run 2
12346, etc.

The benchmark control is cleared automatically after the loop, so later manual
F2 training returns to ordinary random behaviour.

Then analyse:

```bash
python3 -m automation.detector_v2_analyze
```

V2.4 reports old and new paths separately:

- legacy V1
- V2 frame
- V2 bank
- merged legacy/V2
- V2.4 tile probes
- V2.4 accumulator
- V2.4 final pool
- actual F2 raw/filter/ranked/selected
- Ranker V4 full-pool GT rank
- selected-minus-GT patch-feature medians
- benchmark integrity

### Benchmark integrity

Fast scene changes previously left a small number of labelled detector records
buffered (for example 976/1000).

At the end of a completed automated F2 report, V2.4 now force-finalizes all
remaining labelled diagnostics. A missing real evaluation is explicitly marked
rather than silently discarded.

The analyser reports `COMPLETE` only when the expected diagnostic count exists
and none of those records needed a missing-evaluation placeholder.

---

## Machine-readable files

Detector diagnostics remain:

```text
content/ai/detector_v2/shot_diagnostics.jsonl
content/ai/detector_v2/latest_summary.json
```

Automation loop results remain:

```text
content/ai/automation_runs/<session>/
```

V2.4 does not replace those formats; it adds fields to the detector diagnostic
records.

---

## Configuration

High-recall detector extension:

```text
content/ai/detector_v24.json
```

Patch ranker:

```text
content/ai/ranker_v4_config.json
```

The first tuning target should be **candidate recall**, not candidate count.
False candidates are acceptable at this layer if the true hole reaches the
ranker consistently.

---

## Self-test

The package includes a dependency-light self-test:

```bash
python3 automation/detector_v24_selftest.py
```

It verifies:

- compact hole descriptor vs elongated edge descriptor
- one-frame shot-accumulator carry
- weak local tile-hole recovery
- Ranker V4 pairwise learning
- exact-GT patch supervision when no detector candidate is near GT

These are synthetic code-level checks, not a replacement for the physical
projector/webcam benchmark.

---

## Recommended first physical test

1. Copy the package onto `dev`.
2. Restart the game.
3. Confirm the V2.4 and Ranker V4 startup messages.
4. Run:

```bash
python3 -m automation.ai_training_loop 1 10 --seed 12345
```

5. Run:

```bash
python3 -m automation.detector_v2_analyze
```

Compare to the V2.3 deterministic baseline above.

The highest-value numbers are:

```text
V2.4 final BEST/EVER <=42 px
Actual F2 raw <=42 px
candidate_disappeared_before_evaluation
Ranker V4 GT median rank
Ranker V4 Top-1 / Top-3
Actual selected <=42 px
Benchmark integrity
```

---

## Rollback / safety

V2.4 is additive.

To disable only the new detector extension:

```json
// content/ai/detector_v24.json
"enabled": false
```

To disable only Ranker V4:

```json
// content/ai/ranker_v4_config.json
"enabled": false
```

The older Detector V2/V2.3 core remains in the repository and continues to be
the fallback.

No menu JSON, game definition or ordinary game scene is changed by this update.
