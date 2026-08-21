# Detector V2 — hybrid high-recall candidate generator

## Status

Detector V2 is experimental but deliberately fail-safe.

It runs **in parallel with the existing HitScanner candidate generator**.
The legacy detector is still executed first on every detection frame. V2 adds
extra high-recall candidates and the two lists are merged before the existing
tracking and AI ranking stages.

If V2 raises an exception during detection, that frame automatically falls back
to the legacy candidate list.

If V2 cannot be imported/initialised at application startup, the camera package
keeps the legacy detector and prints a warning.

To disable V2 without reverting code:

```json
{
  "enabled": false
}
```

in:

`content/ai/detector_v2.json`

The config is hot-reloaded from disk (approximately once per second).

---

## Files in this update

```text
src/engine/camera/__init__.py
src/engine/camera/candidate_generator_v2.py
content/ai/detector_v2.json
automation/detector_v2_analyze.py
DETECTOR_V2.md
```

No replacement `hit_scanner.py` is included. V2 wraps the current HitScanner
at package initialisation time so the existing detector remains the baseline.

---

## Why V2 exists

The legacy detector has a long hard-filter pipeline:

```text
camera frame
 -> 5x5 Gaussian blur
 -> temporal / blackhat / whitehat signals
 -> global high-percentile thresholds
 -> binary mask
 -> contours
 -> geometry filters
 -> patch verification
 -> candidate list
```

For a hole only a few camera pixels wide, every hard gate can permanently
remove the true hit before the AI ever sees it.

V2 is designed around the opposite priority:

> Candidate generation should prefer recall. Ranking/filtering can remove
> false candidates later.

---

## V2 pipeline

```text
existing HitScanner V1 candidates
              |
              +-------------------------------+
                                              |
camera frame                                  |
    |                                         |
recent frames immediately before audio peak  |
    |                                         |
robust pre-shot reference + noise estimate   |
    |                                         |
small global camera registration              |
    |                                         |
photometric normalisation                     |
    |                                         |
absdiff + darkening + per-pixel z-score       |
    |                                         |
multi-scale local contrast                    |
    |                                         |
soft artifact-mask penalty                    |
    |                                         |
soft pre-existing edge penalty                |
    |                                         |
multi-frame persistence                       |
    |                                         |
robust local saliency threshold               |
    |                                         |
local maxima + spatial quotas + NMS           |
    |                                         |
V2 candidates --------------------------------+
              |
       V1 / V2 hybrid merge
              |
       existing HitScanner tracks
              |
       existing AI runtime/ranking
              |
             hit
```

### Main changes

- Keeps V1 as a baseline/fallback.
- Uses the camera frames immediately before the audio event.
- Uses up to three pre-shot frames for a robust reference/noise estimate.
- Uses a mild 3x3 blur instead of requiring the V2 signal to survive a 5x5 blur.
- Corrects small global camera translations before temporal differencing.
- Measures and removes OpenCV phase-correlation self-bias once per shot.
- Ignores tiny registration shifts below a configurable deadband.
- Uses per-pixel temporal noise to form a z/SNR-like signal.
- Uses multiple local-contrast scales.
- Does not require a true hole to survive one global binary percentile cutoff.
- Finds local saliency maxima.
- Reserves candidates across image tiles so one noisy region cannot consume the
  entire candidate budget.
- Treats the projector artifact mask as a soft prior rather than an absolute
  deletion rule.
- Penalises pre-existing scene edges softly.
- Accumulates persistent evidence across post-shot frames.
- The first observation only seeds persistence; repeated evidence is required
  before persistence gives a score bonus.

---

## Registration notes

`cv2.phaseCorrelate(reference, current)` can report a small non-zero,
shape-dependent offset even for an identical image.

V2 therefore calculates:

```text
registration bias = phaseCorrelate(reference, reference)
measured shift    = phaseCorrelate(reference, current)
corrected shift   = measured shift - registration bias
```

The current frame is warped back into the calibrated pre-shot camera coordinate
system only when:

- response >= `registration_min_response`
- corrected movement >= `registration_min_shift_px`
- corrected movement <= `registration_max_shift_px`

This is important because a bad registration can create artificial
edge-differences that look stronger than a tiny shot hole.

---

## Configuration

File:

`content/ai/detector_v2.json`

Important first-line controls:

```json
{
  "enabled": true,
  "hybrid_with_legacy": true
}
```

### Registration

```json
{
  "registration_enabled": true,
  "registration_max_shift_px": 4.0,
  "registration_min_shift_px": 0.35,
  "registration_min_response": 0.08
}
```

### Candidate sensitivity

The most important tuning values are initially:

```json
{
  "robust_sigma": 3.2,
  "min_saliency": 10.0,
  "min_temporal_change": 1.8,
  "min_zscore": 1.5,
  "strong_temporal_change": 4.0
}
```

Do not blindly lower every value at once. Use the diagnostics described below
to determine whether a missed ground-truth point had:

- weak camera signal,
- sufficient signal but insufficient saliency,
- sufficient saliency but no local-max candidate,
- a candidate that was generated but later lost.

---

## Machine-readable diagnostics

When `diagnostics_enabled` is true, labelled synthetic training shots are
written to:

```text
content/ai/detector_v2/shot_diagnostics.jsonl
```

Each resolved synthetic shot can include:

```text
runtime_session_id
git_commit
shot_id

ground_truth:
    screen_x / screen_y
    camera_x / camera_y
    background
    synthetic hole kind
    radius / strength / opacity

nearest_candidate_distance_px:
    legacy
    v2
    merged

gt_signal_max:
    absdiff
    zscore
    saliency

registration:
    applied frame count
    best response
    max dx / dy

max candidate counts:
    legacy
    v2
    merged

resolved scanner state
detector config snapshot
```

The JSONL file is append-only.

Each program run receives a new `runtime_session_id`, which prevents an
analysis of the newest detector version from silently mixing results from an
older run.

---

## Analyse the latest run

From the repository root:

```bash
python3 -m automation.detector_v2_analyze
```

It writes:

```text
content/ai/detector_v2/latest_summary.json
```

and prints candidate recall for:

```text
10 px
20 px
42 px
```

for:

```text
legacy
v2
merged
```

It also classifies misses/recoveries, for example:

```text
found_by_both
recovered_by_v2
weak_or_no_camera_signal
strong_gt_signal_but_peak_missing
saliency_suppressed
candidate_generation_miss
```

Analyse all historical Detector V2 sessions:

```bash
python3 -m automation.detector_v2_analyze --all
```

Analyse one specific session:

```bash
python3 -m automation.detector_v2_analyze --session SESSION_ID
```

---

## Recommended first benchmark

Do not immediately train the AI for another huge series.

First measure candidate recall with the hybrid detector.

Example:

```bash
python3 -m automation.ai_training_loop 1 10
```

Then:

```bash
python3 -m automation.detector_v2_analyze
```

The key comparison is initially **legacy vs V2 vs merged candidate recall**,
not AI Top-1.

If merged recall rises substantially while AI Top-1 remains low, Detector V2
has done its job and the next bottleneck is ranking/filtering.

If V2 recall itself is still low, inspect the ground-truth signal fields before
changing the AI memory/model.

---

## Rollback

Fast runtime rollback:

1. Open `content/ai/detector_v2.json`.
2. Set:

```json
"enabled": false
```

No game restart should normally be necessary for config reload.

Full code rollback:

- Restore the previous `src/engine/camera/__init__.py`.
- `candidate_generator_v2.py` can then remain on disk unused.

---

## Design invariant for future AI

Do not optimise V2 solely for a small candidate list.

For this stage the desired direction is:

```text
very high candidate recall
        ->
later filtering / ranking precision
```

A true hole that never enters the candidate set cannot be recovered by the AI
ranking layer.
