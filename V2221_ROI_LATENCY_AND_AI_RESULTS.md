# V2.22.1 — ROI, latency and AI Results design note

## Problem observed in first V2.22 live test

The resolver decision itself completed in single-digit milliseconds, but end-to-end shot latency was measured in seconds. At the same time the live detector produced hundreds of raw blobs, including a visible ridge/cluster around a projected playfield edge.

The current detector already knows the calibrated playfield polygon, but its expensive blur/morphology/difference operations are performed on the full camera frame and the ROI mask is applied later. V2.22.1 moves the expensive analysis to the playfield crop.

## Why the crop is perspective safe

The system already has the homography between camera and absolute projector/screen coordinates. V2.22.1 uses the existing inverse homography only to calculate two camera-space quadrilaterals:

1. **outer playfield** — the existing active content rectangle transformed into camera coordinates;
2. **safe playfield** — the same screen rectangle first shrunk by a small screen-space edge guard, then transformed into camera coordinates.

The bounding box of the outer quadrilateral becomes the image crop. The safe quadrilateral becomes the binary search mask inside that crop.

No candidate is converted to game coordinates during the expensive detector stage.

## Coordinate planes

There are three distinct planes and V2.22.1 keeps them explicit:

- **crop-local camera XY** — temporary, used only by OpenCV inside the cropped image;
- **full camera XY** — canonical detector/AI/tracking coordinate plane;
- **screen/game/content XY** — calculated by the existing `HitInput` only after a final hit is emitted.

Every accepted local candidate receives the crop offset before it leaves `_detect_frame_candidates`. Therefore tracking, known-hole logic, SimpleAI, V2.22 ShotResolver and the final HitInput transform continue to see the same full-camera XY as before.

## Edge artifacts

A small movement of the board/projected image can produce a strong long difference ridge at a playfield boundary. Such ridges are not useful shot candidates and can generate many contours/local maxima.

V2.22.1 removes the narrow border **before thresholded contours are generated**. The default guard is 12 screen pixels. This is deliberately conservative and configurable.

## Latency diagnostics

V2.22.1 adds:

- geometry calculation time;
- cropped detector time;
- detector wrapper time;
- update total time;
- crop fraction and safe-pixel count;
- correct per-event HIT/MISS shot IDs;
- detector audio-peak → decision `e2e` time.

A first-frame line is printed for each shot when `analysis_v2221_log=true`:

```
[V2.22.1 ROI] shot=12 mode=homography crop=74.1% guard=12.0px raw=123 kept=87 geometry=0.31ms detector=186.4ms
```

The ordinary HIT line now uses `event.shot_id`, for example:

```
[SHOT #12] HIT: (...) ... e2e=284.7ms
```

This makes it possible to compare detector latency with the V2.22 resolver line for the same shot.

## AI Results crash

The V2.22 bootstrap delta accidentally retained scene-factory cases for `ai_settings` and `ai_training` but omitted the already-existing `ai_results` case. The underlying `AIResultsScene` was not removed; the factory mapping was.

V2.22.1 restores that mapping. No result-file schema or graph calculation is changed by this fix. The first verification is therefore simply that the AI Results scene opens and renders without the `Unknown item type: ai_results` exception.

If the scene opens but a graph later fails on a new result schema, that is a separate compatibility issue and should be fixed from the resulting traceback rather than guessed here.

## What V2.22.1 intentionally does not do

- It does not give V2.21.5 physical-dense authority live.
- It does not change final game-coordinate math.
- It does not change candidate scoring thresholds.
- It does not change the V2.22 resolver weights.
- It does not change F2 training semantics.

The goal of this delta is to make the existing live perception path smaller, cleaner and measurable before the physical expert is added in parallel.
