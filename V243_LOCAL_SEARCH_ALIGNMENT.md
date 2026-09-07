# V2.24.3 — Local Search Alignment & Full-Pipeline Gate

## Why this checkpoint exists

The first physical run of `Hit Context Test (V2.24.2)` was useful precisely
because it did not merely confirm the happy path. Seven game regions were
successfully frozen and transformed at PANG, and the moving target was
classified correctly, but most region shots logged:

```text
[V2.24.1 GLOBAL-FALLBACK] ... reason=outside_detector_roi regions=7
```

There were also shots where no V2.24.1 line appeared at all. Those frames were
inside CandidateGenerator V2's short `waiting_post_peak` period, where the V2
extractor is not called and the already-generated legacy/V1 candidate list is
returned directly.

V2.24.3 fixes the architecture rather than adding another ranker or threshold.

## Root causes addressed

### 1. `content_rect` default violated its own coordinate contract

`content_rect` is viewport-local throughout HitInput/HitScanner. The old default
was effectively:

```python
return load_viewport_rect().copy()
```

If no explicit `content_rect` was saved, this copied absolute viewport x/y into
a rectangle later treated as viewport-local. HitScanner then added viewport x/y
again while transforming the ROI into camera space.

V2.24.3 changes the implicit default to:

```python
pygame.Rect(0, 0, viewport.w, viewport.h)
```

Explicitly saved content rectangles are untouched.

### 2. V2.24.1 gated too low in the detector stack

V2.24.1 wrapped only `CandidateGeneratorV2._extract_candidates()`. That is too
late to constrain:

- legacy/V1 candidates,
- `waiting_post_peak` early returns,
- any path that never reaches V2 peak extraction.

V2.24.3 wraps the live `HitScanner._frame_roi_mask()` instead. The same frozen
camera windows therefore constrain the physical ROI before both legacy and V2
candidate generation.

### 3. `outside_detector_roi` failed open too early

V2.24.1 previously turned a zero local intersection directly into an
unrestricted global extractor call. That allowed an unrelated old-hole/global
candidate to win before the explicit fallback had even been requested.

V2.24.3 changes this to **zero local proposals**. V2.22.5 then performs the one
existing, visible FULL-RESCUE on a later pass. This preserves the safety valve
without hiding the transition.

### 4. Calibration cache consistency

HitScanner reads saved calibration when it builds the physical ROI. The object
snapshot previously used HitInput's cached calibration values. V2.24.3 refreshes
HitInput calibration immediately before shot-time object transformation so both
paths use the same saved generation after recalibration.

## Runtime path

```text
Audio PANG
   |
   +--> freeze game HitRegions
   |       |
   |       +--> refresh calibration
   |       +--> transform 4 corners -> camera AABBs
   |
   v
V2.24.3 full-frame local ROI gate
   |
   +--> V1 / legacy detector
   +--> CandidateGenerator V2 early path
   +--> CandidateGenerator V2 normal path
   |
   v
V2.22.5 local confirmation
   |
   +--> confirmed physical evidence -> resolver -> HitEvent
   |
   +--> zero / unconfirmed
           |
           v
       FULL-RESCUE requested
           |
           v
       V2.24.3 bypasses local ROI
           |
           v
       full detector ROI (global)
```

## New diagnostics

A healthy object-aware first pass should print something similar to:

```text
[V2.24.3 LOCAL-ROI] shot=4 regions=7 merged=6 base=982341 local=132550 (13.5%) base_bbox=(...) region_bbox=(...) local_bbox=(...)
[V2.24.1 LOCAL-SEARCH] shot=4 regions=7 merged=6 valid=100.0% margin=36px
```

The exact V2.24.1 percentage can be high because V2.24.3 has already restricted
the upstream ROI; the important property is that the whole first-pass physical
pipeline is now local.

If transformed regions truly do not overlap the calibrated detector ROI:

```text
[V2.24.3 LOCAL-ROI-EMPTY] ...
[V2.24.1 LOCAL-EMPTY] ...
[V2.22.5 FAST] ... zero proposals -> queue FULL rescue
[V2.24.3 GLOBAL-RESCUE] ... full detector ROI restored
[V2.22.5 FULL-RESCUE] ...
```

That sequence is intentional and auditable.

## Authority invariant

Nothing in V2.24.3 grants object authority:

- no XY snapping,
- no target-centre attraction,
- target/no-shoot roles do not change detector thresholds,
- a candidate still needs physical PRE->POST evidence,
- exact classification remains a game-side operation after HitEvent exists.
