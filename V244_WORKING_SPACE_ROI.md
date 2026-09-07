# V2.24.4 — Detector Working-Space HitRegion Mapping

## Problem found by the V2.24.3 physical run

The physical test showed all expected game regions and a valid homography:

```text
[V2.24.0 GAME-CONTEXT] shot=N game=7 camera=7 transform=homography
```

but every region-enabled shot still reported an empty local object mask:

```text
[V2.24.3 LOCAL-ROI] ... region=0.0% overlap=0 ...
```

The missing piece is V2.22.1 analysis geometry. V2.22.1 intentionally crops the
full camera image before running expensive OpenCV work. Inside the legacy
detector, `_frame_roi_mask()` therefore receives **crop-local image coordinates**.
Candidates are translated back to full camera XY only after detection.

V2.24.0 HitRegions, correctly, are stored in canonical **full-camera XY**. V2.24.3
fed those coordinates directly into a crop-local mask, so the AABBs were clipped
outside the smaller working image.

## V2.24.4 mapping

The first-pass object ROI now follows the complete coordinate chain:

```text
game-local HitRegion
        |
        v
absolute screen XY
        |
        v
full camera XY                canonical shot context
        |
        | active AnalysisGeometryV2221
        | - crop_x0/crop_y0
        | - crop width/height
        | - actual detector work size
        v
analysis / detector working XY
        |
        v
object region mask ∩ V2.22.1 safe playfield mask
        |
        v
physical V1/V2 proposal search
```

For today's V2.22.1 path the working image is normally exactly the crop, so the
scale is 1.0 and the important operation is subtracting `crop_x0/crop_y0`.
V2.24.4 still derives `scale_x` and `scale_y` from the actual work/crop sizes so a
future downsampled worker remains correct without hard-coded `/2` logic.

## Authority and fallback

Nothing about hit authority changes:

- HitRegions only specify where the first physical search should run.
- PRE->POST physical evidence remains mandatory.
- Tracking and V2.22.5 local confirmation remain mandatory.
- Object roles (`target`, `no_shoot`, etc.) do not create a hit.
- Candidate XY is never snapped to an object.
- V2.22.5 FULL-RESCUE receives the original global V2.22.1 ROI.

## New diagnostics

A region-enabled shot should now print both mapping and coverage, for example:

```text
[V2.24.4 ROI-MAP] shot=1 map=homography full=3840x2160 \
  crop=(1700,1000,1500,700) work=1500x700 scale=(1.0000,1.0000) \
  first=static_target camera=(1800,1200,1950,1320) work=(100,200,250,320)

[V2.24.4 LOCAL-ROI] shot=1 regions=7 merged=5 global=79.2% \
  region=12.4% overlap=125000 selected=12.0% margin=36px bounds=(...)
```

The exact percentages and coordinates depend on calibration. The acceptance
signal is that `region` is non-zero and the camera->work numbers are plausible.
