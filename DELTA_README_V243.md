# V2.24.3 — Local ROI Integration Fix

V2.24.3 is the corrective checkpoint after the first physical V2.24.2 Hit
Context Test.

The physical run proved that HitRegions were being snapshotted and transformed,
and that moving/no-shoot/overlap semantics worked, but it also exposed that
V2.24.1's candidate-level mask was too low in the detector stack. Several shots
fell back with `outside_detector_roi`, while V1/early-V2 paths could bypass the
local mask entirely.

## Changes

1. **Implicit content_rect fix**
   - `content_rect` is viewport-local.
   - When no explicit content rectangle exists, V2.24.3 uses
     `(0, 0, viewport.w, viewport.h)` at runtime.
   - The user's settings file is not rewritten.

2. **Latest calibration at PANG**
   - `hit_input.reload_calibration()` runs immediately before the object-context
     shot snapshot.

3. **HitScanner-level local ROI**
   - Frozen camera HitRegions now restrict `HitScanner._frame_roi_mask()`.
   - Therefore legacy/V1, V2 waiting-post-peak and normal V2 all receive the same
     first-pass local search region.
   - V2.24.1's narrower CandidateGenerator-only hook is superseded at runtime.

4. **Zero-overlap recovery**
   - If calibrated object regions and the ordinary global/content ROI have zero
     overlap, the object-region mask is used for the first physical pass instead
     of silently returning to global ranking.
   - This cannot invent a hit: normal PRE->POST evidence remains mandatory.

5. **Global FULL-RESCUE preserved**
   - A V2.22.5 rescue request bypasses the object mask and restores the original
     global ROI before the high-recall extractor runs.

6. **Testscene upgraded**
   - Existing `content/games/hit_context_test_v242.py` is intentionally kept at
     the same path so existing menu entries remain valid.
   - HUD/log label is V2.24.3.
   - Moving target speed is increased.
   - Frozen/current object motion is shown in pixels.

## Install

Extract over the current V2.24.2/dev checkout, then run:

```bash
python3 -m automation.v243_prepare
python3 -m automation.v243_selftest
python3 -m automation.v243_verify_install
python3 -m automation.v243_status
python3 main.py
```

## Expected physical log

A region-enabled shot should now show one of:

```text
[V2.24.3 LOCAL-ROI] shot=... regions=... merged=... global=... region=... selected=...
```

or, if the old global/content ROI still disagrees with calibrated objects:

```text
[V2.24.3 ROI-RECOVERY] shot=... zero_overlap=1 ...
```

A true V2.22.5 full rescue should show:

```text
[V2.24.3 GLOBAL-RESCUE-ROI] shot=...
[V2.22.5 FULL-RESCUE] shot=...
```

The EMPTY/GLOBAL test mode must have no object-local ROI line for that shot.
