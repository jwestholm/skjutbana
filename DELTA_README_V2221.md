# V2.22.1 delta — perspective ROI, edge guard, latency diagnostics, AI Results fix

Apply this ZIP **over a working V2.22 r2 checkout on branch `dev`**.

## Why this delta exists

The first live V2.22 advisory shots showed two independent facts:

1. `ShotResolver` itself was fast (roughly 5–10 ms), while audio-peak → emitted hit took roughly 2–4 seconds.
2. The detector produced many false blobs along the physical/projected playfield edge, especially near the upper boundary when the board/projection moved slightly.

V2.22.1 attacks the expensive live detector path without changing the canonical coordinate system or game API.

It also fixes the V2.22 regression where opening the AI Results scene crashed with `Unknown item type: ai_results`.

## Coordinate invariant — important

V2.22.1 does **not** rectify/warp the camera image into game pixels for candidate detection.

The pipeline is:

```
full 4K camera frame
    -> calibrated playfield quadrilateral in camera space
    -> axis-aligned camera crop around that quadrilateral
    -> expensive CV only inside the crop
    -> perspective-safe inner playfield mask
    -> local crop candidate (x,y)
    -> add crop offset
    -> canonical FULL CAMERA (x,y)
    -> existing tracking / known holes / AI / ShotResolver
    -> existing HitInput homography
    -> screen/game/content coordinates
```

Therefore the existing homography remains the single authority for camera → screen/game conversion, including camera angle/perspective.

## Perspective-aware edge guard

The guard is specified in **screen/projector pixels before inverse homography** (default 12 px). The inner rectangle is then transformed to camera space. This means the excluded strip follows all four physical playfield boundaries even when the camera sees a trapezoid.

The guard does not mean `camera y < 12`; it means “within 12 screen pixels of the calibrated playable boundary”.

## Files in this delta

Changed:

- `src/engine/ai/bootstrap.py`
- `src/engine/ai/runtime_v222.py`
- `automation/v222_physical_probe.py`

New:

- `src/engine/camera/analysis_geometry_v2221.py`
- `src/engine/camera/hit_scanner_v2221.py`
- `automation/v2221_selftest.py`
- `automation/v2221_verify_install.py`
- `V2221_ROI_LATENCY_AND_AI_RESULTS.md`
- `V2221_TEST_PLAN.md`
- `V2221_DOC_NOTES.md`
- `V2221_MANIFEST.json`
- `SHA256SUMS_V2221.txt`

## Default V2.22.1 settings

They are inserted as defaults by `runtime_v222.py`; no manual settings edit is required:

```json
{
  "analysis_roi_crop_v2221_enabled": true,
  "analysis_playfield_edge_guard_screen_px": 12.0,
  "analysis_crop_padding_camera_px": 16,
  "analysis_fallback_edge_guard_camera_px": 8,
  "analysis_v2221_log": true
}
```

Existing explicit values in `content/ai/settings.json` override these defaults.

## Safety / fallback

- If calibrated inverse homography + viewport/content geometry are valid: diagnostics show `mode=homography`.
- If homography cannot be used: it falls back to scanport rectangle.
- If neither can be used: it falls back to the existing full-frame detector.
- AI/ShotResolver authority rules are unchanged; advisory remains advisory.

Read `V2221_TEST_PLAN.md` before live shooting.
