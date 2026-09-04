# V2.24.4 delta — Detector Working-Space ROI Mapping

V2.24.3 made the local-search integration observable. Its physical run then
isolated the remaining coordinate mismatch: game regions were successfully
transformed to full-camera AABBs, but the first-pass object mask was constructed
inside V2.22.1's crop-local detector image. Every object mask therefore became
empty (`region=0.0%`).

V2.24.4 maps the canonical camera AABBs through the live V2.22.1 analysis
geometry before mask construction.

## Runtime changes

- new `src/engine/shot_object_local_v244.py`,
- uses `_v2221_active_geometry` when the detector is inside the analysis crop,
- subtracts crop origin,
- derives work/crop scaling from actual dimensions,
- no hard-coded resolution or `/2`,
- intersects mapped regions with the existing V2.22.1 safe ROI,
- preserves V2.22.5 global FULL-RESCUE,
- leaves hit authority and final XY semantics unchanged,
- adds `ROI-MAP`, `LOCAL-ROI`, `ROI-RECOVERY` and `GLOBAL-RESCUE-ROI` telemetry.

## Testscene

The existing Hit Context Test remains the acceptance scene and is relabelled
V2.24.4. No new game-object architecture is introduced in this delta.

## Install

Extract over the current V2.24.3/dev checkout, then run:

```bash
python3 -m automation.v244_prepare
python3 -m automation.v244_selftest
python3 -m automation.v244_verify_install
python3 -m automation.v244_status
python3 main.py
```

See `V244_TEST_PLAN.md` for the physical test matrix.
