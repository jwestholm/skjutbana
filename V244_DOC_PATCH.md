# V2.24.4 documentation patch

`automation.v244_apply_docs` appends an idempotent V2.24.4 section to:

- `ARCHITECTURE.md`
- `HIT_DETECTION_PLAN.md`
- `CURRENT_STATE.md`
- `ROADMAP.md`

Marker:

```text
<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->
```

The update records the physical V2.24.3 finding that canonical full-camera
HitRegions were being used inside V2.22.1's crop-local detector plane, and the
V2.24.4 correction that maps through live `AnalysisGeometryV2221` before the
object ROI mask is built.
