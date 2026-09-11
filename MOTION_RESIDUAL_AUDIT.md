# Local motion / edge residual audit

This is offline read-only analysis of all 973 eligible tracks in the latest
physical session. It tests whether high-ranking false tracks are explained by
small local edge motion.

## Result

The bounded local phase-correlation shifts are small for both classes. True
track shift magnitude quartiles are approximately 0.028, 0.030, 0.047 px;
selected false winners are 0.037, 0.047, 0.059 px. Raw-to-locally-aligned
residual motion-explained fraction is effectively zero for both groups (true
median -0.001, false median 0.002). Alignment does not materially collapse
the false residuals.

Gradient-predicted residual fit is weak: true quartiles are about -0.16,
-0.09, 0.00; false are 0.02, 0.12, 0.19. This is a small directional hint,
but not enough to explain the seven failures. PRE gradient energy is not
higher for false winners (true median 3.67, false 3.07).

For failed shots 1, 2, 3, 6, 7, 8 and 10, local alignment changes residual
energy by approximately zero and does not provide a consistent pairwise
reversal. The proposed edge-motion mechanism is therefore not supported as
the dominant root cause in this session.

## Interpretation

**PROVEN:** global registration being small does not guarantee local motion is
zero, so local motion was tested explicitly. In this data the measured local
shifts are too small to explain the score winners, and motion compensation does
not materially reduce their residuals.

**NOT SUPPORTED:** a simple translated-edge residual is not the main cause of
the seven final-ranking failures.

No motion-compensated ranker or synthetic motion model was frozen. The next
information gap is residual morphology/source provenance beyond translation:
the false structures may involve deformation, illumination, texture, or
detector-specific proposal semantics rather than a coherent local shift.

Reproduction:

```text
python3 -m automation.motion_residual_audit --root content/ai/physical_traces/session_20260908_194746_b38de674 --features evaluation_runs/registered_impact_20260909/physical_v2/physical_features.json --output evaluation_runs/registered_impact_20260909/motion_audit.json
python3 -m automation.motion_residual_selftest
```
