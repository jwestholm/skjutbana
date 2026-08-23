# Detector V2.6 self-test

Run from repository root:

```bash
python3 automation/detector_v26_selftest.py
```

The test does not use the camera and does not write to the production Ranker V5
model. It uses a temporary model directory.

It checks four V2.6 invariants:

1. a candidate visible in only one camera frame is retained by the shot vault;
2. a dense/noisy region cannot starve a spatially remote vault hypothesis;
3. Ranker V5 can learn a strict ACTUAL-candidate positive against hard negatives;
4. the auto-gate stays closed before the configured evidence count and opens
   only after a sustained measured advantage.

This is a software smoke test, not a substitute for the deterministic physical
projector/camera benchmark.
