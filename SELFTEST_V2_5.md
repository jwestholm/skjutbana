# Detector V2.5 self-test

Run from repository root:

```bash
python3 automation/detector_v25_selftest.py
```

It checks five code-level properties without starting the game/camera runtime:

1. The benchmark-only GT local probe recovers a known synthetic offset.
2. Weighted tile-centre refinement moves a deliberately offset peak toward the
   centre of a compact temporal signal.
3. The V2.5 shadow accumulator groups a stable three-frame hit into one low-
   jitter cluster.
4. Ranker V4 configuration is in `shadow_mode`.
5. A functional shadow-ranker contract test verifies that V4 can disagree with
   the base order without changing the order returned for the actual shot.

Expected ending:

```text
Detector V2.5 self-test: PASS
```

This test does not predict physical camera performance. The authoritative test
remains the seeded F2 benchmark on the projector/webcam setup.
