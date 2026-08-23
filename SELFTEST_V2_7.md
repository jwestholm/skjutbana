# V2.7 self-test

Run from repository root:

```bash
python3 automation/detector_v27_selftest.py
```

The test is intentionally independent of the physical camera. It verifies:

- robust clustering of multiple nearby observations into one hypothesis;
- spatial pooling preserves a remote low-score hypothesis despite dense noise;
- Ranker V6 learns a held-out synthetic hypothesis pattern even when the
  deterministic baseline is deliberately misleading.

These tests verify code invariants only. They do **not** predict the physical
projector/camera hit percentage. The real acceptance test remains the locked
`--seed 12345` 1000-shot series.
