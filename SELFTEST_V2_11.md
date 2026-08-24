# V2.11 selftest

Run from repository root:

```bash
python3 -m automation.ranker_v211_selftest
```

The selftest verifies:

1. Pool-policy leakage features are excluded.
2. Broad-negative sampling reaches multiple false-candidate phenotypes.
3. Development and confirmation split is shot-level with no leakage.
4. On a synthetic artifact-heavy dataset, a physical/listwise rule beats an
   intentionally wrong "strong signal is best" baseline on held-out shots.

The synthetic success percentage is only a software validation. It is not a
prediction of real projector/camera accuracy.
