# V2.9 self-test

Run:

```bash
python3 automation/ranker_v29_selftest.py
```

The test is independent of camera hardware. It verifies:

1. within-shot relative features preserve ordering and tie-safe percentiles;
2. the ranking dataset writer produces an atomic per-shot JSON record;
3. the offline pairwise learner can recover a deliberately hidden synthetic
   pattern when the baseline ranker strongly prefers the wrong candidates.

Expected final line:

`All V2.9 selftests passed.`

This self-test does **not** claim a real camera accuracy. It only checks the
software mechanics needed for the next measured ranking experiment.
