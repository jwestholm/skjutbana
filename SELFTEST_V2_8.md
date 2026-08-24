# V2.8 self-test

Run from project root:

```bash
python3 automation/detector_v28_selftest.py
```

The test checks:

- robust micro-cluster centre;
- a 189-cluster shot is kept intact by the 220-item recall pool;
- overflow selection keeps spatial and evidence diversity on a 4K-like plane;
- Ranker V6 pairwise learning still works.

Then verify the live game process:

```bash
python3 -m automation.detector_v28_verify
```

For the first physical benchmark use only one 100-shot run.
