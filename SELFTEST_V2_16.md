# V2.16 selftest

Run from repository root:

```bash
python3 -m automation.candidate_v216_selftest
```

Expected final line:

```text
All V2.16 selftests passed.
```

The selftest verifies:

- candidate-pack JSON/NPZ roundtrip,
- pre/post/GT patch shapes,
- temporal evidence prefers a newly appearing persistent change over an old static artefact,
- storage-cap GT rescue is explicitly marked diagnostic-only,
- Hole-AI scoring preserves candidate order,
- fusion search can choose pure or mixed evidence and returns normalized weights,
- one-session split is marked provisional,
- hard-negative mining selects far-from-GT candidate mistakes,
- session finalization stays shadow-only.

This test does not require camera/projector hardware and does not claim real hit
accuracy.
