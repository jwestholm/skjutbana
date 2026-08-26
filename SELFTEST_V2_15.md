# V2.15 selftest

From repository root, after V2.14 + V2.15 are installed:

```bash
python3 -m automation.hole_v215_selftest
```

Expected final line:

```text
All V2.15 selftests passed.
```

The test verifies:

- threshold-centred fusion does not favour a differently calibrated model,
- a deliberately complementary paired case can select a real non-trivial blend,
- candidate crops work at image borders,
- both V2.14 models can be loaded,
- candidate order/original coordinates remain untouched,
- all attached V2.15 evidence is explicitly shadow-only.

This is a plumbing/invariant test. It does not claim detector accuracy.
