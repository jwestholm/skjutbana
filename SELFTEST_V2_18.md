# V2.18 selftest

Run:

```bash
python3 -m automation.newhole_v218_selftest
```

The test intentionally creates a candidate-ranking problem where the inherited pointwise probability is wrong but the frozen embedding contains the useful signal. It verifies that:

- listwise training improves candidate Top-1,
- offset refinement is evaluated independently from raw candidate recall,
- model save/load is deterministic,
- known-hole information remains soft diagnostic context and is not a hard exclusion feature.

Expected final line:

```text
All V2.18 selftests passed.
```
