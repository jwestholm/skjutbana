# Detector V2.12 self-test

Run from the **repository root after overlaying the package onto current `dev`**:

```bash
python3 -m automation.offline_v212_selftest --require-v2
```

The mandatory tests verify:

1. historical filename role/pair classification,
2. archive discovery without changing source files,
3. sidecar ground-truth discovery,
4. portable JSONL manifest round-trip,
5. temporal-consensus physical overlay,
6. GT-preserving candidate union,
7. explicit `overlay rescued current detector` metric,
8. compatibility with archives that contain only one before + one after image,
9. **with `--require-v2`: import and replay through the current live `HitScanner` V1→V2 hybrid wrapper.**

The last test is intentionally required on the shooting-PC/full checkout. The V2.12 ZIP only contains changed/new files, so a package-only test environment does not contain `src.engine.camera`.

A successful full-repo run ends with:

```text
[PASS] current live V1->V2 replay adapter: PASSED

All mandatory V2.12 selftests passed.
```

## Synthetic plumbing test

This tests the new offline path without hardware. It is **not an accuracy benchmark**:

```bash
python3 -m automation.offline_v212_synthetic --iterations 100 --out content/ai/offline/v212/synthetic_100.json
```

Use it to catch crashes/regressions in manifest/evidence/fusion code. Never use its percentages as proof of real-world detector quality.
