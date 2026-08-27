# V2.19 Selftest

Run from repository root:

```bash
python3 -m automation.offline_v219_selftest
```

Expected ending:

```text
[PASS] source/family-level media splits + manifest roundtrip
[PASS] media audit catches cross-split exact/near leakage
[PASS] video/animation source stays one media/split unit
[PASS] seed determinism + old-hole/new-hole semantics
[PASS] dynamic-background scenarios preserve physical hole coordinates
[PASS] camera-captured synt_* hole appearances are reused without centre-label shortcut
[PASS] generated worlds compile into existing V2.16/V2.17/V2.18 candidate-pack schema

All V2.19 selftests passed.
```

The selftest is plumbing/semantics only.  It does not claim physical detector accuracy.
