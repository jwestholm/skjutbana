# V2.4 internal validation

Before packaging, the following checks were run successfully in the build
environment.

## Syntax / configuration

All new/modified Python sources compile with Python's `compile()` and both JSON
configuration files parse successfully.

## Functional self-test

```bash
python3 automation/detector_v24_selftest.py
```

Result:

```text
patch descriptor: OK
shot accumulator: OK
tile probe: OK
ranker learning: OK + exact-GT patch supervision
ALL V2.4 SELF-TESTS PASSED
```

## Local tile stress test

A synthetic noisy 160x120 temporal field with several long line artifacts per
frame was generated for 200 random small-hole cases.

The local tile probe recovered a candidate within 5 px of the synthetic hole in
196/200 cases (98.0%) in that artificial test. Average probe count was about 85.

This is only a code-level stress test. The physical projector/webcam benchmark
is authoritative.

## Patch-grid ranker stress test

Ranker V4 was trained on randomized synthetic compact hole patches against
random line, cross, offset-blob, box and ring artifacts using exact-GT patch
supervision. On a separate artificial held-out set containing one hole and 100
artifacts per case, the trained linear patch-grid model placed the hole first in
97.5% of 200 cases and in the Top-3 in 100%.

Again, this validates that the representation and learner can express the
required separation. It does not claim the real projector/camera data will have
the same accuracy.
