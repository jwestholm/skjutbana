# V2.10 self-test

Run from the project root:

```bash
python3 -m automation.ranker_v210_selftest
```

The test creates a synthetic ranking dataset whose true candidate deliberately
has lower support/signal and higher compactness than the baseline's preferred
artifacts. It verifies:

- deterministic development/confirmation split;
- monotonic feature-direction search;
- held-out improvement against a deliberately wrong baseline;
- V8 model JSON persistence.

The synthetic percentages are software tests only and are not claims about the
real camera/projector performance.
