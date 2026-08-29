# V2.23.1 DELTA

Install over V2.23.0 / current V2.22.6 runtime checkout.

This delta changes only the V2.23 training/model layer. It does not alter V2.22 hit authority.

Primary fixes:

1. legacy V2.16/V2.20 import uses `CandidatePackV216.load(path)`;
2. converted legacy records are cached;
3. new F2/manual captures include V2.8 all-hypothesis/recall pools;
4. proposal oracle @5/10/20/42 is reported separately from ranking;
5. meaningless zero-positive champions are quarantined;
6. support + baseline improvement are required for research promotion;
7. protected holdout remains excluded from automatic selection.

First commands after extraction:

```bash
python3 -m automation.v2231_selftest
python3 -m automation.v2231_verify_install
python3 -m automation.v2231_legacy_probe
python3 -m automation.v2231_audit
```
