# V2.23.1 test plan

## A. Install/software

```bash
python3 -m automation.v2231_selftest
python3 -m automation.v2231_verify_install
python3 -m automation.v2230_selftest
```

## B. Canonical legacy import

Probe one pack per root first:

```bash
python3 -m automation.v2231_legacy_probe
```

Then run the complete conversion/audit:

```bash
python3 -m automation.v2231_audit
```

The first run can be slower because 330 JSON+NPZ packs may be converted. A second audit should show cache hits and be substantially cheaper.

Expected key result: legacy `loaded` should no longer be 0/330. Any remaining skips must have explicit `skip_reasons`.

Do not judge a ranker yet if validation oracle20 support is still below the gate.

## C. Offline quick challenger

Only after the audit shows meaningful dev/validation proposal support:

```bash
python3 -m automation.v2231_train --quick
python3 -m automation.v2231_status
```

A pre-V2.23.1 zero-oracle champion should show as quarantined unless replaced by a valid support-gated challenger.

## D. One new F2 session

Start the game and one normal 100-shot F2 session. Verify per-shot lines such as:

```text
[V2.23.1 POOL] shot=... union=... v28_all=... v28_recall=... nearest=... oracle20=...
```

The union candidate count should generally be materially wider than the V2.23.0 average (~97 candidates/shot). Compare final post-run V2.23 status/audit oracle20 and oracle42 against the old session.

## E. Stop condition before overnight work

Do not run `v2231_autotrain --hours ...` until:

1. legacy import succeeds;
2. validation has enough actual <=20 candidates;
3. one new F2 high-recall capture confirms V2.8 pools are present;
4. a challenger has a meaningful baseline comparison.
