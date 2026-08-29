# V2.23.2 verification plan

## Stage A — software installation

Run from repository root:

```bash
python3 -m automation.v2232_selftest
python3 -m automation.v2232_verify_install
python3 -m automation.v2232_status
```

Do not use the old V2.23.1 selftest as the acceptance test after installing 2.23.2: its old promotion test intentionally assumes no fresh-domain gate and is obsolete.

## Stage B — existing-data domain test

Before generating more data, run:

```bash
python3 -m automation.v2232_train --quick
```

The newest existing >=50-shot F2 session must be reported as the fresh domain and excluded from fitting. An old V2.23.1 champion should be quarantined until a challenger passes the new domain gate.

Protected holdout must always report that it was not evaluated for selection.

## Stage C — one-shot framepack/proposal smoke test

Start AI Training and perform one right-click synthetic/manual round. Exit normally. Then run:

```bash
python3 -m automation.v2232_status
python3 -m automation.v2232_proposals --session latest --limit 1 --force
```

Expected:

- one new JSON+NPZ framepack;
- proposal command completes without GT being used for generation;
- counts are shown for current/local/dense/union;
- oracle and nearest-distance diagnostics are printed.

The dense offline calculation can be much slower than live detection; that is expected.

## Stage D — full F2 session

After the one-shot smoke test passes, run a normal F2 x100 session. During capture expect:

```text
[V2.23.2 FRAMEPACK] saved ...
```

Full-frame compression makes this training mode heavier than V2.23.1. That is deliberate; runtime gameplay remains unchanged.

For the first controlled test, after F2 completes it is acceptable to exit the app and run the offline steps manually so logs are easy to isolate:

```bash
python3 -m automation.v2232_proposals --session latest
python3 -m automation.v2232_train --quick
python3 -m automation.v2232_status
```

The F2 report also schedules the same proposal+training cycle in the background while the app remains alive.

## What to compare

For the new 100-shot session compare proposal sources separately:

- current oracle @20/@42;
- local oracle @20/@42;
- dense oracle @20/@42;
- union oracle @20/@42;
- mean candidate counts.

The key question is whether dense/union materially exceeds the ~10% @20 / ~33% @42 live-pool behaviour observed under V2.23.1.

Only after proposal recall is high enough should ranker metrics be interpreted as the dominant problem.
