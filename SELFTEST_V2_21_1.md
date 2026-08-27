# V2.21.1 selftest

From the repository root:

```bash
python3 -m automation.offline_v2211_selftest
```

Expected:

```text
V2.21.1 SELFTEST
================
[PASS] full-frame recorder config is capture-ready
[PASS] collector background parsing
[PASS] one-shot control is atomically disabled
[PASS] automation scene contains recent-PRE + short capture wiring

All V2.21.1 selftests passed.
```

This test is offline. It does not start the camera/projector session.
