# V2.22.4-r2 — cold-start / selftest hot-path fix

This is a small corrective delta **on top of V2.22.4**.

## Why r2 exists

The original `automation.v2224_selftest` used an absolute `<40 ms` timing gate on the very first CV submission. On a cold Python process that first call could include lazy AIRuntime/settings import cost and creation of the first `ThreadPoolExecutor` worker thread. That made the selftest machine-speed dependent even though the detector itself was still asynchronous.

## Runtime changes

- CV worker thread is warmed during `install_v2224_runtime()` at application startup, before any real shot.
- AI shadow worker is warmed at startup as well.
- CV queue limits / frame-spacing settings are cached at install time, so the first PANG cannot trigger lazy settings import work in `submit_if_needed()`.
- Normal detector behaviour, authority, ROI, resolver and candidate algorithms are unchanged.

## Selftest changes

The async test is now semantic rather than benchmark-flaky: the fake detector blocks on an event for up to two seconds and the test proves `submit_if_needed()` returns before that detector is released. This detects accidental synchronous execution without assuming a particular CPU/scheduler speed.

## Install

Extract this ZIP over the repository after V2.22.4. Then run:

```bash
python3 -m automation.v2224_selftest
python3 -m automation.v2224_verify_install
```

When starting the application, expect:

```text
[V2.22.4-r2] async CV + off-critical AI shadow pipeline installed
```
