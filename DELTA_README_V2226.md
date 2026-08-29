# Skjutbana V2.22.6 DELTA

**Base:** V2.22.5 on top of the existing V2.22.4-r2 / V2.22.3 chain.

## Purpose

Fix same-frame track inflation and add raw audio near-miss telemetry.

## Runtime changes

- `src/engine/shot_track_v2226.py`
  - max one `HoleTrack.hits += 1` per physical camera timestamp,
  - same-frame nearby proposals become `v2226_same_frame_support`,
  - preserves strongest authoritative candidate XY rather than averaging duplicate same-frame proposals,
  - logs `[V2.22.6 TRACK]`,
  - logs raw accepted audio triggers and strong rejected near-misses with gate reasons,
  - does not retune audio thresholds.
- `main.py`
  - installs V2.22.6 after V2.22.5.

## Install

Extract over repository root, then:

```bash
python3 -m automation.v2226_selftest
python3 -m automation.v2226_verify_install
python3 -m automation.v2226_apply_docs
python3 main.py
```

Start with only three physical shots. See `V2226_TEST_PLAN.md`.
