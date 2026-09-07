# V2.25.3-r2 — settings packaging repair

## Problem

The first V2.25.3 cumulative archive accidentally contained a tiny unit-test stub at
`src/engine/settings.py`. Extracting the archive therefore overwrote the repository's
real settings module and startup failed when `audio_peak_detector.py` imported
`load_audio_peak_threshold`.

This was a package construction error. It is unrelated to the V2.25.3 detector,
registered-readiness bridge, cross-shot novelty or GameObject design.

## Repair policy

V2.25.3-r2 deliberately does **not** ship `src/engine/settings.py`.

`python3 -m automation.v253_prepare` first runs the r2 repair:

1. Keep the current working-tree settings.py when it is complete.
2. If it is the accidental stub/incomplete, save a one-time `.v253r2-broken.bak`.
3. Walk recent Git history and recover the newest complete committed settings.py.
4. Reapply only the V2.24.3 viewport-local `content_rect` fallback when necessary.
5. Refuse to write anything if no trustworthy complete settings.py can be recovered.
6. Validate the long-lived viewport, scanport, camera, audio and LED settings API.

This preserves local/branch-specific settings code instead of reconstructing it from a
partial package.

## Versioning

Runtime behavior remains V2.25.3. The `-r2` suffix records an installation/package fix.
