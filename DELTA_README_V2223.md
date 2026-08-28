# Skjutbana V2.22.3 DELTA

Install over an already working V2.22.2-r2 checkout.

## Main changes

- top-level `main.py` installs the shot-critical runtime policy before `App().run()`;
- queued audio peaks are serviced before camera housekeeping, scene simulation, automation and rendering;
- normal FPS waiting is skipped while a real shot is pending;
- per-frame camera capability probing is removed from the hot path;
- fixes CameraManager/HitScanner sharing of `_last_pickup_count` so HitScanner can actually consume new timestamped ring frames;
- diagnostic/AI PRE snapshot prefers recent timestamped PRE, preserving static `scene_reference` as fallback/surface baseline;
- AI Training cursor policy: hidden while armed, visible for GT/review, F3 mouse-debug override, F4 latency hourglass;
- weak centre/edge priors are attached as advisory candidate metadata;
- new object-hit region registry + per-shot frozen snapshots + shadow evaluation;
- detailed audio-dispatch / hit / visible latency logs;
- documentation + idempotent doc updater.

## Authority

This version does not grant new authority. Existing global HitScanner/HitInput remains authoritative. Object hit results are shadow only. V2.22 resolver remains governed by its existing mode/settings.

See `V2223_TEST_PLAN.md` before physical testing.
