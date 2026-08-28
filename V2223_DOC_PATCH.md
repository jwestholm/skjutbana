# V2.22.3 repository documentation patch

`automation.v2223_apply_docs` appends the sections below to the repository's existing source-of-truth MD files without replacing their current contents. The operation is idempotent.

## CURRENT_STATE.md

Marker: `<!-- V2223_CURRENT_STATE -->`

Summary added:

- V2.22 ShotResolver remains fast (~4-9 ms measured in physical advisory tests).
- V2.22.1 perspective ROI runs heavy CV over about 13.2% of the current 4K camera frame while restoring full-camera XY before normal homography/game mapping.
- V2.22.2 suppresses major horizontal ridge artifacts.
- V2.22.3 is now implemented: `main.py` installs a top-level shot-critical runtime policy before `App().run()`.
- An audio peak already timestamped by the reader thread is serviced before camera housekeeping, automation, ordinary scene update and rendering.
- Camera capability probing is removed from the per-frame hot path.
- CameraManager no longer consumes HitScanner's `_last_pickup_count`; HitScanner owns its timestamped new-frame cursor.
- Static surface/reference and dynamic recent PRE are explicitly separate evidence concepts.
- Object-hit regions can be frozen at PANG and evaluated in shadow without replacing global XY authority.
- Weak centre/edge spatial evidence is advisory only.
- Runtime gate remains main dispatch p95 <50 ms, resolver p95 <10 ms, shot->HitEvent median <250 ms and p95 <500 ms before serious live-authority/overnight work.

## HIT_DETECTION_PLAN.md

Marker: `<!-- V2223_HIT_DETECTION_PLAN -->`

Adds the object-first/global-localisation split:

```text
PANG
 |\
 | +--> OBJECT HIT PATH (shadow in V2.22.3)
 |      freeze 1..100 game hit regions
 |      ask: did a NEW physical change hit this object?
 |      preserve object-local XY
 |
 +----> GLOBAL HIT PATH
        camera detector + physical AI + ShotResolver
        ask: where did the shot go on the whole playable surface?
```

The paths complement each other. Many games may react to a confident object hit without needing an exact coordinate for irrelevant misses, while calibration, score rings, free targets, damage decals and AI training still require authoritative global XY.

V2.22.3 object results stay shadow only. Physical evidence remains truth; game context cannot invent a hit. A future direct per-object PRE->POST evaluator may replace the candidate-backed object evaluator behind the same API.

Also records the shot-critical runtime rules, recent-PRE/static-surface semantics, re-hit legality and weak centre prior.

## AI_CONTEXT.md

Marker: `<!-- V2223_AI_CONTEXT -->`

Adds:

- `main.py` owns installation of the top-level shot-critical policy; the patched loop implementation still executes as `App.run()`.
- audio reader thread only timestamps/queues; Pygame/game work stays on the main thread.
- `scene_reference` is static surface/projector evidence, not a substitute for current-shot recent PRE.
- hole-likeness and shot-novelty are different signals.
- registered game-object regions are snapshotted before scene movement at PANG.
- object-hit confidence in V2.22.3 is uncalibrated/shadow only.
- cursor is hidden while AI Training is armed, F3 explicitly enables mouse-debug mode, and F4 toggles the visual latency hourglass.
