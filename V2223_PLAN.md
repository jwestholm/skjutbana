# V2.22.3 — Shot-Critical Runtime + Object-Hit Foundation

**Status:** implemented delta, shadow/advisory authority only  
**Date:** 2026-08-28

## Product goal

A real audio shot is a real-time interrupt. The gameplay target remains a final camera/game XY with >=95% correct physical hit detection on unseen sessions, but games may also use a faster object-centric question: **did this frozen game object receive a new physical hit?**

V2.22.3 therefore adds a shot-critical runtime lane and the first object-hit API without changing live hit authority.

## Runtime ordering

`main.py` installs V2.22.3 before `App().run()`. The actual loop still belongs to `App`, but its policy is installed at the program entrypoint so shot priority is a top-level runtime concern, not an AI Training feature.

```text
audio reader thread timestamps PANG
        |
        v
last_peak_ts / queued AudioPeakEvent
        |
        v
FIRST engine decision each loop
        |
        +--> dispatch audio immediately
        +--> freeze object/game hit regions
        +--> cheap camera latest-frame pickup
        +--> HitScanner before scene/update/render
        +--> resolver / HitEvent
        |
        v
only then ordinary automation/scene/render work
```

When a physical shot is unresolved, ordinary scene simulation/rendering may be deferred. The projector therefore keeps displaying the last rendered frame, while the dedicated camera thread continues filling its timestamped ring buffer. F2/synthetic training explicitly bypasses this freeze because it needs to reveal/render its fake hole.

## Camera hot-path correction

The previous `CameraManager.update()` had two problems for hit latency:

1. it probed camera capabilities every frame;
2. it assigned `_last_pickup_count = _read_count`, which consumed the same cursor that `HitScanner.get_new_frames_since_last_pickup()` relies on.

V2.22.3 patches main-thread camera update into a cheap latest-frame pickup only. It does **not** touch HitScanner's pickup cursor. Camera capabilities are still probed at startup and an explicit `refresh_capabilities_v2223(force=True)` method exists for settings/status work, but there is no automatic capability probe in the game loop.

This is expected to make V2.22.2's `frames=0->0` diagnostic become real camera-frame counts.

## PRE/reference semantics

Keep both references; they answer different questions.

- **Static surface/reference:** calibration-time paper joins, repairs, tape, stable surface structure and projector response. Keep it as surface/heightmap-style evidence and fallback.
- **Dynamic recent PRE:** a timestamped camera frame before the current audio peak. It represents what actually existed immediately before this shot, including all old holes.

V2.22.3 makes the diagnostic/AI PRE snapshot prefer a safe ring-buffer frame around ~350 ms before the peak (never later than 80 ms before the peak). The V2.22.1 detector path still has its own dynamic PRE logic. Static `scene_reference` remains the fail-open fallback.

## Hole appearance versus shot novelty

These are intentionally separate concepts:

- `hole_likeness`: does a candidate look physically hole-like?
- `shot_novelty`: did this location change on this shot?

An old hole may score highly for appearance but should not automatically win the NEW-hole task. Known-hole proximity stays soft context because a valid re-hit/hole-in-hole must remain possible when fresh temporal evidence exists.

## Weak spatial prior

V2.22.3 computes viewport-space `center_prior` and `edge_distance_norm` for candidates as **advisory metadata only**. The generic prior reflects the physical observation that ordinary shots are more often central, but it never hard-rejects a strong physical edge hit. Explicit game-object context is stronger than this generic prior.

## Object-hit foundation

Games can now expose a small set of hit regions instead of forcing every game to wait for global localisation logic before it can ask whether an object was hit.

A scene may implement:

```python
def get_hit_regions(self):
    return [
        {
            "object_id": "enemy_17",
            "rect": (x, y, w, h),
            "priority": 1.0,
            "metadata": {"kind": "enemy"},
        },
    ]
```

or register polygons/rectangles directly with `object_hit_registry_v2223`.

At PANG, regions are snapshotted before moving scene objects update. The V2.22.3 evaluator then projects the existing camera candidates to screen space in one batch-style pass and records per-object shadow results:

- object id,
- candidate rank,
- screen/game position,
- object-local position,
- novelty / hole-likeness evidence,
- uncalibrated confidence.

Objects can ask `was_hit(object_id, shot_id)` / `result(...)` without owning OpenCV.

**V2.22.3 object results are SHADOW ONLY.** The ordinary global HitScanner/HitInput path remains authority. A later version may replace the candidate-backed object evaluator with a direct PRE->POST object-region fast path without changing the game-facing API.

## Cursor policy

AI Training uses the cursor as both a contamination risk and a useful diagnostic tool:

- armed/waiting for a real shot: cursor hidden;
- waiting for manual GT click/review: cursor visible;
- `F3`: explicit mouse-shot/debug cursor override;
- `F4`: optional latency wait-cursor toggle.

When latency diagnostics are enabled, the wait cursor starts when the **main thread acknowledges the already-timestamped audio peak**, not when the microphone thread heard it. Therefore a visible delay from PANG to hourglass is itself evidence of pre-dispatch/main-thread latency. Exact telemetry remains authoritative.

## Latency telemetry

New logs:

```text
[V2.22.3 AUDIO-PRIORITY] peak=... main_ack=...ms
[V2.22.3 SHOT] shot=N dispatch=...ms objects=M
[V2.22.3 LATENCY] shot=N state=... dispatch=... camera=... scanner=... hit=...ms
[V2.22.3 VISIBLE] shot=N marker_frame=...ms
[V2.22.3 OBJECT SHADOW] ...   # only when game hit regions exist
```

Runtime gate before serious overnight/live-authority work:

- audio-thread -> main dispatch p95 < 50 ms,
- resolver p95 < 10 ms,
- shot -> HitEvent median < 250 ms,
- shot -> HitEvent p95 < 500 ms.

## Non-goals

V2.22.3 does not:

- give object-hit shadow authority,
- make V2.21.5 physical ranker authoritative,
- hard-delete old-hole regions,
- hard-delete right-side areas,
- let game context invent a physical hit,
- remove mouse/fake-shot beta testing.
