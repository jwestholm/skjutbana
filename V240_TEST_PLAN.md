# V2.24.0 test plan

## Software tests

1. `python3 -m automation.v240_selftest`
   - game-local AABB contract,
   - mapping normalization,
   - viewport offset,
   - four-corner camera AABB transform,
   - camera->game round trip,
   - empty-game behaviour,
   - shot-time snapshot schema,
   - no live-authority surface.

2. `python3 -m automation.v240_verify_install`
   - required files,
   - GameScene provider proxy,
   - OverlayScene provider proxy,
   - legacy V2.22.3 compatibility,
   - existing shot-critical installer still present.

3. Start `python3 main.py` and verify ordinary games/images/video still start.
   Existing games do not need `get_hit_regions()`.

## Optional manual smoke provider

When a future/debug game implements `get_hit_regions()`, start `main.py` and
fire one shot while watching the same terminal. A valid context snapshot prints:

```text
[V2.24.0 GAME-CONTEXT] shot=... game=N camera=M transform=...
```

Expected with valid calibration:

- `game` > 0,
- `camera` > 0,
- transform method `homography`, `scanport` or `homography_fallback`,
- live authority unchanged.

`automation.v240_status` is intentionally static because the runtime registry is
process-local; launching a second Python process cannot inspect snapshots held by
the running game.

## Non-goals

V2.24.0 does not judge local PRE->POST physical evidence. Do not evaluate object
hit accuracy yet; that belongs to V2.24.1/V2.24.2.
