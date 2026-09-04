# V2.24.2 — Game-context verification scene

## Purpose

V2.24.0 introduced the stable HitRegion API, game->camera transforms and
shot-time context snapshot. V2.24.1 uses the frozen camera regions to restrict
the first physical FAST proposal search while preserving global rescue.

V2.24.2 provides the dedicated scene needed to test those two layers with real
shots before creating a shared object engine.

## Scene

Menu title: **Hit Context Test (V2.24.2)**

Game module:

`content/games/hit_context_test_v242.py`

The scene contains:

1. stationary green TARGET,
2. red NO SHOOT,
3. moving green TARGET,
4. overlapping target/no-shoot pair,
5. target near the lower/right viewport edge,
6. `SKJUT PRECIS UTANFÖR` target with a yellow outside challenge area,
7. `E` mode returning `()` from `get_hit_regions()` to force global detection.

## Shot-time visualisation

After a camera/audio hit, the latest frozen `game_regions` snapshot is rendered
as cyan outlines for about three seconds. The current moving object continues to
render normally. A difference between cyan and current position is expected if
the object moved during detector latency.

Console output includes:

```text
[V2.24.2 TEST-HIT] source=... event_shot=... snapshot=... xy=(...) empty=... verdict=... frozen=[...] current=[...]
```

The scene also keeps simple counters for target/no-shoot/outside classifications
and frozen/current differences.

## Authority rule

The scene is diagnostic only. HitRegions remain search context. Final XY comes
from the ordinary physical hit pipeline and is never moved into an object.

## Controls

- `E`: toggle normal HitRegions / EMPTY global-fallback mode.
- `P`: pause/resume moving target.
- `R`: reset scene counters.
- `ESC`: return to menu.
