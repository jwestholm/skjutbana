# V2.24.0 — Game Hit Context foundation

## Why this version exists

V2.23 research proved that game context should be available as a complementary
signal, while the global AI/ranker path should not block game development.
V2.22.3 already contained a dormant object-hit shadow registry and shot-time
snapshot hook. V2.24.0 promotes that foundation into a clean, game-ready
contract rather than creating a second parallel subsystem.

## Canonical new path

```text
game.get_hit_regions()
        |
        | HitRegion(x, y, width, height) in game-local XY
        v
GameScene.get_hit_regions()
        v
OverlayScene.get_hit_regions()
        v
existing V2.22.3 shot-critical snapshot BEFORE scene update
        |
        +--> frozen game AABBs
        +--> game->screen->camera four-corner transform
        +--> frozen camera AABBs
        v
V2.24.1 local physical search (next version)
```

## Compatibility

The existing `HitRegionV2223` screen-polygon registry remains available because
the V2.22.3 runtime and V2.23 AI context provider already consume it. New game
AABBs are also converted to legacy screen polygons inside the snapshot only for
that compatibility path.

New games should never create polygons. The stable import is:

```python
from src.engine.input.hit_regions import HitRegion
```

## Authority

No authority change is made in V2.24.0. Regions are contextual search geometry.
Physical evidence remains mandatory.
