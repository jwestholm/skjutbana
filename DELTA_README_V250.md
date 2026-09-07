# Skjutbana V2.25.0 — Composable Game Object Foundation

This delta is intended to be unpacked directly over a working V2.24.4/dev
checkout. It is cumulative for the V2.24.1–V2.25.0 bridge files included here.

## Why this version exists

V2.24.4 physically validated object-local hit search. Before building production
games, V2.25 defines a reusable gameplay object model that can represent static,
moving, living, breakable, penetrable, layered and event-reactive objects
without a brittle inheritance tree.

It also fixes the missing game-side shot identity: camera HitEvents now carry
the scanner's `shot_id`, allowing exact collision against the PANG-time object
snapshot instead of whichever position an object has later.

## Architecture

```text
physical HitEvent XY + shot_id
        |
        v
ObjectManager
        |
        +--> exact frozen HitShape collision
        +--> front/back hit_depth ordering
        +--> ProjectileProfile vs BallisticBody
        +--> DamageModel / DurabilityLayer(s)
        +--> GameObjectEvent
        +--> ReactionRule
                    |
                    +--> core state/lifecycle changes
                    +--> effect.requested (sound/effect/animation/future services)
```

GameObject is composition-based. Living and breakable are convenience presets.

## Install

```bash
unzip -o skjutbana_v2.25.0_game_object_foundation_delta.zip -d .
python3 -m automation.v250_prepare
python3 -m automation.v250_selftest
python3 -m automation.v250_verify_install
python3 -m automation.v250_status
python3 main.py
```

`v250_prepare` appends documentation sections idempotently and adds
**Game Objects Test (V2.25.0)** to the Games menu without reformatting the rest
of `content/menu.json`.

## Stable public API

Use:

```python
from src.engine.game_objects import ...
```

See `GAME_OBJECT_SYSTEM.md` and `AI_GAME_OBJECTS.md` before creating/migrating a
game.

## Important constraints

- Object geometry is game/viewport-local XY.
- Physical HitEvent XY is never moved/snapped by GameObject logic.
- HitRegions remain detector search context only.
- Camera shots use frozen geometry selected by `shot_id`.
- Mouse/debug shots deliberately use current geometry with `shot_id=None`.
- `caliber_label` is metadata/config selection; gameplay penetration/damage are
  explicit configured values, not inferred real-world ballistics.
- Sound/particles/animation are effect requests only in V2.25.0.
- Continuous object animation during the existing shot-critical wait is not
  changed yet; V2.25.1 is the natural follow-up.

## Diagnostic scene

**Spel -> Game Objects Test (V2.25.0)** demonstrates:

- breakable integrity,
- living health,
- exact ellipse collision,
- solid no-shoot blocker,
- penetrable glass in front of another target,
- moving living object,
- 3 configurable gameplay projectile profiles,
- sound/effect request events,
- shot_id/frozen-snapshot reporting.
