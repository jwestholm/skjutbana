# V2.22.3 architecture notes — shot critical + object hits

## Why the runtime changed

A camera frame/status task is irrelevant compared with a real shot. The microphone reader already runs in its own thread and timestamps a peak as soon as it is observed. V2.22.3 turns that timestamp into the first runtime decision at the next main-thread boundary.

The audio thread remains producer-only. It never calls Pygame or scene logic directly. This avoids cross-thread Pygame hazards while still letting the main loop know that a shot is pending before doing ordinary work.

## Why object hits are a separate path

Global localisation asks:

> Where on the whole playable surface did the shot go?

A game often asks a cheaper question:

> Did any of my 1..100 active objects receive the new shot, and where inside that object?

The object API is therefore designed as a second consumer of physical evidence, not as a replacement for global XY.

### Authority in V2.22.3

```text
GLOBAL HITSCANNER / HITINPUT   -> authority
V2.22 SHOT RESOLVER           -> advisory according to configured mode
V2.22.3 OBJECT HIT ENGINE     -> shadow only
V2.21.x PHYSICAL AI           -> offline/shadow research
```

### Future fast path

Once the object API is validated, the expensive global detector does not necessarily need to finish before a game reacts to an obvious object hit:

```text
PANG
 |\
 | +--> object-region PRE/POST fast path -> high confidence object hit -> game reaction
 |
 +----> global detector / physical AI / resolver -> exact/global XY, validation, misses
```

A scene can also declare later whether an exact miss coordinate matters. For many arcade games, a shot that hits no active object is simply a MISS and gameplay need not wait for global localisation.

## Game API foundation

Import:

```python
from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
```

Direct registration:

```python
object_hit_registry_v2223.register_rect(
    "enemy_17",
    (x, y, w, h),
    owner="my_game",
    metadata={"kind": "enemy"},
)
```

Or scene provider:

```python
def get_hit_regions(self):
    return [
        {"object_id": "enemy_17", "rect": (x, y, w, h)},
        {"object_id": "barrel_2", "polygon": [(...), (...), (...)]},
    ]
```

The registry freezes those regions per `shot_id` before the normal scene update moves anything.

Query after shadow evaluation:

```python
if object_hit_registry_v2223.was_hit("enemy_17", shot_id):
    result = object_hit_registry_v2223.result("enemy_17", shot_id)
    # result.local_x/local_y are 0..1 within the object's bounding box.
```

The current confidence is intentionally uncalibrated and must not be presented as a true probability.

## Geometry

Object regions live in screen/game-facing space. Existing camera candidates remain full-camera XY. V2.22.3 uses the existing canonical camera->screen transform to compare candidates against frozen object geometry. It does not alter the authoritative HitInput coordinate path.

The earlier V2.22.1 ROI crop remains valid: crop-local candidates are restored to full camera XY before they reach this layer.
