# Game Object System — stable developer contract

This file is the long-lived source of truth for game objects from V2.25 onward.
For version-specific rationale and acceptance details, see
`V250_GAME_OBJECT_SYSTEM_PLAN.md`.

## Core rule

**Physical hit detection owns the hit coordinate. Game objects own gameplay
consequences after that coordinate exists.**

A GameObject may narrow V2.24's first physical search through its HitRegion, but
it must never invent, snap or drag a hit.

## Public imports

```python
from src.engine.game_objects import (
    GameObject,
    ObjectManager,
    ObjectGeometry,
    WorldPlacement,
    HitShapeSpec,
    BallisticBody,
    PenetrationMode,
    ProjectileProfile,
    DurabilityLayer,
    DamageModel,
    ReactionRule,
    EffectAction,
    make_static_object,
    make_breakable_object,
    make_living_object,
)
```

Games should prefer this package-level API instead of importing versioned
implementation files.

## Coordinate contract

- `ObjectGeometry`: viewport-local/game-local pixels.
- `HitEvent.game_x/game_y`: same coordinate plane.
- `WorldPlacement`: optional virtual/world semantics only.
- camera/screen/detector working-space transforms remain engine internals.
- `z_index`: render order/default hit depth.
- `hit_depth`: explicit ballistic front/back ordering.

## Object identity

- `object_id` must be unique for one live hit object.
- `generation` protects delayed shots from id reuse.
- `entity_id` groups future multipart entities.
- `part_id` names an optional hit part.
- `object_type`, `role`, `owner`, `tags` describe semantics.

Do not publish several HitRegions with the same `object_id`; use unique part
object IDs and a shared `entity_id` instead.

## Hit geometry

`HitShapeSpec` is exact gameplay collision. The corresponding `HitRegion` is
only its AABB search envelope. The exact shape is frozen inside HitRegion
metadata at PANG and later selected by `HitEvent.shot_id`.

Use polygon points for rotated/irregular collision rather than assuming
`rotation_deg` rotates an AABB.

## Lifecycle and motion

- ACTIVE: update + hit-enabled.
- DISABLED: may remain visible but no longer updates/is hit-enabled.
- REMOVED: removed from manager world.
- STATIC / KINEMATIC are implemented.
- DYNAMIC is reserved for a future physics layer.

## Projectile semantics

`ProjectileProfile` values are **gameplay tuning values**. `caliber_label` is a
name/config selector; the core does not derive real penetration or damage from
calibre names.

`BallisticBody` defines the object's gameplay material/penetration behaviour:

- `NEVER`: stops the projectile;
- `IF_POWER`: passes when remaining configured penetration power reaches the
  configured resistance;
- `ALWAYS`: always passes, while still consuming configured resistance.

`max_object_hits` bounds how many overlapping objects one game shot can affect.

## Damage

Use `DamageModel` + ordered `DurabilityLayer`s. This supports integrity, health,
armour, shield or future custom layers without adding a new GameObject subclass
for each combination.

## Events / reactions

Use `ReactionRule` for declarative object responses. Common triggers:

```text
shot.hit
damage.applied
durability.depleted
object.broken
object.died
object.destroyed
shot.penetrated
shot.blocked
```

Core actions are state/visibility/active/remove/emit-event. Audio, particles,
animation and other media/physics behaviour are `effect.requested` messages for
separate services.

Rules can filter on object state/tags, event payload and projectile tags.

Example:

```python
ReactionRule.on(
    "object.broken",
    EffectAction.spawn_effect("dust"),
    EffectAction.play_sound("crate_break"),
    EffectAction.set_state("broken"),
    EffectAction.set_active(False),
    once=True,
)
```

## Manager usage

```python
manager = ObjectManager()
manager.add(obj)

def get_hit_regions(self):
    return manager.get_hit_regions()

def update(self, dt):
    manager.update(dt)

def on_hit(hit):
    result = manager.resolve_hit(hit, projectile_profile)
```

When `hit.shot_id` is present, resolution uses the exact frozen PANG snapshot.
When it is `None` (mouse/debug compatibility), current geometry is used.

## Effect service boundary

A future sound/particle/animation system should subscribe to the manager's event
bus or consume `manager.effect_requests`. Do not put mixer, file-loading,
particle or rigid-body code into `GameObject`.

## Compatibility

Existing games do not need to migrate immediately. A game that does not use
`ObjectManager` can continue using HitInput exactly as before.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 physical-search note

GameObject semantics remain unchanged. The detector may balance proposals per frozen HitRegion, but exact collision, z-order, damage and penetration run only after physical HitEvent XY exists. See `V251_OBJECT_REGION_PHYSICAL_PROPOSAL.md`.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 detector boundary

The object model is unchanged. V2.25.2 only strengthens physical hit authority before ObjectManager collision.
