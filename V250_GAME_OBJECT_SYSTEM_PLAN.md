# V2.25.0 Game Object System Plan

**Status:** foundation implementation included with this delta  
**Date:** 2026-09-04  
**Depends on:** physically accepted V2.24.4 HitRegion/local-ROI bridge

## 1. Why V2.25 must be broader than `GameObject -> BreakableObject`

A shooting-range object can simultaneously be moving, living, armoured,
breakable, penetrable, a no-shoot target, part of a larger entity and the source
of sound/particles/animation after impact. A deep inheritance tree would force
those independent properties into artificial categories and become difficult to
extend.

V2.25 therefore uses **composition/capabilities** as the canonical model.
Convenience constructors such as `make_living_object()` and
`make_breakable_object()` are presets only; they do not define the architecture.

## 2. Layer boundaries

```text
PHYSICAL HIT ENGINE (V2.24 and older)
    microphone -> camera evidence -> resolved HitEvent XY
                       |
                       | shot_id + frozen HitRegion snapshot
                       v
GAME OBJECT LAYER (V2.25)
    exact shot-time collision
         |
         +-- object/entity/part identity
         +-- projectile profile
         +-- penetration / depth chain
         +-- durability / damage layers
         +-- object events
         +-- reaction rules
                       |
                       v
EFFECT SERVICES (future)
    sound / particles / animation / physics / spawning / decals / scoring
```

The GameObject layer never changes the physical hit coordinate and never tells
the detector that an object *must* have been hit.

## 3. Coordinate model

### 3.1 Authoritative projected geometry

`ObjectGeometry` is always **viewport-local/game-local pixels**. It is the same
coordinate system as `HitEvent.game_x/game_y` and the V2.24 `HitRegion` API.
Game code must not store camera pixels in GameObject geometry.

### 3.2 Optional world semantics

`WorldPlacement` may carry virtual distance/metres or other world-space values
for range-projection games. It is semantic input, not physical-camera geometry.
A future range/world projection service may derive `ObjectGeometry` from it.

### 3.3 Render depth versus ballistic depth

- `z_index`: render order / default depth fallback.
- `hit_depth`: explicit front-to-back ballistic ordering when several objects
  occupy the same game XY.

This lets glass be visually/internally in front of a target without forcing the
renderer and projectile resolver to share every rule forever.

## 4. Identity and multipart objects

Every object has:

- `object_id`: unique runtime hit-object identity;
- `generation`: increments when an id is re-used, protecting delayed shots;
- `entity_id`: logical parent entity (defaults to `object_id`);
- `part_id`: optional part name/id;
- `object_type`, `role`, `owner`, `tags`.

A future character/car/machine can therefore expose several independent hit
objects such as `soldier_7_head`, `soldier_7_body` or `car_4_window`, all with
`entity_id=soldier_7` / `car_4`. This avoids requiring multiple HitRegions with
the same `object_id`, which would conflict with the current V2.24 registry's
identity assumptions.

V2.25.0 does not yet implement an Entity aggregate/shared-health service. It
only makes the identity model compatible with one.

## 5. Exact hit shape versus detector search region

These are deliberately different concepts.

`HitShapeSpec` supports:

- rectangle,
- ellipse,
- circle,
- polygon.

The object publishes an AABB `HitRegion` to V2.24 for fast physical search. The
HitRegion metadata also contains the exact shape snapshot.

At PANG:

```text
GameObject current exact shape
        |
        +--> AABB HitRegion -> local physical detector search
        |
        +--> exact shape metadata frozen in shot snapshot
```

When the HitEvent returns, `ObjectManager` resolves the already-physical
`game_x/game_y` against the **frozen exact shape for that shot_id**. No snapping
or nearest-object attraction is permitted.

## 6. Motion and lifecycle

Lifecycle:

- `ACTIVE`: update + hittable;
- `DISABLED`: object may remain visible but no longer updates/is hittable;
- `REMOVED`: no longer part of manager world.

Motion kind:

- `STATIC`: no automatic movement;
- `KINEMATIC`: simple velocity + game callback;
- `DYNAMIC`: reserved for a later physics service.

V2.25.0 deliberately does **not** change the V2.22.3 shot-critical policy that
may defer ordinary scene updates while a shot is unresolved. The new shot_id
bridge and frozen exact object snapshot are prerequisites for safely allowing
continuous motion later (planned V2.25.1+).

## 7. Projectile / gameplay ballistics model

`ProjectileProfile` carries gameplay values:

- `profile_id`,
- optional `caliber_label` and `diameter_mm` metadata,
- damage + damage type,
- penetration power,
- maximum object hits,
- tags and metadata.

`BallisticBody` carries object-side gameplay values:

- `material_id`,
- penetration mode (`never`, `if_power`, `always`),
- penetration resistance,
- damage multiplier,
- whether the body receives damage.

**Important:** these are game mechanics, not a real-world ballistic simulator.
The engine must not infer penetration/damage from a real calibre name. Games or
content configs explicitly choose gameplay values. This keeps the object model
predictable and avoids embedding weapon-specific physical claims in the core.

### Penetration chain

Overlapping exact hits are sorted front-to-back by `hit_depth`/`z_index`.
For each object:

1. emit `shot.hit`;
2. apply damage if enabled;
3. emit durability/terminal events;
4. evaluate penetration;
5. emit `shot.penetrated` or `shot.blocked`;
6. continue to the next object only when permitted and within
   `max_object_hits`.

This supports, for example, glass in front of a target or a power-up through
which a shot should continue.

## 8. Damage / health / breakable model

V2.25 uses generic ordered `DurabilityLayer`s instead of separate incompatible
health and breakable systems.

Examples:

```text
crate:      integrity
person:     health
armoured:   armour -> health
machine:    casing -> internals
shielded:   shield -> health
```

A `DamageModel` owns one or more layers and a terminal event such as:

- `object.broken`,
- `object.died`,
- `object.destroyed`.

`make_breakable_object()` and `make_living_object()` simply create common
configurations of this generic model.

## 9. Event and reaction engine

Object state changes are event-driven. Core events include:

- `object.spawned`, `object.removed`, `object.state_changed`,
- `shot.hit`, `shot.penetrated`, `shot.blocked`, `shot.stale_object`,
- `damage.applied`, `durability.depleted`,
- `object.broken`, `object.died`, `object.destroyed`,
- `effect.requested`.

`ReactionRule` maps an event to declarative actions. Rules can be restricted by:

- object state,
- object tags,
- event payload values,
- projectile tags.

Core actions executed by `ObjectManager`:

- set object state,
- show/hide,
- enable/disable,
- remove,
- emit another object event.

External/future actions are emitted as `effect.requested`, including:

- `play_sound`,
- `spawn_effect` (dust, sparks, blood, glass, explosion etc.),
- `animation`,
- future impulse/physics/spawn/decal/screen-shake/game-specific actions.

This is intentionally a request boundary. The object system knows *what should
happen* but does not become an audio mixer, particle renderer or physics engine.

## 10. Shot identity bridge required before reusable objects

The V2.24 physical run still showed `event_shot=None`. The scanner itself
already has an `AudioShotEvent.shot_id`; it was lost before `HitEvent` reached
the game.

V2.25.0 adds a backward-compatible runtime bridge:

```text
HitScanner._emit_track_result(track, AudioShotEvent #17)
         |
         +--> thread-local ShotEmissionContext(shot_id=17, peak_ts=...)
         |
         v
HitInput builds ordinary HitEvent
         |
         +--> annotate shot_id=17 BEFORE subscribers are notified
         v
ObjectManager.resolve_hit()
         |
         +--> exact V2.24 snapshot_for_shot(17)
```

Mouse/debug hits remain valid and explicitly get `shot_id=None`, causing current
geometry collision as a compatibility/debug fallback.

No detector/ranker authority changes are introduced.

## 11. Delayed-shot safety

A delayed HitEvent must never damage a newly spawned object that happens to
reuse the same `object_id`. Each `ObjectManager.add()` increments `generation`.
The generation is frozen in the PANG snapshot. A mismatch emits
`shot.stale_object` and skips gameplay damage.

## 12. What V2.25.0 implements now

- stable `src.engine.game_objects` API;
- GameObject identity/lifecycle/motion foundation;
- exact shapes and AABB HitRegion export;
- entity/part identity hooks;
- gameplay projectile/penetration model;
- layered durability/damage model;
- object event bus + declarative reactions;
- sound/visual/animation requests (not playback/rendering yet);
- ObjectManager update/render/hit-region collection;
- exact frozen collision and penetration chain;
- shot-id HitEvent bridge;
- diagnostic Game Objects Test scene;
- compatibility presets for static/breakable/living objects.

## 13. Explicit non-goals for V2.25.0

Do not yet add:

- real audio playback service,
- particle renderer,
- rigid-body physics,
- real-world calibre penetration tables,
- 3D engine / ray casting,
- shared multipart Entity health graph,
- continuous scene motion during the current shot-critical wait,
- replacement/migration of every existing game.

These are extensions behind stable boundaries, not requirements to validate the
foundation.

## 14. Proposed next checkpoints

### V2.25.1 — moving-object continuity
Use the now-authoritative `shot_id` + exact frozen object snapshot to permit
safe game/object movement while camera CV resolves a shot. Verify a moving
object can be visibly far from its PANG position when the HitEvent arrives but
is still resolved against the PANG snapshot.

### V2.25.2 — effect services
Add an effect dispatcher plus optional sound cue service and lightweight
particle/animation requests without putting media code in GameObject.

### V2.25.3 — entity/part aggregation if a game needs it
Parent entity state/shared health, hit-zone multipliers and part-specific
reactions. Build only when a concrete game needs compound entities.

### V2.26+ — game migrations / production games
Migrate or build Pop the Balloon, Shoot/Don't Shoot, Target Distance and other
games on the stable object/event APIs incrementally.

## 15. Acceptance gate

V2.25.0 is acceptable when:

1. legacy V2.24 selftests still pass;
2. scanner camera HitEvent reaches subscribers with exact `shot_id`;
3. mouse hits still work with `shot_id=None`;
4. a moved object resolves against its frozen shot-time shape;
5. stale object generations are rejected;
6. exact ellipse/polygon collision differs correctly from AABB search bounds;
7. penetrable front object can allow a rear object to receive the same shot;
8. blocking object stops the chain;
9. breakable/living terminal events fire once;
10. effect requests contain sound/effect/animation intent without requiring
    those services to exist;
11. returned HitEvent XY is never changed by ObjectManager.
