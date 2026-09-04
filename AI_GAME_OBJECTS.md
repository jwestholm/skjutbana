# AI context — Game Objects V2.25+

Use this as implementation guidance when an AI assistant modifies or creates a
Skjutbana game.

## Invariants that must not be broken

1. `HitEvent.game_x/game_y` is physical hit-engine output. Never snap it to an
   object centre, nearest target or desirable game object.
2. V2.24 HitRegions are search context only. Exact object collision is downstream
   gameplay logic.
3. GameObject geometry is viewport-local/game-local pixels. Never feed camera
   coordinates into it.
4. Preserve `shot_id` from scanner camera hits. It selects the exact PANG-time
   object snapshot.
5. A mouse/debug HitEvent may have `shot_id=None`; current geometry is the
   intentional fallback.
6. Do not infer real weapon performance from `caliber_label`. Projectile damage
   and penetration are explicit game/config parameters.
7. Prefer composition/capabilities over deep subclass trees.
8. `entity_id` groups multipart entities; every hit part still needs a unique
   `object_id`.
9. Keep sound/particles/animation/physics behind `effect.requested` rather than
   embedding those services in GameObject.
10. Existing games must continue to work without adopting the object system.

## Canonical flow

```text
PANG
 -> freeze GameObject HitRegions + exact shape metadata
 -> physical detector chooses a real XY
 -> HitEvent carries shot_id
 -> ObjectManager finds frozen snapshot
 -> exact shape collision at unchanged game XY
 -> front/back penetration chain
 -> damage layers
 -> object events
 -> reaction actions
 -> effect requests / game-specific listeners
```

## Prefer these public APIs

- `src.engine.input.hit_regions.HitRegion`
- `src.engine.input.hit_regions.hit_context_snapshot_for_shot`
- `src.engine.game_objects.*`

Avoid importing versioned internals unless working on the hit/runtime engine
itself.

## Extension guidance

If adding sound, particles, animation, physics, scoring or spawning, first ask
whether it can be an event/effect consumer. Do not expand GameObject into a
service locator.

If adding a new object category, first ask whether it is only a combination of
existing hit-shape, body, damage layers, motion, tags and reactions. Add a preset
before adding a subclass.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 detector boundary

For object-aware hit detection also read `AI_PHYSICAL_REGION_PROPOSAL.md`. HitRegions may partition physical search, but gameplay semantics may never create, move or prefer a physical hit.
