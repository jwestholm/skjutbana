# V2.25.3 physical acceptance test

Use `Game Objects Test (V2.25.3)` and projectile profile 2 (medium).

Repeat the same approximate five-shot series used for V2.25.0–2:

1. BREAKABLE / crate
2. LIVING
3. STATIC NO-SHOOT
4. MOVING LIVING
5. REAR TARGET / penetration

No new photographs are required if the physical aim points remain comparable.

## Required log evidence

For each object-context shot:

- `[V2.25.3 READY-BRIDGE]` should occur after the worker registered frame.
- The main selector must **not** later claim `no_registered_frame` for that same shot.
- `[V2.25.3 NOVELTY]` should show recurring and new camera locations.
- `[V2.25.3 CONFIRM]` should remain bounded (normally <=8 candidates).
- `[V2.25.3 AUTHORITY]` should identify the chosen physical group and distance to
  earlier confirmed hotspots.
- `[V2.25.3 OBJECT-HIT]` should keep `shot=<id>` and `frozen=True`.

## Primary acceptance criteria

1. Worker/main readiness is consistent for all five shots.
2. Final physical XY no longer collapses repeatedly to the same old hotspot.
3. At least the clearly aimed static object shots begin resolving to their real
   GameObjects rather than `objects=[]`.
4. No target/no-shoot semantic bias or coordinate snapping appears.
5. Global FULL rescue still works if local registered authority cannot produce a hit.

Do not tune damage/penetration gameplay until physical XY satisfies these gates.
