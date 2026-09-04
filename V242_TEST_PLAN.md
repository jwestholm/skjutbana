# V2.24.2 physical test plan

This is the first version in the V2.24 sequence where a deliberate physical
shot matrix is useful.

## Prepare

```bash
python3 -m automation.v242_prepare
python3 -m automation.v242_selftest
python3 -m automation.v242_verify_install
python3 -m automation.v242_status
python3 main.py
```

Open:

**Spel -> Hit Context Test (V2.24.2)**

## Recommended short test — 8 shots

1. **Stationary TARGET, centre.**
   - Expect local-search log and HitEvent inside `static_target`.
2. **NO SHOOT, centre.**
   - Physical hit must still be detected; scene verdict should be NO SHOOT.
   - This proves role does not suppress physical evidence.
3. **Moving TARGET.**
   - Shoot while it moves.
   - Cyan frozen box should show the position at PANG; current green box may
     have moved by the time the hit arrives.
4. **Overlap area.**
   - Shoot where target and no-shoot overlap.
   - Frozen diagnostics should list both object ids/roles.
5. **EDGE TARGET.**
   - Shoot near the target centre, then inspect that no clipping/transform error
     occurs near the viewport edge.
6. **Outside challenge.**
   - Shoot in the yellow-marked area immediately outside the green target.
   - The returned hit should remain outside the green HitRegion. It must not be
     snapped into the target.
7. **Ordinary empty space away from all objects.**
   - A physical hit may require V2.22.5 global rescue because the first local
     search is region constrained. Final XY must remain the real hole.
8. Press **E** for `EMPTY REGIONS -> GLOBAL`, then shoot any visible object.
   - No V2.24.1 local region restriction should be used for that shot.
   - The ordinary global detector path must still work.

## Logs worth keeping

```text
[V2.24.0 GAME-CONTEXT]
[V2.24.1 LOCAL-SEARCH]
[V2.24.1 GLOBAL-FALLBACK]
[V2.22.5 FULL-RESCUE]
[V2.24.2 TEST-HIT]
[V2.22.3 LATENCY]
```

## Acceptance

Proceed to V2.25.0 if the short matrix shows:

- no crash or menu/scene integration problem,
- stationary target/no-shoot both produce correct physical XY,
- moving-object cyan snapshot is plausible at shot time,
- overlap preserves both region identities,
- edge target works,
- outside shot is not attracted/snapped into an object,
- EMPTY mode behaves globally,
- global rescue can recover a real shot outside local regions.

If any one of these fails, fix the V2.24 bridge before introducing the shared
object engine.
