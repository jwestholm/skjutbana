# V2.24.1 test plan

## 1. Static verification

```bash
python3 -m automation.v241_selftest
python3 -m automation.v241_verify_install
python3 -m automation.v241_status
```

Expected: all PASS, and status reports live authority unchanged.

## 2. Regression — existing games without HitRegions

Start normally:

```bash
python3 main.py
```

Use a current game that does not implement `get_hit_regions()`.
Expected behaviour is identical to V2.24.0/V2.22.5. There should be no
`[V2.24.1 LOCAL-SEARCH]` line for those shots.

## 3. Object-context smoke test

Use any temporary/test game exposing one or more V2.24 `HitRegion`s. On PANG,
expect both the V2.24.0 context snapshot and V2.24.1 local-search diagnostic:

```text
[V2.24.0 GAME-CONTEXT] shot=... game=... camera=... transform=...
[V2.24.1 LOCAL-SEARCH] shot=... regions=... merged=... valid=...% margin=36px
```

A hit physically inside a region should still require ordinary V2.22.5
confirmation before `HitEvent`.

## 4. Global rescue

Shoot so the first local proposal pass returns no usable physical proof.
Expected if V2.22.5 rescue is triggered:

```text
[V2.24.1 GLOBAL-FALLBACK] shot=... reason=v2225_full_rescue
[V2.22.5 FULL-RESCUE] shot=... using high-recall extractor
```

The rescue must search the original/global detector ROI, not the object mask.

## 5. Outside-region miss / no magnetism

Shoot immediately outside the object's actual geometry. The engine must never
move/snap the XY onto the HitRegion. If the global detector finds the real
physical hole outside the region, that physical XY may still be emitted; exact
game collision should then classify it as a miss.

## 6. Roles

Expose both `role="target"` and `role="no_shoot"`. Both must be present in the
local physical search. Role is interpreted only after physical XY is resolved.

## 7. Moving objects

This version depends on the V2.24.0 shot-time snapshot. Verify logs show the
context was frozen at PANG before normal scene movement. The full moving-object
acceptance test belongs to V2.24.2.

## Return after physical smoke test

Useful log lines:

```text
[V2.24.0 GAME-CONTEXT]
[V2.24.1 LOCAL-SEARCH]
[V2.24.1 GLOBAL-FALLBACK]
[V2.22.5 FAST-EXTRACT]
[V2.22.5 LOCAL-CONFIRM]
[V2.22.5 FULL-RESCUE]
[SHOT #]
```
