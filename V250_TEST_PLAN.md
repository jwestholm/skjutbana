# V2.25.0 Test Plan

## A. Software acceptance

Run:

```bash
python3 -m automation.v250_prepare
python3 -m automation.v250_selftest
python3 -m automation.v250_verify_install
python3 -m automation.v250_status
```

Expected: all PASS.

## B. Existing V2.24 regression

Run at least:

```bash
python3 -m automation.v244_selftest
python3 -m automation.v243_selftest
python3 -m automation.v241_selftest
```

V2.25 must not change detector authority or the V2.24 local ROI geometry.

## C. Game Objects Test scene — mouse/debug smoke test

Start `python3 main.py`, choose **Spel -> Game Objects Test (V2.25.0)**.
Mouse hits deliberately have `shot_id=None`, so they exercise current-geometry
compatibility.

Check:

1. `1/2/3` changes projectile profile.
2. crate loses integrity and emits dust/sound requests when broken.
3. living ellipse uses ellipse collision, not its AABB corners.
4. hard no-shoot blocks a projectile.
5. glass + rear target can both participate in one shot when the profile permits
   enough object hits/penetration.
6. moving target continues moving in ordinary gameplay.
7. `R` resets objects.

## D. Camera/physical shot-id bridge

With camera/audio armed, fire a few shots at the V2.25 test scene or the Hit
Context Test. Expected object log:

```text
[V2.25.0 OBJECT-HIT] shot=<non-None> frozen=True ...
```

The key acceptance point is that scanner hits no longer arrive as
`event_shot=None`.

## E. Frozen moving-object acceptance (after/with V2.25.1 motion continuity)

V2.25.0 stores enough identity/snapshot data for this test, but the existing
shot-critical loop may still defer ordinary scene updates. Once motion is
allowed during resolution, verify:

- object has moved visibly between PANG and HitEvent;
- `shot_id` resolves the PANG shape;
- hit applies to the PANG object generation;
- an object that moved under the hit after PANG is not falsely hit.
