# V2.25.2 test plan

## Software gate

Run:

```bash
python3 -m automation.v252_prepare
python3 -m automation.v252_selftest
python3 -m automation.v252_verify_install
python3 -m automation.v252_status
```

Regression selftests for V2.25.1, V2.25.0 and V2.24.x must remain green.

## Physical test

Use **Game Objects Test (V2.25.2)** and repeat the same five clear shots used for
V2.25.0/V2.25.1:

1. BREAKABLE / crate
2. LIVING
3. STATIC NO-SHOOT
4. MOVING LIVING
5. REAR TARGET / penetration stack

Use projectile profile 2 / `medium` for a comparable run. New photographs are optional
if the physical aim points are approximately the same and unambiguous.

## Primary acceptance observations

For every normal object-context shot:

- `shot_id` remains present and ObjectManager stays `frozen=True`;
- an early legacy-only result may log `EARLY-GATE` but must not emit prematurely;
- at least one `[V2.25.2 REGISTERED-READY]` must occur before normal local authority;
- `[V2.25.2 AUTHORITY]` must identify a candidate with source `region_registered` or
  `legacy_revalidated`;
- final XY must remain the detector coordinate (no snapping);
- five deliberately separated shots must no longer collapse into one small camera area.

If registered-fresh evidence cannot verify a local candidate, the existing V2.22.5
FULL rescue may run globally. That is correct fail-safe behaviour.

## Diagnostic questions

For a miss, determine in order:

1. Did the correct physical region contain any V2.25.1 registered proposal?
2. Did a candidate in that region pass V2.25.2 compact freshness?
3. Did it survive V2.22.5 second-frame persistence?
4. Did a different registered-fresh region have stronger physical authority score?
5. Was FULL rescue invoked?

Do not tune GameObject collision until detector XY is physically correct.
