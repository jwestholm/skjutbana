# V2.24.3 physical acceptance test

Use **Games -> Hit Context Test (V2.24.3)**.

## Short matrix

1. stationary TARGET — shoot near centre,
2. standalone NO SHOOT — shoot near centre,
3. moving TARGET — shoot while it is clearly moving,
4. OVERLAP — shoot in the overlapping area,
5. EDGE TARGET — shoot inside the edge target,
6. OUTSIDE challenge — shoot in yellow but outside green,
7. empty background far from every region,
8. press `E` for EMPTY/GLOBAL and fire one arbitrary normal shot.

## What to capture

Save the console from scene entry through the last shot. The most useful lines:

```text
[V2.24.0 GAME-CONTEXT]
[V2.24.3 LOCAL-ROI]
[V2.24.3 ROI-RECOVERY]
[V2.24.3 GLOBAL-RESCUE-ROI]
[V2.22.5 FULL-RESCUE]
[V2.24.3 TEST-HIT]
[V2.22.3 LATENCY]
```

## Acceptance

- Region-enabled shots must enter LOCAL-ROI or explicit ROI-RECOVERY.
- `outside_detector_roi -> silent global` must no longer be the normal path.
- Standalone target/no-shoot/edge should resolve to the object physically shot,
  not a strong old hole elsewhere.
- The outside challenge must remain outside; no object attraction/snap.
- Moving-target log should report non-zero `motion=...px` when enough movement
  occurred between PANG and HitEvent; frozen geometry must remain the basis for
  classification.
- EMPTY/GLOBAL must expose zero game objects and use the unchanged global path.
- If local physical confirmation fails, a V2.22.5 FULL-RESCUE may run and must
  receive global ROI.

Do not proceed to V2.25.0 object classes until this matrix is clean enough that
remaining misses are detector-quality issues rather than ROI/coordinate wiring.
