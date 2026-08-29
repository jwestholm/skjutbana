# V2.22.5 implementation notes

## Expected log flow — normal fast hit

```text
[V2.22.3 AUDIO-PRIORITY] ... main_ack=...
[V2.22.3 SHOT] shot=N ...
[V2.22.5 FAST-EXTRACT] shot=N extract=... primary=... temporal=... v2out=...
[V2.22.4 CV] shot=N ... worker=... v2=... extract may no longer dominate
[V2.22.5 LOCAL-CONFIRM] shot=N round=1 tested=... confirmed=... time=...
[SHOT #N] HIT ... hits=2 ...
[V2.22.3 LATENCY] ...
[V2.22.3 VISIBLE] ...
[V2.22.4 AI-SHADOW] ...
```

The important improvement is not merely a faster first pass. There should be **no second `[V2.22.4 CV]` for the same normal shot before HIT**.

## Expected log flow — rescue

```text
[V2.22.5 LOCAL-CONFIRM] shot=N ... confirmed=0 ...
[V2.22.5 LOCAL-CONFIRM] shot=N no local proof -> queue FULL rescue
[V2.22.5 FULL-RESCUE] shot=N using high-recall extractor
[V2.22.4 CV] shot=N ...
[V2.22.5 LOCAL-CONFIRM] shot=N ... rescue=1
```

Rescue is intentionally slower and exists to preserve recall while FAST is being evaluated. It is not the desired common gameplay path.

## Coordinate rule

Local confirmation uses full camera coordinates and reads a small patch around each candidate. It does not interpolate or move the authoritative candidate XY. The existing camera -> screen/game homography therefore remains the only coordinate transform used for gameplay emission.

## Reuse for object-first games

The same local PRE->POST primitive is deliberately reusable for future object-hit fast paths:

```text
freeze HitRegion at PANG
        |
map object polygon/ROI to camera
        |
local physical change test
        |
object.was_hit(shot_id)
```

V2.22.5 does not grant this path gameplay authority yet; it builds the low-cost physical confirmation primitive needed for it.
