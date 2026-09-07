# V2.24.1 — Object-aware local physical search

## Purpose

V2.24.0 established `HitRegion`, coordinate ownership and shot-time snapshots.
V2.24.1 uses the frozen camera AABBs to make the **first physical proposal
search local** without granting game context hit authority.

## Integration choice

Do not build a third detector. Reuse the live path that already exists:

```text
V2.22.5 FAST extractor
        +
V2.22.5 local PRE->POST confirmation
        +
V2.22.5 one-shot full high-recall rescue
```

V2.24.1 wraps `CandidateGeneratorV2._extract_candidates` *after* V2.22.5 has
installed its wrapper. On the normal first pass it intersects the existing
`valid` detector mask with expanded/merged camera HitRegions. On a queued
V2.22.5 FULL-RESCUE it passes the original valid mask through untouched.

## Search-window geometry

1. Read the V2.24.0 frozen camera regions for the current `shot_id`.
2. Expand every AABB by a camera-space safety margin (default 36 px).
3. Transitively merge overlapping expanded windows.
4. Clip windows to the existing detector ROI/bbox.
5. Intersect with the detector's existing `valid` mask.
6. Pass that restricted mask into the existing V2.22.5 FAST extractor.

The region mask does not modify saliency, PRE/POST evidence values, candidate
coordinates, track state or resolver weights.

## Fail-open rules

Use the pre-V2.24.1 global path when context is missing, malformed or cannot be
mapped safely. Never turn bad context into a forced miss.

The one V2.22.5 FULL-RESCUE is always global. This is a deliberate architecture
invariant and is covered by selftests/install verification.

## Roles

Roles remain game context only. `no_shoot` regions are included in physical
search just like `target` regions because a physical hit on a civilian/no-shoot
object still needs to be detected before the game can penalise it.

## Authority

V2.24.1 does not add a new source of hit truth. It changes search scope only.
A physical candidate must still be proposed and confirmed by the existing
PRE->POST path. Exact game collision remains downstream of the resolved XY.

## Next

V2.24.2 should add a purpose-built verification scene with moving target and
no-shoot objects, overlaps, edge objects, empty context and deliberate shots
just outside regions. Measure local-search success, global fallback rate, false
object attraction, XY error, latency and transform correctness.
