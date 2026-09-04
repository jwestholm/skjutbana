# V2.25.1 — Balanced Object-Region Physical Proposal / Confirmation

**Status:** implemented foundation / physical acceptance pending  
**Base:** V2.25.0 on top of accepted V2.24.4

## Why this version exists

The first physical V2.25.0 GameObject test verified the important downstream contract:
real camera HitEvents carried `shot_id` and ObjectManager resolved with `frozen=True`.
However, all five deliberately separated physical shots were resolved into a small
camera/game area and therefore produced `objects=[]`. The object system correctly
refused to snap those wrong physical coordinates onto a target.

V2.24.4 also proved that object-local ROI mapping itself was active. The remaining
problem is candidate competition *inside the union of all object regions*: a noisy
or old-hole-heavy region can consume most proposals/confirmations even when the
new physical hole is in another object region.

## Core design

V2.25.1 partitions the first physical pass by frozen HitRegion search area:

```text
PANG / shot_id
      |
      +--> frozen camera HitRegions
              |
              +--> camera -> V2.22.1 work plane (V2.24.4)
              |
              +--> region A -> local robust threshold -> bounded proposals
              +--> region B -> local robust threshold -> bounded proposals
              +--> region C -> local robust threshold -> bounded proposals
              +--> ...
                         |
                         +--> balanced physical pool
                                  |
                         V2.22.5 PRE->POST confirmation
                                  |
                         bounded survivors per region
                                  |
                         physical-evidence track selector
                                  |
                         unchanged final camera XY
                                  |
                         HitEvent -> exact GameObject collision
```

No role is preferred. `target`, `no_shoot`, glass, living and breakable regions use
the same proposal logic. A region gives a *search opportunity*, never a hit verdict.

## Overlapping gameplay objects

Objects occupying essentially the same physical search area are grouped for detector
work. The primary example is a glass panel directly in front of a rear target.
Detector proposal is performed once for that physical XY area, while ObjectManager
later resolves the selected physical point against both frozen gameplay shapes in
z-order and applies penetration rules.

## Physical evidence

Each region gets its own robust saliency threshold and its own registered temporal
normalisation. V2.25.1 uses the same V2.22.5 temporal map:

```text
absdiff * (1 + 0.55 * clip(zscore, 0, 6)) + 0.35 * max(dog, 0)
```

Candidates are annotated with region-normalised evidence but their coordinates are
never modified. After the normal V1/V2 hybrid merge and candidate bank, output is
re-balanced so one search area cannot monopolise the list.

V2.22.5 local PRE->POST confirmation still provides the second observation. V2.25.1
then keeps a bounded number of physically confirmed candidates per region instead
of allowing almost the entire broad candidate list to remain confirmed.

## Final track selection

The original HitScanner selector primarily favours onset time. During object-context
shots V2.25.1 may instead select among V2.25.1-confirmed tracks by physical evidence
strength, then detector score/timing. No role, owner, game score, object priority or
projectile semantics participate.

If no V2.25.1-confirmed track exists, the original selector remains the fallback.

## FULL rescue remains global

The explicit V2.22.5 FULL-RESCUE path bypasses V2.25.1. It must still be able to
search the whole calibrated playfield when the first object-context pass has no
physical proof. This is also what allows a shot outside every game object to remain
physically detectable and later be classified as a gameplay miss.

## Invariants

1. Physical evidence owns final XY.
2. Candidate XY is never snapped, interpolated or moved to an object.
3. Every HitRegion role is treated equally by detector proposal/confirmation.
4. Frozen PANG-time geometry is selected by `shot_id`.
5. Exact GameObject shape collision remains downstream of HitEvent.
6. Penetration/damage/effects never influence detector candidate choice.
7. Re-hit/hole-in-hole remains legal if fresh PRE->POST evidence supports it.
8. No object regions or bad context -> prior global behavior.
9. FULL rescue remains the unchanged global high-recall path.

## Default bounded limits

- region margin: 36 px in canonical camera space
- proposal output: up to 8 candidates per physical region group
- local-confirm survivors: up to 2 per region group
- total balanced confirmation output: up to 8
- near-identical search regions group at 80% intersection/min-area overlap

These are conservative integration defaults, not final tuned accuracy values.
