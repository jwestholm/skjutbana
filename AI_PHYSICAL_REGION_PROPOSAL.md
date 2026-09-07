# AI guidance — V2.25.1 physical region proposal

Read this before changing V2.25.1 or later game-aware hit detection.

## Non-negotiable boundary

A game object may tell the detector **where physical evidence is worth searching**.
It may never tell the detector where the bullet hit. Never snap or move candidate
XY to object centres, shapes, edges, aim points or desired gameplay outcomes.

## Coordinate spaces

Keep these explicit:

```text
game-local -> absolute screen -> canonical full camera
           -> V2.22.1 crop/working space -> CandidateGenerator bbox-local
```

V2.24.4 owns full-camera -> detector-working-space mapping. CandidateGenerator
`valid/saliency/absdiff/...` arrays are bbox-local within that working space.
Do not confuse those planes or derive a scale from the bbox dimensions.

## Region fairness

Process separate frozen physical regions with the same algorithm. Do not rank
`target` above `no_shoot`, living above static, or breakable above background.
Gameplay role is metadata for downstream collision/result handling only.

Near-identical overlapping physical regions may be grouped for detector work. Keep
all object ids in metadata so the downstream exact collision/penetration engine can
still resolve every gameplay object at the chosen physical XY.

## Candidate evidence

Prefer registered PRE->POST physical evidence and region-local robust normalisation.
Candidate annotations may help physical ranking, but must not replace the raw
candidate coordinate. Avoid semantic weights based on score, owner, role, target
value, health, damage or projectile type.

## Confirmation

V2.22.5 confirmation is a second physical observation. V2.25.1 bounds survivors per
region because physical V2.25.0 testing showed that 75-129 of ~100-135 candidates
could pass the prior broad confirmation gate. Do not restore an unbounded union pool
without new physical evidence.

## Rescue

The V2.22.5 FULL-RESCUE path is intentionally global and high recall. A failed local
region pass must be allowed to fall back without object restrictions. Never make the
rescue infer that a shot had to hit an object.

## Shot identity and motion

Preserve scanner `shot_id` through HitEvent. ObjectManager uses it to choose the
frozen PANG-time object shape. V2.25.4 is reserved for continuous moving-object
updates while physical CV resolves; do not solve that by using current object
geometry for an old shot.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 authority follow-up

Region fairness alone is not authority. Legacy/bank coordinates require registered PRE→POST revalidation; see `V252_REGISTERED_FRESHNESS_AUTHORITY.md`.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 follow-up

Per-region proposal remains the recall partition. Final local authority now also uses a
shared worker/main ready bridge and soft cross-shot camera-space novelty.
