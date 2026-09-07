# V2.25.2 — Registered PRE→POST Freshness Authority

## Why this version exists

V2.25.1 physically proved that per-region proposal balancing works: the union of all
GameObject HitRegions was split into bounded physical search areas and confirmation
was reduced from very large pools to at most a few candidates per region. The five-shot
physical test still resolved final XY into the same small lower-left area.

The log exposed two separate authority leaks:

1. **Legacy/bank authority leak.** A candidate could be inside a frozen region and pass
   V2.22.5 confirmation without originating from V2.25.1's registered per-region proposal.
   It could therefore win even when that region itself reported zero registered proposals.
2. **Early `waiting_post_peak` authority.** CandidateGeneratorV2 intentionally returns
   legacy candidates for the first few milliseconds after PANG. Those candidates could
   complete local confirmation and become a hit before any registered V2 PRE→POST frame
   had been evaluated.

V2.25.2 changes authority, not hit geometry and not GameObject semantics.

## New authority rule

For a normal object-context shot, a local candidate may become authoritative only when:

- a registered V2 PRE→POST evidence frame exists for the same `shot_id`;
- the candidate's exact XY (with only a ±3 px evidence probe, never XY movement) has
  compact registered temporal evidence;
- that location also survives the later V2.22.5 persistence confirmation.

A V2.25.1 region proposal starts with registered authority provenance. A legacy/V1/bank
candidate is still useful for recall, but is labelled `legacy_revalidated` only if the
same coordinate independently passes the registered evidence gate. Otherwise it remains
`diagnostic_only` and cannot win the local object-context track selector.

## Registered physical evidence

The evidence is computed from CandidateGeneratorV2's already registered and
photometrically normalised maps:

- center PRE→POST absolute change;
- ring absolute change;
- center-minus-ring compactness;
- center z-score against the immediate PRE-stack temporal noise model;
- directional darkening and compact darkening;
- immediate PRE-stack noise at the same location.

Immediate PRE noise is a **soft penalty**, not a hard exclusion. This is important for
moving projected media and future moving targets.

No role, owner, health, damage, projectile profile or game score is included.

## Early legacy gate

If an object-context shot has only the early legacy/V1 frame, `_best_track_for_event`
returns no track yet. This does not delete the proposals; it only prevents premature
emission. Once a registered V2 frame has been evaluated, registered-fresh tracks may
win. A bounded fail-open exists to avoid deadlock if a registered frame never arrives.

The explicit V2.22.5 FULL rescue is excluded from this gate after it has actually run.
It remains the old global physical high-recall path.

## Invariants

- Final XY is always an observed detector coordinate.
- Evidence search offsets are telemetry only; they never move candidate XY.
- `target` and `no_shoot` are physically equal.
- Overlapping glass/rear-target regions can share detector work while retaining all
  object IDs for downstream penetration.
- Re-hits remain legal.
- FULL rescue remains global.
- GameObject exact collision/damage/penetration stays downstream of HitEvent XY.
