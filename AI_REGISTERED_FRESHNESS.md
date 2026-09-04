# AI guidance — V2.25.2 registered freshness authority

Read this before modifying object-aware physical hit selection after V2.25.1.

## Never change these boundaries

1. **Game context never invents a hit.** HitRegions only define physical search areas.
2. **No role weighting.** `target`, `no_shoot`, living, breakable and glass receive the
   same detector treatment.
3. **No XY snapping.** An evidence probe may inspect a few pixels around a candidate,
   but returned `camera_x/camera_y` and resulting HitEvent XY must stay unchanged.
4. **Legacy is proposal, not local authority.** V1/legacy/bank candidates may aid recall
   only after independent registered PRE→POST revalidation for object-context authority.
5. **Immediate PRE matters.** V2.25.2 uses CandidateGeneratorV2's registered immediate
   PRE stack/noise model for authority. Do not replace it with the older ~hundreds-of-ms
   diagnostic PRE snapshot as the sole freshness truth.
6. **Second-frame persistence still matters.** V2.22.5 local confirmation remains a
   persistence check, but its permissive gate alone is not authority.
7. **FULL rescue remains global.** Do not constrain the explicit high-recall rescue to
   GameObject regions.
8. **Moving content stays possible.** Temporal PRE noise is a soft penalty. Never hard
   delete moving/dynamic regions merely because their local background is unstable.

## Provenance labels

- `region_registered`: candidate came from V2.25.1 registered region proposal and also
  passes V2.25.2 freshness.
- `legacy_revalidated`: legacy/V1/bank coordinate independently passes the registered
  freshness evidence.
- `diagnostic_only`: useful for telemetry/recall analysis but not local authority.

## What to inspect in physical logs

Look for:

- `[V2.25.2 EARLY-GATE]` on a fast legacy-only first frame;
- `[V2.25.2 FRESHNESS]` per physical region;
- `[V2.25.2 REGISTERED-READY]` before local authority;
- `[V2.25.2 REGISTERED-CONFIRM]` showing old broad confirmation reduced to
  registered-fresh survivors;
- `[V2.25.2 AUTHORITY]` naming the physical group/source actually selected.

If a wrong region wins, compare center/ring compactness, z-score and PRE noise. Do not
solve it with target-role bonuses or object-centre snapping.
