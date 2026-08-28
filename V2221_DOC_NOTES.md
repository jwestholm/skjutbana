# V2.22.1 documentation notes

Reviewed against the current `dev` hit-detection direction and the V2.22 resolver documents.

## Documentation decision

No large existing project MD is overwritten by this delta. The hit-detection roadmap has accumulated versioned experimental history, and replacing it wholesale from a delta would create unnecessary merge risk.

V2.22.1 therefore adds two focused authoritative notes:

- `V2221_ROI_LATENCY_AND_AI_RESULTS.md` — architecture/invariants and crash cause;
- `V2221_TEST_PLAN.md` — exact verification procedure.

These should be referenced from the broader hit-detection roadmap when V2.22.1 is committed.

## Current architectural rule to carry forward

**Full camera XY remains canonical until final hit emission.**

ROI cropping is an implementation optimization only. Future physical-dense or game-context experts should publish votes in full camera coordinates to `ShotResolver`; only the final chosen hit needs the existing camera→screen/game transform.

## Follow-up after verification

If V2.22.1 materially reduces live detector latency without coordinate regression, the next documentation milestone should describe the parallel physical-expert worker and the session-based/night-run data strategy. Those are intentionally not declared live-authoritative in this delta.
