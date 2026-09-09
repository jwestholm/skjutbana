# Overnight candidate-to-track research, 2026-09-08/09

## Measurement contract

Instrumentation is opt-in through physical trace capture. It records every
tracking input/association and the complete active track pool at the selector,
including ineligible tracks and the exact predicate rejecting them. Eligibility
is **before readiness**, which the live resolver tests only for the winner.
`local_confirmed`, `state=stable`, and emitted `state=confirmed` are distinct.

`CURRENT_EXACT_REPLAY` independently recomputes eligibility, tuple ordering,
stable ties, track id and coordinates from the snapshot. Any mismatch raises;
missing legacy input is unavailable. The recorder freezes this snapshot before
emission and keeps historical stage fields unchanged. No research ranker is
installed in live authority.

Association records identify each consumption attempt, event producer and
dispatch owner, source, current score, before/after XY and best score, distance,
merge threshold and actual branch (`no_track_within_merge_radius`,
`same_frame_support`, `later_frame_nearest_track`). Source history also includes
same-frame support that does not replace the representative. Terminal snapshots
own frozen ledgers; pending overlapping event ledgers remain available.

The post-PRE physical run can be reconstructed from complete proposal and local
confirmation inputs despite its top-eight debug export. This resolves the prior
unknown fates through additional evidence; it does not alter original traces,
labels or independent validation metrics. Full results follow below.

## Producer ownership transport correction

The earlier boundary and rejection fixes remain valid, but the new integration
regression exposed an incomplete transport step: `AsyncDetectorV2224.apply_result`
added `v2224_producer_shot_id` only to `scanner.last_candidates`; the runtime then
passed **`result.candidates`** to tracking and to the local-confirmation seed.
That list lacked the tag. A pending old track could consume a newer result at
nearby XY, reach two hits, and pass the old event's selector/readiness gates.

**PROVEN and FIXED:** `apply_result` now tags the actual consumed result list and
copies it to diagnostics. This completes the existing ownership contract;
scoring, candidate generation and ranking rules are unchanged. Integration tests
cover 6→7 and 14→15 intervals and valid delayed pre-boundary delivery. The new
ledger distinguishes runtime eligibility from causal ownership, so exact replay
cannot hide an ownership defect by calling an unsafe live result safe.

This transport correction has no new physical validation. The latest clean
10-shot result is preserved as an independent validation of its original commit.
