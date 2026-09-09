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
