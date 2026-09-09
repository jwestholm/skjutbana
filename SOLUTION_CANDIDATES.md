# Architectural solution candidates

## Current path review

`HitScanner` receives an audio event, snapshots PRE state, dispatches V1/V2/
FAST proposals, carries candidates through the vault/bank, merges observations
into `HoleTrack`, applies local confirmation/readiness, and finally calls
`_best_track_for_event` in `src/engine/camera/hit_scanner.py`. That final method
orders eligible tracks by `(abs(onset_dt), -track.best_score)`. The V2 score is a
proposal saliency/change score with source-specific carry and penalty paths; it
is not a calibrated probability of a new physical hole. The architecture
therefore discards common physical identity evidence before the final decision.

| stage | purpose | output | final authority? |
|---|---|---|---|
| V1/V2/FAST | high-recall proposal discovery | source-specific candidates/scores | no |
| vault/bank/rescue | preserve candidates across frames | carried candidates | no |
| tracking | temporal/spatial grouping | tracks and history | no |
| local confirmation | reject obvious local mismatches | confirmation fields | gate only today |
| readiness | enough observations / timing | eligible tracks | gate only |
| final selector | choose one event result | max historical best score | currently yes; this is the defect |

## Candidate approaches

1. **High priority — common impact verifier.** Keep all detector proposals and
   tracks, then evaluate every eligible track with shared PRE→POST morphology
   and causal temporal evidence. Risk: current scalar common evidence is not
   sufficient; cost is roughly one patch per track plus temporal frame reads.
2. **High priority — small real-patch verifier.** Train a source-independent
   classifier or pairwise ranker on physical true tracks and hard negatives,
   with leave-shot/session-out evaluation. Risk: only nine physical positives
   in the latest session; requires more labelled sessions for promotion.
3. **Medium — new-damage segmentation.** Build stable PRE→POST residual masks,
   connected components, then associate components to eligible tracks. Risk:
   long-horizon evidence did not separate current false winners; morphology
   and illumination remain difficult.
4. **Medium — two-stage shortlist plus verifier.** Preserve a high-recall union
   (at least 50/all tracks in current data), then run an expensive common
   verifier. Risk: runtime and shortlist truncation can discard rank-50 truth.
5. **Medium — pairwise ranking.** Compare eligible candidates using common
   evidence rather than absolute score. Risk: pairwise labels are sparse and
   event-specific.
6. **Low — source normalization.** Already tested and insufficient; it cannot
   repair within-source detector ranking.
7. **Low — motion compensation / long-horizon wait.** Rejected by current
   physical evidence.

## Offline experiments completed

- Static common compactness: improves @10/@20 recovery but wins none of seven
  failed pairs.
- Causal persistence: improves @3–@10 recovery but still loses all seven
  failed pairs.
- Tiny leave-shot-out common-feature centroid verifier: the true track enters
  the top ten in only one of ten events. This is an information-test failure,
  not a deployment candidate.
- Local translated-edge compensation and long-horizon stable-state deltas were
  also rejected in earlier audits.

## Recommended concrete architecture

Keep proposal scores discovery-only. Add a diagnostic/research verifier
interface after eligibility with explicit inputs: shared PRE/POST frame ids,
camera XY, causal cutoff, source-independent patch features, and provenance.
Evaluate it offline first; promote only after leave-session-out physical
validation. The next engineering investment should be richer physical patch
labels and causal verifier tracing, not another detector-score formula.

## Implemented research infrastructure

`src/engine/offline/common_verifier.py` now provides a disabled-by-default
common verifier interface, shared causal patch context and four reusable patch
representations. `automation/rich_patch_experiments.py` runs strict
leave-one-shot-out information tests without source or detector-score inputs.
The richer tests do not yet justify a frozen verifier.

## Historical-data decision

The available real data is sufficient to build a conservative patch research
set, but not to validate a final verifier. The next concrete engineering step
is to preserve this dataset schema and add complete eligible-pool/confirmation
metadata to future sessions, then train/evaluate a verifier with
leave-session-out splits.
