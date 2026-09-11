# Common impact ranking research

This document records read-only replay over the complete 973-track eligible
pool from the latest physical session. No live selector changed.

## Feature coverage

Registered static image features (`ring_affine_compact`, darkening, center
darkening, concentration) are available for all 973 eligible tracks in the
recorded pool. They are causally usable only where the corresponding PRE and
decision-time POST imagery exists; this session provides those frames for the
complete diagnostic pool. Temporal persistence is currently available only for
the 18 labelled true/winner examples, so it cannot be fairly ranked over all
eligible tracks.

## Recovery ranks

True-track ranks for shots 1,2,3,4,5,6,7,8,9,10:

| method | @1 | @3 | @5 | @10 | @20 | @50 | ranks |
|---|---:|---:|---:|---:|---:|---:|---|
| raw best score | 2 | 2 | 2 | 3 | 3 | 6 | 53,59,9,1,85,55,33,23,1,24 |
| registered compact | 0 | 2 | 3 | 5 | 7 | 8 | 10,3,18,7,84,21,3,19,4,102 |
| registered dark contrast | 0 | 1 | 1 | 3 | 4 | 8 | 43,19,44,9,69,7,2,37,68,33 |
| registered center dark | 0 | 1 | 1 | 3 | 5 | 8 | 42,19,37,8,85,10,3,28,64,17 |
| registered concentration | 0 | 2 | 3 | 3 | 5 | 8 | 19,5,30,11,90,23,2,36,3,88 |

Compactness is the most useful static common signal at @10–@20, but it does
not make any of the seven failed physical events top-ranked. Its improvement
comes from moving shots 1, 2, 6, 7 and 9 substantially upward; it also leaves
shot 10 at rank 102 and shot 5 at 84. The pairwise winner remains false for
all ten events under every tested static feature (the two raw successes are
already correct).

## Interpretation

Compactness rewards a localized center-vs-ring residual, which is physically
plausible, but it is not a sufficient new-impact identity test. False tracks
can also have compact residuals, and temporal persistence is not available for
the full pool. Static common evidence should therefore be a proposal-stage
screen or diagnostic, not a final authority yet.

The preferred architecture remains source-specific proposals followed by a
common evidence stage, but the common stage needs complete causal temporal
features before a final rank can be evaluated fairly.

## Status

**PROVEN:** common compactness moves several physical true tracks upward
without source-specific scores.

**STRONG HYPOTHESIS:** source-independent residual evidence is the correct
architectural direction.

**NOT PHYSICALLY VALIDATED:** no common ranker, threshold, rejection policy or
NO_VALID_IMPACT behavior is frozen or promoted.

Reproduction:

```text
python3 -m automation.common_impact_ranking --input evaluation_runs/registered_impact_20260909/physical_v2/physical_features.json --output evaluation_runs/registered_impact_20260909/common_ranks.json
```

## Full-pool temporal update

Causal temporal features are now available across all 973 eligible tracks. A
persistence-count rank improves top-k recovery over raw score, but remains
false in all seven failed pairwise events. Temporal evidence is therefore
necessary diagnostic information but not sufficient final authority.

Local-motion diagnostics do not explain why static/temporal common features
still lose the seven pairwise cases. The measured motion-explained fraction is
near zero for both true and false tracks.

Stable-state/next-event PRE diagnostics do not resolve the common-ranking gap:
future-aware center-dark ranks reach only @5=3 and @10=3 across the ten true
tracks. No stable-state final scorer is justified.
