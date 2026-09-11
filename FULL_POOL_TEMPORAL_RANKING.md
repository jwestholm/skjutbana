# Full-pool temporal ranking research

This is a read-only replay over all 973 eligible tracks in the latest physical
session. It uses only recorded frames at or before each event's saved decision
timestamp for the causal results. No live selector changed.

## Coverage and timing

All 973 tracks have static local temporal measurements from the saved
pre-snapshot and post frames. Every event has causal POST coverage: 4–8 POST
frames are available before the live decision, with decision horizons from
0.370 to 0.744 seconds after the audio peak. The last causal POST frame is
0.327–0.679 seconds after peak. Later POST frames were retained separately as
post-hoc diagnostics and were not mixed into causal ranks.

This proves a common temporal measurement is technically reconstructable for
this trace format. It does not prove that all measurements were available to
the original selector implementation, only that the captured frames precede
the saved decision boundary.

## True-track recovery

Ranks for shots 1,2,3,4,5,6,7,8,9,10:

| causal feature | @1 | @3 | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|---:|---:|
| raw best score | 2 | 2 | 2 | 3 | 3 | 6 |
| static compactness | 0 | 2 | 3 | 5 | 7 | 8 |
| causal peak center dark | 0 | 1 | 2 | 3 | 4 | 6 |
| causal persistence count | 2 | 3 | 3 | 5 | 7 | 8 |
| causal peak compactness | 1 | 3 | 4 | 6 | 7 | 8 |
| causal median center dark | 0 | 2 | 3 | 7 | 7 | 9 |
| causal median compactness | 2 | 4 | 4 | 6 | 6 | 8 |

Temporal persistence is the strongest causal common feature at @3–@10, but it
does not recover any of the seven failed pairwise winners. It selects the two
already-correct shots 4 and 9; all seven failures remain false winners.

## Pairwise result

For failed shots 1,2,3,6,7,8,10, persistence selects the false winner in all
seven cases. Peak darkening and peak compactness also select the false winner
in all seven. Median compactness selects the true track for shots 2 and 7,
but not the other five. No tested single common feature is a final authority.

## Interpretation

The full-pool result strengthens the architectural direction of a common
impact stage: temporal evidence is available for every eligible track and
improves top-k recovery. It also rules out the simple hypothesis that the
seven false winners are merely non-persistent changes. They can have equal or
stronger causal persistence at their representative coordinates.

The remaining likely distinction is richer than scalar persistence: temporal
spatial identity, impact onset geometry, source/track provenance interaction,
or a mismatch between representative coordinates and the actual changed
patch. No rank fusion or threshold is frozen.

## Status

**PROVEN:** causal full-pool temporal features can be reconstructed for all
973 eligible tracks in this session; persistence improves top-k recovery but
does not solve the seven pairwise failures.

**RESEARCH_ONLY:** common temporal ranking and any latency tradeoff.

**NOT PHYSICALLY VALIDATED:** no temporal selector or `NO_VALID_IMPACT`
decision is promoted.

Reproduction:

```text
python3 -m automation.full_pool_temporal_ranking --root content/ai/physical_traces/session_20260908_194746_b38de674 --features evaluation_runs/registered_impact_20260909/physical_v2/physical_features.json --output evaluation_runs/registered_impact_20260909/full_pool_temporal.json
```

## Local-motion follow-up

A complete 973-track local phase-correlation audit found negligible residual
collapse after bounded local alignment. Temporal persistence remains useful
for top-k diagnostics, but the false winners are not simply translated-edge
artifacts in this session.

## Long-horizon follow-up

Future-aware next-event PRE and later post frames were evaluated separately.
Persistence beyond the causal window does not reliably favor physical tracks;
selected nuisance winners can remain persistent. No longer-delay policy is
supported.
