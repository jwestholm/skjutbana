# Candidate to track survival audit

## Scope

This audit covers the ten-shot independent validation
(`session_20260908_194746_b38de674`). Nine shots have a causally available
candidate within 42 px. The trace records 114 retained candidates but only the
final eight tracks in each diagnostic snapshot.

## Funnel

| stage | count |
|---|---:|
| physical shots | 9 oracle-positive |
| causal candidate generated | 9 |
| candidate demonstrably present in exported track set | 2 |
| confirmed exported track | 2 |
| eligible exported track | 2 |
| selected correctly | 2 |

For shots 1, 2, 3, 6, 7, 8 and 10, the nearest causal candidate is present
in the retained candidate pool but has no spatially corresponding track in any
saved top-eight snapshot. The evidence therefore supports
`GENERATED_BUT_NOT_TRACKED` as the operational classification, but the exact
runtime loss cannot be proven from these traces because non-top-eight tracks
were not exported. They are recorded as `UNKNOWN (export gap)` for strict
causal accounting.

## Nine-shot fate table

| shot | nearest causal source / distance | track fate | CURRENT winner | primary loss |
|---:|---|---|---|---|
| 1 | V26/V1, 2.24 px | no matching exported track | FAST, 171.54 px | UNKNOWN (export gap) |
| 2 | V26/V1, 3.80 px | no matching exported track | FAST, 464.70 px | UNKNOWN (export gap) |
| 3 | V26/V1, 3.22 px | no matching exported track | FAST, 115.60 px | UNKNOWN (export gap) |
| 4 | V26/V1, 3.29 px | track 276, confirmed/eligible | V1, 3.29 px | selected correctly |
| 6 | V26/V1, 2.60 px | no matching exported track | FAST, 107.39 px | UNKNOWN (export gap) |
| 7 | V26/V1, 3.49 px | no matching exported track | FAST, 144.30 px | UNKNOWN (export gap) |
| 8 | FAST/V2, 10.08 px | no matching exported track | FAST, 378.92 px | UNKNOWN (export gap) |
| 9 | V26/V1, 4.15 px | track 765, confirmed/eligible | V1, 4.15 px | selected correctly |
| 10 | FAST/V2, 11.49 px | no matching exported track | FAST, 193.90 px | UNKNOWN (export gap) |

The two visible true tracks had two hits on two unique frames and were
confirmed. False winners likewise commonly had two hits on two frames, so
confirmation and recurrence alone do not identify the physical hit.

## Slot and duplicate conclusions

The exported snapshots contain exactly eight tracks, preventing a defensible
count of all created/confirmed/eligible tracks, duplicate clusters, or slot
evictions. Retained candidate counts and source flags prove that FAST proposals
are plentiful, but do not prove that they consumed runtime slots or absorbed a
V1/Vault candidate. Candidate ordering and `best_score` history are visible for
selected tracks only; no source transition or complete history is available for
the seven missing true candidates.

`best_score` is the historical maximum on exported tracks. This can preserve an
early FAST advantage, but its causal impact cannot be isolated without complete
track histories.

## Required tracing improvement and research direction

Future captures must export the complete causally eligible final pool, including
every track's source history, current and best scores, first/last timestamps,
unique frame hits, confirmation fields, rejection reason and final rank. Until
that exists, source-aware FAST ablation results remain top-eight diagnostics.
The next research-only hypothesis should be complete-pool source-independent
deduplication or current-shot ranking, selected only after the missing-stage
evidence is captured. No live policy change is justified by this audit.
