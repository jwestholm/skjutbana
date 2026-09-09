# Candidate-to-track survival audit

## Resolved by verified input reconstruction — 2026-09-09

The original export-gap finding in commit `8da2868` was correct for the
eight-track view alone. The subsequent audit uses the **complete recorded
proposal and local-confirmation input lists**, runs the actual V2.22.6 tracker,
and verifies track ids, XY, scores, timestamps, hits, state and counters against
the original trace. No original trace, label or independent metric was changed.

Latest dataset: `session_20260908_194746_b38de674`; independent evaluation:
`physical_20260908_195522_ab2304cd`. All ten selected tracks/coordinates and
136 pending-state track checkpoints plus 85 tracking counters match exactly.
Ids are aligned by exact XY correspondence across every first-frame exported
track, never by candidate index. Multiple confirmation rounds are replayed.
Unrecorded idle-frame history is not invented; invisible irrelevant old tracks
cannot be reconstructed. The results below are verified recorded-input
reconstruction, not frame regeneration or new physical validation.

## PROVEN funnel

10 physical shots → 9 causal proposals @42 → 9 consumed by tracking →
9 created/associated tracks → 9 locally confirmed → 9 eligible → 2 selected.

All nine correct-candidate tracks have two hits on two distinct frames.
Eligibility is the selector predicate before readiness; `state=confirmed`
after emission is not local-confirmation proof. Shot 5 lacks a causal @42
candidate (nearest 53.26 px) and is excluded from the nine-shot funnel.

## Exact candidate fate

Coordinates below are full-camera pixels. Scores are proposal score → final
historical best. First-seen time is the proposal frame, also the creation time.
All nine tracks are locally confirmed and eligible. E means outside exported
eight but available to live ranking; G means correctly selected.

| Shot | GT XY | Nearest causal XY; distance | Proposal source / score | First seen after peak | Track id / XY; distance | Best score history | Rank / fate |
|---:|---|---|---|---:|---|---|---|
| 1 | (2173.09, 1316.63) | (2172.84, 1314.40); 2.24 | V1 via vault / 2.786 | 0.059891 s | 55 / (2172.84, 1314.40); 2.24 | 2.786 → 6.647 | 53 / E |
| 2 | (2372.24, 1267.30) | (2368.67, 1266.00); 3.80 | V1 via vault / 1.183 | 0.043359 s | 157 / (2368.67, 1266.00); 3.80 | 1.183 → 6.183 | 59 / E |
| 3 | (2253.30, 1318.30) | (2254.90, 1321.10); 3.22 | V1 via vault / 3.122 | 0.043362 s | 210 / (2260.50, 1328.50); 12.48 | 10.315 → 15.315 | 9 / E |
| 4 | (2205.50, 1395.11) | (2203.50, 1392.50); 3.29 | V1 via vault / 13.397 | 0.063031 s | 276 / (2203.50, 1392.50); 3.29 | 13.397 → 18.314 | 1 / G |
| 6 | (2103.19, 1242.98) | (2101.50, 1241.00); 2.60 | V1 via vault / 2.828 | 0.023004 s | 540 / (2101.50, 1241.00); 2.60 | 2.828 → 6.455 | 55 / E |
| 7 | (2469.88, 1188.06) | (2467.50, 1185.50); 3.49 | V1 via vault / 6.279 | 0.039236 s | 632 / (2467.50, 1185.50); 3.49 | 6.279 → 11.279 | 33 / E |
| 8 | (2086.61, 1466.08) | (2096.50, 1468.00); 10.08 | FAST / 4.698 | 0.044357 s | 720 / (2096.50, 1468.00); 10.08 | 4.698 → 9.698 | 23 / E |
| 9 | (2426.29, 1422.52) | (2422.20, 1423.19); 4.15 | V1 via vault / 30.452 | 0.056578 s | 765 / (2422.20, 1423.19); 4.15 | 30.452 → 34.216 | 1 / G |
| 10 | (2271.50, 1424.43) | (2267.00, 1435.00); 11.49 | FAST / 10.535 | 0.046535 s | 886 / (2267.00, 1435.00); 11.49 | 10.535 → 13.279 | 24 / E |

**Primary loss stage for shots 1, 2, 3, 6, 7, 8, 10: final ranking.**
There is no candidate-to-track disappearance in these cases.

Shot 3 is the association exception: its V1 observation (2254.90,1321.10)
is 9.28 px from a higher-scoring FAST-created representative (2260.5,1328.5).
It becomes same-frame support and does not move that representative. The
track stays 12.48 px from GT, still an @42 positive, and ranks ninth.
This loses fine localization (@5/@10), not @42 survival. Shot 10 also has
same-frame support, but its nearest FAST representative retains its XY.

## Complete pool and duplicate counts

Each count pair is FAST / other; shot 5 other provenance is UNKNOWN.
Created counts refer to creation provenance; final source can change.
Confirmed means the last candidate has explicit local-confirmation proof.
FAST clusters are actual 12-px association groups, not claims that images
depict the same physical feature. Every retained proposal was consumed.

| Shot | Retained / consumed | Created | Local confirmed | Eligible | Debug 8 | FAST proposal clusters (multi / max support) | Winner |
|---:|---|---|---|---|---|---|---|
| 1 | 42 / 72 | 37 / 60 | 36 / 52 | 37 / 60 | 8 / 0 | 37 (5 / 2) | FAST |
| 2 | 30 / 83 | 27 / 72 | 23 / 60 | 26 / 73 | 7 / 1 | 27 (3 / 2) | FAST |
| 3 | 35 / 53 | 31 / 47 | 31 / 36 | 31 / 47 | 7 / 1 | 31 (4 / 2) | FAST |
| 4 | 31 / 89 | 26 / 81 | 26 / 61 | 26 / 81 | 7 / 1 | 27 (4 / 2) | V1 |
| 5 | 0 / 129 | 0 / 104 | 0 / 88 | 0 / 104 | 0 / 8 | 0 (0 / 0) | UNKNOWN |
| 6 | 34 / 71 | 30 / 64 | 25 / 56 | 30 / 64 | 8 / 0 | 30 (4 / 2) | FAST |
| 7 | 32 / 59 | 30 / 50 | 30 / 48 | 30 / 50 | 8 / 0 | 30 (2 / 2) | FAST |
| 8 | 36 / 84 | 34 / 71 | 33 / 54 | 34 / 71 | 8 / 0 | 34 (2 / 2) | FAST |
| 9 | 77 / 50 | 65 / 41 | 65 / 40 | 65 / 41 | 7 / 1 | 65 (10 / 3) | V1 |
| 10 | 39 / 82 | 34 / 69 | 33 / 70 | 33 / 70 | 8 / 0 | 34 (5 / 2) | FAST |

Totals: 1,128 retained/consumed proposals; 973 final active tracks.
FAST contributes 356 proposals grouped into 315 association clusters;
39 clusters contain multiple FAST proposals (maximum three). The tracker has
**no track-slot cap**. FAST owns 312/973 final tracks (32.1%) but 68/80 debug
positions (85%); shot 5 accounts for eight unknown-source positions.
No evidence supports eviction or unconsumed good proposals here.

## Ordering, ownership and score history

- Inputs are sorted by descending score before nearest-track association.
  Source has no explicit priority. Equal-score ties retain input order.
- Association radius is 12 px. Same-frame duplicates increment support only;
  they do not move XY or increment temporal hits. Later-frame observations
  update XY with alpha 0.35 and replace the current candidate.
- `best_score = max(previous best, incoming score)` on both support and
  temporal matches. This behavior can be isolated synthetically, but **0/973**
  final tracks have best_score above their current candidate score in this run.
  Historical-score poisoning does not explain these physical losses.
- Two representative source changes occur (FAST→V1 in shots 2 and 10); neither
  is a nearest-GT track. Shot 3 has V1 support absorbed by a FAST representative.
- Every initial track within an event has the same onset timestamp. Therefore
  the BASE selector tuple `(abs(onset), -best_score)` reduces to score ordering
  for this event pool. No earlier-source onset advantage explains the failures.

## Replay ablations

Accuracy entries are hits at 5 / 10 / 20 / 42 px, denominator ten physical shots.
Alternatives are checked with the actual readiness predicate at the recorded
decision timestamp; every selected alternative in this table is ready. No
future continuation is invented for an unready alternative.
Error statistics exclude abstentions and use the ordinary median plus nearest-
rank p95. TOP8 below is explicitly derived from the verified final snapshot.
The old ablation artifact is unchanged; its upper-middle “median” for even N
was not a statistical median. This table uses corrected, newly named metrics.

| Evidence / policy | Selected | Hits @5/10/20/42 | Mean | Median | p95 | >100 |
|---|---:|---|---:|---:|---:|---:|
| TOP8_DIAGNOSTIC_REPLAY: CURRENT | 10/10 | 2 / 2 / 2 / 2 | 171.10 | 135.75 | 464.70 | 8 |
| TOP8_DIAGNOSTIC_REPLAY: EXCLUDE_FAST | 5/10 | 2 / 2 / 2 / 2 | 71.52 | 80.48 | 142.47 | 2 |
| TOP8_DIAGNOSTIC_REPLAY: PREFER_NONFAST | 10/10 | 2 / 2 / 2 / 2 | 135.36 | 134.83 | 378.92 | 7 |
| COMPLETE_ELIGIBLE_REPLAY: CURRENT | 10/10 | 2 / 2 / 2 / 2 | 171.10 | 135.75 | 464.70 | 8 |
| COMPLETE_ELIGIBLE_REPLAY: EXCLUDE_FAST | 10/10 | 2 / 2 / 2 / 2 | 145.04 | 124.29 | 388.31 | 6 |
| COMPLETE_ELIGIBLE_REPLAY: PREFER_NONFAST | 10/10 | 2 / 2 / 2 / 2 | 145.04 | 124.29 | 388.31 | 6 |
| COMPLETE_ELIGIBLE_REPLAY: SIGNED_LOCAL_CONTRAST | 10/10 | 0 / 0 / 0 / 0 | 242.85 | 248.46 | 557.42 | 8 |

Complete non-FAST alternatives exist in all ten shots. Both FAST ablations
change **zero of the seven** failures to a physical-near winner. Simply removing
FAST is not sufficient. The one additional fixed contrast hypothesis fails
0/10 on this physical development set and is not a promotion candidate.

## Historical cross-check

With exact synchronous confirmation-frame proof, the same reconstruction
verifies 22/22 dart audio events and 10/10 earlier audio events. For physical
scorecards the two dart false events remain excluded:

| Session | Causal oracle @42 | Tracked | Locally confirmed / eligible | Correct | Oracle-positive ranking losses |
|---|---:|---:|---:|---:|---:|
| Prior dart 20 physical shots | 12/20 | 12/12 | 12/12 | 1/20 | 11 |
| Earlier 10 physical shots | 9/10 | 9/9 | 9/9 | 1/10 | 8 |
| Latest post-PRE 10 | 9/10 | 9/9 | 9/9 | 2/10 | 7 |

The repeated ranking loss therefore predates the PRE fix. Biathlon5 (six audio
events) and smoke2 (two) lack decision snapshots and local-confirmation input
lists; their complete track fates remain UNKNOWN.

Correction of denominator only: the previously quoted dart source rates
FAST 0/16 and non-FAST 1/6 count all 22 audio events. Physical-only rates are
**FAST 0/15; non-FAST 1/5**. Original independent accuracy remains 1/20.

## Reproduction and next step

```bash
python3 -m automation.track_survival_replay \
  --root content/ai/physical_traces/session_20260908_194746_b38de674 \
  --output evaluation_runs/track_replay_NEW
```

Full JSON includes candidate/provenance, association history, current/best score,
first/last times, confirmation metrics, eligibility and rank for every track.
Existing output directories are refused. See [OVERNIGHT_TRACK_RESEARCH.md](OVERNIGHT_TRACK_RESEARCH.md) for synthetic tests, ownership transport correction,
performance, limitations and the next physical diagnostic procedure.


The subsequent ownership regression extends the boundary to track association:
known different producers no longer share XY/hits/best_score. The original
physical inputs still replay exactly; the seven event-local ranking losses
remain unchanged. This correctness fix is separate from the rejected ranking
hypothesis and requires fresh physical validation.
