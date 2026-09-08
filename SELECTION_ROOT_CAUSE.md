# Selection root-cause analysis — physical 2026-09-08 ten-shot set

The frozen input is `content/ai/physical_traces/session_20260908_144822_d6713dec`
and the evaluation is `evaluation_runs/physical_20260908_145817_9cdbf1e1`.
The full per-shot machine-readable audit is
`evaluation_runs/physical_20260908_145817_9cdbf1e1/selection_audit.json`.

## Root cause

The immediate bottleneck is deterministic track selection, not candidate
generation, for most of this session. The global target path has no frozen object
camera regions (`context.game_hit_regions` is unavailable), so V2.25.1, V2.25.2
and V2.25.3 selectors do not run. Each explicitly calls the previous/base selector
in that case.

The base selector in `src/engine/camera/hit_scanner.py:485-498` considers every
active track whose `first_seen_ts - event.peak_ts` is within the association
window, skips only already-emitted tracks, and compares:

```text
(abs(onset_dt), -track.best_score)
```

The V2.22.6 tracker in `src/engine/shot_track_v2226.py:169-243` sorts each frame's
candidates by descending detector `score`, creates tracks in that order, and keeps
the maximum score observed as `best_score`. Therefore simultaneous tracks normally
tie on onset and the strongest detector-score artifact wins. The final readiness
test only requires two distinct observations after 300 ms or the configured frame
count (`hit_scanner.py:500-506`); it does not require the winner to have the
strongest local new-hole evidence.

This explains the physical numbers directly. For shots 1, 2, 3, 5, 7, 8 and 10,
the GT-nearest retained candidate has detector scores roughly 0.57–2.63 and is
buried at positions 69, 101, 85, 86, 85, 87 and 81. The selected tracks have
scores roughly 8.08–37.49 and are hundreds of pixels away. Shot 4 and shot 9 are
proposal misses at the 42 px criterion, so selection cannot fix them. Shot 6 is
the positive control: its GT candidate is retained at position 1 with score 36.5,
the highest score in the pool, and the selector emits it. It is the only clear
final success. The ordering/score relationship is therefore consistent across
the control and failures.

The correct candidates are not simply rejected by the confirmation calculation.
The saved local-confirmation observations contain candidates at or near the GT for
the successful and many failed shots. However, those candidates retain much lower
detector scores, and the global selector never compares local-confirmation
strength before `best_score`. The trace's top-eight track list is only a debug view;
it cannot prove that every active track was absent. It does prove that selected
artifacts reached confirmed emission while the GT candidate did not become the
winner.

## What is and is not active

The per-shot audit shows no usable V2.25 region/freshness/novelty authority fields
for this global target. The V2.25 selectors are source-level fail-throughs here,
not the cause of the chosen XY. Recurrence/novelty values are present as candidate
features (many GT candidates have `v2222_novelty_factor=0.1` while selected
artifacts are usually `1.0`), but the base global selector does not use them.
Registration/crop fields are preserved in the audit; they are evidence about the
detector coordinate path, not a selection rule.

The candidate timestamp is the detector frame timestamp and the track timestamps
are consistent with the corrected async handoff. Tracks generally begin at the
same shot frame and receive one later observation. This means the failures are not
explained by the earlier timestamp overwrite regression. Stale/old-hole evidence
is still a physical feature concern, but this dataset proves the final comparator
currently gives detector score and onset priority over that evidence.

## Per-shot comparison

The JSON audit includes full retained candidate feature dictionaries, selected
candidate/track dictionaries, nearest visible track, all visible track snapshots,
AI challenger output, context and trace completeness. The headline errors are:

| Shot | GT retained position / distance | Deterministic error | AI error | Shadow replay error |
|---:|---:|---:|---:|---:|
| 1 | 69 / 2.63 px | 1029.60 px | 1063.83 px | 2.63 px |
| 2 | 101 / 2.56 px | 179.12 px | 258.38 px | 87.10 px |
| 3 | 85 / 22.94 px | 249.47 px | 254.10 px | 59.70 px |
| 4 | 51 / 41.56 px | 911.67 px | 915.27 px | 262.60 px |
| 5 | 86 / 2.75 px | 801.07 px | 750.16 px | 253.30 px |
| 6 | 1 / 3.11 px | 3.11 px | 3.11 px | 3.11 px |
| 7 | 85 / 3.73 px | 679.91 px | 747.63 px | 45.30 px |
| 8 | 87 / 3.77 px | 707.10 px | 731.47 px | 52.00 px |
| 9 | 105 / 49.63 px | 592.77 px | 635.71 px | 91.50 px |
| 10 | 81 / 6.41 px | 672.08 px | 665.54 px | 105.80 px |

At 42 px, retained/confirmed coverage is 9/10 while deterministic final selection
is 1/10. The current AI challenger is also 1/10 and should not be promoted.

## One safe replay experiment

I ran exactly one shadow-only hypothesis: choose from saved local-confirmation
candidates using

```text
confirm_center_abs + confirm_darkening - confirm_compact
```

with detector score as a tie-break. This is a physically meaningful PRE→POST
evidence experiment, but it is not installed in `HitScanner` and it does not use
ground-truth coordinates at runtime. Its status is `PHYSICAL_REPLAY_CHALLENGER`.

Against this frozen set:

| Selector | Top1 @5 | Top1 @10 | Top1 @20 | Top1 @42 | Mean error | Errors >100 px |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic | 1/10 | 1/10 | 1/10 | 1/10 | 582.59 px | 9 |
| Canonical AI | 1/10 | 1/10 | 1/10 | 1/10 | 602.52 px | 9 |
| Replay challenger | 2/10 | 2/10 | 2/10 | 2/10 | 96.32 px | 3 |

The replay result is promising as a diagnostic direction but is not physical
validation, does not establish generalisation, and is not live-authoritative. It
also loses shot 6, so it is not ready for promotion. The next experiment should
first improve the trace so every active track's local confirmation and final
eligibility are persisted, then compare a confirmation-first selector against the
same frozen set and a new physical set.

## Scope decision

No candidate-generation policy, threshold, AI model, or live selector was changed.
Only the read-only audit and replay tooling was added. This keeps the evidence
focused on the confirmed selection bottleneck.
