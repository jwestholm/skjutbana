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

## Track-history ownership correction

A final integration check proves a second, distinct ownership requirement:
checking only the last candidate tag does not isolate a track's accumulated XY,
hits and best_score. With an old pending track, a nearby newer producer, then
valid older camera evidence delivered late, the original tracker associated all
three observations into one track. The final old tag hid mixed history.

The before/after component episode uses actual `apply_result` transport and
V2.22.6 tracking, with no GT in decisions:

| Earlier event result | Before ownership-aware association | Corrected |
|---|---:|---:|
| XY (camera pixels) | (20.91, 30) | (20, 30) |
| best_score | 40 from newer producer | 12 from own evidence |
| frame hits | 3 across producers | 2 from own producer |
| track count | 1 | 2 |
| newer event retains its own onset/track | no | yes |

**PROVEN component correctness defect, FIXED:** V2.22.6 association now skips
tracks with a known different producer event. It retains the same distance,
ordering, representative update, score and readiness rules within an event.
Unknown legacy producer identities retain their existing behavior; new async
results carry explicit producer tags, including confirmation seeds and rescue.
There is no time-based blacklist, coordinate blacklist or new scoring policy.
Association ledgers record nearby excluded track ids, producer ids, distances
and the actual `producer_shot_id_mismatch` reason.

Tests cover both 6→7 and 14→15 intervals, a later old-frame delivery, preservation
of the newer event's own onset, unchanged same-event delayed operation and
completed-snapshot immutability. This is an adversarial delivery-order regression;
it is **not** a claim that mixed-history ordering caused the latest seven physical
failures. Those latest event-local pools still reproduce exactly. The correction
extends the already-authorized ownership boundary, separately from research
ranking. Fresh physical validation remains required.

## Physical candidate survival and exact replay

**PROVEN (code + verified recorded-input replay):** the seven unresolved physical
candidates were never lost from tracking. All nine causal @42 positives become
tracks, acquire local confirmation on two distinct frames, and remain eligible.
Shots 1/2/3/6/7/8/10 rank 53/59/9/55/33/23/24. Shots 4/9 rank first.

The compact nine-row fate table, full per-source counts, association exception
for shot 3, and ablation metrics are in [TRACK_SURVIVAL_AUDIT.md](TRACK_SURVIVAL_AUDIT.md).
Machine-readable reports contain every candidate feature and track association.
All ten CURRENT selections match exactly, as do 136 saved track checkpoints and
85 tracking counters. The previous top-eight-only conclusion is superseded by
this additional reconstruction evidence. Original trace files and independent
validation metrics are unchanged.

The complete-pool FAST-exclusion/fallback ablations still get 2/10, with mean
145.04 px, median 124.29, p95 388.31 and six errors over 100 px. Every event now
has an available non-FAST alternative; none of the seven failures is repaired.
This rules out missing non-FAST slots as a sufficient explanation.

### Mechanism scorecard

| Mechanism | Evidence and finding |
|---|---|
| A. FAST duplicate flooding | 356 proposals form 315 actual association clusters; 39 multi-proposal clusters, maximum three. No runtime track-slot cap; 9/9 good candidates survive. Slot crowding is not the observed loss. |
| B. Ordering changes creation | PROVEN generically: descending score, stable input-order ties; tested with reversed inputs. All physical tracks share one onset per event, so there is no earlier-FAST onset advantage here. |
| C. V1 absorbed by FAST | PROVEN for shot 3; V1 supports an existing FAST representative 9.28 px away. That track is still 12.48 px from GT and eligible at rank 9. |
| D. Good candidate not consumed | Rejected for all nine latest oracle-positive shots: exact inputs and counters reproduce consumption. |
| E/F. Radius/representative drift | Radius is 12 px, later-frame XY alpha is 0.35; same-frame support does not move XY. Shot 3 loses @5/@10 precision through representation but retains @42. Not the seven @42 losses. |
| G. Historical best score | Historical maximum is the implemented rule and can be reproduced as an isolated hazard. Zero of 973 final physical tracks have best_score greater than current score, so it does not explain this session. |
| H. Confirmation and representative differ | Confirmation searches ±4 px for the strongest 3×3 absolute change, retaining proposal XY. Tracking may then average XY independently. This is a proven semantic displacement, not proof of a different physical hole. |
| I. Stale evidence beats current evidence | Possible in general; unconfirmed strong V1 tracks block ready weaker tracks in synthetic texture/line cases. Historical best advantage is absent from the latest physical pool. |
| J. Correct eligible track is outranked | PROVEN primary loss for all seven latest physical oracle-positive failures and repeated in older physical sessions. |

**Best-supported root cause:** the global BASE selector ranks surviving tracks by
onset and best score; within these common-onset pools, it compares detector scores
across source semantics, with only a bounded confirmation bonus. That ordering
fails to distinguish physical impact evidence from nuisance change. FAST occupies
85% of the debug-eight positions despite only 32.1% of active tracks. FAST alone
is not the whole explanation because non-FAST ablation also fails.

## Local-confirmation semantics

The active V2.22.3 PRE capture prefers a safe recent ring-buffer frame; it falls
back to static scene reference only if no safe frame is available. V2.22.5 holds
this full-camera PRE array per event. Its local confirmation compares that PRE
directly to the current full-camera frame, without the V2 registration transform
or global photometric normalization. The V2 coordinate correction repaired V2
history sampling; it did not change this separate local-confirmation formula.

The center is chosen by maximum 3×3 **absolute** difference within ±4 px.
Measurements use center radius 2 and ring radii 4–8. Confirmation requires:

```text
center_abs >= 1.65 AND
(compact >= 0.45 OR peak_abs >= 3.4 OR darkening >= 1.30)
bonus = clip(0.35*center_abs + 0.60*max(compact,0), 0, 5)
```

`darkening` is mean `max(PRE-POST,0)`; compact is center absolute residual minus
ring absolute residual. Brightening can pass with darkening zero. This is tested
explicitly. It explains why wrong winners 7 and 10 remain valid proposals, but
zero darkening is not a safe rejection policy: the near-GT candidate of physical
shot 10 also has confirmation darkening zero. Confirmation is evidence of local
change, not an isolated test that a new bullet hole appeared at exact track XY.

Physical positive controls do not supply a universal local-confirmation cutoff:
shots 4/9 have V1 proposal scores 13.397/30.452, becoming 18.314/34.216, while
confirmation center_abs is only 7.154 for both. Shot 9's PRE change 26.59 is valid.
The other true candidates often have weaker raw scores despite local proof.

## Synthetic harness and evidence level

`automation.synthetic_track_research` uses the project's `SyntheticHoleOverlay`
(renderer used for projected training), the real hybrid V1/V2/vault detector,
V2.22.1 crop transform, V2.22.2 cleanup, the FAST extractor on a real worker thread
named `shot-cv-v2224`, V2.22.5 local confirmation and V2.22.6 tracking. It uses
explicit frame/audio times and calls the real selector/readiness functions.
CURRENT_EXACT_REPLAY verifies every recorded synthetic decision. No camera,
projector, microphone, AI-training loop or physical labels are accessed.

Real image inputs are native-resolution 640×480 crops at camera rectangle
(1800,1030)–(2440,1510), from two recorded `pre_history` frames within 320 ms
before each physical peak. Both frames precede the physical shot. Their natural
sensor/registration/illumination variation remains in designated cases. The
renderer adds deterministic synthetic truth at seeded coordinates. Other cases
use controlled light/dark/texture/line backgrounds. Half use a nonzero analysis
crop origin (17,13); half use zero origin. Geometry is explicit, not the live
projector homography. Synthetic distances remain native camera-pixel units.

The scene matrix includes light, dark, real dart imagery, old holes, new holes
8 px from old holes, target edges, tape/lines, texture, low/strong contrast,
subpixel registration jitter, +3/+18 intensity illumination changes, delayed
worker delivery, no-impact noisy-frame events, and exactly unchanged images.
The `sequential_nearby` stress stratum is an adjacent-old-hole image fixture;
a separate regression exercises two actual sequential synthetic impacts on the
same scanner, including idle track aging and known-hole state. Another regression
runs the exact 6→7 / 14→15 false-audio boundary patterns followed by a new impact.
Those timing episodes are reported separately from the independent image batch.

**Limitations:** deterministic worker completion replaces live frame scheduling;
there is no hardware camera latency, microphone waveform, projector update or
physical tear/bright-backlight simulation. No-hit cases that fail readiness after
the allowed local rounds are reported as non-emissions, not measured wall-clock
timeouts. The stress loop does not recreate the entire App/rescue scheduler.
Therefore these results are integrated component tests, not live-path-equivalent
physical replay or physical validation.

## Development and holdout discipline

One coherent research ranking was tested: `SIGNED_LOCAL_CONTRAST`, the recorded
name for `confirm_darkening - confirm_ring_abs`, restricted to locally confirmed
tracks, with existing rank tuple as tie-break. It has no source weights, GT
features, coordinate blacklist or trained AI. This is a contrast score and can
be negative; it is not an unrectified signed pixel delta.

It was chosen after physical reconstruction proved final-ranking loss, then
frozen before synthetic holdout. Seeds are DEVELOPMENT 20260908 and HOLDOUT
20260909. The same real background library is shared; holdout tests new impact
locations/nuisance combinations, not unseen camera installations. Development
includes the smoke cases. The 510-event stress run is a development superset,
not another independent holdout. Holdout was evaluated once; no parameters were
changed in response to its result.

Frozen research configuration SHA-256:
`df2760029c3e2513b62551cd34c733cfabcf5bad367370d4968f822fccf5829a`.

The experiment gets **0/10 @42 on physical development**. Synthetic holdout does
not improve @42 and worsens mean error. It is **RESEARCH_ONLY / NOT APPROVED**;
there is no best research candidate suitable for live installation.

## Synthetic measurements

Tolerance columns list hits at 5 / 10 / 20 / 42 px. Error statistics exclude
non-emissions; accuracy denominators include them. p95 is nearest-rank.

| Set / policy | Impacts | Hits @5/10/20/42 | Mean / median / p95 px | >100 | Non-emissions |
|---|---:|---|---|---:|---:|
| development: causal retained oracle | 90 | 88 / 88 / 88 / 89 | 2.049 / 0.078 / 2.000 | 1 | 0 |
| development: CURRENT | 90 | 80 / 80 / 80 / 81 | 21.767 / 0.143 / 174.270 | 6 | 2 |
| development: SIGNED_LOCAL_CONTRAST | 90 | 85 / 85 / 85 / 85 | 14.634 / 0.134 / 85.796 | 3 | 0 |
| holdout: causal retained oracle | 150 | 142 / 142 / 144 / 146 | 3.095 / 0.000 / 12.748 | 1 | 0 |
| holdout: CURRENT | 150 | 139 / 139 / 139 / 141 | 12.894 / 0.000 / 100.439 | 8 | 1 |
| holdout: SIGNED_LOCAL_CONTRAST | 150 | 140 / 140 / 140 / 141 | 19.372 / 0.000 / 100.568 | 8 | 0 |
| stress_development_510: causal retained oracle | 450 | 430 / 433 / 435 / 440 | 2.521 / 0.000 / 4.173 | 3 | 0 |
| stress_development_510: CURRENT | 450 | 405 / 405 / 405 / 406 | 26.298 / 0.063 / 256.564 | 36 | 4 |
| stress_development_510: SIGNED_LOCAL_CONTRAST | 450 | 417 / 417 / 417 / 417 | 19.932 / 0.000 / 216.760 | 30 | 0 |

| Set | Generated/retained @42 | Tracked | Locally confirmed | Eligible | Selected @42 | No-impact events / ready false emissions |
|---|---:|---:|---:|---:|---:|---:|
| development | 89/90 | 89 | 89 | 89 | 81 | 12 / 6 |
| holdout | 146/150 | 146 | 146 | 146 | 141 | 20 / 10 |
| stress_development_510 | 440/450 | 440 | 440 | 440 | 406 | 60 / 30 |

Candidate survival through eligibility is 98.89% of impacts on development,
97.33% on holdout and 97.78% in stress. CURRENT selection @42 is respectively
90.00%, 94.00% and 90.22%; no intermediate @42 track loss occurs.

The 510-event batch contains **450 rendered impacts and 60 forced no-impact
audio events**. All 30 exactly unchanged images avoid readiness; all 30 natural
pre-frame-variation/no-impact events produce a ready candidate. This is a
conditional visual false-event result, not a measurement of audio triggering.
Four image cases select an unconfirmed strong V1 track despite a lower-ranked
confirmed near-GT track; these are readiness-blocked non-emissions, not timed
out hardware shots. The remaining 446 rank winners are FAST. Many correct
synthetic FAST winners directly refute a blanket “disable FAST” interpretation.

### Stress breakdown by scene

Each impact scene contains 30 cases. Oracle and research figures use @42.
The near-old/sequential fixture distinction is described above.

| Scene | Oracle | CURRENT | Research | CURRENT mean px | >100 | Non-emissions |
|---|---:|---:|---:|---:|---:|---:|
| light | 30 | 30 | 30 | 0.12 | 0 | 0 |
| dark | 30 | 30 | 30 | 0.27 | 0 | 0 |
| real_dart | 29 | 29 | 29 | 16.29 | 1 | 0 |
| old_holes | 29 | 27 | 27 | 40.91 | 3 | 0 |
| near_old | 30 | 27 | 28 | 32.78 | 3 | 0 |
| sequential_nearby | 30 | 29 | 29 | 8.55 | 1 | 0 |
| target_edge | 30 | 30 | 30 | 1.31 | 0 | 0 |
| tape_lines | 30 | 29 | 30 | 0.38 | 0 | 1 |
| texture | 30 | 28 | 30 | 0.79 | 0 | 2 |
| low_contrast | 24 | 3 | 11 | 239.47 | 23 | 0 |
| strong_contrast | 29 | 29 | 29 | 14.76 | 1 | 0 |
| jitter | 30 | 29 | 29 | 8.35 | 1 | 0 |
| illumination_small | 30 | 29 | 30 | 0.34 | 0 | 1 |
| illumination_large | 30 | 30 | 28 | 0.09 | 0 | 0 |
| delayed_result | 29 | 27 | 27 | 26.63 | 3 | 0 |

### Reproducing the physical signature

Development contains three cases and the protected holdout four cases with
all three features: near-GT retained evidence, a wrong FAST winner, and a good
eligible track outside the debug top eight. Development indices 43/60/94 have
good-track ranks 38/20/18. Holdout indices 43/77/111/162 have ranks 59/104/17/86.
All are low-contrast synthetic impacts. Complete ledgers show ranking loss,
not eviction, on these cases. Similarity of signatures does not establish that
all physical nuisance structures are reproduced by the renderer.

The final harness additionally invokes the actual async `apply_result` transport.
The 17-case development parity runs after producer transport and track-owner
isolation reproduce
all prior choices, coordinates, oracle/funnel values and association counts.
The final metrics path also calls the runtime readiness predicate for every policy (rather than relying on the equivalent two-hit shortcut for these default fixtures); the same 17 development cases retain exact choices, readiness and funnel values. Holdout was not re-tuned or rerun after seeing its results.

## Historical cross-check

The earlier 20-shot physical funnel is 12 causal positives → 12 tracked →
12 locally confirmed/eligible → 1 correct. The earlier ten-shot development
funnel is 9 → 9 → 9 → 1. All recorded input replays match their original winners.
The prior 20-shot rate FAST 0/16 / non-FAST 1/6 included false audio events.
For the actual 20 physical shots it is FAST 0/15 / non-FAST 1/5. This corrects
an audit denominator, not the independent 1/20 validation result. Biathlon5
and smoke2 remain insufficient for complete-pool reconstruction.

## Performance and trace cost

The complete physical inputs average 97.3 tracks/event (max 107),
with 225.2 association records/event (max 351).
Across ten repetitions per event, alternating instrumentation on/off, mean
per-event median tracking+selection+snapshot time is **28.10 ms off vs
30.29 ms on**, an incremental **2.19 ms**, for the final implementation with
current async producer tags applied to the verified event batches. This final
benchmark ran without a concurrent synthetic batch. The earlier untagged-input
measurement (preserved separately) was 23.49/25.37 ms, a 1.88 ms difference.
Coordinates are identical in every toggled pair. This measures the captured input batches plus one selector/snapshot call per
event, isolating trace cost from detector work. It does not measure repeated
resolver polling or live frame-scheduling impact.

Incremental complete audit JSON averages 0.639 MB compact /
0.952 MB with the recorder's indented format.
Projected additional JSON for 10 / 20 / 100 events is 9.52 / 19.03 / 95.17 MB
(excluding already-existing images, maps and legacy stage snapshots).
Indented JSON encoding averages 32.04 ms/event at finalization,
after the decision. Detector work in the 510-event synthetic batch averages
135.53 ms/event. The first implementation copied the ledger twice and added
about 16.7 ms; immutable completed rows now avoid that redundant critical-path
copy. No verbose per-candidate console logging was added.

**NOT YET VALIDATED:** real camera detector latency with complete tracing.
The new physical capture must check this before further behavior changes.


## Reproduction commands and retained artifacts

All generated reports remain local under
`evaluation_runs/overnight_tracks_20260908/`; none are committed. Final physical
CLI reports are `physical_complete_funnel_final`, `historical_dart_cli_final` and
`historical_ten_cli_final`. Development, holdout, stress and overhead outputs
retain their separate names. Existing output paths are refused.

```bash
python3 -m automation.track_survival_replay \
  --root content/ai/physical_traces/session_20260908_194746_b38de674 \
  --output evaluation_runs/track_replay_NEW
python3 -m automation.track_survival_replay \
  --root content/ai/physical_traces/session_20260908_163626_9b47fac6 \
  --exclude-events 7 15 --output evaluation_runs/dart_track_replay_NEW
python3 -m automation.track_survival_replay \
  --root content/ai/physical_traces/session_20260908_144822_d6713dec \
  --output evaluation_runs/earlier_track_replay_NEW
python3 -m automation.synthetic_track_research \
  --root content/ai/physical_traces/session_20260908_194746_b38de674 \
  --split development --count 102 --output evaluation_runs/synthetic_dev_NEW
python3 -m automation.synthetic_track_research \
  --root content/ai/physical_traces/session_20260908_194746_b38de674 \
  --split holdout --count 170 \
  --frozen-manifest evaluation_runs/synthetic_dev_NEW/manifest.json \
  --output evaluation_runs/synthetic_holdout_NEW
python3 -m automation.synthetic_track_research \
  --root content/ai/physical_traces/session_20260908_194746_b38de674 \
  --split development --count 510 --output evaluation_runs/synthetic_stress_NEW
python3 -m automation.track_audit_benchmark \
  --root content/ai/physical_traces/session_20260908_194746_b38de674 \
  --repeats 10 --tag-producers --output evaluation_runs/track_overhead_NEW.json
```

The holdout command is for independent reproduction of the frozen experiment,
not another opportunity to develop against its disclosed cases. The renderer
and policy manifest are deterministic; wall-clock overhead is machine dependent.
The physical replay command supports emitted global BASE traces with sufficient
recorded input; absent/ambiguous input and failed checkpoints raise explicitly.
New physical exports also verify complete snapshots directly, without legacy
reconstruction. No image regeneration is claimed for historical physical replay.

The final historical CLI cross-check also evaluates the already frozen policies:

| Physical development session | CURRENT @42 / mean | Exclude FAST @42 / mean | Signed contrast @42 / mean |
|---|---|---|---|
| Earlier ten | 1/10 / 582.59 | 0/10 / 327.92 | 1/10 / 240.04 |
| Prior dart twenty (false 7/15 excluded) | 1/20 / 509.36 | 3/20 / 325.49 | 2/20 / 398.58 |
| Latest ten | 2/10 / 171.10 | 2/10 / 145.04 | 0/10 / 242.85 |

These are additional development ablations, not independent validation results.
They reinforce that neither source exclusion nor this one contrast feature is
an adequate final policy.

## Remaining uncertainties and next physical capture

- **PROVEN:** the seven latest good tracks survive to final ranking. The missing
  top-eight entries are a diagnostic export issue, not evidence of track loss.
- **STRONG HYPOTHESIS:** nuisance image change versus localized impact evidence
  remains insufficiently separated by current scores/confirmation. Low-contrast
  synthetic failures reproduce the signature, but do not identify the physical
  nuisance as specifically tape, old holes, camera motion or illumination.
- **RESEARCH_ONLY:** signed local contrast was frozen and evaluated, then rejected
  for promotion. There is no best research policy ready for physical authority.
- **NOT YET VALIDATED:** complete trace timing and the consumed-result/track-history ownership
  corrections on live hardware; generalization of synthetic images to real impact
  damage; unrecorded idle state in old traces; complete fates in biathlon5/smoke2.
- Exact replay intentionally preserves selector-specific behavior. The standalone
  legacy V2.25.2 selector lacks the producer-id predicate used by BASE/V2.25.1/
  V2.25.3. Current startup installs V2.25.3; this audit does not certify running
  V2.25.2 as a standalone authority configuration. Its policy was not changed.
- Synthetic batches exercise detector/tracker components, not the full App/rescue
  scheduler. The separate sequential and false-boundary regressions cover event
  episodes. Non-emission and forced no-impact readiness are not measured live
  timeout/audio false-trigger rates.

A new physical **diagnostic** capture is justified after the useful offline
reconstruction and frozen holdout work. Keep CURRENT ranking unchanged. Use
20 deliberate shots: ten spread across light/dark and low-contrast regions,
plus five pairs of nearby sequential impacts at normal reload pace. Label each
new impact separately; record event mapping if false audio triggers occur.
The purpose is to verify complete capture/ownership and observe real nuisance
and impact evidence before designing another research policy.

Run the following on the shooting machine later, with the application initially
closed. Replace NEW_SESSION with the session printed by `start`; use fresh
output names. These commands were documented, not executed against hardware
or user settings during this investigation.

```bash
python3 -m automation.physical_test start --prepare-only
python3 main.py
```

After the shots, close the application to flush capture. In a terminal:

```bash
python3 -m automation.physical_test check --root content/ai/physical_traces/NEW_SESSION
python3 -m automation.physical_test label --root content/ai/physical_traces/NEW_SESSION
python3 -m automation.physical_test evaluate \
  --root content/ai/physical_traces/NEW_SESSION \
  --output evaluation_runs/physical_NEW
python3 -m automation.track_survival_replay \
  --root content/ai/physical_traces/NEW_SESSION \
  --output evaluation_runs/physical_tracks_NEW
python3 -m automation.physical_test stop --root content/ai/physical_traces/NEW_SESSION
```

Require healthy persistence, matching source/frozen hashes and complete exact
replay before interpreting accuracy. Inspect capture and worker latency before
assuming instrumentation has negligible live timing cost. The next engineering
hypothesis should isolate registered, polarity-aware localized novelty anchored
to the same observation; there is no authority change approved by these results.

## Verification and local commit boundaries

Required suites passed: `overnight_selftest` (15), `evaluation_selftest` (13),
`physical_trace_selftest` (4), `async_track_timing_selftest` (3).
Additional suites passed: `track_audit_selftest` (15),
`synthetic_track_selftest` (7), `causal_candidate_selftest` (10),
`pre_spatial_mapping_selftest` (5), `physical_trace_label_selftest` (6),
`physical_trace_target_selftest` (4), `physical_trace_stress_selftest` (3),
and V2.22.1–V2.22.6 / V2.25.1–V2.25.3 selftests. Exact replay checks also run under `python -O`;
legacy reconstruction explicitly refuses that mode because its saved-state
checkpoint assertions must stay enabled (one legacy reconstruction test is
therefore explicitly skipped under `-O`; the other fourteen pass). All 17 changed/new Python files compile; `git diff --check` passes.

The expanded checks found a pre-existing V2.22.3 test fixture mixing saved
machine calibration at snapshot time with identity geometry at evaluation time.
The fixture now explicitly uses the same identity plane for both. Calibration,
settings and object-shadow runtime behavior are unchanged.

Local commits separate complete tracing/replay (`139c0b4`) from the proven
consumed-result ownership fix (`53781fb`) and track-history ownership guard
(`7d7d1a4`), followed by the verified reconstruction,
synthetic research, regression and documentation work. No push, merge, physical
trace, generated evaluation output or `content/ai/settings.json` staging occurred.


## Source/test file inventory

- `automation/physical_trace_export.py`
- `automation/synthetic_track_research.py`
- `automation/synthetic_track_selftest.py`
- `automation/track_audit_benchmark.py`
- `automation/track_audit_selftest.py`
- `automation/track_survival_replay.py`
- `automation/v2223_selftest.py`
- `src/engine/camera/hit_scanner.py`
- `src/engine/offline/track_replay.py`
- `src/engine/physical_trace.py`
- `src/engine/shot_async_v2224.py`
- `src/engine/shot_cross_thread_novelty_v253.py`
- `src/engine/shot_fast_v2225.py`
- `src/engine/shot_region_freshness_v252.py`
- `src/engine/shot_region_proposal_v251.py`
- `src/engine/shot_track_v2226.py`
- `src/engine/track_audit.py`

Permanent documents updated: this report, `TRACK_SURVIVAL_AUDIT.md`,
`FAST_SELECTION_AUDIT.md`, `SCORE_ROOT_CAUSE.md`, `CAUSAL_CANDIDATE_AUDIT.md`,
`CURRENT_STATE.md` and `ARCHITECTURE.md`. README is unchanged.
