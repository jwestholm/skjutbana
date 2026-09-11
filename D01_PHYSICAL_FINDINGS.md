# D01 physical accuracy and clean-change findings

## 2026-09-10 continuation: exact losses, bounded alternatives, no promotion

**The oracle/selection gap is not closed. 100% physical correctness remains the
target; 95% is the minimum, and neither is achieved.** The only live-path change
in this continuation is opt-in hybrid-merge diagnostics. Candidate scores,
coordinates, quota rules, association, ranking and authority are unchanged.
No new central subsystem, gameplay shortcut, weapon classifier or display mode
was introduced. The canonical challenger remains SHADOW-only with its existing
model/hash; new models are explicitly offline research artifacts.

Starting checkout: `codex/score-root-cause`,
`0e095b9e2a58e8eea6477c1d9ccb018bef4678d4`. All continuation outputs are new files
under `/data/skjutbana/evaluation_runs/D01_evidence_20260910_081143/`.
The September 9 findings below remain the preserved baseline; the exact event-3
loss and D02 recommendation in this section supersede their earlier uncertainty.

### What the new measurements do and do not establish

- CURRENT repeats exact recorded-input selection for all 39 complete decisions:
  D01 6, S01 10, S02 13, POST_FIX 10. S01 event 11 is still incomplete and excluded.
- Research fits use the existing primary sessions H10/H20/S01/POST_FIX; an
  evaluated primary session is entirely excluded from its fit. D01/S02 never
  enter fitting. Development inspection still means these are not independent
  physical validation results.
- OBSERVATIONS32 adds at most 32 distinct recorded coordinates, at most three
  per eligible parent. It preserves the current XY, checks producer/cutoff and
  independently recomputes the existing offline 3-frame/90-ms support rule.
  It does not inherit local confirmation from a different coordinate. Existing
  association history suffices; no new track class is needed.
- CONTOURS32 adds a separate fixed 32-budget raw-contour rescue, with unchanged
  common verifier. LEGACY150_PLUS50 instead tests the same total 200 retention
  cap with 150 score-head candidates and 50 spatial/size-stratum reserves.
  These are separate hypotheses, not a combined D01-tuned live algorithm.
- The cap experiment reconstructs the saved first candidate frame. Events 1–5
  match all 200 ordered cleanup-input XY/scores. Event 6 matches 199/200 and is
  explicitly partial: one off-GT proposal differs, with cause unresolved.
  Capture configuration was not fully saved; reconstruction uses hashed checkout
  configuration. Exact output matching establishes equivalence for those stages,
  not raw-frame asynchronous replay of the entire detector.
- `retention_v1` is preserved but its CURRENT summary is invalid: a generic zero
  rejection threshold was incorrectly applied to negative rank keys. The driver
  now uses an ordering-only threshold and asserts exact CURRENT winner equality.
  `retention_final` is authoritative; `retention_v2` repeats the same results.
  Both channel runs also repeat identical non-timing results.

### Six physical regression cases

| Event | Exact traced loss / result | Nearest current → OBS32 pool XY px | CURRENT error → COMMON_FIXED / OBS32 error px | Relevant rank before → OBS32 |
|---|---|---:|---:|---|
| 1 light_flat | Good original observation becomes same-frame support; score-selected representative remains worse | 12.835 → **2.892** | 466.489 → 279.900 / 279.900 | Current representative 50; good @5 coordinate absent → 12 |
| 2 dark_flat | Protected CURRENT positive; common physical ranking loses it | 8.340 → 8.340 | **9.146** → 197.077 / 197.077 | Selected positive 1 → best @10 positive 89; **regression** |
| 3 printed_line | Legacy point removed by **hybrid_capacity_exhausted** after source quota selection | 15.383 → 15.383 | 201.816 → 681.164 / 681.164 | Best @20 positive 90 → 5, still not selected |
| 4 old_hole_nearby | Explicit local confirmation survives; incompatible source score ranking loses it | 4.781 → 4.781 | 427.863 → **4.781 / 4.781** | Nearest good track 49 → 1 |
| 5 low_contrast | Useful raw ranks 228/276/279 fail legacy 200 cap; tested small reserves do not rescue them | 63.134 → 63.134 | 183.203 → 81.930 / 81.930 | No eligible @42 positive before or after |
| 6 repeat_grouping | Explicit local confirmation survives; final source-score ranking loses it | 5.968 → 5.968 | 198.629 → 693.308 / 693.308 | Nearest good track 77 → 16 |

Event 1: track 48 starts at `(1980.5,1318)`, 12.835 px from GT, score 6.256743.
The useful `(1969.5,1322)` observation is 2.892 px from GT and survives cleanup
with score 1.569889. Its distance to the track is 11.705 px, within association's
12 px radius. Same-frame support neither adds an independent temporal hit nor
replaces XY. Later local confirmation at the worse coordinate has score
10.513423; local confirmation at the good coordinate has score 5.883148 and again
only supplies same-frame support. The track is eligible, rank 50, behind track 1
with score 24.718453 at the same 0.015308-second onset. This is lossy coordinate
collapse, not evidence that the observation was deleted: the existing audit
retains it. The bounded alternative recovers its @5 availability but not top-1
selection. Neither “best physical-change” nor “best spatial” has a validated
definition yet; preserving measured alternatives is the smallest research step.

Event 2: CURRENT track 97 has error 9.145695 and rank 1, score 19.630435;
the nearer 8.339725 track 180 ranks 73. All common/centered-PRE/simple-confirmation
variants tested here lose the CURRENT @10 success. This explicitly fails the
protected regression gate, irrespective of comparator averages.

Event 3: the `(2512.5,1312.5)` contour, 7.327634 px from GT, is legacy pre-limit
rank 171 (`legacy:170`), score 7.654583. It has no spatial-agreement operation:
it is neither deduplicated nor geometrically replaced. 200 legacy + 180 FAST
inputs yield 363 merged candidates after 17 agreements. Its merged rank is 307,
legacy-pool rank 174. The 50-slot V2 reserve then 150 additional legacy selections
fill all 200 places, leaving no global-fill capacity. The final set has 160 V1
and 54 V2 provenances, with 14 overlaps. The new ledger identifies the exact
operation, parameters, input identity, merged rank and absence of output rank.
This is the existing intended quota policy losing a useful weak contour, not
an accidental cleanup deletion that justifies removing a filter blindly.

Event 4: good track 349 has center/ring/compact/darkening confirmation
14.769/4.000/10.769/7.077 versus winner 297's
4.615/1.454/3.161/3.538. Yet its best score is 6.280923 versus 19.078257 at tied
0.011163-second onset. The good coordinate reaches rank 1 under the frozen common
verifier. V2 was in its pre-15-ms waiting path on the reconstructed first frame,
so missing V1/V2 provenance is reported UNKNOWN, not guessed. No old-hole
exclusion explains this loss and no old-hole rule was added.

Event 5: raw contour count is 310; nearest legacy/hybrid retained coordinate on
the saved frame is 58.170 px, worsening to nearest eligible 63.134 px. The earlier
256 rescue contains an 11.657 px coordinate at rescue rank 171; shrinking to 32
loses it. The 150+50 fixed-cap trial also leaves nearest 58.170 px. Geometry/size
diversity alone is insufficient here. No increase from 200 to a huge live pool
was made, and these failed trials were not iteratively retuned to D01.

Event 6: good track 554 has center/compact/darkening 6.077/3.616/1.077 versus
winner 486's 4.846/2.774/0.615, but best score 5.377161 versus 21.279051.
The nearest good track ranks 77 in CURRENT, 12 in COMMON_FIXED and 16 after
observation expansion. Repeated shots remain supported; proximity to existing
damage was not made a rejection condition.

### Actual selection, oracle, error tails and comparator regressions

Tuples are **@5/@10/@20/@42 camera pixels**. Oracle here means each method's
evaluated coordinate pool; it is not selection or new physical validation.
CURRENT causal proposal oracle remains D01 2/4/5/5, S01 6/6/7/9,
S02 3/3/3/5, POST_FIX 7/7/9/9. The smaller CURRENT eligible oracle is below.

| Session (physical n) | Method | Selected | Pool oracle | Mean / median / nearest-rank P95 px | >100 px |
|---|---|---|---|---|---:|
| D01 (6) | CURRENT | 0/1/1/1 | 1/3/5/5 | 247.858 / 200.223 / 466.489 | 5 |
| D01 | COMMON_FIXED | 1/1/1/1 | 1/3/5/5 | 323.027 / 238.489 / 693.308 | 4 |
| D01 | OBSERVATIONS32 | 1/1/1/1 | **2/4/5/5** | 323.027 / 238.489 / 693.308 | 4 |
| D01 | CONTOURS32 | 1/1/1/1 | 2/4/5/5 | 303.801 / 180.810 / 693.308 | 4 |
| S01 (9) | CURRENT | 2/2/3/3 | 6/6/7/9 | 204.109 / 205.897 / 463.951 | 5 |
| S01 | COMMON_FIXED | 5/5/5/5 | 6/6/7/9 | 162.297 / 4.300 / 463.951 | 4 |
| S01 | OBSERVATIONS32 | 5/5/5/5 | 6/6/7/9 | 162.297 / 4.300 / 463.951 | 4 |
| S01 | CONTOURS32 | 4/4/4/4 | 6/6/7/9 | 220.919 / 227.318 / 578.617 | 5 |
| S02 (10) | CURRENT | 1/1/1/1 | 3/3/3/5 | 187.926 / 163.723 / 426.564 | 7 |
| S02 | COMMON_FIXED | 3/3/3/3 | 3/3/3/5 | 143.010 / 126.152 / 385.398 | 7 |
| S02 | OBSERVATIONS32 | 3/3/3/3 | 3/3/3/5 | 143.010 / 126.152 / 385.398 | 7 |
| S02 | CONTOURS32 | 3/3/3/3 | 3/3/3/5 | 162.253 / 148.431 / 385.398 | 7 |
| POST_FIX (10) | CURRENT | 2/2/2/2 | 6/6/9/9 | 171.099 / 135.749 / 464.701 | 8 |
| POST_FIX | COMMON_FIXED | 3/3/3/3 | 6/6/9/9 | 195.744 / 165.108 / 490.276 | 7 |
| POST_FIX | OBSERVATIONS32 | 3/3/3/3 | **7/7/9/9** | 195.744 / 165.108 / 490.276 | 7 |
| POST_FIX | CONTOURS32 | 3/3/3/3 | 6/6/9/9 | 195.744 / 165.108 / 490.276 | 7 |

Regressions must not be hidden by the aggregate: COMMON_FIXED/OBS32 lose D01:2
(9.146→197.077), S01:3 (17.546→227.318) and POST_FIX:4 (3.290→490.276).
CONTOURS32 additionally loses S01:2 relative to COMMON_FIXED (3.116→270.544),
and worsens S02:11 from 102.194→294.623. All per-event selections, distances,
positive ranks, counts and timings are in `retention_final/results.json`.
No implemented live selection changes means live before/after is CURRENT/CURRENT.
The earlier best D01 flow result remains 2/2/2/2, with oracle 3/5/6/6; nothing
here supersedes its selection count or makes that rejected flow variant safe.

The separate `observation_readiness_audit.json` checks every complete physical
event again: OBS32's pool and ready-coordinate oracles agree at all four radii
in all four sessions. D01's extra ready counts are 11/11, 25/25, 15/16, 15/18,
16/16 and 20/20. Thus the restored event-1 coordinate passes the stated offline
support probe, although that probe still is not regenerated live confirmation.

### Bounded retention and runtime

Counts include false events because they also cost work. COMMON_FIXED has the
same pool as CURRENT. Additional proposal timings include the offline support
probe; feature timings exclude common context loading and base feature extraction.

| Session | CURRENT mean/max count | OBS32 mean/max | CONTOURS32 mean/max | OBS32 proposal + extra features ms | CONTOURS32 proposal + extra features ms |
|---|---|---|---|---|---|
| D01 | 97.83/109 | 115.50/134 | 118.00/130 | 4.67 + 54.89 | 15.49 + 62.50 |
| S01 | 68.30/100 | 75.10/110 | 87.90/118 | 1.78 + 20.19 | 8.87 + 58.95 |
| S02 | 130.77/146 | 162.31/178 | 147.69/170 | 7.56 + 99.12 | 12.19 + 52.75 |
| POST_FIX | 97.30/107 | 116.20/136 | 113.20/125 | 4.30 + 57.22 | 8.63 + 48.78 |

D01 raw counts are 203/307/262/295/310/317. Every legacy and reconstructed hybrid
output stays at 200. LEGACY150_PLUS50 takes 0.48–1.10 ms and does not improve any
event's nearest retained coordinate; first-frame hybrid oracle stays 2/4/5/5.
Because this stage pilot fails its D01 recall objective, it was not promoted or
run as a regenerated asynchronous comparator selector. Its final selection and
comparator effect are **unmeasured**, not claimed unchanged from matching oracle.

Hybrid trace on/off outputs also equal the HEAD implementation on 200 generated
fixture comparisons and the reconstructed D01:3 inputs. Forty-sample warm
microbenchmarks: HEAD median/P95 15.29/16.33 ms, changed trace-off 15.46/15.83 ms,
trace-on 16.66/17.22 ms. Event-3 ledger JSON is 456,072 bytes. This opt-in capture
cost excludes serialization/disk I/O and does not establish live latency parity.
Numerical fixtures are software tests, not physical validation.

### Clean change and independent local confirmation

An isolated centered-PRE ablation distinguishes temporal variation from stable
PRE-reference bias. It centers registered PRE residuals around their own median
before RMS noise estimation, preserving the original function as baseline.
It uses the same soft attenuation, fixed candidate pools and logistic method;
there is no global edge blacklist, threshold increase or GT-dependent retention.

| D01 map | Raw mass retained | GT-local mass retained | Mean S/B |
|---|---:|---:|---:|
| Raw (preserved baseline) | 100% | 100% | 2.969 |
| Background correction | 82.15% | 89.39% | 3.273 |
| Persistence | 72.09% | 81.49% | 3.410 |
| PRE RMS baseline | 32.09% | 40.06% | 3.788 |
| **Centered PRE (new)** | **43.44%** | **55.84%** | **3.918** |
| Crop-half compensation | 40.01% | 54.32% | 4.076 |
| Broad flow | 15.58% | 14.34% | 2.679 |

Centered PRE leaves nonzero GT evidence in all six shots and suppresses 56.56%
of raw change, but changes **none** of the D01 selected coordinates. Fixed-pool
PRE_RMS_BASELINE → PRE_CENTERED selected tuples: D01 1/1/1/1→1/1/1/1;
S01 **3/3/3/3→2/2/2/2**; S02 3/3/3/3→3/3/3/3;
POST_FIX 2/2/2/2→2/2/2/2. Pool oracles/counts remain the CURRENT eligible values.
S01 mean/median/P95/>100 worsens 234.884/274.908/463.951/6 →
317.964/324.745/752.024/7. D01 both are 439.259/459.550/693.308/5;
S02 both 143.010/126.152/385.398/7; POST_FIX both
202.376/196.707/472.558/8. All four known S01/S02 false events remain accepted.
S02 S/B also falls from 1.633 to 1.548. Thus better D01 map appearance/GT retention
does not justify a live change.

Centered map construction averages D01 647 ms, S01 538 ms, S02 626 ms,
POST_FIX 587 ms; corresponding causal-context loading/registration averages
850/419/831/504 ms and extra feature extraction 25/17/33/27 ms. These are offline
Python measurements with cache/scheduling variation, not a live latency claim.
`channels_final/report.json` records all event maps, retained masses, timings,
selection errors and comparator results.

Two separate deterministic rankings use the recorded local-confirmation compact
inner-minus-ring signal or darkening, with no source score. Compact ranking
selects D01 0/0/0/0, S01 2/2/3/3, S02 1/1/1/1, POST_FIX 1/1/1/1.
Darkening selects D01 0/0/0/0, S01 1/1/2/3, S02 0/0/0/0, POST_FIX 0/0/0/0.
Full error tails/per-event rows are in the same report. Stronger single local
scalars are insufficient; event 4's useful local proof does not imply a universal
compactness winner. Both also lose protected event 2 and accept all known false
events. No hardcoded old-hole/repeated-shot behavior was used.

### Board motion and NO_IMPACT

Existing global registration, overlapping left/right crop-half registration,
local flow and their temporal summaries were reused. The centered channel does
not establish panel compensation. D01 lacks capture-time physical Board Space
and seam calibration; current settings were not substituted. Earlier crop-half
S/B improvement remains a camera-crop result; exact panel/seam attribution is
unavailable. Broad flow remains rejected for severe GT attenuation and comparator
regression.

A new deterministic nearest-centroid classifier uses standardized motion-only
features (global/half transform differences, temporal first/last/mean/variation),
primary-session fitting and a 3×3 normalized **crop proxy**, not physical Board
Space. Two hundred train-label permutations provide an empirical null:

| Held-out/development session | Correct cell | Training-majority baseline | Uniform expected | Permutation mean correct / tail p |
|---|---:|---:|---:|---|
| D01 | 0/6 | 3/6 | 0.67 | 0.855 / 1.000 |
| S01 | 6/9 | 5/9 | 1.00 | 2.035 / 0.010 |
| S02 | 1/10 | 4/10 | 1.11 | 1.900 / 0.876 |
| POST_FIX | 3/10 | 8/10 | 1.11 | 2.265 / 0.383 |

The isolated S01 result does not generalize. Motion-alone localization remains
unsupported; the earlier D01 ridge result merely tied majority at 3/6. No motion
prior or persistent/adaptive model was installed.

NO_IMPACT remains unresolved. S01 event 4 and S02 events 2/8/10 are accepted by
CURRENT and every tested alternate here. The new comparisons are ranking
experiments with the existing default zero probability threshold, not newly
calibrated event-rejection gates. Their CURRENT winners explicitly have
`v2225_local_confirm=1`, with compact signals 5.844/2.473/5.613/3.895 and darkening
5.846/2.000/0.615/10.077 respectively. Local confirmation is therefore not proof
of a physical shot, much less an emitted `state=confirmed`. Candidate pools have
71/146/133/112 eligible coordinates. A source-independent selection score alone
does not supply an event-level “nothing impacted” alternative. Existing visual
gates already reject genuine shots when tightened; timing-only rejection would
invalidate rapid real shots. D01 has no new negatives; no events were relabeled
or invented. Missing observed sound-only versus motion-only controls and capture
geometry limit attribution, not the demonstrated false acceptance itself.

### Architecture, next capture, reproducibility and verification

The expanded current/history inventory and smallest consolidations are in
[PHYSICAL_BOARD_STATE_ARCHITECTURE.md](PHYSICAL_BOARD_STATE_ARCHITECTURE.md).
Reuse HitInput/ArUco/AnalysisGeometry/WorkingSpaceMap, current references/holes,
the existing audio waveform settings scene and GameObject lifecycle. No missing
central replacement was found that is necessary to do this accuracy work.
Canonical priorities are updated in ROADMAP/ARCHITECTURE and AGENTS.

**Revised smallest D02: six diagnostic events, not created or captured here.**
Existing D01 already supplies six positive regression cases; the next uncertainty
is separating weak true change from acoustic and panel-motion false events.
One weak-flat shot and one printed-line/near-hole shot are initial positive
controls; four negative controls separate timing context and two motion regions.
This reduces the previous eight-event proposal's duplicated physical shots,
not its already scarce negative controls. It is a discrimination pilot, not
training sufficiency, an accuracy estimate or 95% acceptance evidence.

1. Stable-board sound-only control, isolated: acoustic trigger without impact.
2. Low-contrast physical shot: weak persistent change versus PRE variability.
3. Stable-board sound-only control following that shot: distinguish lingering
   change from new impact without imposing a fixed minimum shot interval.
4. Printed-line/near-existing-hole physical shot: real edge/old-damage change
   must survive the same physical verifier. Label exact actual XY.
5. Observed left-panel board movement with no new damage: structural false change.
6. Observed right-panel/seam movement with no new damage: whether regional motion
   compensation generalizes beyond a single direction/region.

Keep support, lighting and geometry fixed, independently log every physical
action/control and preserve UNKNOWN when observation is uncertain. A control
without an audio trigger still needs an explicit observation/capture mechanism;
do not silently call a missing trace a true negative. Before any capture, verify
that the actual H/inverse, frame orientation, physical bounds/seam and
viewport/content/projected-image revision are saved; this provenance is currently
missing from D01 and is not implemented by this documentation. Start/end stable
baseline observations are metadata, not invented labeled events. Add repeats
only after this pilot identifies the unresolved mechanism.

Reproduce into **new** output directories (existing outputs are refused):

```bash
python3 -m automation.evidence_retention_research \
  --source /data/skjutbana/evaluation_runs/accuracy_95_100_20260909_174137/dataset_history_early \
  --source /data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/dataset_history_early \
  --output /data/skjutbana/evaluation_runs/D01_evidence_reproduction_retention
python3 -m automation.evidence_channel_research \
  --source /data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/clean_change_v2 \
  --output /data/skjutbana/evaluation_runs/D01_evidence_reproduction_channels
```

Each final run saves protocol, source dataset hashes, fitted parameters,
per-event results and trace/GT hash checks. Final review adds model/training/source
hashes in each `review_manifest.json`; `completion_manifest.json` records compiled
files, checks and the verified existing canonical SHADOW model pins.
`retention_final/forensics.json` holds
the complete relevant track histories; `cap_reconstruction.json` contains input
identities, merge operations, both quota ablations and exact-match limits.
`repeatability.json` confirms identical repeated non-timing results.

Broad verification: **36 suites, 214 unittest cases plus 168 explicit checks,
all PASS**. Exact suite names/counts/logs are preserved in `tests/results.json`
and runnable `run_checks.py` in the continuation directory. This includes
measurement, trace, causal timing, no-impact truth, finalization/reset/adversarial
workflow, PRE spatial mapping, V2.22/V2.24/V2.25, clean change and the new
coordinate/retention tests. The 200 additional generated differential comparisons
prove diagnostic output equality against HEAD, not physical accuracy.
All eight changed/new Python files compile and `git diff --check` passes.
Binding-based finalization rechecked all six D01 mappings into a new external
`D01_finalization_recheck.json` with quality PASS; prior reports were preserved.

`integrity_after.json` verifies all **4,954** initial file hashes, no additions
under the four explicit session roots, unchanged settings and unchanged trace
symlink target. This inventory did not enumerate/open S03. No D02 plan/session,
commit, push or merge was made. The initial local settings modification and
untracked physical-trace symlink remain user state and must not be staged.

## Preserved 2026-09-09 baseline findings

**100% correct emitted physical hits is the target. 95% is the minimum. Neither
has been achieved. CURRENT authority and both existing shadows are unchanged.**

Work continues from `63f3372` on `codex/score-root-cause`. D01 is development data,
not independent validation of the experiments selected during this pass.
All generated data, failed attempts, diagnostics, parameters, input hashes and
test logs are preserved under:

`/data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/`

## Session, human observations and integrity

D01 is `session_20260909_194947_D01_60806ff2`, explicitly bound by
`evaluation_runs/D01_binding.json`. There are exactly **six captured events and
six physical shots**, with precise manual GT. Event 1–6 map to planned shots
1–6: light_flat, dark_flat, printed_line, old_hole_nearby, low_contrast,
repeat_grouping. No labels or captured artifacts were changed.

The existing `D01_finalized.json` was already FINALIZED. It was inspected and
preserved. The new production workflow independently generated the same semantic
mapping as `D01_labels.json` and the existing finalization. New reports are
`D01_finalization_verified.json` and `D01_finalization_final_check.json` under the
run directory. Trace, label, frame, replay-readiness, patch and temporal quality
all **PASS**, with resolved IDs `[1,2,3,4,5,6]` and no reasons.

Before event 1 the user accidentally bumped a mirror, producing a “clonk”. The
user reports that the runtime log appears to contain no separate captured event
for it; the session contains only the six shot events. Event 1 was a real shot
with hard/direct wall support and noticeably stronger echo/resonance. Events
2–6 were real shots with the user's hand between the rifle/support and wall.
These observations are preserved as human context, not definitive audio classes,
weapon labels or a causal explanation of any detector failure.

D01 has **no intentionally collected no-impact examples**. The mirror bump is
not a labeled negative. Neither PRE noise samples nor numerical test fixtures
are manufactured no-impact validation data.

The initial audit hashed **765 session files plus eight existing input files**:
settings, target image, menu, plan, binding, labels, quality and finalization.
All 773 hashes match after analysis, with no added session files. The external
finalization report additionally contains its own all-artifact hashes. Hashing
proves stability over this pass; there was no earlier independent D01 hash
baseline proving its entire pre-existing history. The physical-trace symlink
still targets `/data/skjutbana/physical_traces`.

CURRENT exact recorded-input replay matches all **6/6** selected tracks and
recorded emissions, including repeated identical replay. Decision digest:
`851dea7850faf509b75895ca5c7b6e92735982ab16302b96cf840f945bbadf42`.
This reconstructs selector predicates/order from recorded inputs, not a rerun of
the live asynchronous detector from raw frames. S01/S02/POST_FIX replay remains
10/10, 13/13 and 10/10 observable decisions respectively; S01 event 11 remains
incomplete and is excluded, not scored as a miss.

## Accuracy and stage distinctions

All tuples below mean **@5 / @10 / @20 / @42 camera pixels**, expressed as correct
events, not percentages. Every D01 accuracy denominator includes all six shots.

| D01 method / stage | Oracle | Selected/emitted | Interpretation |
|---|---|---|---|
| Recorded CURRENT | causal proposals 2/4/5/5 | **0/1/1/1** | Actual preserved physical outputs |
| Existing eligible track representatives | **1/3/5/5** | CURRENT 0/1/1/1 | Association can change which XY represents evidence |
| Frozen common snapshot verifier, current pool | 1/3/5/5 | 0/0/0/0 | No D01 fitting |
| Frozen common early-reference verifier, current pool | 1/3/5/5 | **1/1/1/1** | Recovers event 4, loses CURRENT's event 2 |
| Existing earlier-reference expanded union | 1/3/5/5 | 0/1/1/1 | Extra proposals do not fix event 5 |
| Bounded raw-contour rescue + early common verifier, union | **2/5/6/6** | 1/1/1/1 | Proposal recovery alone is not success |
| PRE variability/persistence + augmented verifier, union | 2/3/5/5 | 1/1/1/1 | Existing physical current coordinates + new bounded proposals |
| Flow-compensated channel + augmented verifier, union | **3/5/6/6** | **2/2/2/2** | Best observed D01 result; research-only; substantial counterevidence below |

CURRENT mean/median/P95 errors: **247.857708 / 200.222893 / 466.488855 px**.
The frozen early verifier's corresponding values are
323.026625 / 238.488568 / 693.307610 px. Its @5 gain is not a general improvement
in D01 errors or @42 accuracy. Best-flow selection recovers events 4 and 5 and
is still only **33.33%**, versus the 100% target and 95% minimum. All six flow
union oracle positives at @20/@42 also have ready candidates under the stated
offline readiness rule; mere candidate existence is still not a correct output.

### Per-event findings

Ranks identify the first positive at @42, including distant positives within
that broad tolerance. Parentheses give the stricter @10 rank where different.
“—” means no eligible positive, not unavailable label truth.

| Event / category | CURRENT error px | Nearest causal / eligible px | CURRENT positive rank | Early verifier error / rank | Best flow error / rank | Useful source and loss |
|---|---:|---:|---:|---:|---:|---|
| 1 light_flat | 466.489 | 2.892 / 12.835 | 50 (@10 —) | 279.900 / 28 | 165.627 / 146 | V1 near-GT contour survives cleanup; same-frame association retains a less accurate representative; final ranking loses |
| 2 dark_flat | **9.146** | 8.340 / 8.340 | **1** | 197.077 / 83 | 452.019 / 20 (@10 39) | Useful FAST and V1; CURRENT succeeds @10, both common research selectors lose it |
| 3 printed_line | 201.816 | 15.383 / 15.383 | 20 (@10 —) | 681.164 / 65 | 681.164 / 96 (@10 295) | FAST/V1; a 7.328 px legacy contour survives its own cap but disappears before hybrid cleanup input |
| 4 old_hole_nearby | 427.863 | **4.781 / 4.781** | 20 (@10 49) | **4.781 / 1** | **3.330 / 1** | Legacy upstream geometry is identifiable, but track source tags say UNKNOWN; local confirmation is explicit |
| 5 low_contrast | 183.203 | 58.170 / 63.134 | — | 81.930 / — | **2.188 / 1** | Three useful legacy contours are lost at the 200-candidate cap; flow/contour rescue can recover a coordinate |
| 6 repeat_grouping | 198.629 | 5.968 / 5.968 | 76 (@10 77) | 693.308 / 12 | 266.379 / 26 (@10 27) | V1 survives cleanup/local confirmation but loses final ranking |

The complete tables at all four radii, scores, readiness, selected coordinates,
source features and candidate counts are in `forensics_complete/report.json`,
`frozen_D01_initial.json` and `clean_change_v2/results.json`.

### Proven proposal/retention boundaries

- Event 1's 2.892 px V1 contour associates with track 48 at distance 11.705 px,
  inside the existing 12 px merge radius. It is recorded as same-frame support;
  the representative remains `(1980.5,1318.0)`, 12.835 px from GT. Thus causal
  oracle and eligible-XY oracle differ at @5/@10. This is observed association,
  not a proposal disappearance inferred from the old top-eight debug display.
- Event 3's 7.328 px contour is pre-limit rank 171 and survives legacy retention.
  It is absent from the hybrid cleanup input. The exact hybrid union/dedup/cap
  operation is not individually traced, so that narrower loss interval is
  proven but its unique internal cause is unavailable.
- Event 5 has three area-valid contours within 42 px at pre-limit ranks
  **228, 276, 279**, beyond `candidate_limit=200`. The nearest is 11.657 px away,
  area 4. All three disappear from the saved legacy retained pool, before
  novelty/ridge cleanup. This is a proven cap loss. At the GT radius-four patch,
  saved general change has mean 6.64/max 10, but immediate PRE-shot delta is
  zero. These are different references/evidence channels; nonzero static change
  does not establish a new projectile change at the labeled coordinate.
- Every D01 event has a raw contour within 20 px; some are degenerate. The full
  hybrid RAW pool and discarded FAST/V2 coordinates remain UNAVAILABLE. Do not
  claim all channels have complete proposal tracing.

The @42 CURRENT funnel is 6 physical → 5 causal-positive → 5 with eligible,
locally-confirmed representatives → 1 selected. There is one proposal failure
and four oracle-positive selection failures. Emitted `state=confirmed` alone
was never counted as local confirmation.

## Clean physical change experiment

The source-independent channel consumes registered causal camera PRE/POST and
PRE history. No GT, detector score, source identity, old-hole coordinate or
time-after-prior-shot rule enters its maps. Experiments are fixed, sequential
ablations: raw mean absolute residual; local background correction (sigma 12);
signed temporal median/persistence; soft per-pixel PRE-variability attenuation;
then two overlapping crop-half translations or broad Farneback flow compensation.
The motion alternatives each start from the same PRE-variability baseline.

There is no global threshold increase, small-component veto, edge blacklist or
absolute-contrast requirement. Proposal selection uses multiscale inner-versus-
surround support, 4×4 spatial quotas, eight-pixel spacing and a fixed 256 budget.
The readiness probe is the existing research three-frame/90 ms local-contrast
rule. It is not a regenerated live local-confirmation/resolver path.

The quantitative maps cover **71 complete decision contexts: 65 physical and six
existing no-physical events**. Historical primary sessions are H10/H20/POST_FIX/
S01. S02 and D01 never enter fitting; each primary evaluation excludes its entire
session. Common-model parameters/hyperparameters remain fixed. The augmented
comparison uses the same logistic method with 15 extra patch-local map/noise/
edge values. Training-only GT patches never enter evaluation oracle or selection.
Expanded-map proposal tests cover POST_FIX/S01/S02/D01; intermediate background
and persistence ablations evaluate fixed current coordinates only.

### Map retention and separability

Means are per physical event. GT-local means use an eight-pixel disk; background
excludes the 42-pixel GT neighborhood. “Severe” means GT-local mass falls below
25% of raw. This is an explicit diagnostic attenuation criterion, not a runtime
rejection or proof that each residual pixel was projectile evidence.

| D01 channel | Raw mass retained | GT-local mass retained | Mean signal/background | Severe GT attenuation |
|---|---:|---:|---:|---:|
| Raw | 100% | 100% | 2.969 | 0/6 |
| Local background correction | 82.15% | 89.39% | 3.273 | 0/6 |
| + persistence | 72.09% | 81.49% | 3.410 | 0/6 |
| + PRE variability | **32.09%** | **40.06%** | **3.788** | **0/6** |
| Crop-half compensation + PRE variability | 40.01% | 54.32% | 4.076 | 0/6 |
| Broad flow compensation + PRE variability | **15.58%** | **14.34%** | **2.679** | **6/6** |

PRE variability retains nonzero GT-local evidence in all six events, with
individual GT mass retention 37.91%,45.06%,42.25%,39.92%,35.38%,39.84%.
Mean raw change falls 67.91%; that does not mean every removed pixel was noise.
Flow removes 84.42% of total change but disproportionately suppresses GT-local
evidence. A cleaner-looking diagnostic can therefore be worse evidence.

| Session | Raw → PRE-variability S/B | PRE-variability raw / GT mass retained | Severe PRE-variability / flow attenuation |
|---|---:|---:|---:|
| POST_FIX (10) | 3.221 → 8.237 | 16.24% / 40.43% | 0/10 / 10/10 |
| S01 (9) | 2.260 → 5.025 | 18.00% / 40.31% | 1/9 / 7/9 |
| S02 (10) | 1.141 → 1.633 | 24.79% / 33.83% | 1/10 / 8/10 |
| D01 (6) | 2.969 → 3.788 | 32.09% / 40.06% | 0/6 / 6/6 |

Candidate burden did **not** decrease: D01 raw maxima average 8,632; PRE
variability 9,380; crop-half compensation 9,334; flow 9,781. Each spatially bounded
map supplies 256 candidates, giving 353.83 mean union coordinates versus 97.83
current eligible tracks. Low-amplitude fluctuations can form more local maxima
after normalization. Visual cleanup is not candidate-count reduction.

### Proposal and final-selection effects

The following use each channel's augmented verifier on the union pool. They
include all physical events, not just oracle-positive cases.

| Session | Raw union oracle / selected | PRE-variability oracle / selected | Crop-half oracle / selected | Flow oracle / selected |
|---|---|---|---|---|
| POST_FIX | 7/7/9/9 / 2/2/2/2 | 7/7/9/9 / 2/2/2/2 | 7/8/9/9 / 2/2/2/2 | 8/9/10/10 / **1/1/1/1** |
| S01 | 6/6/9/9 / 2/2/2/2 | 6/6/9/9 / 2/2/2/2 | 6/6/9/9 / 1/1/1/1 | 6/7/9/9 / **4/4/4/4** |
| S02 | 3/3/3/8 / 3/3/3/3 | 3/3/3/7 / 2/2/2/2 | 3/3/4/6 / 2/2/2/2 | 3/3/3/8 / **1/2/2/2** |
| D01 | 2/3/5/5 / 0/0/0/0 | 2/3/5/5 / 1/1/1/1 | 3/4/5/5 / 0/0/0/0 | 3/5/6/6 / **2/2/2/2** |

Original recorded comparison baselines remain:

| Session | CURRENT selected/emitted | Causal proposal oracle | Eligible-XY oracle | Frozen early common current-pool selection |
|---|---|---|---|---|
| POST_FIX (10) | 2/2/2/2 | 7/7/9/9 | 6/6/9/9 | 3/3/3/3 |
| S01 (9 complete) | 2/2/3/3 | 6/6/7/9 | 6/6/7/9 | 5/5/5/5 |
| S02 (10) | 1/1/1/1 | 3/3/3/5 | 3/3/3/5 | 3/3/3/3 |
| D01 (6) | 0/1/1/1 | 2/4/5/5 | 1/3/5/5 | 1/1/1/1 |

S01/POST_FIX common results use held-session-out fits. The frozen D01/S02 model
was trained on all primary sessions; these are different fits under the same
method. S02 event 1's prior possible-multiple-change caveat remains unconfirmed
and its native label is unchanged.

**Decisions:** keep PRE variability/persistence as a measured evidence channel,
not an accuracy solution. Keep contour rescue as a bounded recall diagnostic.
Reject broad flow as a promotion candidate despite its D01 2/6 result: GT
attenuation, comparison-session regression and false-event acceptance remain.
Crop-half compensation improves mass separation but fails final selection;
do not promote it. Pure map-strength ranking reaches at most 1/6 D01 and flow
map-strength ranking reaches 0/6 with one unready winner; it is not a verifier.

### Edge context, motion prior and runtime

In D01, median radius-four PRE edge strength is **7.271 at GT versus 4.387 at
CURRENT false winners**. In S02 the relationship reverses (3.355 versus 9.584).
An isolated three-edge-feature ablation still selects 0/6 D01; adding only PRE
noise features selects 1/6. Blanket edge downweighting is unsupported and could
discard real edge hits. The actual label visualization also percentile-stretches
change for visibility; red intensity is not a calibrated impact probability.

The exploratory motion-only ridge test uses normalized recorded crop coordinates,
not physical Board Space. D01 coarse 3×3 predictions are all center: **3/6**, equal
to the training-majority baseline, permutation-tail p=0.353. S01 is 3/9 versus
majority 5/9; S02 3/10 versus 4/10; POST_FIX 3/10 versus 8/10. No defensible
location prior is established. Physical seam, board boundaries, projected-image
identity/transform and calibration are unavailable in these trace contexts;
current settings were not substituted. No motion model changes physical output.

D01 mean single-thread stage timings: context loading/global registration
**799 ms**; background/persistence/PRE variability **534 ms**; additional
crop-half transforms **550 ms** or flow **725 ms**; each 256-proposal extraction
about **106 ms**. Incremental cached feature extraction averages 422 ms for PRE
variability and 687 ms for flow. These are measured research stages, with shared
caches and disk reads; their sum is not a certified standalone/live latency.
Full per-event timing is in `clean_change_v2/measurements.json`.

Diagnostics for all six D01 shots and representative comparator shots are under
`clean_change_v2/diagnostics/`. Each shows PRE, causal POST, raw and cleaned change,
unchanged GT and a close-up; raw/clean share a display scale. Large diagnostics
are not stored in Git. The first extraction stopped on unavailable matplotlib;
its log/output remains preserved. OpenCV-based diagnostics completed in the
separate `clean_change_v2/` run without adding dependencies.

## No-impact, authority and frozen references

D01 supplies no new negative events. The common and augmented union models
continue to accept **S01's 1/1 and S02's 3/3** known no-physical events; the two
historical H20 false events are also accepted. No timing-based rejection was
introduced. The mirror bump has no independent captured/labeled event and is
not counted. New no-impact collection remains necessary.

The canonical challenger remains `mode=SHADOW`, `status=OFFLINE_CHALLENGER`:
model JSON SHA-256 `2078c398f141e8f7f63e52849697f163552165878602510e17408b5e9ce1678e`,
model NPZ `1c58d31d44ec46d7db5f7aa3281ed7c7314c61c1f36f1c3c8475d4282bb10305`.
Frozen confirmation hash remains
`123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f`.
The common early-reference parameter hash remains
`3cd60866944f6de82500e149aec18bc0ca04f3cece014fa9ed1be62dea4db546`.

The existing dataset extractor gained only an opt-in D01 alias; its historical
default cohort is unchanged. Original frozen manifests are preserved. New
`frozen_*_source_recheck.json` manifests explicitly record that additive source
hash change and their parent hashes. Both still reproduce all 13 saved S02
decisions with unchanged digest
`be3439eb0d0ef8fd1de781e9b08ec2912105d142f2a42fe6ffdb125095a7ee87`.
New clean-model parameters remain local research artifacts, never installed.

## Workflow fix and Testtavla preservation

The original integration hole was real: production labeling wrote individual
GT files but finalization required an aggregate JSON that only selftests built.
Binding-based finalization now constructs the manifest from existing GT and
assignments, verifies exact plan/session/class identity, checks fresh quality,
hashes all captured artifacts and refuses any output overwrite. It preserves
NO_PHYSICAL_SHOT semantics and requires explicit ordinal mapping where absent.
Historical `--labels ... --quality ...` remains compatible.

```bash
RUN=$(mktemp -d /data/skjutbana/evaluation_runs/D01_recheck_XXXXXXXX)
python3 -m automation.physical_collection finalize \
  --plan research/physical_capture_plans/D01.json --session D01 \
  --binding evaluation_runs/D01_binding.json --in-capture-order --preview
python3 -m automation.physical_collection finalize \
  --plan research/physical_capture_plans/D01.json --session D01 \
  --binding evaluation_runs/D01_binding.json --in-capture-order \
  --output "$RUN/D01_finalized.json"
```

For non-order mappings use repeated `--shot-map PLANNED=EVENT`; for external human
assignments add `--mapping`. No manually written aggregate file is required.
The report embeds `label_manifest`, fresh `quality` and input hashes. See
[PHYSICAL_CAPTURE_RUNBOOK.md](PHYSICAL_CAPTURE_RUNBOOK.md) and CLI `--help`.

The actual 1536×1024 `assets/images/testtavla.png` and existing direct child
`img_testtavla` of Bilder (`id=images`) were inspected and preserved byte-for-byte.
Its title/description/path/preview/contain settings match the request, with an
additional black background color. Image SHA-256:
`a1bb501adfa862a8f93437dd8adb8ab8c68453de0dd0b6318bcf6d22e7a8d343`.
The image's printed suggestion to collect later no-impact events does not mean
D01 contains them.

The D01 plan is a small reproducible source specification with null runtime
bindings. Since `evaluation_runs/` is ignored and has no tracked plans, an exact
copy is preserved at `research/physical_capture_plans/D01.json`, SHA-256
`b3ae8ac1c02192d5161dab2a4c5e13c96b6fbee67325599edd322caeb1c90338`.
Original plan/binding/reports remain local and unchanged. No runtime binding,
large dataset, generated trace or model is included in the source allowlist.

## Earlier eight-event D02 proposal — superseded by the 2026-09-10 pilot above

**Eight deliberately observed captured events: four physical shots plus four
known no-impact controls. Do not create/capture D02 until D01 is reviewed.**

1. Keep target, lighting, camera/projector position and hand-buffered support
   fixed. Save actual capture-time calibration, viewport/content placement,
   physical bounds and a visible seam reference alongside the session. The
   existing trace's `calibration=UNAVAILABLE` must not silently become board-space
   truth. If this provenance cannot be captured, keep those analyses unavailable.
2. Fire two new shots on a low-contrast flat region and two on a printed-line/
   existing-hole neighborhood. Identify and label each new change immediately;
   for each pair make the second near the first with a separately visible new
   change. Do not repair between the pair. This repeats D01's cap-loss condition
   and tests whether compensation preserves a real nearby/edge impact.
3. Collect two independently observed sound-only events with the board stable,
   and two observed board/seam-motion events with no new damage. Include one
   isolated and one post-shot instance of each type. Count an example only if
   it is actually captured and human-resolved. Record what happened; do not infer
   NO_PHYSICAL_SHOT from missing detector output or timing.
4. Preserve all PRE/onset/POST frames and an independent event log; label any
   uncertain newness UNKNOWN/AMBIGUOUS. Keep runtime ranking frozen. No minimum
   inter-shot rule, forced one-impact mapping or S03 consumption is allowed.

This isolates real weak/new change from movement/structure and ordinary audio
triggers with two physical repeats per condition. D01's single hard-supported
shot cannot establish an acoustic/support effect; varying that now would add a
confound. Eight events provide a diagnostic comparison, not evidence of >=95%
reliability or a calibrated board-motion model.

## Reproduction and verification

Use new directories; never overwrite the recorded baselines.

```bash
export OPENBLAS_NUM_THREADS=1
RUN=$(mktemp -d /data/skjutbana/evaluation_runs/D01_repeat_XXXXXXXX)
BASE=/data/skjutbana/evaluation_runs/accuracy_95_100_20260909_174137
FROZEN=/data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752
python3 -m automation.d01_accuracy_analysis --output "$RUN/exact_and_forensics"
python3 -m automation.accuracy_physical_dataset --sessions D01 --reference snapshot --output "$RUN/dataset_snapshot"
python3 -m automation.accuracy_physical_dataset --sessions D01 --reference history_early --output "$RUN/dataset_history_early"
python3 -m automation.accuracy_contour_rescue --source "$RUN/dataset_history_early" --output "$RUN/dataset_contour_rescue"
python3 -m automation.accuracy_frozen_replay --manifest "$FROZEN/frozen_history_early_source_recheck.json" --evaluation-dataset "$RUN/dataset_history_early" --session D01 --output "$RUN/frozen_early"
python3 -m automation.accuracy_frozen_replay --manifest "$FROZEN/frozen_snapshot_source_recheck.json" --evaluation-dataset "$RUN/dataset_snapshot" --session D01 --output "$RUN/frozen_snapshot"
python3 -m automation.accuracy_frozen_replay --manifest "$FROZEN/frozen_history_early_source_recheck.json" --evaluation-dataset "$RUN/dataset_contour_rescue" --session D01 --output "$RUN/frozen_contour"
python3 -m automation.clean_change_research --datasets "$BASE/dataset_snapshot" "$RUN/dataset_snapshot" --output "$RUN/clean_change"
python3 -m automation.clean_change_analysis --root "$RUN/clean_change" --output "$RUN/clean_ablation_motion"
```

The external-dataset frozen replay command first reproduces the original S02
reference and verifies its artifact hashes, then evaluates D01 with those exact
parameters. It never fits on D01. All three production command results match
the initial independent frozen evaluation semantically. The historical replay
CLI is unchanged; mismatched feature/reference schemas are refused.

Verification: **232 unittest cases and 104 explicit checks across 34 suites,
all PASS**. The new aggregate regression suite has 29 cases, including the exact
six-saved-label CLI flow, duplicate/missing mappings, duplicate coordinate labels,
unknown/ambiguous statuses, malformed JSON/nonfinite values, false-event conflicts,
wrong binding/root, recovery, external assignments, S03 refusal before trace
access, overwrite refusal, preview safety and unchanged session hashes. The 16
clean-change cases cover weak two-polarity impacts, PRE instability, persistence,
edge retention, brightness drift, immutable inputs, coordinate normalization,
causal timestamps, proposal bounds/determinism, GT isolation and motion regression.
Numerical fixtures test software; physical measurements use only recorded labels.

| Test suite (`python3 -m automation.<name>`) | Passing cases/checks |
|---|---:|
| physical_collection_selftest | 9 cases |
| physical_workflow_adversarial_selftest | 5 cases |
| physical_label_reset_selftest | 15 cases |
| physical_trace_label_selftest | 6 cases |
| physical_postflight_selftest | 12 cases |
| physical_capture_tools_selftest | 2 cases |
| physical_trace_selftest | 4 cases |
| physical_trace_target_selftest | 4 cases |
| physical_trace_stress_selftest | 3 cases |
| physical_finalize_manifest_selftest | 29 cases |
| track_audit_selftest | 15 cases |
| overnight_selftest | 15 cases |
| common_verifier_selftest | 3 cases |
| async_track_timing_selftest | 3 cases |
| causal_candidate_selftest | 10 cases |
| candidate_root_cause_selftest | 2 cases |
| physical_dataset_selftest | 2 cases |
| registered_impact_selftest | 12 cases |
| temporal_impact_selftest | 5 cases |
| physical_patch_verifier_selftest | 2 cases |
| full_pool_temporal_selftest | 3 cases |
| common_impact_ranking_selftest | 2 cases |
| accuracy_research_selftest | 25 cases |
| proposal_trace_selftest | 4 cases |
| score_comparability_selftest | 2 cases |
| pre_spatial_mapping_selftest | 5 cases |
| clean_change_selftest | 16 cases |
| evaluation_selftest | 13 cases |
| v2221_selftest | 14 explicit checks |
| v2222_selftest | 23 explicit checks |
| v2224_selftest | 24 explicit checks |
| v2225_selftest | 21 explicit checks |
| v2226_selftest | 22 explicit checks |
| motion_residual_selftest | 4 cases |

Exact suites/counts and final compilation/diff/integrity evidence are recorded
in the run's `tests/final_results.json`, `compilation.json`, `diff_check.txt`,
`integrity_final.json` and `commit_reviewed_work.sh`. Nothing is pushed or merged.
Settings and the physical-trace symlink are excluded from the reviewed source
allowlist. The authorized local staging attempt failed because `.git/index.lock`
is on a read-only filesystem. The index remains empty and HEAD remains
`63f337215dee279bf4151d7d47049d12be59d8a2`; this pass is not committed or staged.
The exact guarded script checks the branch, HEAD, empty index and all 25 reviewed
file hashes before staging only those files. Run its read-only `--check`, then
`--commit` outside the sandbox when ready:

```bash
bash /data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/commit_reviewed_work.sh --check
bash /data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/commit_reviewed_work.sh --commit
```

**S03 was not opened, inspected, evaluated, tuned against or modified.** No D02
session or binding was created. CURRENT authority has no accuracy promotion.

## Reviewed source files

The reviewed source allowlist contains 25 files; generated output stays under `/data`.

- `ACCURACY_95_100_RESEARCH.md`
- `ARCHITECTURE.md`
- `COMMON_VERIFIER_ARCHITECTURE.md`
- `CURRENT_STATE.md`
- `D01_PHYSICAL_FINDINGS.md`
- `PHYSICAL_BOARD_STATE_ARCHITECTURE.md`
- `PHYSICAL_CAPTURE_RUNBOOK.md`
- `PHYSICAL_COLLECTION_PLAN.md`
- `ROADMAP.md`
- `SCORE_ROOT_CAUSE.md`
- `assets/images/testtavla.png`
- `automation/accuracy_frozen_replay.py`
- `automation/accuracy_physical_dataset.py`
- `automation/accuracy_research_selftest.py`
- `automation/clean_change_analysis.py`
- `automation/clean_change_research.py`
- `automation/clean_change_selftest.py`
- `automation/d01_accuracy_analysis.py`
- `automation/physical_collection.py`
- `automation/physical_finalize.py`
- `automation/physical_finalize_manifest.py`
- `automation/physical_finalize_manifest_selftest.py`
- `content/menu.json`
- `research/physical_capture_plans/D01.json`
- `src/engine/offline/clean_physical_change.py`
