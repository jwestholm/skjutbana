# Current project state

## CURRENT FACTS — 2026-09-09

- **PROVEN physical validation:** post-PRE session
  `session_20260908_194746_b38de674` has 10 physical shots, 10 audio events,
  zero false events. Causal oracle is 7/10 @5/@10 and 9/10 @20/@42.
  CURRENT remains **2/10**, mean 171.10, median 135.75, p95 464.70 px.
- **PROVEN reconstructed funnel:** 10 physical -> 9 causal proposals @42 ->
  9 tracked -> 9 locally confirmed -> 9 eligible -> 2 correctly selected.
  All seven losses occur at final ranking, at ranks 9–59, outside the debug
  top eight. Complete input replay matches all ten winners and 136 saved
  track checkpoints plus 85 tracking counters. This is recorded-input
  reconstruction, not regenerated
  physical detection or a new physical validation.
- **PROVEN instrumentation:** complete active pools, eligibility/rejection
  reasons, association ledgers and compact source histories are captured.
  CURRENT_EXACT_REPLAY raises on mismatched track ids, coordinates or ranks.
- **PROVEN prior fixes:** PRE camera-plane mapping is fixed; its historical
  residual median was 192.385 -> 1.538. FAST saturation disappeared. Pending
  confirmation boundaries and producer rejection gates are implemented.
  The additional producer-transport gap is fixed: the actual worker result
  list consumed by tracking/confirmation now carries the event tag too.
  Tracks also reject association with a known different producer, preventing
  cross-event XY/hit/best-score history from surviving a later tag change.
  Pending-event regressions cover both 6→7 and 14→15 patterns and delayed
  valid pre-boundary delivery. Fresh physical validation is still required.
- **PROVEN bottleneck:** surviving correct tracks lose final ranking. FAST
  wins all seven oracle-positive failures, but complete-pool FAST exclusion
  still gives 2/10. No slot cap or historic-best-over-current score advantage
  explains these seven losses. Appearance/confirmation discrimination remains
  the selection problem to isolate.
- Previous 20-shot independent validation remains 1/20 @42 for CURRENT and
  both shadows. Audio events 7 and 15 remain NO_PHYSICAL_SHOT. Its causal
  oracle remains 12/20 @42. Historical results and inputs are preserved.
- **RESEARCH_ONLY, rejected for promotion:** fixed local-contrast ranking is
  0/10 on physical development. Synthetic development is 85/90 vs CURRENT
  81/90 @42; protected holdout ties at 141/150 and worsens mean error. The
  510-event development stress run has 450 impacts plus 60 no-impact events.
  CURRENT is 406/450 @42, with 30/60 forced no-impact events becoming ready.
  No AI, canonical challenger, or frozen CONFIRMATION_SELECTION_SHADOW is
  promoted or changed. Frozen confirmation hash:
  `123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f`.
- **Next step:** collect 20 deliberate physical diagnostic shots with the
  complete tracing and corrected producer transport/association; verify exact replay and
  capture latency first. Include light/dark, low-contrast and nearby repeated
  impacts. Keep the existing ranking. The next offline hypothesis should
  examine registered, polarity-aware localized novelty/confirmation; no
  research ranking is ready for a physical promotion trial.
- **Latest offline evidence:** registered PRE→POST residual features reproduce
  saved confirmation values, but physical true tracks and false winners remain
  weakly separable; synthetic separability is much stronger and therefore has
  a clear domain gap. No registered-impact ranker is frozen or promoted.

See [OVERNIGHT_TRACK_RESEARCH.md](OVERNIGHT_TRACK_RESEARCH.md) and
[CAUSAL_CANDIDATE_AUDIT.md](CAUSAL_CANDIDATE_AUDIT.md). Checkpoints below are
historical and do not supersede CURRENT FACTS.

<!-- V2.24.0 GAME_HIT_CONTEXT -->
## V2.24.0 checkpoint

Game-hit context foundation is available. Existing games remain valid without
changes. Future games may return `HitRegion` AABBs in game-local coordinates;
shot-critical snapshotting and game->camera transformation are prepared for the
next local-physical-search stage. No local object-aware hit authority is enabled
in V2.24.0.

<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->
## V2.24.1 checkpoint — object-aware local physical search

V2.24.0 has been physically smoke-tested on the shooting PC: its selftests pass
and the application starts. V2.24.1 now uses frozen camera HitRegions to
constrain the first V2.22.5 FAST physical proposal search. Overlapping regions
are merged after a camera-space safety margin. Existing V2.22.5 PRE->POST local
confirmation remains the physical gate and its FULL-RESCUE remains global.

Games with no HitRegions retain the existing global detector behaviour. No new
AI or game-context hit authority is enabled. Next planned checkpoint is V2.24.2
with a dedicated game-context verification scene before V2.25.0 introduces the
small shared GameObject/Hittable/Breakable/ObjectManager layer.

<!-- V2.24.2 GAME_CONTEXT_TESTSCENE -->
## V2.24.2 checkpoint — dedicated HitRegion testscene

V2.24.1 installed and started successfully after V2.24.0. No physical V2.24.1
shot series was required before proceeding because V2.24.2 provides the scene
needed to exercise HitRegions intentionally.

The new scene is available under Games after running
`python3 -m automation.v242_prepare`. It exposes target/no-shoot/moving/overlap/
edge/outside-region cases plus an EMPTY-regions global-fallback mode. The scene
logs returned HitEvent XY and shows the frozen shot-time region geometry in
cyan for direct visual comparison.

No V2.25 object system or new AI authority is enabled yet. The next decision is
based on physical V2.24.2 results: fix the V2.24 bridge if required, otherwise
proceed to V2.25.0 GameObject/HittableObject/BreakableObject/ObjectManager.

<!-- V2.24.3 LOCAL_ROI_ALIGNMENT -->
## V2.24.3 checkpoint — physical V2.24.2 findings and correction

A physical V2.24.2 run on 2026-09-04 produced nine camera shots. HitRegion
snapshots reported seven game and seven camera regions, the moving target was
classified successfully in two shots, EMPTY/GLOBAL correctly produced
`objects=0`, and shot-critical timing remained healthy. However, no
`V2.24.1 LOCAL-SEARCH` line appeared in the run. Most region shots instead
logged `outside_detector_roi`, and two early frames bypassed the V2.24.1
extractor through CandidateGenerator V2's `waiting_post_peak` path.

V2.24.3 therefore corrects ROI coordinate alignment and moves object-local
search to the whole live HitScanner ROI. It also changes zero local overlap from
a silent global first pass to an explicit V2.22.5 FULL-RESCUE transition.

Next action: run `Hit Context Test (V2.24.3)` with the short physical matrix in
`V243_TEST_PLAN.md`. Do not start V2.25.0 until normal region shots visibly use
the local ROI path and returned XY matches the physical holes reasonably.

<!-- V2.24.3 LOCAL_ROI_INTEGRATION_FIX -->
## V2.24.3 checkpoint — local ROI integration fix

The first V2.24.2 physical test verified game->camera snapshots, moving-target
classification, overlap/no-shoot semantics and EMPTY/global compatibility, but
it also exposed that the intended V2.24.1 local-first detector was frequently
bypassed. Most shots logged `outside_detector_roi` and no V2.24.1 LOCAL-SEARCH.

V2.24.3 fixes the implicit viewport-local content-rect origin, refreshes HitInput
calibration before the shot snapshot and applies the frozen object mask at
HitScanner ROI level. FULL-RESCUE remains global. The V2.24.2 testscene is
retained and upgraded with faster movement plus frozen/current motion distance.

Next action: repeat the short physical Hit Context Test matrix. Only after clean
local-ROI results should development proceed to V2.25.0 reusable objects.

<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->
## V2.24.4 checkpoint — detector working-space ROI mapping

The V2.24.3 physical run isolated the remaining integration bug: HitRegions
were present and transformed, but every region-enabled shot logged
`region=0.0% overlap=0`. V2.22.1 intentionally executes detection on a smaller
crop-local image and restores full camera coordinates afterwards; V2.24.3 fed
full-camera AABBs directly into that local mask.

V2.24.4 maps full-camera HitRegions to the active analysis working space using
`AnalysisGeometryV2221` crop origin and actual work/crop scale. The testscene
and logs now expose full-frame size, crop rectangle, work size, scale and an
example camera->work region mapping. V2.25.0 remains gated on one clean physical
acceptance run of this bridge.

<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->
## V2.25.0 checkpoint — reusable game objects

V2.24.4 is accepted as the game-context bridge. V2.25 introduces a stable
`src.engine.game_objects` package with exact shapes, entity/part identity,
gameplay projectile penetration, layered durability, event/reaction handling,
effect requests and ObjectManager resolution against PANG-time snapshots.

A shot-id bridge fixes the prior `event_shot=None` limitation without replacing
HitInput or changing detector authority. Continuous object motion during the
shot-critical wait remains a later checkpoint.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 checkpoint — region-balanced physical hit proposals

V2.25.0 GameObject `shot_id` and frozen collision were physically verified. Detector
XY remained the blocker: five widely separated real shots were selected in one small
camera area. V2.25.1 adds balanced per-region proposal/confirmation on top of the
accepted V2.24.4 working-space mapping. Physical acceptance is pending a repeat of
the five-object test.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 checkpoint — registered PRE→POST authority

V2.25.0 shot-id/frozen GameObject resolution is physically verified. V2.25.1 region
proposal balancing is physically verified but did not correct final XY because local
authority still leaked through early/legacy candidates. V2.25.2 closes that authority
boundary. Physical five-shot acceptance is pending.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 checkpoint — cross-thread novelty authority

V2.25.0 shot_id/frozen GameObject resolution is physically verified. V2.25.1 per-region
proposal works. V2.25.2 registered evidence runs physically but its readiness bridge
was instance-local and its freshness gate remained too permissive. V2.25.3 fixes both
at the authority layer; five-shot physical acceptance is pending.

<!-- V2.25.3-r3 FULL_FILE_DELIVERY -->
## V2.25.3-r3 – full-file delivery

Packaging-only correction after the V2.25.3 runtime work. Future delivery for this development line uses complete replacement files only: no prepare/apply scripts and no menu/settings mutation helpers. `content/menu.json` is shipped as a complete schema version 1 file with the diagnostic games already present. Central configuration files are not replaced unless the version actually requires a source change.


## Evaluation foundation — measurement only

A versioned stage-observation scorecard, JSON/terminal CLI, provenance manifests,
and measurement selftests are available; see [EVALUATION.md](EVALUATION.md).
Historical V2.23 framepacks can measure saved candidate-pool recall at separately
reported 5/10/20/42 camera-pixel tolerances. They do not record complete stage
boundaries, effective producer settings/models/calibration, or final emissions.
Unavailable stages remain explicit. No detector behavior, parameters, models or
ranking changed. Full live-path replay and V2.25.3 physical acceptance remain
pending; projected F2 data is not physical validation.

The initial existing-data run covered 101 projected captures (100 F2, one single
projected): saved-pool recall was 1/101 @5 px, 2/101 @10 px, 6/101 @20 px and
26/101 @42 px. POST coverage was 93 with one frame, six with two and two with
three. These are historical snapshot measurements, not V2.25.3 detector results.

## Offline ten-iteration ranker research

Completed ten isolated native-F2 ranking trials on chronological 100-shot TRAIN,
100-shot DEVELOPMENT and 100-shot protected final holdout sessions. No live code,
settings, model registry or detector thresholds changed. Iteration 10 (geometry-only
linear ranker) won only the development MRR20 tie-break: Top1@20 stayed 0/100.
Holdout Top1/Top3/Top10@20 all remained 0/100; MRR20 improved, while Top1@42 regressed
1/100 to 0/100. No primary accuracy improvement or promotion is justified. Candidate
coverage remains the dominant limitation (6/100 holdout @20). See
[OFFLINE_10ITER.md](OFFLINE_10ITER.md); models and full JSON results are isolated under
`evaluation_runs/offline_10iter_20260907/`. V2.25.3 physical acceptance remains pending.

Iteration 10 is NOT approved for live use. All experiment data is projected F2
data and does not establish physical/live performance.

## Overnight physical audit and isolated challenger — 2026-09-07

On `codex/overnight-ai`, the biathlon audit assigns five holes to event groups
`{1,2},3,4,5,6`; event 2 is the likely redundant trigger 100.66 ms after event 1.
Original labels and traces remain untouched; ordinal-to-event mapping is external.
The first pair cannot be acoustically disambiguated without missing audio logs.
The second physical shot (event 3) has explicit local-confirmation evidence near
human GT but loses selection to the first hole. Legacy `state=confirmed` means
emitted; the saved top-eight track view is not a complete confirmation pool.

A frozen, SHADOW-only adapter consolidates the existing V2.23 linear listwise model.
Six historical conditional-ranking trials used chronological whole sessions;
stronger regularization gives only a small development MRR20 tie-break gain,
with Top1@20 unchanged (1/19 oracle-positive). Reused holdout Top1@20 remains 0/6.
No model/config is promoted to live authority. See `OVERNIGHT_AI_REPORT.md`
for the completed session report and caveats when available.

Measurement improvements transport worker pipeline snapshots, capture the actual
local-confirmation output and deterministic emission-boundary selection, and attach
per-event audio thresholds/cooldown evidence. Canonical scoring runs only when
finalizing the trace, with explicit retained-pool and post-decision semantics.
`automation.physical_test` supplies start/check/label/classify/evaluate/stop helpers.
All generated evaluations remain under ignored `evaluation_runs/`.

### Frozen confirmation selection shadow

The accepted development replay formula is now frozen in
`src/engine/ai/confirmation_selection_shadow.py` as
`CONFIRMATION_SELECTION_SHADOW` with status `PHYSICAL_REPLAY_CHALLENGER` and a
configuration hash. Physical traces record it beside `CURRENT_DETERMINISTIC` and
`CANONICAL_AI_SHADOW`; it cannot alter emitted coordinates. The physical-test
evaluator reports overall and conditional selector metrics plus retained-pool oracle
availability. The 10-shot replay remains development evidence; the next labelled
session is the independent validation dataset.

### Detector score root-cause audit

`automation.physical_score_audit` now decomposes the V2 score formula and reports
candidate-source distributions, per-shot selector comparisons, and track-score
evidence without changing runtime behavior. The 10-shot development trace shows
FAST V2.22.5 candidates saturated near 35–38 while genuine V2.6 vault candidates
are commonly below 3; the next research question is source-balanced ordering.
The frozen confirmation selector remains unchanged and shadow-only.

### Async track timestamp correction

A reproduced installer-composition regression let V2.22.6 overwrite V2.22.4's
camera-frame timestamp handoff and its “worker waiting is not a negative frame”
rule. The wrappers now share one timestamp-consumption helper; older ready results
also use the currently installed tracking method. Two failing-before/passing-after
regressions cover delayed-result association and waiting-frame aging. This changes
runtime timing semantics, not detector/audio thresholds or AI authority. Physical
acceptance remains pending; historical traces are not rewritten or claimed fixed.

## Temporal impact evidence — 2026-09-09

- **RESEARCH_ONLY:** temporal forensic extraction over the latest physical frames finds exploratory pairwise AUC about 0.71 for post-impact dark-residual persistence and 0.70 for impact-onset darkening. Before-frame stability alone is weak (about 0.58). The sample is small and not a validation result.
- **NOT PHYSICALLY VALIDATED:** no temporal ranker or synthetic-model change is frozen or promoted. Some true tracks have weak persistence at their saved representative coordinate, so coordinate drift/reference timing remain unresolved.

- **Observation-coordinate audit:** available association histories show only
  limited measurable drift (about 8–10 px in three examples); most physical
  tracks expose one coordinate. Confirmation-search best XY is not preserved
  for most rows, so observation-level temporal ranking remains unvalidated.

- **Final-ranking audit:** live eligible ordering is exactly
  `(onset_distance_to_peak, -track.best_score)`. For every seven @42-positive
  failure, onset is tied and the lower true-track `best_score` loses. Selected
  false-winner coordinates do not recur within 40 px across this ten-shot
  session; no recurrence penalty is justified.
