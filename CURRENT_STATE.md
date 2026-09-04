# CURRENT_STATE.md fixture

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
