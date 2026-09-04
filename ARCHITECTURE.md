# ARCHITECTURE.md fixture

<!-- V2.24.0 GAME_HIT_CONTEXT -->
## V2.24.0 — Game Hit Context

Games may expose an optional `get_hit_regions()` provider. Regions are simple
viewport-local/game-local AABBs. `OverlayScene -> GameScene -> game` proxies the
provider to the shot-critical runtime. At the audio-shot boundary the existing
V2.22.3 object snapshot freezes geometry before normal scene movement.

Coordinate ownership is strict:

`game-local AABB -> viewport/screen -> calibrated camera AABB`.

The engine transforms all four rectangle corners and bounds the result in camera
space; games never provide camera coordinates. The region is search/context
geometry only. Final hit authority still requires physical camera evidence.

<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->
## V2.24.1 — Object-aware local physical search

V2.24.0 freezes optional game `HitRegion` AABBs at PANG and transforms them to
camera coordinates. V2.24.1 consumes those frozen camera AABBs in the live hit
pipeline. Expanded/merged regions are intersected with the detector's existing
valid perspective ROI and constrain only the first V2.22.5 FAST proposal pass.

Authority remains physical: HitRegions are search context, not hit truth. The
existing V2.22.5 PRE->POST local confirmation remains mandatory and its single
FULL-RESCUE pass is explicitly global/unmasked. Missing, invalid or
untransformable context fails open to the pre-V2.24.1 global path.

Runtime layering is therefore:

`PANG -> frozen game context -> local physical proposal -> physical confirmation
-> resolver/HitEvent`, with `global detector rescue` as the fail-safe branch.

<!-- V2.24.2 GAME_CONTEXT_TESTSCENE -->
## V2.24.2 — Game-context verification scene

V2.24.2 adds a diagnostic game module that exercises the V2.24 HitRegion
contract without introducing the V2.25 object engine. The scene provides
stationary target/no-shoot regions, a moving target, overlapping regions, an
edge target, an outside-region challenge and an EMPTY-regions mode.

The scene subscribes to the normal HitEvent path. When a camera/audio hit
arrives it compares final game XY against the frozen shot-time `game_regions`
when that snapshot is available. Frozen positions are drawn temporarily in
cyan, making movement between PANG and delayed HitEvent delivery visible.

This checkpoint changes no detector or game authority. Its job is to expose
transform, shot-time, local-search/fallback and false-attraction behaviour
before V2.25.0 introduces reusable GameObject/HittableObject classes.

<!-- V2.24.3 LOCAL_ROI_ALIGNMENT -->
## V2.24.3 — full-pipeline object-local ROI gate

Physical V2.24.2 testing showed that V2.24.1 constrained only the V2 peak
extractor. Legacy/V1 candidates and CandidateGenerator V2's early
`waiting_post_peak` return could therefore remain global. V2.24.3 moves the
HitRegion gate up to the live `HitScanner._frame_roi_mask()` layer so the same
frozen camera-space regions constrain the complete first-pass physical pipeline.

The one V2.22.5 FULL-RESCUE deliberately bypasses the local ROI and restores the
normal detector ROI. Valid region context with zero ROI overlap now yields an
empty first pass rather than silently running global detection.

The documented coordinate contract is also enforced: `content_rect` is
viewport-local. With no explicitly saved content rectangle the default is
`(0, 0, viewport.w, viewport.h)`. Shot-time object transforms refresh HitInput
calibration before projecting corners so object AABBs and scanner ROI use the
same saved calibration generation.

<!-- V2.24.3 LOCAL_ROI_INTEGRATION_FIX -->
## V2.24.3 — HitScanner-level object-local ROI

Physical V2.24.2 testing showed that V2.24.1 was installed too low in the
proposal pipeline: CandidateGeneratorV2 could be masked while legacy/V1 and the
V2 waiting-post-peak path still saw the ordinary global ROI. V2.24.3 moves the
region restriction to `HitScanner._frame_roi_mask()`, before those proposal
branches split.

The normal first pass is now:

`global calibrated/content ROI -> intersect frozen camera HitRegions -> V1/V2 physical proposals`

If the two masks have zero overlap, the calibrated object-region mask is used as
an explicit ROI-recovery first pass rather than silently falling back global.
This still cannot create a hit: normal PRE->POST physical evidence and track/local
confirmation remain required. V2.22.5 FULL-RESCUE bypasses the object mask and
receives the original global ROI.

V2.24.3 also corrects the implicit content rectangle at runtime. `content_rect`
is viewport-local; when no explicit rectangle exists its correct default is
`(0, 0, viewport.w, viewport.h)`, not a copy carrying viewport.x/y.

<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->
## V2.24.4 — canonical camera to detector working-space ROI

The V2.24.3 physical probe exposed a coordinate-plane mismatch. V2.24 game
HitRegions are transformed to canonical full-camera coordinates, while V2.22.1
runs the expensive detector on a crop-local analysis image and translates
candidate XY back to full-camera coordinates only after detection.

V2.24.4 therefore makes the missing transform explicit:

`game-local -> screen -> full camera -> V2.22.1 analysis/crop-local -> physical detector`

The live `AnalysisGeometryV2221` supplies crop origin and dimensions. Object
AABBs are translated by the crop origin and scaled only if the actual working
image differs from the crop size. No camera resolution or fixed `/2` scale is
hard-coded. V2.22.5 FULL-RESCUE remains global and bypasses the object mask.

<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->
## V2.25.0 — composable GameObject foundation

V2.24.4 physically validated the game-context bridge, so V2.25 adds a gameplay
object layer downstream of physical HitEvent authority. The canonical model is
composition-based: identity/geometry, exact hit shape, ballistic body, layered
durability, motion and reactions are independent capabilities rather than a deep
inheritance tree.

Camera hits now carry scanner `shot_id` through a backward-compatible HitEvent
bridge before subscribers are notified. ObjectManager uses that id to resolve
exact collision against V2.24's frozen PANG-time shape metadata. Mouse/debug
hits retain `shot_id=None` and use current geometry.

Object effects are event requests. Sound, particles, animation and future
physics stay separate services rather than becoming GameObject responsibilities.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 — balanced physical proposal per frozen object region

V2.25.0 physically verified scanner `shot_id` -> HitEvent -> frozen GameObject
resolution, but the detector still allowed one noisy area inside the union of all
object HitRegions to dominate candidate selection. V2.25.1 partitions the normal
first physical pass by frozen physical search area. Each area receives its own
registered PRE->POST threshold and bounded proposal/confirmation quota.

Near-identical overlapping physical regions (for example glass directly in front of
a rear target) are grouped for detector work but retain all object identities for
downstream exact collision and penetration. Target/no-shoot/object type never changes
detector evidence weight. V2.22.5 FULL-RESCUE remains global.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 — registered freshness authority for object-context hits

V2.25.1 physically bounded candidates per frozen HitRegion but exposed an authority
leak: legacy/bank tracks inside a region could still win, including before a registered
V2 PRE→POST frame had run. V2.25.2 keeps those candidates for recall but requires normal
object-context authority to be independently supported by CandidateGeneratorV2's
registered immediate PRE→POST maps and then by V2.22.5 second-frame persistence.

The selector contains no target/no-shoot/game weighting and never changes XY. The
explicit V2.22.5 FULL rescue remains the global physical fallback.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 — shared registered readiness and cross-shot physical novelty

V2.25.2 exposed an async ownership bug: CandidateGenerator marked readiness on the CV
worker scanner while authority read a different main scanner instance. V2.25.3 moves
that state to a lock-protected process-local bridge keyed by shot id and peak timestamp.
It also compares confirmed candidate locations across prior shots in canonical camera
coordinates so persistent hotspots receive a soft recurrence penalty. Re-hits remain
legal through registered signature-gain recovery. FULL rescue remains global.
