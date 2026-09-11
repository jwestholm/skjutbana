# Architecture

## 2026-09-10 inventory and evidence ownership

Before introducing central classes/subsystems, inspect current implementation,
all relevant Git history, existing plans and responsibilities split across
classes. Reuse/consolidate those owners first. The complete current/history
inventory is in [PHYSICAL_BOARD_STATE_ARCHITECTURE.md](PHYSICAL_BOARD_STATE_ARCHITECTURE.md),
including the locally supplied September 9 design plan, coordinate tests,
display limitations, surface revisions, audio calibration and game mechanics.

This accuracy pass adds no central class. Existing `track_audit` association
history supplies bounded alternative coordinates offline. CandidateGeneratorV2's
existing hybrid merge now has an opt-in observational ledger with input identity,
geometry operations, merge/quota parameters and retained/output rank. IDs stay
outside candidate dictionaries; scores, geometry and order are unchanged. It
closes event 3's diagnostic blind spot, not its physical selection failure.
`automation/evidence_retention_research.py` and `evidence_channel_research.py`
hold separate, immutable-output offline hypotheses. None is a live authority.

Camera ↔ Board ↔ Game is the intended public model. Current HitInput H/inverse,
ArUco calibration, scanport/viewport/content transforms, AnalysisGeometry crop,
WorkingSpaceMap and frozen GameObject regions already implement most mechanics.
Expose rather than replace them; add explicit physical Board normalization and
capture-time revisions where absent. Five tests exercise current round trips,
orientation and four-corner bounds. Synthetic geometry tests do not establish
actual D01 calibration. Window desktop position is not physical truth; changes
to the projected image's physical geometry can require recalibration.

Board consolidation should initially expose existing geometry/reference/hole
owners with revision metadata. Distinguish GEOMETRY, PHYSICAL BUILD, SURFACE
STATE and RECENT EVENT STATE. Scan/revision at game start, trusted incremental
updates while playing, repairs typically between games, rescan next start.
Known holes, tape, projected edges, seams and noisy regions are context, never
automatic vetoes. Existing emitted-track hole updates do not establish trusted
persistent learning. Recent/long-term adaptive models remain research.

Audio waveform settings remain present; there is no missing physical weapon
profile subsystem to replace. Prefer deterministic signal/profile matching if
later justified, optional to hit detection. Gameplay ProjectileProfile remains
separate from measured physical weapon identity.

**ENGINE PROVIDES MECHANICS. INDIVIDUAL GAME PROVIDES RULES.** Reuse GameObject,
HitEvent, scenes/rendering and lifecycle; games own Dart/Zombie/Cowboy rules.
Fictional z-order never derives from camera XY. An AI-readable capability SDK,
game attention priors and camera-2 aim priors are later/future ROADMAP entries,
not this pass's implementation. Exact physical coordinates remain canonical:
100% correctness is the target and 95% only the minimum acceptable outcome.

## Physical accuracy research architecture — D01

The current work preserves the live detector and evaluates causal physical
change offline. Binding-based finalization now builds its manifest from saved
labels/assignments and records fresh quality and hashes in an immutable external
report. Explicit ordinal mapping prevents event IDs from becoming guessed truth.

`src/engine/offline/clean_physical_change.py` reuses `EvidenceContext` for local
background correction, temporal persistence, PRE variability and broad motion
compensation. It has no live imports/callers. Selection and proposal availability
are evaluated separately, including severe attenuation of real GT evidence.
The source-independent common verifier remains research-only.

Existing surface references, projector-response masks, hole histories,
coordinates, audio waveform calibration and GameObjects were inventoried before
considering new central concepts. See
[PHYSICAL_BOARD_STATE_ARCHITECTURE.md](PHYSICAL_BOARD_STATE_ARCHITECTURE.md).
Board-state consolidation, calibrated motion priors and adaptation remain future
measured steps. Exact physical XY is canonical; game z-order/rules stay downstream.
The target is 100% physical correctness, with 95% the minimum; S03 is untouched.

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

## Evaluation foundation (codex/eval-loop)

`src.engine.offline.evaluation` extends the offline measurement layer with a
versioned observation contract and stage scorecards. `automation.evaluate_pipeline`
adapts existing V2.23 framepacks or explicit stage traces. It uses existing camera
pixel metrics and framepack readers; no live detector code is changed. Saved pools
remain distinct from raw/filtered/retained/confirmed/selected/emitted observations.
See [EVALUATION.md](EVALUATION.md) for evidence levels, provenance and replay gaps.

## Physical trace causality and authority boundaries

Camera evidence time, worker delivery time, scanner observation time and decision
time are separate clocks/roles. `decision_input.timestamp` is captured by the
emission hook and identifies the consumed proposal snapshot; a candidate's frame
timestamp alone does not prove delivery before selection. Physical trace windows
can outlive terminal events and overlap later audio events. Shared diagnostic
pools must carry producer ownership and must not be treated as earlier authority
inputs. Causal audit/export fields are additive; historical stage metrics retain
their named snapshot meaning. New captures preserve the first terminal outcome.

The V2.22.1 working-space contract applies to **every image read by a detector**,
including PRE frame history, not only current/reference masks and output XY.
The 2026-09-08 audit found a V2 PRE-history violation of that contract, fixed
separately in `c7854ab`. Pending-event boundary and track-ownership corrections
were also implemented separately from the original measurement work. See
`CAUSAL_CANDIDATE_AUDIT.md` for the preserved evidence.

The accepted pending-event ownership correction (`5d45527`) makes the next
audio peak an event boundary for local confirmation while preserving delayed
worker results captured before that boundary. The V2 PRE mapping correction in
`c7854ab` translates crop-local detector regions into full-camera
frame-history coordinates exactly once for normal and fallback references.

## Complete decision tracing and replay

With physical tracing enabled, V2.22.6 emits a per-consumption association ledger
and selectors snapshot all active tracks with exact rejection predicates/rank
tuples. `last_stable_tracks` remains the legacy score-sorted top-eight debug view;
it is never the authoritative full pool. A local-confirmed candidate, stable
track, rank-eligible track, ready track and emitted `state=confirmed` differ.
Readiness is checked on the ranked winner after selection.

The recorder freezes `decision_input.complete_track_audit` before emission and
preserves the first decision. Terminal non-emission snapshots use a separately
named field. Compact source/owner history includes support observations, while
current representative ownership identifies the observation supplying XY.
Completed immutable ledger rows are shared until trace serialization to avoid
repeated critical-path copies. The completed event's scanner ledger is released;
pending overlapping owners remain.

`CURRENT_EXACT_REPLAY` reconstructs selector predicates and ordering, including
stable tie order, and raises on mismatched track ids/coordinates. Export only
claims exact verification for a complete snapshot; older traces remain explicitly
unavailable unless a separate recorded-input reconstruction verifies them.

Producer ownership must travel on the **consumed** worker result list. Tagging
only `scanner.last_candidates` is insufficient because both tracking and local
confirmation seed from `result.candidates`. The latter is now tagged on delivery.
This is transport metadata enforcing existing authority boundaries, not a new
source-ranking policy.

Association also respects known producer identity: a candidate cannot update a
track produced by another audio event. This protects accumulated XY, frame hits
and best_score even if an older frame arrives after newer event activity.
Same-event spatial association and untagged legacy inputs preserve their existing
rules. The trace records nearby tracks excluded by the producer predicate.
