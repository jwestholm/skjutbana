# Physical board research architecture and inventory

The 2026-09-09 design map supplied by the user is a research direction, not
authorization to replace the engine. This inventory accompanies the D01 pass.
The objective is **100% correct emitted physical results; 95% is the minimum**.
No central class or live policy was added. S03 remains entirely untouched.

## 2026-09-10 inventory before consolidation

Inventory covered current source, targeted `git log --all -G`/path/deletion
history, ROADMAP/ARCHITECTURE and the complete locally supplied
`SKJUTBANA_CODEX_ARCHITECTURE_PLAN_2026-09-09.md`. The latter is design context,
not another implementation contract. No new central class is needed for the
measured D01 fixes: existing track audit, proposal trace and coordinate APIs
already provide the necessary extension points.

### Coordinate, display and placement ownership

| Existing component | Actual responsibility | Gap / smallest reuse |
|---|---|---|
| `HitInput._canonical_camera_to_screen` / `_canonical_screen_to_camera` | ArUco homography and inverse, with existing scanport fallback | Expose these existing operations through a documented facade only after calibration/revision semantics are explicit; do not replace them |
| `HitInput._screen_to_spaces` | Screen, viewport-local game and viewport-local content rectangle, normalized content positions | Keep content offsets distinct from desktop window position |
| `ArucoCalibrator`, calibration scenes | 24-marker board calibration; H/inverse, points, frame size, viewport, reprojection metadata | Older/manual calibration paths partially duplicate responsibility; consolidate persistence/provenance first |
| `AnalysisGeometryV2221` | Full canonical camera ↔ crop-local translation, perspective ROI, projected-playfield barrier/guard | Its 12-screen-pixel guard is not a surveyed physical board boundary |
| `WorkingSpaceMapV244` / object-local search | Camera → cropped/scaled working image, frozen camera search regions | An explicit inverse/round-trip contract can reuse origin and scales; avoid another mapper |
| `object_hit_v2223` / V2.24 HitRegions | Game rectangles → camera AABBs using all four corners; camera point → game; frozen at shot boundary | AABB is a search bound, not an invertible representation of the original rotated/perspective shape |
| CameraManager | Camera frame rotation and mirrors | Orientation belongs to the camera-coordinate/calibration revision |
| `PhysicalSetupSettingsScene`, range projection settings, `WorldPlacement` | Wall distance, physical viewport size/bottom height, game placement and fictional extra depth | These are not a measured physical surface height map; GameObject z/depth stays fictional game state |
| App/Pygame + SDL2 window position support | Existing fixed-size window, development positioning including automation `setWindowPos` | No existing projector-monitor/fullscreen CLI found. Desktop XY is not physical camera truth |

History confirms earlier implementations rather than a missing replacement:
`a48f94a` (ArUco refinement), `185aa41` (V2.24 game hit context),
`0aa9c55` (GameObject foundation), `052ba66`/`af7a685`/`e76043b` (physical
distance/placement), and `c7854ab` (PRE/current spatial-plane correction).
That PRE correction is a reason to preserve the current working-space bridge.

`automation.coordinate_roundtrip_selftest` adds five tests of these APIs:
Camera → Game → Camera for two explicit homographies/render sizes with nonzero
viewport/content offsets; a supplied rectangular Board fixture ↔ Camera/Game;
camera/crop and scaled-region inverses; four-corner bounds; all four supported
rotations with all four mirror combinations. Perspective numerical tolerance is
0.002 camera pixels. The Board fixture is not D01 board calibration, and the
1920×1080 fixture is not physical fullscreen acceptance.

The eventual Camera ↔ Board ↔ Game API needs a capture-time physical boundary,
calibration identity, orientation, render/content geometry and inverse validity.
Existing H maps camera to calibrated rendering; it does not independently prove
normalized physical Board Space. Reuse H and crop transforms with explicit
provenance, then add Board normalization. Window movement must never be injected
as camera XY. However, moving/resizing projected content can change the physical
projection: invalidate/revalidate calibration instead of promising an old H
remains valid for arbitrary monitor/resolution/letterboxing changes.

### Surface, history and revision ownership

`HitScanner` already owns scene/surface reference images, capture buffers and
capture kind. `capture_surface_refresh(reset_holes=False)` supports refreshing
appearance without discarding hole history; `reset_hole_map` also clears active
tracks. `known_holes` is bounded to 512 entries with coordinates, score, time and
hit count, and supports nearby/re-hit handling. `_remember_known_hole` currently
uses emitted tracks: it is not independently calibrated trusted-event learning.

AITraining's `_ref_white`/`_ref_black` and projector-response `absdiff` feed an
artifact mask. Existing response threshold 12, 5-pixel erosion and the <40%
active-area fallback are heuristic response handling, not a measured tape map.
The V2 consumer has soft artifact weighting. Banks/vaults are short-lived
candidate history; V2.22.2 novelty/ridge cleanup and V2.25.3 process-local shared
novelty are additional partial state. Audit conditional stale-hole/ridge vetoes
before changing context semantics; do not create another blanket exclusion.

No explicit persistent repair/tape polygons, material revision, calibrated seam
structure, physical height map, unstable-region model or board/weapon response
model was found in the inspected current/history paths. Camera image edges are
available; exact projected-image edges require the projected frame and mapping.

The smallest useful consolidation is initially a **read-only view plus revision
metadata over existing owners**, not a new copy of images/holes or a new
`PhysicalBoardState` subsystem. Keep four distinctions:

- GEOMETRY: physical valid region, calibration, camera/projector placement.
- PHYSICAL BUILD: cardboard/mat, panels/seam, backing/support and material change.
- SURFACE STATE: paper, repairs, existing holes, projected image, appearance.
- RECENT EVENT STATE: causal PRE/POST change, candidate history, trusted impacts.

Intended lifecycle: game/session start scans the current surface and establishes
a revision; many shots update only highly trusted new changes; patching usually
happens between games; the next start rescans/revisions. Old holes, tape, lines
and seams remain context, never automatic vetoes, including direct re-hits and
hole enlargement. This lifecycle is planned, not newly implemented here.

Fast adaptation may update confirmed holes/recent appearance. Slow adaptation
may update long-term noise, panel response and weapon-conditioned motion.
`LONG_TERM_MODEL + RECENT_MODEL` must be revision-aware and evaluated across
time/material changes. Neither low-confidence predictions nor an emitted
`state=confirmed` may self-train persistent physical state. No adaptation was
enabled in this pass.

### Weapon/audio: the calibration visualization still exists

`AudioPeakSettingsScene` and `AudioPeakDetector.get_waveform_snapshot` still
provide waveform/threshold setup, peak, RMS and noise feedback. Detector code
uses PCM input, peak/RMS, noise-floor/crest handling (including the existing 1.7
crest condition), device selection and cooldown. V2.22.6 adds trigger/near-miss
telemetry. Existing trigger thresholds are not a weapon identity database.

History: `87ce579` introduced audio input; `f9d5cae` added the audio settings
scene on 2026-03-09; `4989c96` added audio-input switching. Current and all-history
targeted searches covered weapon/gun/rifle/pistol, WeaponProfile,
ProjectileProfile, calibration, waveform, FFT/spectrum, peak/RMS/crest and saved
profiles. Weapon-related hit-scanner comments describe variable audio delay
(for example `d3c0555`); they do not implement recognition. No removed weapon
calibration/profile replacement was found. The visible calibration tool did
not disappear in the inspected history; there is no removal date/reason to give.

Reuse this scene and waveform tap if a physical weapon profile becomes useful.
First evaluate deterministic peak/RMS/crest, duration, attack/decay, band energy,
spectral shape and transient timing with saved template/similarity matching.
No FFT-based identity/profile matcher was found or added. Only consider ML after
simple methods fail on independently labeled examples. Optional future
`WeaponContext(weapon_id, confidence, event_type, pressure/state)` can inform
expected hole appearance or board response; it must not gate hit detection.
Existing `game_objects/projectiles.py:ProjectileProfile` models game damage and
penetration, not measured physical weapon/audio characteristics.

### Game mechanics, rules and deferred capabilities

The foundation is distributed across App/scenes, the game wrapper's
`create_game(game_root, viewport)` and enter/exit/update/render/event lifecycle,
HitEvent, ObjectManager, GameObject shapes/components, ObjectEventBus and frozen
HitRegions. Objects support active/disabled/removed states, static/kinematic
movement, damage/reactions and fictional `z_index`/`hit_depth`. Video playback
exists. `EffectAction.play_sound` is an effect request; a complete generic
effect/audio dispatcher is still roadmap work. Player/score/countdown state
exists in games such as Pop the Balloon; do not invent an already-complete
shared scoring/player/timer service or duplicate the engine under a cleaner name.

**ENGINE PROVIDES MECHANICS. INDIVIDUAL GAME PROVIDES RULES.** DartGame owns
301/501, bull, double-out and turns; ZombieGame owns waves, health, headshots and
lives. CowboyGame may show a target for about three seconds, apply success on
hit, and fire/apply failure on timeout. Those are individual game rules. Camera
coordinates never determine fictional z-order.

Later GameAttentionMap may freeze per-shot active regions, priorities and
contributors. Overlapping objects should cause one physical-area search while
retaining their contributor stack for game resolution. Attention is a prior;
shots outside targets remain possible. Game-requested precision/latency modes
must preserve the exact physical baseline.

Future capability documentation (`GAME_ENGINE_API.md` and/or
`game_engine_capabilities.json`, not created now) can expose actual events,
inputs, objects, lifecycle, rendering, timers, scoring and sound/video services
to an AI generating game rules. A missing generic capability may justify an
engine addition; a game-specific rule belongs in that game. Camera 2 may later
provide an AimPrior from known geometry, weapon pose/fiducial and board plane.
Misadjusted sights and shooter error keep it a prior; physical evidence wins.

## Existing responsibilities and reuse points

| Responsibility | Current implementation / history | Consequence |
|---|---|---|
| Static scene and physical surface references | `HitScanner.scene_reference_gray`, `surface_reference_gray`, reference capture and `_rebuild_artifact_mask()`; `ai_training.py` white/black reference capture and projector-response debug map. History includes `79e7df6` and `485092d`. | Consolidate ownership later; distinguish calibration appearance from immediate causal PRE. Static reference alone cannot prove a new impact. |
| Holes and recent physical state | `HitScanner.known_holes`, tracking, V2 banks/vaults, V2.22.2 novelty cleanup, V2.25.3 shared novelty state. | Multiple partial histories already exist. Inventory update/reset rules before consolidation. A hole is context; nearby/reopened hits must remain legal. |
| Coordinates and bounds | `input/hit_input.py`, camera ArUco calibrator, calibration scenes, saved viewport/scanport/content rectangles, V2.22.1 analysis geometry, V2.24 object-to-camera transforms. | Reuse these transforms; do not create a parallel coordinate engine. A future normalized physical board plane needs a saved physical boundary and round-trip tests. |
| Board/panel motion | Existing `common_verifier.py`, `accuracy_verifier.py`, motion-residual and temporal offline audits; new `clean_physical_change.py` experiments. | Whole-frame registration exists. Split-crop/flow tests are research evidence; no persistent motion model exists in the inspected paths. |
| Audio visualization | `audio/audio_peak_detector.py:get_waveform_snapshot`, `scenes/audio_peak_settings.py`; history `f9d5cae` (audio settings), `4989c96` (audio hit switch). Peak/RMS/noise/crest diagnostics already exist. | Extend the existing calibration flow if physical evidence later justifies weapon identification. No physical WeaponProfile was found by current-code and all-history string/path searches. |
| Projectile identity | `game_objects/projectiles.py:ProjectileProfile` explicitly models gameplay damage/penetration. | This is not a measured physical weapon/audio profile. Do not silently reuse game parameters as physical truth. |
| Game objects | Existing model/manager/geometry/events and frozen HitRegions. | Preserve downstream ownership of game rules and fictional z order; no game implementation in this pass. |

The targeted all-history deletion search for weapon/audio/calibration/surface/
board paths found no removed replacement to revive. This is a bounded inventory,
not proof that every historical idea was implemented or absent.

## Direction after measurement

If consolidation is justified, one owner can expose physical bounds, surface
revision, static/projected appearance, holes/repairs, uncertainty and temporal
stability. Existing buffers and canonical transforms should supply it. Paper,
cardboard, camera/projector geometry, lighting or major repair changes require
revision-aware history. The user's workflow repairs before/between games and
accumulates holes during play; never assume repair after every shot.

Visual change, board motion, board memory and weapon/event context are separate
evidence families. Correlated scores within a family are not independent votes.
Known edges, holes, tape and motion are context, never hard vetoes. A calibrated
confidence gate and independent truth are prerequisites for persistent learning;
an emitted `state=confirmed` or an uncalibrated 0.95 score is insufficient.

Future motion learning should ask whether regional movement predicts impact
location across sessions better than empirical spatial priors. Keep recent and
long-term models separate and test drift/revision changes chronologically before
enabling adaptation. Lighting and support conditions must be observed metadata,
not inferred weapon or pressure labels.

D01 has saved camera frames/crop evidence but `calibration=UNAVAILABLE` and no
saved projector transform or physical seam. The new experiments therefore use
camera PRE structure and two overlapping crop halves. They cannot claim exact
projected-edge attribution, normalized physical board coordinates or calibrated
panel motion. Do not substitute today's settings for missing capture provenance.

See [D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md) for measured clean-change,
motion and ranking results, their limits and the proposed next capture.
