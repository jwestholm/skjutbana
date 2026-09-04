# HIT_DETECTION_PLAN.md fixture

<!-- V2.24.0 GAME_HIT_CONTEXT -->
## V2.24.0 — Game-context AABB contract

New games SHOULD expose approximate hit-search geometry with
`game.get_hit_regions()`. Each region is an inexpensive `(x, y, width, height)`
AABB in viewport-local/game-local coordinates. No mesh/polygon/image mask is
required from games.

At shot time the engine freezes the list before scene update and transforms the
four corners through the existing screen/camera calibration into a camera-space
AABB. If the game has no regions, or the transform is unavailable, detection
must use the ordinary global path. A region may prioritize/localize physical
search but MUST NOT invent a hit or snap an unsupported shot onto an object.

V2.24.0 establishes the contract only. V2.24.1 adds local physical PRE->POST
search plus global fallback.

<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->
## V2.24.1 — Game-context local physical proposal search

V2.24.1 activates the first object-aware physical-search stage without changing
hit authority. Frozen V2.24.0 camera HitRegions receive a safety margin, overlap
is merged, and the result masks the existing V2.22.5 FAST extractor's `valid`
ROI for the first proposal pass.

Non-negotiable rules:

- region means **search here first**, never **hit this object**;
- physical PRE->POST evidence remains mandatory;
- candidate XY is never snapped to an object/region;
- target/no-shoot/breakable roles do not alter physical evidence thresholds;
- no context / invalid transform uses the existing global path;
- V2.22.5's one FULL-RESCUE is always global and bypasses the object mask.

V2.24.2 is the acceptance stage: moving regions, target/no-shoot, overlaps,
edges, shots immediately outside regions, empty context and transform
round-trips. Measure local-search success, global fallback, false object
attraction, XY error and latency before deeper object-aware optimisation.

<!-- V2.24.2 GAME_CONTEXT_TESTSCENE -->
## V2.24.2 — Physical game-context acceptance scene

The dedicated `Hit Context Test (V2.24.2)` scene is the acceptance harness for
V2.24.0/1. Physical testing must cover:

- stationary target and no-shoot regions,
- moving target with shot-time frozen geometry,
- overlapping target/no-shoot regions,
- region near the viewport edge,
- a shot immediately outside a visible target region,
- EMPTY regions, which must use the ordinary global detector,
- game->camera->game transform sanity and normal HitEvent delivery.

Success means object context reduces/structures search without snapping final XY
or turning a nearby object into hit truth. V2.24.2 remains diagnostic only;
V2.25.0 is still the first shared object/item engine checkpoint.

<!-- V2.24.3 LOCAL_ROI_ALIGNMENT -->
## V2.24.3 — local-search alignment correction after physical V2.24.2 test

The first physical Hit Context Test established that game regions were frozen
and transformed, and the moving-target snapshot path worked, but it also
exposed that the V2.24.1 gate was too low in the detector stack. Most tested
region shots fell through `outside_detector_roi`, while early V2 frames could
return global legacy candidates without calling the wrapped extractor.

V2.24.3 correction:

1. enforce viewport-local default `content_rect=(0,0,w,h)` when no explicit
   content rectangle is saved;
2. refresh HitInput calibration before shot-time game->camera transformation;
3. gate the entire live scanner ROI with frozen HitRegions + margin, covering
   legacy/V1, early V2 and normal V2 paths;
4. preserve the one explicit V2.22.5 FULL-RESCUE as global;
5. if valid local context has zero overlap, return an empty local first pass and
   let V2.22.5 request rescue instead of silently accepting global candidates;
6. keep object roles and exact collision outside physical detector authority.

V2.25.0 remains blocked until the short V2.24.3 physical matrix demonstrates
actual `LOCAL-ROI`/`LOCAL-SEARCH` engagement and no object attraction.

<!-- V2.24.3 LOCAL_ROI_INTEGRATION_FIX -->
## V2.24.3 — local ROI integration correction

V2.24.2 physical acceptance found repeated `outside_detector_roi` fallbacks even
though all seven game regions transformed successfully. The fix is not a new
ranker. Region restriction now happens at HitScanner ROI level so V1, early V2
and normal V2 share the same first-pass search area.

Acceptance requires visible `[V2.24.3 LOCAL-ROI]` or `[V2.24.3 ROI-RECOVERY]`
lines on region-enabled shots, correct physical XY for the dedicated test
objects, and an explicit global ROI when V2.22.5 FULL-RESCUE is requested.
V2.25.0 remains blocked until this physical bridge behaves correctly.

<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->
## V2.24.4 — detector working-space mapping correction

V2.24.3 produced `camera=7` for all test objects but `region=0.0%` inside the
local ROI stage. This was not a failure of the game->camera homography. The
region AABBs were full-camera coordinates while the ROI mask was being built
inside the V2.22.1 crop-local detector plane.

V2.24.4 maps the frozen full-camera HitRegions through the live V2.22.1 analysis
geometry before constructing the first-pass mask. Physical acceptance requires
non-zero region coverage on region-enabled shots, correct target/no-shoot/
outside classifications, and an unchanged global FULL-RESCUE path.

<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->
## V2.25.0 — hit engine / game object boundary

The V2.24.4 physical acceptance established non-zero object-local detector ROI
and correct game-context operation. V2.25 does not change detector authority.
It only preserves scanner shot identity through HitEvent so downstream gameplay
can select the exact frozen object snapshot.

Invariant: HitRegion means `search here first`; ObjectManager receives the
already-resolved `HitEvent.game_x/game_y` and never snaps or moves that point.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 — object-region proposal fairness

Physical V2.25.0 testing showed all five deliberately separated shots resolving into
a small camera area despite correct V2.24.4 ROI mapping. The normal FAST path had
`primary=0` and the old local-confirm gate allowed very large candidate sets to
survive. V2.25.1 therefore changes candidate competition, not GameObject collision:

- split the frozen union into individual physical region groups;
- compute region-local robust proposal thresholds;
- retain a bounded proposal quota from every region;
- re-balance V1/V2/bank candidates by physical region;
- keep a bounded number of V2.22.5 PRE->POST confirmations per region;
- among V2.25.1-confirmed tracks, prefer stronger physical confirmation/evidence;
- never use role/owner/game score/projectile semantics to choose detector XY;
- preserve global high-recall FULL rescue.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 — close early/legacy local-authority leak

Physical V2.25.1 logs showed successful per-region balancing, but final hit XY still
collapsed into one small area. Some shots completed confirmation without any
`REGION-PROPOSAL`, proving CandidateGeneratorV2's early `waiting_post_peak` legacy path
could emit before registered evidence. Other shots selected a region even when that
region had zero registered proposals.

V2.25.2 therefore adds registered compact freshness at the same XY, immediate PRE-stack
noise as a soft penalty, early emission gating until a registered frame exists, and
registered-fresh-only local authority. Legacy coordinates may be revalidated; they are
not discarded. FULL rescue remains global.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 — fix async readiness; suppress repeated physical hotspots softly

Physical V2.25.2 logs contained both `REGISTERED-READY` and later
`reason=no_registered_frame` for the same shot. The registered frame was real, but its
ready flag lived on the worker scanner instance. This release makes readiness shared
across worker/main and replaces local fail-open with the existing global rescue.

Because V2.25.2 marked almost every candidate fresh on the worn board, V2.25.3 adds
cross-shot camera-coordinate recurrence as a physical prior. It is soft, supports rehit
recovery, and contains no gameplay semantics.
