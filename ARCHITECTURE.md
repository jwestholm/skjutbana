# Architecture

## Offline physical AI lab: two learned roles — 2026-10-03

`automation.physical_ai_dataset` adapts the existing exact EvidenceBundle witness
journals into an immutable bounded candidate cache. It does not regenerate or
move coordinates, replace common_verifier, or create another temporal owner.
The scalar feature schema contains allowlisted raw evidence plus explicit missing
indicators; labels, camera XY, session identity and retrospective POST checks are
not model features. Raw exact-XY/frame/version/source/availability provenance is
kept separately. Correlated measurements remain one evidence family, not votes.

Two complementary learned roles reuse `training_v223`:

- **Vision AI:** existing V2.23.5 registered physical image maps and V2.23.6 spatial
  convolution can generate full-map peaks independently of retained detector
  candidates. The physical adapter uses a fixed chronological frame policy and
  canonical camera origin, without snapping to dense candidates or game targets.
  Its first/last history representation is not qualified clean causal PRE.
- **Evidence Fusion NN:** existing RankModelV223 MLP representation/serialization
  and V2.23.5 gradients/mining support a labelled event-balanced training adapter.
  It learns candidate-local nonlinear evidence scores, preserving observed XY.
  A research threshold can withhold UNKNOWN; it never establishes NO_IMPACT truth.

`automation.physical_ai_lab` is an offline runner, not a runtime verifier. The F2
TCP automation remains for projected synthetic sessions; it is not invoked to
construct physical benchmark truth. Completed trial JSON is the recovery journal
of record. Adam/RNG/epoch checkpoints are atomic, completed fold models are hash
bound, and leaderboard/failure-wall summaries are rebuildable. Champion history
is append-preserved and lives only under /data; no runtime registry is written.
Evidence/Fusion runs now index compact trial summaries and full champion state in
transactional SQLite (`physical_ai_lab_index`). Publication atomically updates
count, comparison and role metadata; JSON views contain top20/recent100 results.
Immutable result/model/champion files preserve full reproducibility; an interrupted
publication is recovered from the next durable result, without rescanning history.
Legacy status is read-only and can recover missing role metadata from the bound
dataset. SIGINT stays deferred through finalization. Existing80k artifacts are not
automatically rewritten; code-identity guards still reject mixed-code resumes.

The split unit is the whole physical session. Each recorded outer fold has three
training sessions, one development-reporting session and one unseen-by-weights
scoring session. Repeated outer scoring drives research model selection, so its
aggregate is DEVELOPMENT evidence, not an independent held-out validation claim.
S03 names/paths/aliases are refused before source reads and never join the cache.

**Upstream leakage rule:** a Vision feature used for any downstream event must
come from a model that excluded that event's entire physical session and the
outer held/development sessions. Training-fold Vision features need nested
cross-fitting, not predictions from a Vision model fitted on the same events.
The contract checker refuses such leakage. The Fusion TRAIN adapter now extends
the peak-stage cross-fitting by one level, described below. Legacy standalone OOF
TRAIN rows remain excluded; only explicitly qualified nested rows are admitted.

### Nested Vision → Evidence TRAIN cache — 2026-10-07

`physical_ai_fusion_crossfit` prepares fold-specific TRAIN rows using existing
Vision/peak models, RankModel, common physical evidence and frozen union helpers.
For outer Evidence TRAIN={A,B,C}, HELD=H, dev=D, a proposal for C uses base
Vision(A,B) and reranker(A,B). That reranker's training peaks on A come from
Vision(B-only), and on B from Vision(A-only). Thus neither C nor H/D enters any
upstream fit, scaler or miner. Repeat for A/B and each outer fold. Existing pair
models are reused; five singleton Vision fits and fifteen inner rerankers complete
the cache. No hyperparameter search, new evidence family or runtime owner is added.

The cache retains exact XY/source/frame provenance, model hashes, training events,
transitive model lineage and separate fold-specific feature indices. Load checks
actual model metadata/hashes and excludes outer HELD/dev at every level. GT is
only supervised training truth/evaluation, never an inference feature. Existing
scalar features do not contain source/session identity or proposal-oracle values.
TRAIN union deduplicates exact XY only, with pool32/final3 unchanged. HELD uses the
original frozen Classic+Vision event/feature rows exactly. Train-only Evidence
scaling/mining and threshold0.5 remain unchanged; model comparisons are development
CV, not untouched validation. UNKNOWN does not establish NO_IMPACT.

The new cache binds the80k final champion as its safety baseline. The first two
controlled smoke fits were rejected; neither trained weights nor research
champions change CURRENT authority. Source qualification and historical timing
limitations survive nesting; cross-fitting cannot manufacture missing evidence.

Physical evidence remains authoritative; neither learned role invents GT,
changes gameplay coordinates, vetoes old holes/edges, nor auto-promotes live hit
authority. Current exact-witness semantics and timing limitations are unchanged.

### Optional observed-window descriptors — 2026-10-08

`common_verifier.query_temporal_observation` extends the existing EvidenceBundle
query over an already recorded exact witness. It describes the bounded (at most50
observations) same-XY core-minus-ring signal with fixed first3/last3 signed change
and weakest same-polarity tail change divided by endpoint MAD (floor0.75). Frame
identities/order/cutoff are checked. Fewer than6 observations remain explicitly
missing. Ordinals never become timestamps, and the first window is not certified
clean PRE. The query leaves witness support/revocation untouched and returns no
new physical witness. Both scalars share one correlated contrast family.

The optional `physical_ai_temporal_representation` sidecar applies this contract
to Classic and Vision coordinates identically, retaining their provenance outside
model features. It reuses nested TRAIN lineage and the existing linear listwise
selector; default feature projection and CURRENT remain unchanged. Dataset,
witness sources, schema and code are hash-bound for resume. This is diagnostic
only: the three-seed repair resolved vector collisions but regressed selection,
so no live or research-champion promotion follows. Source-independent physical
evidence does not mean treating an unqualified temporal change as new damage.

### Bounded Vision proposal extension — 2026-10-03

The existing runner accepts a role adapter from `automation.physical_ai_vision`;
it keeps the same locking, time budget, deterministic trial identity, checkpoint,
resume and immutable-result lifecycle. There is no new runtime detector or
central verifier. `vision-prepare/run/resume/status` are subcommands of that lab.

Registered maps are cached at native camera-crop resolution, hashed together with
the unchanged physical benchmark. Existing 5×5 spatial convolution can use stride
1/2/4 (5/10/20-camera-pixel receptive field). Direct learned maxima map back using
the recorded crop origin and pixel-center convention, never dense-candidate/GT
snapping. One ranked output of at most eight peaks supplies nested budgets 1/3/8.
Classic ∪ Vision is an offline proposal benchmark and frozen Evidence HELD input;
the nested cache above also supplies safe TRAIN proposals. The union deduplicates
only identical XY; near-duplicate rate is
reported without averaging or replacing exact points.

Weights, channel statistics, positive sampling and model-scored hard-negative
reweighting use TRAIN sessions only. Operator-supported NO_IMPACT samples can
train the negative loss; UNKNOWN truth is excluded. The existing V2.23 gradients,
convolution, serialization and stable mining order are reused. Fixed-budget
champion tracks compare combined @5, unique saves, worst-session @5, combined
@10, localization error, distractors and proposal/control counts in that order.
First-model championship is a reference, not an improvement claim. Every trial
retains comparison reasons, fold models, scaler lineage and per-event proposals.
Control proposals are not control hit emissions or validated rejections.

Each proposal identifies VISION source, exact camera XY, rank, uncalibrated logit,
model/fold, training and held session, source session, code/data/model hashes,
frame identifiers and causal-status limitations. Independent per-image registered
map normalization is GT-free preprocessing; learned channel statistics never fit
held/development images. Repeated OOF model comparison remains development CV.
Standalone OOF proposals do not solve nested upstream leakage for a downstream
Evidence model; TRAIN consumption requires the additional cache described above.

### Vision peak-stage nested reranking — 2026-10-03

`automation.physical_ai_vision_peaks` is a role adapter to the same lab runner,
not another detector, Evidence Fusion service or candidate class. It extends
existing VISION proposal dictionaries with raw/reranked rank, measured local
features and both model lineages. `RankModelV223` and its existing event-balanced
training/mining/checkpoint code are reused with role VISION_PEAK_RERANK; no runtime
model registry is written. The signed-log transform and TRAIN-only scaler are
the existing implementations. No change to common temporal evidence ownership.

The internal bound is32 points in either of two explicitly different orders:
raw top32 → learned ordering → original-radius NMS, or original-score NMS → first
32 survivors → learned ordering. Neither intermediate pool is emitted. Final
top1/3/8 are prefixes of at most eight exact peak coordinates. The second order
can reach deeper raw ranks by discarding redundant peaks first; comparisons must
report that pool difference rather than crediting every gain to learned scoring.
Heatmap maxima enumeration precedes bounded retention; full NMS history is a
diagnostic, while the inference pool stops at32. NMS radius is unchanged,
including stride rounding.

Each outer fold preserves the existing TRAIN/dev/held split. For reranker TRAIN
events, an inner Vision model fits the other two outer-TRAIN sessions; held/dev
and the predicted TRAIN session are all excluded. Thus three genuinely trained
inner models per outer fold produce cached features, not relabelled global OOF
features that may have trained on the outer held session. Held peak features use
the original outer Vision model. Normalizers and mining follow the same exclusion.
Checkpointed inner/outer models and cache shards are hash-bound and validated.

Twenty-four features describe the heatmap, peak geometry and the eight existing
registered change maps. GT, absolute XY, object/zone distances and model/fold IDs
never enter the vector. Frame/cutoff uncertainty stays in provenance. Temporal
newness/onset/revocation are unqueried, not inferred from these correlated image
features. Model logits remain uncalibrated; a returned proposal is not a hit.

Peak champions start from the frozen base Vision result; losing its existing
unique saves at a given output budget blocks replacement. Per-event diagnostics
record pool absence, actual suppression and rank failure separately. Repeated
OOF model selection remains development research. This peak chain alone does not
qualify downstream Evidence TRAIN rows; the additional cross-fitting above is
required. Neither chain confers live authority.

### Representation and full-heatmap training adapter — 2026-10-04

`automation.physical_ai_vision_repr` composes the existing Vision outer-fold
trainer, nested peak-cache builder and reranker inside the same lab lifecycle.
`vision-repr-prepare/run/resume/status` bind a hashed native cache and frozen
baseline. No new neural framework, candidate type or runtime evidence owner.

Each trial fits five outer and fifteen inner Vision models, then five rerankers.
Full-map mining loads only each model's TRAIN events, excludes UNKNOWN truth,
and reuses V2.23.6 maxima extraction and patch training. One mining round retains
at most64 supervised >42px false peaks per TRAIN event. It never enlarges the
inference pool. Warmup and refinement have independent atomic optimizer/RNG
checkpoints; resume restores the warmup before regenerating its deterministic
negative bank. Scalers stay fitted on TRAIN patches, not mined/held populations.

Native stride1/2, small existing convolution widths, registered channel subsets,
target radius, BCE/pair loss and bounded mining count are searchable. `context9`
keeps native persistent abs/dark/bright/compact maps in their original slots and
puts their smoothed context in the other four input slots. The downstream peak
features still query the original registered maps, avoiding a silent feature
meaning change. No temporal variant is claimed: first3/last3 source qualification
remains unresolved, and KOMP frame ordinals are not acquisition timestamps.

NMS remains6 camera pixels (existing stride rounding), internal survivors<=32,
final output prefixes1/3/8 with3 primary. Exact image-derived XY is never moved.
Proposals retain raw/pool/NMS/final rank and Vision/reranker/model/data lineage.
TRAIN peak features are generated by inner models excluding both the predicted
session and outer held/dev; every upstream scaler/miner has the same exclusion.
OOF architecture comparison is development selection, never untouched validation.

The research champion starts at the frozen32 result. Its deterministic policy
compares final top3@5, bounded union pool@5, worst-session@5, top1, @10, burden and
runtime. Losing protected saves blocks ties/regressions; at most one protected
trade is permitted only with a strictly higher primary score. All changes are
reported and old champion records remain immutable. No live registry is written.
The measured four-variant smoke was negative; additional code is not accuracy.

## Shared exact physical evidence contract, offline only — 2026-09-28

`offline.common_verifier.query_physical_witness` now returns candidate-local
records through existing `EvidenceBundle.witnesses`. These bundles may have no
reference/fused image; they are measurement records, not weighted hit authority.
`CommonTemporalHistory.witness_query` supplies the same query from its existing
bounded frame owner. There is no second ring or live hook. The pure query also
accepts explicitly unclocked ordinal research histories without inventing receipt
or exposure timestamps.

Event/candidate/version and exact full-camera XY identify a witness. Raw signal,
actual frame identities, temporal measurements, producer/reference provenance,
availability and revocation travel together. A replaced coordinate/history cannot
inherit support. Existing `observation_hypotheses` retains supported exact points
through source-score deduplication under the unchanged 24/32 bound. Support is a
revocable physical observation, never permanent HIT or proof of NO_PHYSICAL_HIT.
Only the offline chronological replay currently consumes this contract. Detailed
family semantics and reuse ownership are in COMMON_VERIFIER_ARCHITECTURE.md.

## Kompetensprov AK physical-XY consumer contract — 2026-09-26

Kompetensprov uses one ordinary HitRegion covering its rendered HF2020 image,
through the existing GameScene/OverlayScene/PANG registry and V2.24/V2.25 camera
search path. The image AABB is search context; transparent silhouette pixels are
not automatically hits. Scoring zones do not partition detector work or provide
physical scores. No separate GameObject manager or physical-query authority is
introduced. The existing FULL-rescue path stays global and conditional; an early
local false acceptance can still prevent rescue, so misses require measurement.

The game consumes the final HitEvent unchanged. Its precise zone mask samples
the physical game point in the matching shot's frozen target image rectangle,
using half-open bounds; no same-shot target assist, ballistic displacement or
bullet-radius expansion substitutes a hit. This direct physical classification
supersedes the game's historical RULES.md V1 virtual-trajectory/line-overlap
description. Numeric zone points remain gameplay rules. ShotRecord additionally
retains camera XY, source, hit timestamp and classification rectangle; shot_id
comes from the existing V2.25 bridge. No Camera/Game mapping is duplicated.

The existing physical trace owner records `context.shot_game_context` from the
registry at event creation, checking the peak timestamp and copying its game and
camera regions (including mask identity). Missing/stale snapshots remain null;
no debug scanner field or latest-session discovery substitutes provenance.

An explicit process-only KOMPETENSPROV_AK capture enables the game's static/silent
diagnostic mode. Existing `owns_hit_presentation` suppresses global overlays;
ENTER starts/ends the diagnostic without an automatic 45 s scene change. Ordinary
competition keeps its clock. This is capture/presentation behavior, not a causal
detector promotion. Independent physical GT and actual outside-rescue performance
remain necessary; unit tests and D02 counterfactual regions cannot prove accuracy.

## 2026-09-24 mapped window setup and calibration quality

Normal App window automation positions an existing window. The standalone
foundation tool creates a window via SDL_VIDEO_WINDOW_POS. A mapped SDL position
is a readback, not a guarantee that a new decorated-window creation request at
those numbers places the same client pixels there. The latest X11 run shifted
by +37 desktop Y after creation. SDL's X11 backend handles decoration offsets
when moving windows; initial creation and WM reparenting follow a different path.
See [SDL X11 window implementation](https://raw.githubusercontent.com/libsdl-org/SDL/SDL2/src/video/x11/SDL_x11window.c)
and [Pygame creation-position parsing](https://raw.githubusercontent.com/pygame/pygame/2.6.1/src_c/display.c).
The old capture lacks measured frame extents; no exact titlebar constant is assumed.

Existing ArucoCalibrator.geometry_context remains the provenance owner. Optional
read-only X11 introspection crosschecks SDL position against native client-root
origin and records actual frame extents, without native-window mutation. Setup
comparison checks client/window sizes, mapped position, display/desktop geometry,
window mode, viewport/content and camera orientation. Incidental non-geometric
surface flags and native window IDs do not define calibration identity.

The foundation tool retains env-before-set_mode placement, waits for WM/SDL
readback and permits one bounded recreation based on this run's measured
creation/readback displacement. It does not use a live Window.position setter,
change the normal app's placement API, or alter camera homography. Failure to
restore the saved setup yields SETUP_MISMATCH before camera startup. Setup changes
at SPACE or during recording also invalidate the measurement.

Calibration quality is audited separately over ALL stored marker correspondences,
including RANSAC exclusions, using inverse reprojection error in CAMERA pixels.
The existing fit uses a 3 CLIENT px RANSAC threshold and a legacy mean 25 CLIENT
px save ceiling. Those are not equivalent to the physical foundation's maximum
3 CAMERA px tolerance. No fit algorithm or save threshold was retuned. New fits
record units, inlier marker IDs and foundation-fit diagnostics. Foundation status
precedence is SETUP_MISMATCH, CALIBRATION_FIT_POOR, PHYSICAL_GEOMETRY_FAIL, PASS.
Good round trips and a saved calibration alone cannot establish physical PASS.

## 2026-09-24 coordinate and acquisition foundations

The existing coordinate owners remain canonical. GameObject-local geometry is
translated through the game viewport into Pygame **client pixels**, then through
HitInput's inverse camera calibration to the oriented full camera plane. Existing
AnalysisGeometry/WorkingSpaceMap perform crop/scale mapping. HitInput forward
mapping returns client pixels; subtracting the viewport yields game coordinates.
Can uses the entire game viewport; content_rect is not another Can blit offset.
Scanport/physical barrier restrict inspection without changing client coordinates.
Fictional z-order remains game state. No duplicate CoordinateMapper was added.

ArucoCalibrator projects in the same client plane. Desktop window position,
decoration and monitor placement affect the physical projection at calibration,
not the numeric origin of client pixels. They must be recorded and the physical
mapping revalidated after changes, never added as a guessed camera translation.
New calibration fits use one forward homography and its mathematical inverse;
independent reverse RANSAC had broken round trips. Correspondences use the actual
integer blit pixel centers. Existing stored matrices retain their historic meaning
until the user deliberately recalibrates. `geometry_context` records existing
settings, loaded matrices, window/client/display/orientation and a stable identity.
It does not measure a titlebar or establish physical setup correctness by itself.

AudioPeakDetector and CameraManager retain public wall-clock timestamp semantics.
Additional monotonic metadata describes receipt, processing and delivery. Audio
sample indices and offsets permit within-stream chronology; receipt-anchored
first/last/trigger estimates explicitly leave ADC time and upstream backlog
unknown. Partial raw pipe reads are accumulated into the existing bounded block.
A 512-block PCM diagnostics deque is bounded (~8.2 s at 16 kHz mono, 262 KB PCM).
The installed audio telemetry patch preserves the same provenance. CameraFrame
holds read intervals, frame sequence/stream, backend raw timestamp values,
orientation/shapes and publication/first-pickup times. Backend timestamps have
unverified units and do not become exposure authority. PhysicalTraceRecorder
copies this metadata; neither source changes coordinate ranking or confirmation.

The no-shoot diagnostic reuses these owners and existing ArUco rendering/detection.
It neither starts HitScanner nor binds physical events nor writes settings. Paired
flash/tone measurements describe receipt differences including output latency and
propagation. The offline temporal-step PRE experiment uses existing clean-change
functions within frozen merged context and retains its decision cutoff. It does
not modify live PRE or transfer retrospective certainty into global accuracy.
EXACT_COORDINATE remains global physical authority (100% target, 95% minimum);
OBJECT remains a scoped query, with both physical mapping and PRE now prerequisites.

## 2026-09-15 OBJECT per-object evaluation and captured retry

The first human-observed duck false negative exposed a reproduced API ordering
defect: the initial OBJECT implementation reduced all crop candidates to three
before asking which frozen shapes they intersected. OBJECT now evaluates the
bounded evidence for every frozen object before that public output limit.

Merged crops/registration/local evidence are still shared. Component retention
reserves two slots per object within 24 per crop, with an explicit maximum of
128 when many objects share a crop. Each object keeps at most three actual
observed coordinates. A confirmed exact-shape observation gets a slot before
nearby alternatives; physical evidence scores are unchanged. Exact membership
cannot confirm a candidate. Public output ranks supported exact intersections
first, then remaining physical evidence, up to the requested limit (maximum 3).
Deduplication cannot transfer a coordinate across a shape boundary. Existing
ObjectManager repeats exact frozen collision and accepts only one coordinate.

Per-object diagnostics retain bounded raw/observed candidates and frame ledgers,
including capacity, persistence, collision and public-limit decisions. CURRENT's
V251/V252/V253 producers are a separate path, not serial inputs to OBJECT. In
particular V252 neighborhood freshness is not exact-XY OBJECT confirmation.
Game geometry never enters global coordinate truth, learning or selector scores.

HitShapeSpec gains only a bounded union of existing shapes (no nested unions).
The duck's original body/head/beak drawing primitives now define its frozen
collision and search AABB. Decorative shadow stays outside that definition.
No registry, mapper or physical-state owner was replaced. Pixel rasterization
still has a boundary; it is not a calibrated physical hit tolerance.

The capture launcher extends physical_collection and the existing recorder.
It creates/validates a unique CAN_OBJECT binding without writing AI settings,
and main claims it before hardware imports. OBJECT crop inputs carry camera
bounds/timestamps/hashes in `object_evidence_crops`, separate from the ordinary
global `frames` list. CURRENT remains immediate-comparison diagnostics without
holding OBJECT gameplay. Its later pixels may contain the reaction, as logged.

## 2026-09-15 OBJECT lifecycle and output observability

The existing audio pending queue drains oldest-first. Shot preparation now uses
that oldest event's timestamp rather than the latest peak, correcting queued-shot
snapshot/PRE ownership without changing trigger thresholds or CURRENT ranking.

Small optional `on_object_shot_frozen(snapshot)` / `on_frame_presented(timestamp)`
hooks pass through existing scene wrappers. Can freezes the last rendered image
while owned OBJECT queries are pending, including render-time animations. Its
`owns_hit_presentation` flag keeps outer global hit/debug overlays out of camera
evidence. The physical query is final before gameplay/SFX; final queries cannot
consume reaction pixels. After result/intro transitions, transient visuals clear;
the first presented static live image supplies `physical_query.surface_after_ts`.
PRE must follow this boundary and the previous shot. Camera/projector buffering
still requires physical measurement; presentation time is not sensor exposure.

R, distance changes, player transitions and exit retire unresolved queries through
the existing registry's owner token and clear frozen/comparison images. Final
diagnostics remain. Existing round/shot deduplication and generation checks remain;
OBJECT resolution additionally skips targets made non-hittable by another frozen
shot. Ordinary `ObjectManager.resolve_hit` retains its previous default.

The existing 350 ms deadline now covers delivery and ongoing CV, checked between
crop/candidate operations. Exhaustion returns UNCERTAIN and releases arrays rather
than accepting a partial pool late. OpenCV calls are not preemptible. Region/pixel/
frame/concurrent-query/top-3 bounds remain. A constant per-crop valid-pixel count is
cached rather than recounted per candidate. No evidence/ranking threshold changed.

`AudioPeakDetector.play_feedback` extends the existing audio owner with known-output
dispatch/observability, not a new audio service. It records cue, owner/source shot,
API start, nominal end, availability and cost. At main-thread audio dispatch,
nearby output is attached to raw-event diagnostics. The two-second lookup does
not reject audio or alter cooldown/noise estimation. Coincidence cannot prove
acoustic causation or exclude a simultaneous genuine shot. No echo gate is promoted
without measured microphone/speaker timing and response. Can's explicit
`SKJUTBANA_CAN_SFX=0` removes its own output; default sound remains enabled.

The joined diagnostic waits for CURRENT and first reaction presentation when
available, and reports frozen shapes, top-3 XY/provenance, collision, timing,
output context, agreement and projection-contamination flags. Separate
`[AUDIO-PLAYBACK-CONTEXT]` records preserve later coincident raw events. Neither
collision nor playback timing becomes physical truth. Comparison mode holds game
visuals through CURRENT; normal immediate reaction can affect its later inputs.

## 2026-09-14 OBJECT physical capability / Can Shootout

**EXACT_COORDINATE** remains the ordinary global physical-coordinate authority.
**OBJECT** asks whether new localized physical change occurred within frozen
game-object search regions (plus technical context), and returns up to three
actual observed camera XY hypotheses. It is a scoped gameplay capability with a
lower precision requirement, not evidence that global accuracy has improved.
Its engineering thresholds and confidence strengths are physically unvalidated.

The existing `ObjectHitRegistryV2223` owns each query inside its existing shot
snapshot. Games request it with `object_query_regions(get_hit_regions(), top_k=3,
margin_camera_px=36)`. Existing GameHitRegion/CameraHitRegion transforms and a
frozen HitInput camera-to-screen transform preserve shot-time calibration and
exact GameObject shapes. One small `ObjectPhysicalQuery` holds bounded PRE/POST
crop arrays; there is no new object registry, mapper, HoleMap or Track subsystem.

The existing recent camera ring supplies the latest safe PRE surface after the
previous shot, plus at most two preceding PRE frames for variability context.
There is no startup scene-reference fallback. Unchanged holes stay in PRE; known
holes supply diagnostics only. New adjacent damage and observable enlargement
are allowed. Nothing from OBJECT changes persistent surface/hole truth or trains
the global detector. A re-hit leaving no visible change remains unresolvable.

Expanded camera AABBs merge to disjoint crops; nominations use the union mask,
with surrounding pixels reserved for local evidence. Existing AnalysisGeometry,
WorkingSpaceMap, V2 registration/photometric normalization and V252 disc/ring
measurements are reused. PRE is registered into each actual POST coordinate plane.
Persistence is measured again at the same observed XY in a distinct causal POST;
confirmation is never transferred from a nearby coordinate. Limits are 64 regions,
1,000,000 processed pixels, 24 component nominations per crop/frame, eight POST
frames, four concurrent queries, top three output coordinates and a 350 ms deadline.

The main loop advances this scoped work before ordinary scanner submission.
The existing EventBus delivers `physical.object_result` with PHYSICAL_HIT,
NO_PHYSICAL_HIT or UNCERTAIN. NO_PHYSICAL_HIT means no supported new change **in
the requested scope/window**, not global NO_IMPACT. Confidence is an uncalibrated
evidence strength. Result metadata includes frame/delivery times, current and
previous same-XY evidence, contributors, exact-shape membership and processing cost.
Frame timestamps come from CameraManager's read completion, not sensor exposure;
camera buffering and real microphone-to-frame alignment still require physical
measurement. Missing recent PRE produces UNCERTAIN rather than invented history.

`ObjectManager.resolve_object_query` tries confirmed hypotheses against exact
frozen shapes and existing projectile mechanics, accepts one hypothesis and
deduplicates shot IDs. Technical search margins never enlarge gameplay targets.
Can Shootout owns scoring, round progression, reactions and cached original SFX.
Its old center-snapping/shadow assist is removed from normal play; the historical
demo module and its working-space mapping regression remain available but are
no longer installed automatically. Ordinary global callbacks are comparison-only
in this game. No global ranking/selection algorithm or learning policy changed.

Normal immediate projected reactions can change later frames seen by CURRENT.
For a paired diagnostic capture, `SKJUTBANA_OBJECT_COMPARE=1` holds the frozen game
image until CURRENT finishes (bounded three-second fallback); sound stays immediate.
Both paths are logged independently. Trace diagnostic fields and object collisions
are excluded from global selector metrics and never constitute human physical GT.

## 2026-09-14 bounded evidence and capture ownership

`src/engine/offline/observation_hypotheses.py` is a pure helper over the existing
complete track audit. Track identity is retained; current/first-alternative/
best-locally-confirmed/latest coordinate slots contain actual observations only.
It scans history in linear time with four XY slots and at most twelve witnesses
per eligible track, then emits at most 32 alternative parent/XY pairs. Frame and
delivery cutoffs and all available producer/dispatch owners are checked. Explicit
local confirmation stays attached to its exact XY. No new Track subsystem or
live hypothesis policy was introduced. Grouped conditional coordinate ranking
and event-level NO_IMPACT are separate research questions.

PhysicalTraceRecorder now owns only a fresh capture root for its runtime. It
refuses a pre-populated root and atomically claims each shot directory before
queueing writes. This prevents runtime shot-ID reuse from replacing original
frames while leaving old labels behind. Binding/backup overwrites are refused.
The D02 provenance helper adds a small external geometry/notes sidecar using
existing binding resolution and preflight, not a BoardState replacement.

The historical can-game OBJECT_DEMO extension is gameplay-only. Its difference
maps and candidate coordinates are in CandidateGeneratorV2's working crop plane;
frozen camera HitRegions must pass through the existing V2.24.4 WorkingSpaceMap
before reading those arrays. The narrow mapping fix leaves the returned canonical
candidate object unchanged. Frozen object geometry, canonical XY and demo results
can explain disagreement, but neither object intent nor assisted gameplay supplies
physical labels, detector metrics or self-learning truth. The current extension
scores camera AABBs; that is not proof of an exact physical object-shape impact.

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

The 2026-09-25 causal runtime prototype adds a bounded `CommonTemporalHistory`
companion in `offline/common_verifier.py`, reusing that frame-context boundary
and the existing camera-local patch sampler. It accepts already computed raw
grayscale and exposes exact cutoff-bounded candidate queries; it has no live
ingestion hook. Batched reductions preserve the reference evidence exactly but
complete-query p95 remains 119 ms, above the intended budget. Eventual integration
must reuse one camera/scanner grayscale owner with frame provenance and immutable
lifetimes, not add independent PRE stores to OBJECT or a future verifier. See
[COMMON_VERIFIER_ARCHITECTURE.md](COMMON_VERIFIER_ARCHITECTURE.md) for the measured
ownership contract and limitations. CURRENT hit authority remains unchanged.

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
