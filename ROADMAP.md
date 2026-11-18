# Roadmap

## Proposal: hide decision latency with a "bullet-time" projectile animation — 2026-11-18 (user)
When a game can wait for a sure answer (e.g. Skyttegrav 3D, up to ~3 s), it shows a short
matrix-style animation of the projectile flying into the scene towards the target while the
evidence chain finishes; the hit/miss reaction plays when the animation lands. Engine side: an
opt-in "decision pending" callback with the expected remaining time from the evidence scheduler
(EVIDENCE_CHAIN.md §3.3), so the game can time the animation. Caveat: everything projected changes
the camera image — the animation must avoid the analysed region (or the region is frozen / known to
the detector), same rule as HUD text and markers (AUDIT_FINDINGS §3b, §11). Fits "early gameplay
decision + shadow analysis" and per-game latency classes.

## Game design rule: spread the aim points over the whole board — 2026-11-18 (user)

The physical board wears out where games put their targets. Photo 2026-11-18 (tape removed):
the centre column is shot to pieces while the edges are almost untouched; patched/torn areas
also hide new holes from the camera (only 7/17 AR15 holes were visible on taped areas).
Rules for every new game, test and engine helper that places something to shoot at:
- Do not always spawn targets in the same place, and do not centre by default. Spread spawn
  positions over the whole usable board (e.g. random over the full viewport, rotating layouts,
  random offset of fixed figures/zones between rounds).
- Prefer under-used areas when possible (future engine feature: a per-board wear/hole-density map
  from the real-hole library / known holes, offered to games as an opt-in spawn helper).
- Fixed layouts (Kompetensprov figure, capture target crosses) should get an offset/rotation
  option so the same paper spots are not hit every session.
- Engine provides the mechanism (wear map, spawn helper); games choose whether to use it.

Measured wear (2026-11-18, before re-papering; `python3 -m automation.bench board-snapshot`,
`/data/skjutbana/evaluation_runs/board_wear_20261118/board_wear_heatmap.png`): 443 hole blobs in
the viewport (torn centre undercounted); the central 50 %×50 % box (25 % of the area) holds
**55 %** of the holes, the central 25 % box (6 % of the area) 18 %; the middle two column
quarters hold 70 % and the middle row third 57 %.
Uses agreed with Codex (in priority order):
1. Opt-in spawn helper fed by a MEASURED wear map (board-snapshot), not by detector emissions
   (the engine currently emits even without a visible hole).
2. Maintenance/review: re-paper alert from a defined damaged-area measure (validate against the
   operator first); session-start/end snapshots with `--compare` as review aid for labelling —
   not automatic ground truth.
3. Region quality may inform games/precision choice but never vetoes a hit (AGENTS rule).
4. NOT as an aim/position prior in winner selection (reinforces centre bias, favours old damage).
Snapshots only at session start/end, or in a pause the game explicitly owns that cannot overlap
a shot.

## Hit-detection precision levels — 2026-11-18 (decided with the user + Codex)

Games declare the question they need answered; the engine picks the method.
Levels (`hit_precision`): **SHOT** (a shot happened, audio), **HIT_MISS** (hit on
the target or not, fast), **OBJECT** (which object incl. miss; e.g. cans, enemies,
guard scenario with a drawn "warning shot" box / person / rest of board), **ZONE**
(which scoring zone; Kompetensprov A/B/C, rings), **EXACT** (mm XY; darts).
Separate choices, not levels: aggregation (`per_burst` | `series`: one precision
flash at an explicit series end), latency class (realtime / normal / after),
uncertainty policy (guess / UNKNOWN / request flash). IMPACT ("every real shot
leaves a hole") is a background validity check that must be earned with image
evidence, never an automatic veto. Legacy aliases: `shot_only`->SHOT,
`exact`->EXACT, `region`->OBJECT (ambiguous, compatibility only).
Decisions: Kompetensprov = ZONE + series flash at ceasefire. Skyttegrav stays
SHOT_ONLY-authoritative until it publishes enemy regions in a SHADOW OBJECT trial
that passes physical acceptance.

TODO:
- [ ] More detection levels may be added later as engine features. Every new level
      MUST also get support in the AI training/test mode (in-game AI test scene and
      the offline bench/nightly): its own synthetic test targeting (e.g. object edges,
      zone borders), its own labelled AI data, its own frozen holdout and its own
      acceptance criteria. A level without test + AI support is not done.
- [ ] Per-level test driver for real games (synthetic shots + real-hole stamps
      injected into a running game; score object/zone/exact/impact per level).
- [ ] Per-level datasets and acceptance in the nightly (report per level).
- [ ] Skyttegrav: publish per-frame enemy regions (SHADOW OBJECT trial).
- [ ] Kompetensprov: use the series measurement for final scoring once validated
      physically (today it is recorded alongside, scoring unchanged).

## Real-hole library (where real holes live and how they are kept) — 2026-11-18

Real bullet holes are measured data, never generated: keep and grow them.
- **Location (outside the repo, not committed):** `/data/skjutbana/hole_library/`
  - `stamps.npz` + `stats.json` + `contact_sheet.png` + `README.md`: raw ratio patches
    (post/pre) around labelled real holes, extracted by `extract.py` from
    `/data/skjutbana/physical_traces/` and the 2026-10-13 GT capture.
  - `clean_stamps.npz`, `clean_stamps.json`, `clean_stamps.png`,
    `clean_stamps_vs_source.png`, `stamps_png/NNN.png`: background-free stamps
    (darkening map D, 0 outside the hole; PNG = black with alpha D), built by
    `python3 -m src.engine.offline.hole_stamps`.
- **Adding holes:** every physical session recorded with `SKJUTBANA_SOURCE_RECORD=1`
  and labelled (`ground_truth.jsonl`) is a source. Record the WEAPON per sequence
  (airgun 4.5 mm, CO2 AR15, CO2 pistol) so stamps can be grouped by weapon instead
  of scaling airgun holes. Re-run extraction into a NEW versioned directory; never
  overwrite or delete existing stamps (training data).
- **Users of the library:** offline bench (`PhysicsAssumptions.hole_model="stamps"`),
  nightly training data, the in-game AI test scene (projected holes, F8 profile).

TODO:
- [ ] Extraction from source sessions (`/data/skjutbana/source_sessions/*` with
      `ground_truth.jsonl`) with weapon labels; per-weapon stamp sets.
- [ ] IDEA (game feature, engine-provided): games may use real-hole stamps as
      bullet-hole decals ("shot-up wall", damage on objects), so in-game damage looks
      like the real holes. Engine provides a decal API over the library; games decide
      where/when (engine mechanics, game rules).

## Practical gameplay + physical accuracy priorities — 2026-10-08

**The long-term target remains 100% correct physical results; 95% is the minimum practical acceptance level.**
Exact physical coordinates remain a core capability, but they no longer block useful gameplay when a game asks a simpler question such as `OBJECT_HIT`. Candidate oracle is still not selected accuracy, and game context must never manufacture physical truth.

The project must remain enjoyable to build and use. Near-term work therefore prioritizes playable physical games and reusable engine capabilities before more long-running exact-coordinate research.

> **Status 2026-11-18:** the audio-lateness root cause discussed in this NOW section was
> found and fixed on 2026-11-17 (ffmpeg PCM bursts -> `parec` 10 ms blocks; camera
> BUFFERSIZE/exposure fixes). Physical shooting is requested again (pilot with source
> recording, see PHYSICAL_CAPTURE_RUNBOOK.md top). Offline bench, source recording, lens
> model, precision flash and levels: see CURRENT_STATE.md top sections.

### NOW — ONE ENEMY / ONE HITBOX / DID WE HIT IT?

- **Physical root-cause finding (2026-10-18):** D02 exact-label frames show the
  impact-site transition about1.4–1.6 s before its receipt-anchored audio event;
  exposure/audio device clocks are absent, so cause is unresolved. A separate
  future-frame PRE fallback bug is fixed with an audio−200 ms receive-time cutoff,
  but historical D02 did not use that path. D02 event3's correct 2.10 px candidate
  survived and lost final ranking; event1's closest pre-limit proposal was dropped
  by cap. Do not tune/promote from these two shots. Highest-value next software
  step is opt-in timestamp-domain instrumentation; highest-value offline step is a
  bounded pre-limit retention replay. See `/data/skjutbana/evaluation_runs/physical_root_cause_deep_dive_20261018_01/NEXT_ACTIONS.md`.

- **2026-10-17 checkpoint:** Skyttegrav 3D is integrated with SHOT_ONLY; the
  operator verified one physical shot pauses it. Preserve this game integration.
  It does not establish hit position. Continue generic physical evidence work on
  `codex/physical-ai-lab-foundation`, with no new selector search or live promotion.
- **Single-region foundation (2026-10-14):** prioritize one frozen exact enemy
  region over seven-object discrimination. Reuse `ObjectPhysicalQuery`; no separate
  single-region authority or game implementation. Offline replay now isolates one
  saved region and emits explicit family ledgers/spatial residuals.35 paired-data
  region runs show30 legacy supports, but no certified physical HIT. All actual
  truth remains UNKNOWN. Removing multi-object ambiguity is not damage detection.
- **Render provenance foundation implemented/offline verified (2026-10-17):**
  opt-in immutable engine presentation history now binds existing shot-freeze,
  mapping and recorder owners; it is never detector input. CPU flip timestamps do
  not prove projection at impact. External SHOT_ONLY rendering remains UNAVAILABLE;
  do not substitute the parent launcher's surface. Next measured boundary is
  calibrated optical transfer + independent physical GT, not a new score formula.
  Existing Optical/HIL and GT instruments support that work when hardware testing
  is possible; no new physical capture is requested now. Projector explanation
  stays UNAVAILABLE. Keep region margin outside hit truth.
- **Transfer falsification (2026-10-18):** existing neutral physical frames do not
  support dependable per-pixel WHITE/BLACK-to-gray prediction; a synthetic
  gamma/geometry mismatch overwhelms a weak local change. Do not subtract raw
  render pixels or add a residual score. Next discriminating measurement, when
  optical hardware is available, is repeatable no-shot probe/recovery plus repeated
  neutral states with camera sequence/timing provenance. Selected accuracy remains
  unchanged and unavailable for that unlabeled capture.
- Physical acceptance remains pending, with no shooting requested now. Preserve the
  separate3-shot GT observability gate. Later at most10 single-enemy shots:5 inside,
  5 just outside; record independent actual impact, frozen geometry, guarded PRE,
  POST and settled GT. Target5/5 inside and0/5 outside false HIT is a diagnostic
  milestone, not general95–100% physical accuracy. No Infinite Trench changes or
  live promotion until measured evidence supports them.

Previous multi-object work and its negative results remain preserved below:

- **Optical/HIL measurement foundation (2026-10-14):** passive variability,
  projector response and saved-calibration roundtrip tools are implemented and
  synthetic/offline verified. Physical optical validation remains pending. These
  instruments generate no projectile labels and cannot replace the pending
  three-shot GT acceptance. Native4K hashing currently limits throughput; explicit
  backpressure/write gaps make a run INCOMPLETE, never silently full-rate. No
  detector tuning or physical shooting while the operator is unavailable.
- **GT timing/provenance hardened offline (2026-10-14):** PRE now pins the existing
  clock-qualified ring reference at event dispatch, not after potentially long live
  processing. The 500ms receipt/publication guard remains; V4L2 POS_MSEC lacks verified
  clock/source flags and is not exposure truth. `physical_gt_audit` reports legacy
  PREs UNQUALIFIED without mutation. No new shots while the operator is unavailable.
  Physical acceptance remains pending: later only three shots, absence of the new
  hole in guarded PRE, visibility in a neutral frame, manual XY only when clear.
  No detector changes/search; preserve the failed −30..−50ms physical PRE finding.
- **Independent GT observability (2026-10-13):** operator inspection found the
  prior paired crops unlabelable. Preserve all six UNKNOWN labels; no actual-hole
  oracle is available. The first three-shot full-frame run exposed noncausal PRE:
  the new hole was already visible. GT-only selection now requires receipt and
  publication500ms before audio, otherwise UNAVAILABLE; physical causality remains
  explicitly unverified. Preserve the run; do not infer a clean PRE from its name.
  The opt-in `play-can --gt-review --gt-shots 3` extension retains the same neutral
  review sequence (clean surface / near old damage / dark detail).
  After all live work ends it saves native PRE/LIVE_POST and separate settled
  WHITE/BLACK/MID_GRAY human-review images. Use `physical_trace_label --gt-review`;
  require3/3 independently visible holes before collecting more evaluation data.
  This supersedes the older “label existing crops/no new shooting” next step below.
  No live authority change, model search or accuracy claim; full procedure in runbook.
- **Paired capture measured (2026-10-12):** `can_play_candidate_compare_20261011_203905` provides five paired events plus one log-only event. Operator truth is explicitly UNKNOWN for all six. Early unregistered mature pools exist on1/3/5; registered delivery takes1.50–1.94s, largely worker compute. Confirmation keeps8 unchanged coordinates and sometimes loses exact-shape coverage, but we cannot say which physical signal survived. Do not promote mature or change retention from these unlabeled counts. **Immediate next step:** operator review with the now-ready `physical_trace_label --paired-object-crops` adapter and separate source-bound labels, then paired oracle analysis only where holes can be independently identified. No new shooting is justified yet; full-board/missing event6 evidence cannot be reconstructed. See the paired report in CURRENT_STATE.md.
- **Current blocker: candidate ownership and physical discrimination.** OBJECT_HIT makes its own OBJECT_LOCAL_CHANGE proposals; it does not consume the mature V1/V2/FAST/bank/vault pool. V2.25.1 region proposals are inside the mature worker. can_play_03/04 contain eight operator-verified real object hits: GT object supported 8/8, but gameplay UNKNOWN 8/8 due competing support. The false-audio control correctly has zero object support. No exact hole XY is available.
- The original candidate reconciliation compared saved ledgers and immutable historical stage caches. Complete mature pools were absent in can_play_03/04; the paired capture above now fills that gap for five newer events, but lacks independent truth. Historical registered mature jobs took ~1.35–1.48s; neither those jobs nor the newly measured deliveries can simply replace the350ms object query. Use `object_candidate_reconciliation --paired` for saved exact-shape membership and delivery boundaries. No physical capture or live promotion was performed by either analysis task.
- Preserve negative results: waiting for multi-support decay, simple signed freshness, early→late birth, dense residual, settled permanence and simple scalar ranking did not safely resolve object identity. Do not rerun them. Do not assume controlled illumination is required. Preserve multiple-supported → UNKNOWN; no highest-score shortcut or selector/locality research.
- Existing **Skjutprylar** remains the gameplay integration. The saved-crop replay and exact scalar/batch equivalence remain available; no deadline was extended. Optional bounded candidate sidecars use the existing trace writer; default gameplay/candidate authority is unchanged. See CURRENT_STATE.md and PHYSICAL_CAPTURE_RUNBOOK.md.
- Reuse the existing GameObject / HitRegion / shot-id / frozen-shot geometry and object-local physical search. Do **not** build a parallel detector and do not rebuild existing games such as the can-shooting game.
- Add/finish `OBJECT_HIT` as an opt-in engine semantic: the game supplies active object regions; the physical engine inspects only those relevant regions and returns the physically supported object, `UNKNOWN`, or no hit.
- `OBJECT_HIT` must not be implemented as global exact-XY selection followed by nearest-object snapping. Region context narrows where physical evidence is evaluated; it never creates a hit.
- Keep the normal exact-coordinate result and best physical candidate XY as diagnostics so gameplay success never hides general detector weakness.
- Use existing games first, especially the can-shooting game and Shoot/Don't Shoot, for small physical acceptance runs. The immediate practical goal is that static object gameplay is fun and reliable enough to use with friends.
- Treat global FULL-RESCUE carefully: for `OBJECT_HIT`, evidence outside all requested active object regions must not create an object hit. Do not change ordinary exact-coordinate semantics without independent evidence.
- Prefer short physical sessions that directly answer gameplay questions. Stop spending whole Codex windows on single scalar-ablation research unless a concrete gameplay blocker requires it.

### NEXT — broaden physical evidence and collect useful volume

- Implement a **breadth-first evidence pass**. Evidence families may begin as version 0.1 if they are physically meaningful, source-independent where possible, explicit about missing/abstain, and regression-safe. Do not require each family to be fully optimized before the next one exists.
- Prioritize distinct evidence families rather than many correlated scalar variants. Candidate families include chronological onset/birth, persistence/revocation, exact witness lineage, morphology/damage compatibility, core-vs-ring locality, radial/edge structure, signed/unsigned change, local stability/noise, board/panel motion, projector/edge nuisance, known-hole/re-hit/enlargement context, surface revision, chromatic response and camera/compression nuisance.
- Extend the existing EvidenceBundle/common-verifier path rather than creating a redundant MasterVerifier. Every evidence output should carry a small contract: values, validity/missingness, evidence family, candidate/region, causal time window, live availability and measured compute cost.
- Build a compact **evidence ledger** so every useful physical shot can record object regions, candidates, evidence outputs, selected result, timing and model/version provenance without retaining huge raw artifacts by default.
- Keep gameplay and exact-coordinate scoreboards separate: `OBJECT_HIT`, `ZONE_HIT`, `TARGET_OR_MISS` and `EXACT_COORDINATE` are different requested semantics using the same physical engine underneath.

### NEXT — make latency and storage first-class engine constraints

- Introduce/finish measured performance modes such as `FAST`, `BALANCED` and `MAX_ACCURACY`. A game should be able to request both result semantics and latency/accuracy budget, for example `OBJECT_HIT + FAST`.
- Add an evidence scheduler/budget model. Cheap high-value evidence runs first; expensive evidence is skipped, deferred or run in shadow when its marginal value does not justify latency for that request.
- Support **early gameplay decision + shadow analysis**: once sufficient physical evidence exists for a fast object hit, the game may react immediately while slower evidence continues asynchronously for diagnostics/training. Late evidence must not retroactively move gameplay hits.
- Measure per-evidence runtime and marginal value. Remove or defer evidence that is too expensive for its contribution, especially in action games where hit/no-hit must resolve quickly.
- Storage policy: always keep compact metadata/evidence/timing; save image crops or short PRE/POST windows selectively for UNKNOWN, disagreement, novel/failure cases and sampled events; reserve long full-resolution traces for explicit research captures. Large raw data and evaluations stay under `/data/skjutbana/`, never Git.
- Evaluation output must remain bounded/append-friendly. Do not repeat the historical unbounded JSON/leaderboard growth pattern.

### LATER — offline Physical Teacher and nightly learning

- Reuse saved physical PRE/POST histories to reconstruct board changes chronologically instead of treating every shot as an isolated classification problem. Later frames may be used **offline only** to determine whether a suspected physical change persists.
- Build an offline read-only Physical Teacher over existing surface/history owners rather than a new competing board-state subsystem. It may combine genuinely different evidence families, spatial hypothesis clustering, physically legitimate perturbation/replay stability and future persistence to create confidence-gated pseudo-labels.
- Use model/seed/registration perturbation as a robustness measurement, not as fake independent votes. Correlated models/evidence remain one evidence family.
- Feed high-confidence teacher labels plus human GT into nightly model/evidence evaluation. Ambiguous cases should enter a small active-learning review queue instead of forcing labels.
- Night jobs should compare evidence families, fusion/selectors and runtime/value tradeoffs using compact databases/top-N/failure artifacts rather than preserving every intermediate tensor/image forever.

### LATER — continue general exact-coordinate improvement

- Resume exact-coordinate proposal/selection work after practical `OBJECT_HIT` gameplay is usable and the richer evidence ledger provides more development data.
- Current exact-coordinate research remains preserved, including the safe offline linear selector challenger, the frozen candidate-oracle gap, negative temporal/locality feature results, and leakage-safe Vision/Evidence infrastructure. Do not redo completed sweeps.
- Completed narrow locality-midrank ablation (2026-10-08): all three seeds reach24/25/26/27 selected, but false controls remain1/3/1. Rejected; safe23/24/25/26 remains. Stop locality scalar refinement. A separate absolute-evidence emission gate is a parked design option only; no new model/search was started. Full report: `/data/skjutbana/evaluation_runs/locality_midrank_ablation_20261008_01/REPORT.md`.
- General detector improvements should benefit both exact-coordinate and object-level semantics where possible. Gameplay priors may reduce the search area, but improvements should stay in shared physical evidence/proposal infrastructure rather than object-specific hacks.
- `NO_IMPACT` remains a separate capability. UNKNOWN is preferable to a fabricated hit.
- S03 stays untouched until a genuinely frozen candidate is ready for independent validation.

### FUTURE — richer gameplay and adaptive board understanding

- Continuous moving-object support using shot-time frozen geometry and the existing GameObject lifecycle.
- Revision-aware board/surface models: current paper/cardboard, holes, repairs/tape, projected image, unstable regions and long-term board response. Reuse existing known-hole/reference ownership before consolidation.
- Optional weapon/audio context using deterministic signal processing first; hit detection must not depend on perfect weapon classification.
- Camera-2 AimPrior and GameAttentionMap remain priors/capabilities, never physical truth.
- AI-readable GameEngine capability documentation and rapid generation of new games after the reusable mechanics are stable.

**ENGINE PROVIDES MECHANICS. INDIVIDUAL GAME PROVIDES RULES.** Games own scoring, lives, animations, target/no-target logic and win conditions. The engine owns physical hit semantics, object/zone resolution, timing budgets and reusable services. Fictional z-order never derives from camera XY.

Inventory and ownership constraints:
[PHYSICAL_BOARD_STATE_ARCHITECTURE.md](PHYSICAL_BOARD_STATE_ARCHITECTURE.md).
The historical gameplay checkpoints below do not supersede these priorities.


<!-- V2.24.0 GAME_HIT_CONTEXT -->
## Game-ready hit-engine path (V2.24+)

1. **V2.24.0** — optional game `HitRegion` AABB contract, wrapper proxies,
   shot-time snapshot, game/screen/camera transforms, documentation.
2. **V2.24.1** — object-aware local physical search in camera regions with
   global fallback and shadow metrics.
3. **V2.24.2** — moving-region/debug verification scene and physical tests.
4. **V2.25.0** — common GameObject / Hittable / Breakable / ObjectManager layer.
5. Build new games on the stable HitEvent + object-context interfaces while AI
   research continues behind the same hit engine.

<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->
## Game-ready hit-engine path — V2.24.1 checkpoint

- [x] **V2.24.0** — optional `HitRegion` API, game-local coordinates,
  four-corner game->camera transform and shot-time snapshot.
- [x] **V2.24.1** — region-first physical FAST proposal search, merged camera
  windows, mandatory physical confirmation and global V2.22.5 fallback.
- [ ] **V2.24.2** — dedicated moving/overlap/no-shoot/edge/outside-region test
  scene and physical verification metrics.
- [ ] **V2.25.0** — small shared GameObject / HittableObject /
  BreakableObject / ObjectManager layer.
- [ ] Resume broader game production on the stable HitEvent + HitRegion
  contracts while AI/dense/direct-heatmap research continues behind the same
  hit engine.

<!-- V2.24.2 GAME_CONTEXT_TESTSCENE -->
## Game-ready hit-engine path — V2.24.2 checkpoint

- [x] **V2.24.0** — HitRegion API, transforms and shot-time snapshot.
- [x] **V2.24.1** — object-aware local physical search with global fallback.
- [x] **V2.24.2** — dedicated target/no-shoot/moving/overlap/edge/outside/empty
  game-context verification scene.
- [ ] **V2.24.2 physical acceptance** — run the short prescribed shot matrix and
  inspect local-search, fallback, frozen geometry and returned XY.
- [ ] **V2.25.0** — first shared GameObject / HittableObject / BreakableObject /
  ObjectManager layer, assuming V2.24.2 acceptance is clean.

<!-- V2.24.3 DIRECT_ROADMAP -->
## V2.24.3 corrective checkpoint

V2.24.2 physical testing exposed an ROI integration defect. V2.24.3 fixes implicit content-rect origin, moves local region restriction to HitScanner ROI level, preserves global FULL-RESCUE and upgrades the physical testscene. Physical acceptance of V2.24.3 precedes V2.25.0.

<!-- V2.24.4 DIRECT_ROADMAP -->
## V2.24.4 — detector working-space mapping

V2.24.3 physical testing isolated a canonical-camera vs V2.22.1 crop-local
coordinate mismatch. V2.24.4 maps frozen camera HitRegions into the active
detector working image before building the first-pass mask. Physical acceptance
of this bridge is the final V2.24 gate before V2.25 reusable game objects.

<!-- V2.24.3 LOCAL_ROI_INTEGRATION_FIX -->
## Game-ready hit-engine path — V2.24.3 correction

- [x] **V2.24.0** — HitRegion API, transforms and shot-time snapshot.
- [x] **V2.24.1** — first object-local proposal implementation.
- [x] **V2.24.2** — dedicated physical game-context testscene.
- [x] **V2.24.2 physical probe** — exposed ROI integration/bypass defect.
- [x] **V2.24.3 code** — HitScanner-level local ROI, implicit content-rect fix,
  latest-calibration refresh and global FULL-RESCUE preservation.
- [ ] **V2.24.3 physical acceptance** — repeat target/no-shoot/moving/overlap/
  edge/outside/EMPTY matrix and verify LOCAL-ROI/ROI-RECOVERY/global rescue logs.
- [ ] **V2.25.0** — GameObject / HittableObject / BreakableObject / ObjectManager
  after V2.24.3 physical acceptance.

<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->
## Game-ready hit-engine path — V2.24.4 working-space correction

- [x] **V2.24.0** — HitRegion API, transforms and shot-time snapshot.
- [x] **V2.24.1** — first object-local proposal implementation.
- [x] **V2.24.2** — dedicated physical game-context testscene.
- [x] **V2.24.3 physical probe** — isolated full-camera vs crop-local ROI mismatch.
- [x] **V2.24.4 code** — map frozen camera AABBs into V2.22.1 analysis working
  space; preserve physical authority and global FULL-RESCUE.
- [ ] **V2.24.4 physical acceptance** — verify `ROI-MAP`, non-zero `region`,
  target/no-shoot/moving/overlap/edge/outside and EMPTY/global behaviour.
- [ ] **V2.25.0** — GameObject / HittableObject / BreakableObject / ObjectManager
  after V2.24.4 physical acceptance.

<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->
## V2.25 object-system path

- [x] V2.24.4 — physical local-ROI bridge accepted.
- [x] V2.25.0 — composable GameObject foundation, exact frozen collision,
  shot-id HitEvent bridge, gameplay penetration/damage, reactions/effect requests.
- [x] V2.25.1 — region-balanced physical proposal/confirmation inserted after physical XY failures.
- [x] V2.25.2 — registered PRE→POST freshness authority and early legacy gate.
- [ ] V2.25.4 — continuous moving-object updates using exact shot_id snapshot collision.
- [ ] V2.25.5+ — effect/audio dispatcher and multipart aggregation when concrete games need them.
- [ ] Build/migrate production games incrementally on the stable object API.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 detector/object bridge correction

- [x] V2.25.0 — composable GameObject foundation and physical shot-id/frozen bridge.
- [x] V2.25.1 — build balanced per-object physical proposal/confirmation; physical
  acceptance pending.
- [x] V2.25.2 — registered freshness authority + early legacy emission gate, inserted after V2.25.1 physical testing.
- [ ] V2.25.4 — continuous moving-object updates while CV resolves, using frozen shot-id snapshots for exact collision.
- [ ] V2.25.5 — effect/audio dispatcher foundation if needed by the first migrated game.
- [ ] Migrate/build production games only after V2.25.2 physical XY is accepted.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 authority correction

- [x] V2.25.0 — composable GameObject foundation + shot-id/frozen bridge.
- [x] V2.25.1 — balanced per-region physical proposal/confirmation.
- [x] V2.25.2 — registered freshness authority + early legacy emission gate; physical acceptance pending.
- [ ] V2.25.4 — continuous moving-object updates while CV resolves, retaining frozen PANG collision.
- [ ] V2.25.5 — effect/audio dispatcher foundation when required by the first migrated game.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 authority correction

- [x] V2.25.0 — composable GameObject foundation + frozen shot bridge.
- [x] V2.25.1 — balanced physical region proposals.
- [x] V2.25.2 — registered PRE→POST freshness gate.
- [x] V2.25.3 — worker/main readiness bridge + cross-shot physical novelty; acceptance pending.
- [ ] V2.25.4 — moving-object continuity after physical XY acceptance.
- [ ] V2.25.5 — effect/audio dispatcher when required by a migrated game.

<!-- V2.25.3-r3 FULL_FILE_DELIVERY -->
## V2.25.3-r3 – full-file delivery

Packaging-only correction after the V2.25.3 runtime work. Future delivery for this development line uses complete replacement files only: no prepare/apply scripts and no menu/settings mutation helpers. `content/menu.json` is shipped as a complete schema version 1 file with the diagnostic games already present. Central configuration files are not replaced unless the version actually requires a source change.
