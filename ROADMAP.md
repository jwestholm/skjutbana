# Roadmap

## Accuracy-first priorities — 2026-09-10

**100% correct end-to-end physical output is the target; 95% is the minimum.**
Exact physical camera XY is canonical. Candidate oracle is not selected accuracy.
Before a new central class, inventory current code, Git history, existing plans
and split responsibilities, then prefer consolidation over parallel architecture.

### NOW — preserve evidence and close the selection gap

- D01 event 1: retain useful observed coordinates through association without
  fake temporal support or inherited confirmation. OBS32 restores 2.892 px
  evidence but still ranks it 12th; existing track audit is the reuse point.
- Events 3/5: bounded retention through **both** legacy and hybrid caps. Event 3
  now has exact quota-loss diagnostics. Naive 150+50 spatial/size reserves and
  32-contour rescue failed event 5; do not deploy them or just enlarge the cap.
- Events 4/6: source-independent physical ranking/common verifier. Preserve
  event 2's 9.146 px CURRENT success as an explicit regression gate, and report
  S01/S02/POST_FIX individual regressions, counts, runtime and error tails.
- Clean physical change: PRE stability, persistence, local normalization,
  inner/ring and multiscale connected evidence. Centered PRE retains more D01
  GT but regresses S01; broad flow remains rejected. Context edges/holes/tape
  cannot automatically veto real hits.
- Defensible NO_IMPACT: existing S01/S02 false events still pass explicit local
  confirmation. Investigate event evidence/abstention; no unconditional minimum
  shot interval, relabeling or invented negatives. Rapid/grouped shots stay legal.
- All alternatives remain offline until comparator and held-session evidence
  justifies a live trial. S03 remains completely untouched.

### NEXT — small accuracy-relevant foundations

- A minimal Board view/revision contract over existing geometry, references and
  hole/context owners; audit HoleMap update/reset trust and existing stale/ridge
  cleanup first. No giant PhysicalBoardState rewrite.
- Board/panel-motion research with saved physical bounds/seam and calibration.
  Test compensation and a 3×3 Board prior separately; current crop-proxy motion
  results do not establish either calibrated panel response or general location.
- Canonical Camera ↔ Board ↔ Game API by exposing existing HitInput/ArUco/crop
  transforms. Extend the new round-trip tests with capture-time geometry and
  inverse validity/revision handling; do not replace working mappings.
- Reuse the existing audio waveform calibration scene for optional classical
  weapon profiles (peak/RMS/crest, timing, band energy/spectral/template matching)
  only when relevant to the physical verifier. Weapon identity is not mandatory.
- Review the revised six-event D02 diagnostic proposal in D01_PHYSICAL_FINDINGS:
  two physical controls plus four observed sound/motion controls. Save missing
  capture provenance first. No D02 plan/binding/capture was created in this pass.

### LATER — measured adaptation and runtime polish

- Revision-aware recent/long-term board appearance, response and noise models.
  Fast trusted hole updates; slow panel/weapon response adaptation. Only highly
  trusted confirmed physical events may update persistent state.
- Frozen per-shot GameAttentionMap: active regions/priorities/visible-object
  contributors. Search overlapping physical area once; retain contributor stack
  for game resolution. Attention is a prior and global physical search survives.
- Game-requested precision/latency modes, preserving exact physical output.
- WINDOWED / PROJECTOR_FULLSCREEN, potentially `--fullscreen` then
  `--display projector --fullscreen`. Inventory found window positioning but no
  finished fullscreen/display-index CLI. Reuse App display setup; keep desktop
  position out of physical coordinates and revalidate mappings when projection,
  monitor/resolution or letterboxed content geometry changes.

### FUTURE — downstream capabilities

- Camera-2 AimPrior: known camera/board geometry plus weapon pose/fiducial may
  estimate aim. Sight error/shooter error remain possible; physical evidence wins.
- AI-readable GameEngine capability/API documentation, potentially
  `GAME_ENGINE_API.md` / `game_engine_capabilities.json`. Inventory actual events,
  inputs, objects, scenes/lifecycle, timers, scoring, sound/video and services.
  Generate game-specific rules using existing mechanics; extend the engine only
  for a genuinely reusable missing capability. No SDK implementation now.
- Rapid AI-generated games and generative video/game continuation after the
  physical foundation is reliable.

**ENGINE PROVIDES MECHANICS. INDIVIDUAL GAME PROVIDES RULES.** DartGame owns
301/501, double-out, bull and turns. ZombieGame owns waves/health/headshots/lives.
CowboyGame owns the cowboy appearance, roughly three-second deadline, success on
hit and firing/failure on timeout. Generic GameObjects/events/timers support
those rules; camera coordinates do not supply fictional game z-order.

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
