# Roadmap

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
- [ ] V2.25.3 — continuous moving-object updates using exact shot_id snapshot collision.
- [ ] V2.25.4+ — effect/audio dispatcher and multipart aggregation when concrete games need them.
- [ ] Build/migrate production games incrementally on the stable object API.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 detector/object bridge correction

- [x] V2.25.0 — composable GameObject foundation and physical shot-id/frozen bridge.
- [x] V2.25.1 — build balanced per-object physical proposal/confirmation; physical
  acceptance pending.
- [x] V2.25.2 — registered freshness authority + early legacy emission gate, inserted after V2.25.1 physical testing.
- [ ] V2.25.3 — continuous moving-object updates while CV resolves, using frozen shot-id snapshots for exact collision.
- [ ] V2.25.4 — effect/audio dispatcher foundation if needed by the first migrated game.
- [ ] Migrate/build production games only after V2.25.2 physical XY is accepted.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 authority correction

- [x] V2.25.0 — composable GameObject foundation + shot-id/frozen bridge.
- [x] V2.25.1 — balanced per-region physical proposal/confirmation.
- [x] V2.25.2 — registered freshness authority + early legacy emission gate; physical acceptance pending.
- [ ] V2.25.3 — continuous moving-object updates while CV resolves, retaining frozen PANG collision.
- [ ] V2.25.4 — effect/audio dispatcher foundation when required by the first migrated game.
