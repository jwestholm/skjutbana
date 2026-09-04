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
