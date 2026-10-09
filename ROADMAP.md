# Roadmap

## Practical gameplay + physical accuracy priorities — 2026-10-08

**The long-term target remains 100% correct physical results; 95% is the minimum practical acceptance level.**
Exact physical coordinates remain a core capability, but they no longer block useful gameplay when a game asks a simpler question such as `OBJECT_HIT`. Candidate oracle is still not selected accuracy, and game context must never manufacture physical truth.

The project must remain enjoyable to build and use. Near-term work therefore prioritizes playable physical games and reusable engine capabilities before more long-running exact-coordinate research.

### NOW — make OBJECT_HIT practically usable in existing games

- `can_play_02`: five UNKNOWN outcomes, two deadlines, no independent object GT. Deterministic saved-crop replay is available via `automation.object_hit_replay`; all available live evidence ledgers reproduce. Clock-neutral support5/4/3/1/6 remains ambiguous; two immediate PRE observations do not certify physical newness. No new gate is justified. Exact batching reduces measured replay CPU247–255ms to113–115ms without pruning or changing the350ms deadline. **Next: run the local analyzer, then a short independently observed crop retest for latency/object decisions.** Longer diagnostic PRE history is a requirement for subsequent support-rule research, not a capability already provided by the crop flag. Preserve multiple-supported -> UNKNOWN; no highest-score shortcut or selector/locality research.
- Existing **Skjutprylar** uses explicit single-object/UNKNOWN results over the region-local query; ambiguous objects abstain. `physical_collection play-can` launches normal gameplay with compact logs/15-shot notes and optional bounded actual-input crops. Physical reliability and post-batching live latency remain unverified; see CURRENT_STATE.md and PHYSICAL_CAPTURE_RUNBOOK.md.
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
