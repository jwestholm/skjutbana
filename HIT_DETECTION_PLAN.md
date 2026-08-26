# Hit Detection Master Plan — current source of truth

**Status:** Active plan from Detector V2.12 onward.  Older detector/roadmap MD files remain useful history, but if they conflict with this file, use this plan for the current hit-detection work.

## 1. Actual product goal

This is a **physical shooting range**. A microphone supplies the shot-time anchor, a camera observes the projected shooting surface, and the projector/game engine knows what was displayed. The immediate objective is:

> For every real shot that creates a new hole inside the playable area, find the **new** hole among 0..n old holes and emit the correct game-space hit automatically.

Target for game authority:

- aspirational: 100% correct hit position,
- minimum release gate: **>=95% final correct hit detections** on genuinely unseen physical sessions with mixed backgrounds,
- no assumption that one detector must reach 95% by itself.

The intended solution is an **evidence-fusion system**. Classical CV, direct image AI, temporal evidence, projector knowledge and game context may complement each other. Context is a soft prior, not permission to invent a hit.

## 2. Core architecture

```text
microphone shot peak
        |
        v
known shot time T
        |
        +---------------- physical evidence ----------------+
        |                                                   |
        |  current V1/V2 camera detector -> candidates      |
        |  direct before/after image evidence -> heatmap    |
        |  future Hole-AI -> candidate scores / heatmap     |
        |  future AI proposal network -> extra candidates   |
        |                                                   |
        +---------------- context evidence -----------------+
        |                                                   |
        |  expected projector frame / video frame           |
        |  game objects + hitboxes + likely target objects  |
        |  shooter/session spatial prior                    |
        |                                                   |
        +-------------------------+-------------------------+
                                  |
                                  v
                         candidate/evidence union
                                  |
                                  v
                         learned fusion/ranking
                                  |
                                  v
                       authoritative camera X,Y
                                  |
                             homography
                                  |
                                  v
                          viewport/game X,Y
```

### Non-negotiable design rules

1. **Same perception code offline and live.** Offline replay adapts frame input; it must not grow a second copy of the live detector.
2. **Never delete truth early to make candidate lists look clean.** Candidate generation is recall-oriented. Ranking/fusion is allowed to be selective later.
3. **Every evidence source keeps provenance.** We must always be able to answer why a candidate existed and which source rescued it.
4. **Physical evidence and context are measured separately.** A game hitbox may increase probability; it must not make an otherwise impossible physical change true.
5. **Synthetic data trains; unseen real data decides.** Generated results are never the final acceptance test.
6. **Split by physical session, not random image.** Near-identical frames from one session may not leak into train and holdout.
7. **One new evidence source at a time.** Establish baseline -> add one source -> measure complementarity -> keep/reject -> only then combine further.
8. **"Hole" and "new hole" are different labels.** Static Hole-AI must treat old + new physical holes as hole-like positives. A temporal NEW-hole model may correctly treat old holes as negatives because they did not appear at the current shot. Never convert `far from current GT` directly into a static non-hole label.
9. **Known-hole state is soft context, not truth.** Reuse `HitScanner.known_holes`; do not build a competing registry. It is session-local/incomplete, so before/after novelty must still handle holes that existed before the scene started and hole-in-hole cases.

## 3. Metrics — always report the funnel

A single Top-1 percentage hides where the failure happens. Every serious benchmark should eventually report:

- shots with valid ground truth,
- V1/V2 candidate recall at <=5/10/20/42 camera px,
- each independent source's recall at the same radii,
- **union recall** across candidate sources,
- source complementarity: `A-only`, `B-only`, `both`, `neither`,
- conditional ranking accuracy when GT exists in the union,
- final selected-hit accuracy,
- median/P95 position error,
- candidate count and compute time,
- results by background class,
- results by old-hole density / overlap class,
- results by session (not only overall average),
- live latency from audio peak to emitted hit.

The most important early metric is **union candidate recall**. A ranker cannot select a hole that no source proposed.

## 4. Data strategy

### A. Existing image banks — use each for what it really contains

The shooting computer currently has two historically different image banks:

- `content/ai/shot_diag/`: human diagnostics. These images contain drawn crosshairs/text and are **not model-training input** or an honest full-frame replay benchmark. Keep them for debugging only.
- `content/ai/holes/`: raw 128x128 camera patches plus JSON sidecars. Current inventory observed on the shooting PC is about **15,134 `synt_*` synthetic-projector/camera hole patches** and **37 `hole_*` real physical-hole patches**.

The `synt_*` patches are especially useful because the synthetic overlay went through the real projector, physical surface and camera before the patch was saved. They therefore contain real optics/exposure/noise even though the hole itself was synthetic.

The 37 real `hole_*` patches are too few to train a robust model, so V2.13 treats them as a **golden real holdout by default**. They are never used for model fitting or threshold selection.

Important format fact: the hole bank is centred by construction. V2.13 must therefore never train on the raw 128x128 source image as a binary shortcut. The source image is only a reservoir from which candidate-centred crops are sampled at varied offsets.

### B. Background diversity

Do **not** throw away the mostly-white synthetic camera archive. Add diversity in layers:

1. existing `synt_*` patches from white / white_grid / checker_anim and the smaller black/checker/gray/bubbles sets,
2. background-class holdouts to measure whether the model learned the hole rather than one projection pattern,
3. later: extracted/calibrated hole appearance composited onto many game/projector backgrounds,
4. later: raw camera before/after captures with several static/dynamic projected backgrounds,
5. a smaller but sacred mixed-background **physical** holdout.

### C. Million-iteration work

Do not rerun 4K computer vision unnecessarily for every optimizer step.

- Full-frame replay: compile physical evidence/candidate pools from thousands/tens of thousands of real shots.
- Candidate/patch/fusion training: reuse compiled data and run hundreds of thousands or millions of cheap iterations.
- Re-run full frames only when detector/image-processing code changes.

This separates expensive perception from cheap learning and is the path to day-scale autonomous experiments.

## 5. Evidence-source roadmap

### Source 0 — Current V1/V2 detector

**Status:** existing/live baseline. V2 is high-recall candidate generation with multiple signal/rescue paths and candidate bank.

### Source 1 — V2.12 Temporal-consensus physical overlay

**Status:** implemented offline in V2.12.

Inputs:

- pre-shot frame stack,
- post-shot frame stack.

Separate component overlays:

- `absdiff`
- `darkening`
- `persistent_zscore`
- `temporal_consensus`
- `local_contrast`
- `physical_fusion_v212`

Purpose: independently propose locations where a new physical change appears after T and stays in the same registered camera location. First question is **not** whether this source is perfect; it is whether it rescues GT on shots current V2 misses.

### Source 2 — Hole-AI appearance model

**Status:** V2.13–V2.15 implemented and frozen for candidate-level experiments.

This source answers only:

> `P(this candidate looks like a physical bullet hole)`

Old and new physical holes are both positive hole-appearance evidence. The model must never be taught that an old hole is a `non-hole` merely because it is far from the current shot GT. V2.15 mild+standard remains a shadow evidence source.

### Source 3 — NEW-hole before/after model

**Status:** V2.17 implemented offline/shadow.

This source answers a different question:

> `P(a hole-like physical change appeared at this candidate at the current shot)`

Inputs are matched pre-shot + post-shot candidate patches. For this task an old unchanged hole is correctly negative. V2.17 therefore can safely learn from V2.16 wrong candidates without corrupting static Hole-AI semantics. It also predicts an X/Y refinement offset.

### Source 4 — AI direct proposal / heatmap

**Status:** later.

Purpose: allow AI to propose a location current CV never emitted. This is how combined recall can exceed the classical detector ceiling.

### Source 5 — Projector-aware residual

**Status:** later, after physical replay works.

The program knows the rendered/projected frame. Learn/model what the camera should see and suppress explained scene/video motion. Unexplained persistent physical residual is strong hole evidence.

### Source 6 — Game-object/context prior

**Status:** later.

Examples:

- active shootable object bounds,
- no-shoot object bounds,
- object importance / likely aim target,
- current game state.

Use as a **soft prior/tie-breaker**. A miss beside an enemy must remain a miss.

### Source 7 — Shooter/session spatial prior

**Status:** later and deliberately weak.

The centre of the target/play area may statistically be more likely, but corners remain possible. This source may resolve ambiguous physical candidates; it must never hard-exclude an unusual shot.

## 6. Version/work TODOs

### V2.12 — Offline Replay & Evidence Foundation

- [x] Add this stable master plan.
- [x] Add read-only archive inspector.
- [x] Pair common `pre/before/ref` and `post/after` file conventions.
- [x] Discover sidecar ground truth when present.
- [x] Report standalone labelled hole patches separately for future Hole-AI.
- [x] Write portable JSONL `ShotCase` manifest.
- [x] Add hardware-free adapter for the **current live HitScanner V1→V2 hybrid wrapper** (same detector code, replay-provided pre-shot/ROI).
- [x] Keep V1/V2 provenance separately measurable inside the current-detector result.
- [x] Prevent replay from writing live detector diagnostics.
- [x] Add direct-image temporal-consensus evidence components.
- [x] Convert fused evidence map to permissive candidate proposals.
- [x] Create current-live-detector + overlay **union recall pool** without changing live authority.
- [x] Measure current detector, V1 provenance, V2 provenance, agreement, every physical component overlay, fused overlay and union independently.
- [x] Measure `overlay rescued CURRENT LIVE detector miss` explicitly.
- [x] Add evidence-map visualizer.
- [x] Add synthetic plumbing/stress test.
- [ ] Run archive inspector on the shooting PC and adapt filename pairing if its historical format differs.
- [ ] Run first real offline V2 vs overlay vs union benchmark.
- [ ] Freeze a first real-session holdout before tuning V2.12 weights.

**V2.12 status note:** the discovered `shot_diag` archive is annotated human-debug imagery rather than raw full-frame replay input, so the original replay gate cannot yet be satisfied from that archive. V2.12 remains the correct full-frame framework and will resume when raw before/after frames are available. This does **not** block the separate V2.13 Hole-AI learning proof because `content/ai/holes` is raw patch data with a different purpose.

### V2.13 — Hole-AI Learning Proof + Dataset Discipline

- [x] Inventory `synt_*` and real `hole_*` separately.
- [x] Make `shot_diag` explicitly diagnostic-only in the active plan.
- [x] Deterministic **session-level** synthetic train/validation/test split.
- [x] Separate novel-background holdout (default: black/checker/gray/bubbles).
- [x] Keep real `hole_*` assets as golden holdout only.
- [x] Prevent the centred-128x128 shortcut by candidate-centred crop jitter.
- [x] Generate same-image local candidate negatives away from GT.
- [x] Train a small dependency-free pixel MLP (numpy/OpenCV only).
- [x] Add auxiliary X/Y offset prediction to force localisation.
- [x] Compare Hole-AI against a simple non-learning centre-contrast baseline.
- [x] Report synthetic session holdout, unseen-background holdout, real holdout and explicit off-centre stress.
- [x] Run V2.13 on the 15,134-image shooting-PC archive and inspect learning curves / real holdout.
- [x] Decide KEEP/REJECT from unseen results, not training accuracy.

**Observed V2.13 result on the shooting PC (2026-08-26):**

- validation AUC: **0.874595**
- synthetic-test AUC: **0.837996**
- strict novel-background AUC: **0.529693**
- real holdout: AUC **0.924489**, recall **0.918919** (34/37 real holes)
- off-centre stress AUC: **0.800802**

**Decision: KEEP the pixel-AI path.**  V2.13 proves useful synthetic->real transfer and survives the centre-position anti-cheat test.  The dominant weakness is now clearly background generalisation, not lack of learnability.

### V2.14 — Background-Generalising Hole-AI

**Status:** implemented offline.  This version deliberately attacks the measured V2.13 failure before live/shadow integration.

- [x] Preserve the same strict novel-background holdout (black/checker/gray/bubbles) so V2.13 and V2.14 remain comparable.
- [x] Keep all real `hole_*` patches out of training and model selection.
- [x] Replace intensity-heavy input with local physical maps: local residual, DoG, morphological black-hat and gradient energy.
- [x] Add procedural/background-remix augmentation that preserves compact hole residuals while changing projector/background structure.
- [x] Add strong photometric/shadow/projected-edge/noise variation during training.
- [x] Select the best epoch using clean validation **plus procedural domain-stress validation**, never the strict novel or real holdouts.
- [x] Keep candidate jitter and X/Y offset localisation from V2.13.
- [x] Add explicit procedural-domain stress reporting and per-background metrics.
- [x] Compare V2.14 report automatically with the saved V2.13 report when available.
- [x] Add mild/standard/strong profile sweep whose winner is ranked without strict holdout data.
- [x] Run V2.14 `standard` on the complete shooting-PC archive.
- [x] Run the three-profile sweep after standard remained just below the candidate-shadow gate.
- [x] KEEP V2.14 background-invariant representation; reject `strong` profile and carry `mild` + `standard` forward for a paired-complementarity test.

**V2.14 first gate for the next integration step:** strict novel-background AUC >=0.70, real-hole recall >=0.85 and off-centre AUC >=0.78.  This is intentionally a candidate-shadow gate, not the final >=95% game-authority gate.

**Observed V2.14 full standard result on the shooting PC (2026-08-26):**

- validation AUC: **0.879472**
- synthetic-test AUC: **0.842980**
- strict novel-background AUC: **0.692312** (V2.13: 0.529693; +0.162619)
- real holdout: AUC **0.913075**, recall **0.905405**
- off-centre AUC: **0.772425**

**Observed V2.14 sweep (8 epochs/profile):**

| profile | non-holdout selection | strict novel AUC (report-only) | real recall (report-only) | decision |
|---|---:|---:|---:|---|
| mild | **0.7734** | **0.741301** | 0.824324 | keep for paired test |
| standard | 0.7275 | 0.681095 | **0.945946** | keep for paired test |
| strong | 0.6779 | 0.643597 | 0.851351 | reject |

**Important V2.15 audit finding:** the original V2.14 sweep changed the single `seed` per profile, and that seed also controlled whole-session train/validation/test assignment. Therefore the `mild` and `standard` non-holdout selection scores above are not perfectly apples-to-apples. Strict novel-background and real holdouts remained valid because they never entered training/model selection, so the qualitative finding (mild generalises better to unseen backgrounds; standard transfers better to the real-hole bank; strong augmentation is too destructive) remains useful. Starting with V2.15, **split seed and model/training seed are separate**, and paired models must prove identical session-assignment provenance before any ensemble is selected.


### V2.15 — Paired Hole-AI Evidence + Experimental-Discipline Fix

**Status:** implemented offline/shadow foundation.  V2.15 intentionally comes *before* full-frame hard-negative capture because the V2.14 sweep first needs an honest paired error-overlap test.

- [x] Separate `split_seed` from model/training seed in V2.14 training.
- [x] Fix future V2.14 sweeps so all profiles use one identical whole-session split.
- [x] Add `hole_v215_pair_train` to retrain only `mild` + `standard` on the same session assignment.
- [x] Persist and verify split provenance in model metadata.
- [x] Refuse V2.15 ensemble selection when mild/standard provenance does not match.
- [x] Evaluate both models on **exactly the same candidate-centred examples**.
- [x] Measure score correlation, disagreement, `standard-only` rescues, `mild-only` rescues, both-hit and neither-hit.
- [x] Report oracle `either model` recall explicitly as diagnostic only (never as an achievable live score).
- [x] Search a one-parameter threshold-centred mild/standard blend, including pure endpoints, using **clean validation + procedural domain-stress only**.
- [x] Freeze weight/threshold before evaluating synthetic test, strict novel backgrounds, real holes and off-centre stress.
- [x] Add a reusable `HolePatchEnsembleV215` shadow annotator that preserves candidate ordering/coordinates and only appends evidence fields.
- [x] Run paired mild+standard training on the shooting PC.
- [x] Run V2.15 ensemble/complementarity report on the full archive.
- [x] KEEP two-model inference for candidate-level testing: paired error overlap shows material complementarity on difficult domains.

**V2.15 candidate-shadow gate:** non-holdout ensemble selection must be no worse than 99.5% of the best pure endpoint, strict novel AUC >=0.70, real-hole recall >=0.85, and off-centre AUC >=0.76.  Passing means only that patch evidence is worth feeding into later candidate-level shadow analysis.

**Observed corrected V2.15 paired result on the shooting PC (2026-08-26):**

- shared whole-session split verified: **True**,
- mild non-holdout select **0.777557**, strict novel AUC **0.705364**, real recall **0.918919**,
- standard select **0.731138**, strict novel AUC **0.692606**, real recall **0.918919**,
- selected blend: **62.5% mild + 37.5% standard**, selection **0.715013** vs best pure **0.713042**,
- fused strict novel AUC **0.704868**, real-hole AUC **0.926771**, off-centre AUC **0.788914**,
- complementarity is substantial on hard domains: strict novel **0.214592**, procedural stress **0.209167**,
- all V2.15 gates and `hole_v215_verify` passed with `shadow_only=True`.

**Decision:** V2.15 is frozen/KEEP for candidate-level experiments.  Stop optimizing patch-only metrics for now.  The next question is whether its evidence improves ranking among the *actual detector candidates*.

### V2.16 — Candidate-Level Shadow Capture + Real Hard Negatives

**Status:** first candidate-level dataset captured and benchmarked on shooting PC (seed 65432, white, 100 shots).

- [x] Instrument only `AutomationAITrainingScene` so normal game/manual hit authority remains untouched.
- [x] Save candidate-centred pre/post patches for actual ranked V1/V2/funnel candidates plus raw detector extras.
- [x] Save GT distance, <=10/20/42 labels, candidate provenance and JSON-safe original detector features.
- [x] Save dedicated GT patches even when no candidate lands on GT.
- [x] Make optional storage-aware full-frame pre/post capture available, disabled by default.
- [x] Explicitly mark any GT-nearest row retained only because of storage cap as diagnostic-only so capture cannot fake live recall.
- [x] Score the same candidate with V2.15 Hole-AI, local temporal before/after evidence and V9 when available.
- [x] Benchmark current ranked pool separately from raw+ranked union.
- [x] Add transparent fusion-weight search including pure endpoints; no assumption that more sources are automatically better.
- [x] Split by whole capture session once >=3 sessions exist; mark one/two-session shot splits **provisional**.
- [x] Add real detector hard-negative mining and optional PNG export.
- [x] Keep all V2.16 output shadow/offline only; no candidate reorder or coordinate override.
- [x] Capture first 1x100 candidate dataset with a completely new seed (`65432`).
- [x] Run V2.16 candidate benchmark and inspect current vs Hole-AI vs temporal vs V9 vs fusion.
- [x] Mine 800 real detector hard-negative **candidates** from that run.
- [x] Audit label semantics before using those candidates for learning.
- [ ] Capture at least two more independent physical/projected sessions before treating confirmation/holdout as non-provisional.

**Observed V2.16 seed-65432 result (2026-08-26):**

- 100 labelled shot packs, **38,423 candidates**, avg **384.23/shot**, capture errors 0,
- 23 diagnostic forced-GT-nearest rows (excluded from live-pool/oracle metrics),
- ranked-pool <=20px oracle recall: development **36.67%**, confirmation **25%**, holdout **45%**,
- raw+ranked union <=20px oracle: **40% / 35% / 50%**,
- V9 improved Top-1 vs current in confirmation (**10% vs 5%**) and holdout (**15% vs 10%**),
- V2.15 Hole-AI alone had **0% Top-1** on this hard real candidate pool,
- first local temporal heuristic also had **0% Top-1**,
- selected fusion therefore assigned **0 weight to Hole-AI and temporal**, keeping V9/current only,
- all live-authority gates correctly remained false because there is only one capture session.

**Important semantic correction after V2.16:** a far-from-current-GT candidate may be an old real bullet hole. The 800 mined rows are therefore `NOT CURRENT NEW HOLE`, not automatically `NON-HOLE`. Static Hole-AI must not be retrained with them as label 0.

### V2.17 — Old-hole-safe NEW-hole Learning

**Status:** implemented offline/shadow; this is the next active experiment.

- [x] Keep V2.15 Hole-AI as an appearance model where old + new holes are positive.
- [x] Introduce a separate before/after `NewHoleAIV217` for `P(current NEW hole at candidate)`.
- [x] Use dedicated GT pre/post patches as positives even when V1/V2 missed GT.
- [x] Use V2.16 mined far-from-current-GT candidates as valid `NOT-NEW` negatives.
- [x] Explicitly allow those negatives to be old holes; never export them as static `non-hole` labels.
- [x] Add offset refinement and post-frame persistence to the temporal model.
- [x] Reuse the existing V2.16 provisional/session split discipline.
- [x] Snapshot the existing `HitScanner.known_holes` registry before future F2 rounds as soft provenance only.
- [x] Capture a **true recent pre-shot camera frame** for future candidate packs in addition to the legacy stable/reference PRE. This is required for moving game/video backgrounds; it does not change live detector behaviour. Legacy seed-65432 packs remain valid via fallback.
- [x] Document that the current registry is session-local/incomplete and does not include every physical hole present at scene start.
- [x] Train V2.17 on the existing seed-65432 candidate packs.
- [x] Compare V2.17 candidate Top-1/Top-3/median-rank against V2.16 temporal/Hole-AI/V9 baseline. Result: patch AUC learned strongly, but pointwise candidate Top-1 remained 0%; ranking objective must change.
- [x] V2.17 learned useful novelty, but its pointwise ranking failed; explicitly postpone overnight optimization until V2.18 changes the objective to same-shot listwise ranking.

### V2.18 — Candidate-aware NEW-hole listwise ranking + offset refinement

**Status:** implemented offline/shadow.

- [x] Freeze/reuse the useful V2.17 before/after representation instead of discarding it.
- [x] Train candidates as whole per-shot ranking groups, not independent binary examples.
- [x] Use graded relevance through 42 px so near candidates can be refined rather than mislabeled.
- [x] Add hard pairwise pressure against real far candidates that fool the pointwise score.
- [x] Train a residual candidate->GT offset head and report raw vs refined oracle separately.
- [x] Add disk caching for the expensive frozen V2.17 candidate embeddings.
- [x] Keep known-hole registry as soft diagnostic context; no hard old-hole exclusion.
- [ ] Run V2.18 on seed-65432 and compare current/V2.17/V2.18 Top-1, Top-3, median rank and refined oracle.

### V2.19 — Offline champion/challenger / overnight learning loop

- [ ] Mine/reweight hard NOT-NEW ranking groups from frozen candidate packs.
- [ ] Train multiple challenger seeds/configurations for hours/days.
- [ ] Evaluate only on protected confirmation/holdout sessions.
- [ ] KEEP only challengers that beat the current champion without breaking representation/background gates.
- [ ] Resume safely after interruption and keep complete experiment provenance.

### V2.20+ — Direct AI proposals, learned multi-source fusion, projector/context priors, live shadow

- [ ] Add AI heatmap proposals that can rescue CV omissions.
- [ ] Add projector-frame residual source.
- [ ] Add game-object prior.
- [ ] Add spatial prior.
- [ ] External physical shadow benchmark on completely unseen sessions/backgrounds.
- [ ] Increase AI authority only after frozen real holdout gates pass.

### Game authority gate

Before camera/AI selection becomes authoritative in normal games:

- [ ] >=95% final correct hit detection on unseen physical mixed-background test sessions,
- [ ] no catastrophic background class,
- [ ] acceptable wrong-hit rate and latency,
- [ ] mouse/API/camera hits still use the same game hit interface,
- [ ] fallback/debug path remains available.

## 7. Immediate working procedure

For every new detector/evidence idea:

1. state one hypothesis,
2. run frozen offline baseline,
3. add **one** source/change,
4. replay the same development shots,
5. inspect source-only and union recall,
6. inspect rescued and newly-broken shots,
7. keep or reject,
8. only then optimize ranking/fusion,
9. validate on an untouched real session,
10. occasionally confirm with physical projector/camera F2; do not use physical F2 as the everyday optimizer loop.

This file should be updated when a gate is passed or the strategy changes materially.


## V2.17 measured conclusion (2026-08-26)

V2.17 confirmed that before/after NEW-hole information is learnable (confirmation AUC **0.935817**) but pointwise classification did **not** solve same-shot candidate ranking (confirmation/holdout Top-1 **0%**). This is why overnight automation was deliberately postponed one version: running millions of iterations against the wrong pointwise objective would optimize the wrong task. V2.18 changes the objective to per-shot listwise ranking and learned offset refinement; only after that demonstrates useful candidate-level movement should V2.19 automate champion/challenger learning overnight.
