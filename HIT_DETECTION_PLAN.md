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

### A. Real archive — highest-value anchor

The shooting computer contains many real hole/before images. V2.12 first inventories these **read-only** and creates a portable manifest.

Use real data for:

- detector replay,
- real hole morphology,
- hard negatives,
- validation and regression,
- final unseen holdouts.

Mostly-white historical data is still valuable. It tells us what real holes, paper tearing, focus, camera noise and exposure look like in this exact installation.

### B. Background diversity

Do **not** throw away the white archive. Instead add diversity in layers:

1. real white before/after pairs,
2. extracted real hole/delta patches composited onto varied projected backgrounds,
3. real camera captures with several static/dynamic projected backgrounds,
4. a smaller but sacred mixed-background physical holdout.

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

### Source 2 — Hole-AI patch model

**Status:** next after archive/replay baseline is measured.

Input should include **before and after pixels**, not only an after image. Initial job:

> Given a candidate-centered before/after patch, estimate P(new bullet hole here).

Train on real hole pairs + hard negatives. Use session-level train/validation/test splits. First use as an extra ranking/fusion feature; do not give authority immediately.

### Source 3 — AI direct proposal / heatmap

**Status:** later.

Purpose: allow AI to propose a location current CV never emitted. This is how combined recall can exceed the classical detector ceiling.

### Source 4 — Projector-aware residual

**Status:** later, after physical replay works.

The program knows the rendered/projected frame. Learn/model what the camera should see and suppress explained scene/video motion. Unexplained persistent physical residual is strong hole evidence.

### Source 5 — Game-object/context prior

**Status:** later.

Examples:

- active shootable object bounds,
- no-shoot object bounds,
- object importance / likely aim target,
- current game state.

Use as a **soft prior/tie-breaker**. A miss beside an enemy must remain a miss.

### Source 6 — Shooter/session spatial prior

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

**Gate to V2.13:** replay a useful real dataset with stable metrics and show whether Source 1 adds complementary recall. Do not tune blindly before this result exists.

### V2.13 — Real Dataset Compiler + Recall Lab

- [ ] Turn discovered archive into deterministic train/development/holdout session splits.
- [ ] Cache/compile expensive full-frame evidence and candidate pools.
- [ ] Add per-background/per-hole-density reports.
- [ ] Mine real V2 misses and overlay misses as named error classes.
- [ ] Optimize detector thresholds only against development sessions.
- [ ] Target union recall >=95% first, then drive toward 98-99%.

### V2.14 — Hole-AI v1

- [ ] Build before/after patch dataset around GT and hard negatives.
- [ ] Train a small image model offline.
- [ ] Prove held-out-session improvement over physical features alone.
- [ ] Add Hole-AI score as shadow evidence only.

### V2.15 — Learned Fusion

- [ ] Learn source weights/calibration from V2 + temporal overlay + Hole-AI.
- [ ] Optimize selection only after union recall is high.
- [ ] Report calibration/confidence, not only rank.

### V2.16+ — Direct AI proposals, projector/context priors, live shadow

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
