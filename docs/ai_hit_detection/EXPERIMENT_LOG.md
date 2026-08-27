# Experiment log — V2.12 through V2.20.2

This is a compact historical record of what was tried, what was measured and what was kept/rejected.

---

## V2.12 — Offline Replay & Evidence Foundation

Built:

- archive inspector,
- normalized `ShotCase`,
- current detector replay,
- temporal physical evidence maps,
- candidate union/rescue metrics,
- visualizations.

Key discovery:

- `shot_diag` is diagnostic imagery with overlays and is not clean raw full-frame training truth.

Decision:

- KEEP offline replay/evidence foundation.

---

## V2.13 — Hole-AI learning

Small NumPy/OpenCV MLP with candidate-centred jitter and anti-centre offset task.

Observed:

- validation AUC ~0.875,
- synthetic test AUC ~0.838,
- novel-background AUC ~0.530,
- real holdout AUC ~0.925,
- real holdout recall 34/37 (~0.919),
- off-centre AUC ~0.801.

Decision:

- useful synthetic->real appearance signal exists,
- background generalisation is the dominant weakness.

---

## V2.14 — Background-Generalising Hole-AI

Added:

- local residual,
- DoG,
- black-hat,
- gradient channels,
- procedural/remixed backgrounds,
- stronger domain stress.

Standard run:

- validation AUC ~0.879,
- synthetic test ~0.843,
- novel background ~0.692,
- real holdout AUC ~0.913,
- real recall ~0.905,
- off-centre ~0.772.

Sweep later revealed a methodology issue: profile seed also changed the whole-session split. V2.15 corrected this.

Decision:

- keep mild + standard representations,
- reject overly strong augmentation.

---

## V2.15 — Paired Hole-AI ensemble + split discipline

Fixed:

- split seed independent from training seed,
- shared split provenance for paired models.

Observed corrected paired run:

- mild select 0.777557,
- standard select 0.731138,
- both real recall 0.918919,
- selected blend 62.5% mild + 37.5% standard,
- fused strict novel AUC ~0.7049,
- real-hole AUC ~0.9268,
- off-centre AUC ~0.7889.

Decision:

- KEEP static hole appearance as candidate evidence,
- stop optimizing patch-only metrics until candidate-level evidence is tested.

---

## V2.16 — Real candidate capture + hard candidates

Physical seed `65432`, white background, 100 shots.

Observed:

- 100 labelled shot packs,
- ~38,423 candidates,
- avg ~384.23/shot,
- 23 diagnostic forced-GT-nearest rows excluded from honest live-pool metrics.

Ranked oracle <=20 px:

- development 36.67%,
- confirmation 25%,
- holdout 45%.

Raw+ranked union oracle <=20 px:

- 40% / 35% / 50%.

V9 was the useful existing source; Hole-AI and first temporal heuristic did not help Top-1 enough.

800 “hard negatives” were mined, followed by an important semantic correction: they are **NOT CURRENT NEW HOLE**, not guaranteed non-hole.

Decision:

- candidate recall is a major ceiling,
- preserve mined candidates for NEW-hole learning only.

---

## V2.17 — NEW-hole temporal AI

Created a separate before/after model for current-shot novelty.

Physical seed-65432 patch classification:

- development AUC ~0.924,
- confirmation AUC ~0.936,
- holdout AUC ~0.846.

But same-shot ranking failed:

- Top-1 0% across development/confirmation/holdout in the candidate-ranking benchmark.

Decision:

- novelty is learnable,
- independent pointwise classification is the wrong objective for hundreds of same-shot candidates.

---

## V2.18 — Listwise candidate ranker + offset head

Reused V2.17 representation and trained whole-shot ranking groups.

Physical seed-65432 development improved strongly:

- Top-1 13.33%,
- Top-3 18.33%,
- median GT rank 5.5.

But confirmation/holdout did not generalise enough and did not beat the best existing physical V9/fusion source.

Decision:

- KEEP listwise architecture,
- do not run long optimizer against one physical 100-shot session,
- create much more problem diversity first.

---

## V2.19 — Offline media-world engine

Added:

- deterministic seed worlds,
- media manifest and source/family splits,
- image/video support,
- old holes + incomplete known-hole state,
- near-old/hole-in-hole/edge challenges,
- generated world -> live V1/V2 replay -> V2.16-compatible candidate packs.

Initial visual QA exposed defects:

- output collapsed to grayscale,
- hole residuals looked like black/white streaks,
- PRE/POST camera changes were too independent/global.

Decision:

- architecture good,
- visual simulator needed repair before benchmarking/training.

---

## V2.20 / V2.20.1 — RGB, compact holes, QA and performance

Changes:

- preserve RGB world,
- derive grayscale only for legacy detector,
- shared camera state for static PRE/POST,
- compact procedural hole stamps calibrated from `synt_*` statistics,
- deterministic hole QA,
- fix expensive whole-frame copying and repeated image decode/resize,
- add GT debug/crop visual QA.

Decision:

- visual realism/performance became sufficient to proceed to measured experiments.

---

## V2.20.2 — transparent sprite backgrounds + richer hole texture

Changes:

- load RGBA still images with alpha,
- composite transparent regions over deterministic coloured/textured procedural background,
- avoid black empty background,
- add subtle frayed/light paper rim and tonal flecks to hole appearance.

Decision:

- stop cosmetic simulator iteration for now,
- begin quantitative training/transfer test.

---

## Generated candidate training experiment

Compiled generated seeds `1..100` into V2.16-compatible candidate packs.

Inspect:

- 100 groups,
- 38,400 candidates,
- 98 groups had candidate <=20 px and <=42 px,
- embedding cache built successfully.

V2.18 32-epoch result:

- development Top-1 96.67%,
- confirmation Top-1 100%,
- holdout Top-1 95%,
- median rank 1 throughout.

Decision:

- listwise objective can learn generated-world ranking almost perfectly.

---

## Frozen generated validation experiment

Compiled seeds `9000001..9000100` separately.

Inspect:

- 100 groups,
- 38,400 candidates,
- 100/100 candidate <=20 px,
- 100/100 candidate <=42 px.

Frozen V2.18 model result:

- development Top-1 98.33%,
- confirmation 100%,
- holdout 100%,
- median rank 1.

Decision:

- not simple seed memorisation; generalises across generated seeds.

---

## Generated-trained model -> physical V2.16 session

Frozen synthetic-trained V2.18 model, no retraining.

Physical ranked Top-1:

- development 0%,
- confirmation 0%,
- holdout 0%.

Physical median GT rank:

- 110 / 125 / 104.

Raw+ranked union median GT rank:

- 235 / 287 / 185.5.

Current physical Top-1 remained 6.67% / 5% / 10%.

Physical union oracle remained 40% / 35% / 50%.

Decision:

- **major synthetic/physical domain gap confirmed**,
- proposal recall remains a separate hard ceiling,
- move to V2.21 physical-domain bridge + direct proposals,
- do not start synthetic-only overnight optimization yet.
