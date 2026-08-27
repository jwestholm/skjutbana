# V2.21 plan — Physical Domain Bridge + Direct Proposals

## Purpose

V2.21 exists because the latest result separated the problem into two hard bottlenecks:

1. **proposal recall** — the true physical hit is absent from the candidate pool too often,
2. **domain transfer** — V2.18 ranks generated worlds nearly perfectly but gives 0% Top-1 on the existing physical session.

V2.21 should not be another “train longer” version. It should create the infrastructure and evidence needed to attack both bottlenecks directly.

---

## Phase 0 — audit what physical packs actually contain

Before writing a full-frame proposal model, inspect the physical V2.16 `.npz` contents.

Required questions:

- Is full `recent_pre_gray` present?
- Is one or more full `post_gray` frame present?
- Are only candidate/GT patches stored?
- Are timestamps preserved?
- Can the current pack reconstruct a full-frame PRE->POST evidence image?

### Decision branch

If full frames exist:

- reuse them immediately for V2.21 domain-gap and direct-proposal development.

If full frames do not exist:

- extend shadow capture with an opt-in storage-aware **full-frame V2.21 capture mode**,
- capture only the minimum additional physical data needed to build/evaluate proposal logic,
- keep normal game storage behaviour unchanged.

Do not assume the old diagnostics are raw frames: many `shot_diag` images contain overlays/text and are not suitable training truth.

---

## Phase A — domain-gap profiler

### Goal

Quantify why generated-world rank evidence differs from physical rank evidence.

### Inputs

- V2.20.2 generated train groups,
- frozen generated validation groups,
- physical development candidate groups,
- protected physical confirmation/holdout for report-only comparison.

### Compare at candidate level

For each source/domain, record distributions for:

- V2.17 embedding dimensions,
- V2.17 score,
- current detector score/rank,
- V9 features if present,
- local mean/std,
- signed PRE->POST change,
- absolute change,
- darkening,
- black-hat / DoG / gradient features,
- persistence across POST frames,
- distance to nearest known hole,
- candidate density per shot,
- GT candidate rank/distance.

### Suggested metrics

- mean/std and robust quantiles,
- standardized mean difference,
- Wasserstein distance,
- KS statistic,
- feature correlation shifts,
- generated-vs-physical classifier AUC as a **domain-gap diagnostic**.

If a trivial classifier can distinguish generated vs physical candidate patches with near-perfect accuracy, inspect the features/images responsible before more training.

### Critical shortcut audit

Explicitly test whether the generated data exposes unintended cues such as:

- exact GT-centred rendering artefact,
- fixed hole-scale distribution,
- simulator-specific bright rim,
- deterministic local noise halo,
- temporal noise only around GT,
- different quantisation/interpolation around generated holes,
- features that depend indirectly on GT labels.

The synthetic 98–100% result is strong enough that a shortcut audit is mandatory.

---

## Phase B — direct full-frame proposals

### Why this comes before major ranker optimization

Current physical raw+ranked union oracle <=20 px:

- development: 40%,
- confirmation: 35%,
- holdout: 50%.

A 100%-accurate ranker cannot beat these ceilings.

### Baseline proposal pipeline

Start simple and measurable:

```text
recent PRE
   +
POST stack
   |
   +--> robust temporal baseline/alignment
   +--> absdiff
   +--> signed darkening
   +--> persistent darkening/change
   +--> local contrast / morphology
   +--> optional V2.17 dense/local score
   |
   +--> fused heatmap
   +--> local maxima
   +--> non-max suppression
   +--> proposal confidence
   |
   +--> AI_DIRECT candidate list
```

Then merge:

```text
V1/V2 raw/ranked candidates + AI_DIRECT proposals
```

### First report

For each physical split report:

- V1/V2 oracle @5/10/20/42,
- AI_DIRECT oracle @5/10/20/42,
- union oracle,
- proposals/shot,
- rescued shots,
- newly introduced false proposal density,
- runtime.

### V2.21 proposal gates

Initial useful gate:

- physical confirmation and holdout union oracle >=70% @20 px.

Strong next gate:

- >=85%.

Pre-authority target:

- >=95% on multiple independent physical sessions.

The first version does not need beautiful ranking. It needs **coverage**.

---

## Phase C — physical-domain bridge

### Goal

Teach V2.18-like ranking evidence what the actual camera domain looks like without consuming protected physical tests.

### Preferred generator if full frames are available

Use only allowed physical development frames as bases:

```text
real recent PRE frame
       |
       +--> retain real paper/projector/camera texture
       +--> retain old physical holes
       +--> apply measured small temporal nuisance
       +--> insert one new synthetic hole at a new labelled location
       |
       +--> physical-domain generated POST
```

Important: the generated hole must be inserted in a way calibrated to real camera observations, not simply pasted as the exact V2.20.2 appearance that the model already overfits.

### Temporal nuisance model

Estimate from physical development data only:

- exposure/gain drift,
- sensor noise,
- blur/focus variation,
- projector flicker/banding if present,
- small frame-to-frame shifts,
- moving-content residual where applicable.

Use shared parameters when physically justified; do not independently randomize PRE and POST into unrealistic pairs.

### Mixed-domain training

Initial experiment could compare:

- 100% V2.20.2,
- 100% physical-domain generated,
- mixed 75/25,
- mixed 50/50.

Selection must be based on frozen generated validation + protected physical confirmation, not on physical holdout.

### Success criterion

A physical-domain bridge is useful only if it improves physical ranking over:

- current detector,
- original physical-trained V2.18 baseline,
- pure V2.20.2-trained V2.18,

while preserving reasonable generated validation behaviour.

---

## Phase D — candidate-aware fusion after proposals improve

Once AI_DIRECT raises oracle recall, feed all proposal sources into the same-shot ranker/fusion stage.

Candidate provenance should be explicit:

- `current_ranked`,
- `current_raw`,
- `v9`,
- `ai_direct_temporal`,
- later `projector_residual`.

The ranker should learn source reliability but must not depend on source identity as a shortcut to GT.

---

## Proposed implementation files

Names are suggestions; keep repo style consistent.

### Core

- `src/engine/offline/domain_gap_v221.py`
- `src/engine/offline/physical_domain_v221.py`
- `src/engine/offline/direct_proposal_v221.py`
- `src/engine/offline/direct_proposal_benchmark_v221.py`

### Automation

- `automation/domain_gap_v221.py`
- `automation/physical_pack_v221_inspect.py`
- `automation/physical_bridge_v221_generate.py`
- `automation/direct_proposal_v221_benchmark.py`
- `automation/offline_v221_selftest.py`

### Optional capture change

Only if the existing V2.16 packs lack full frames:

- extend `CandidateShadowRecorder`/AI-training capture with opt-in full-frame PRE/recent-PRE/POST storage,
- do not change normal live/game behaviour.

---

## Required V2.21 selftests

1. no use of physical confirmation/holdout in bridge generation,
2. provenance survives generated variants,
3. source physical frame family never crosses forbidden split,
4. direct proposals are deterministic for frozen inputs,
5. proposal benchmark does not inject forced GT candidates,
6. no GT coordinate is exposed to proposal scoring,
7. hole-in-hole case remains representable,
8. known-hole context remains soft,
9. generated validation remains read-only,
10. live detector authority/order remains unchanged.

---

## Stop conditions

Do not launch V2.22 overnight training if either remains true:

- physical candidate union oracle is still around 35–50%,
- physical Top-1 transfer remains at/near zero after physical-domain bridge experiments.

In that case, inspect rescued/missed shots and improve evidence/capture rather than increase training volume.
