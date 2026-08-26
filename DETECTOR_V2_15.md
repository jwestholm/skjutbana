# Detector / Hole-AI V2.15 — Paired Evidence and Honest Ensemble Test

## Why V2.15 exists

V2.13 proved that pixel learning transfers from projector-generated synthetic
holes to real projectile holes. V2.14 then improved strict unseen-background
AUC substantially. The V2.14 sweep suggested two useful behaviours:

- `mild`: stronger strict novel-background behaviour,
- `standard`: stronger real-hole recall,
- `strong`: worse overall and rejected.

That **suggests** complementary evidence but does not prove it.  V2.15 therefore
measures the two retained models on exactly the same examples before any live
integration.

V2.15 also fixes an experimental issue discovered while auditing V2.14: the
old sweep changed the seed per profile and that seed also changed the
whole-session split. Starting now, **split seed and model/training seed are
separate**.

## What changed

### 1. Shared whole-session split

`run_training_experiment_v214()` now accepts an optional `split_seed`.

- `split_seed` decides which complete sessions are train / validation / test.
- `seed` still controls model initialization, training order and randomisation.

Future V2.14 sweeps now keep one split across all profiles.

### 2. Paired mild + standard retraining

Run:

```bash
python3 -m automation.hole_v215_pair_train
```

This trains only `mild` and `standard`, with:

- the **same** whole-session split,
- different model/training seeds,
- the same strict novel-background classes,
- the same 37 real-hole holdout,
- no real/novel data in model selection.

Outputs:

```text
content/ai/reports/v215_pair/
  mild/hole_patch_ai_v214.npz
  mild/hole_v214_report.json
  standard/hole_patch_ai_v214.npz
  standard/hole_v214_report.json
  pair_summary.json
```

Both model files now record `split_seed` and `session_assignment` in metadata.

### 3. Paired complementarity / ensemble experiment

Run:

```bash
python3 -m automation.hole_v215_ensemble
```

Both models receive **the exact same candidate-centred patches**. The report
measures:

- score correlation,
- mean probability disagreement,
- both-hit positives,
- standard-only positive rescues,
- mild-only positive rescues,
- neither-hit positives,
- false-positive disagreement,
- an oracle `either-model` recall.

The oracle number is diagnostic only. It answers "is there complementary
information here?" It is **not** a live accuracy claim.

## Honest blend selection

V2.15 searches a single standard-vs-mild weight, including both pure endpoints.
The models can have different calibration, so raw probabilities are not simply
averaged. Each score is first converted to a logit margin around that model's
own learned V2.14 threshold:

```text
margin_standard = logit(P_standard) - logit(threshold_standard)
margin_mild     = logit(P_mild)     - logit(threshold_mild)
```

Then:

```text
fused_margin = w * margin_standard + (1-w) * margin_mild
```

`w` and the final fused threshold are selected using only:

1. clean validation,
2. procedural domain-stress copies of validation.

**Synthetic test, strict novel backgrounds, real holes and off-centre stress are
not allowed to choose `w`.** They are evaluated only after the blend is frozen.

If the best non-holdout result is a pure endpoint (`w=0` or `w=1`), V2.15 says
so. We do not keep two networks merely because an ensemble sounds attractive.

## Candidate shadow annotator

`src/engine/ai/hole_patch_ensemble_v215.py` provides
`HolePatchEnsembleV215.annotate_candidates(gray, candidates)`.

It can attach fields such as:

```text
hole_v215_standard_probability
hole_v215_mild_probability
hole_v215_fused_probability
hole_v215_disagreement
hole_v215_offset_dx / dy
hole_v215_refined_camera_x / y
hole_v215_above_threshold
hole_v215_uncertain
hole_v215_shadow_only = true
```

Important safety rule: **candidate order and original camera coordinates are
preserved**. V2.15 does not change the authoritative selected hit.

The offset head is also fused as shadow evidence. A later candidate-level
benchmark can determine whether that refinement really reduces localisation
error before it is allowed to affect coordinates.

## Edge candidates

Live candidates may sit near the edge of camera/scanport images. V2.15 uses
reflect padding when extracting a candidate patch so an edge candidate is not
silently discarded or given an artificial black frame.

## Gate

The generated report currently uses this candidate-shadow gate:

- non-holdout blend selection >= 99.5% of the best pure endpoint,
- strict novel-background AUC >= 0.70,
- real-hole recall >= 0.85,
- off-centre AUC >= 0.76.

Passing means only:

> Hole-AI paired evidence is worth carrying into the next candidate-level
> shadow/replay step.

It does **not** mean >=95% final hit detection and does not grant game/live
authority.

## What V2.15 deliberately does not do

V2.15 does not yet have honest full-frame detector hard negatives. The current
`content/ai/holes` bank is excellent for patch learning but cannot reconstruct
all wrong V1/V2 candidate locations from the original camera frame.

Therefore V2.15 does **not** pretend to measure final V1/V2 ranking gain.

The next step after the paired report is V2.16:

1. save opt-in raw clean pre/post frames,
2. save real V1/V2/overlay candidate-centred patches plus GT distance,
3. mine high-ranked wrong detector candidates,
4. score those actual candidate pools with Hole-AI,
5. measure GT rank / Top-1 / Top-3 / localisation before any live authority.

That is the bridge from "AI recognises hole morphology" to "AI actually helps
choose the new bullet hole in a game".
