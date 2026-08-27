# Data and evaluation policy

This file exists to prevent accidental leakage and semantic mistakes while the detector evolves.

## 1. Label semantics

### Static Hole-AI

Question:

> Does this patch contain a physical hole-like structure?

Positive:

- old physical hole,
- new physical hole.

Negative:

- true non-hole structures only.

A far-from-current-GT candidate must **not** automatically be labelled non-hole; it may be an old hole.

### NEW-hole AI / listwise ranker

Question:

> Is this candidate the hole created by the current shot?

Positive:

- current-shot GT / near-GT candidate.

Negative / NOT-NEW:

- old hole,
- texture,
- projector edge,
- noise,
- any candidate unrelated to the current new hole.

`NOT_NEW` and `NON_HOLE` are intentionally different concepts.

## 2. Known-hole registry

Known-hole distance can be used as a soft feature/diagnostic.

It must not be treated as complete truth because:

- the scene can start with physical holes already present,
- the registry resets with sessions/scenes,
- detections can be missed,
- a re-hit can occur close to an existing hole.

No hard exclusion based only on proximity.

## 3. Real-hole patch bank

Real `hole_*` patches are scarce and should remain protected by default.

Use them for:

- real appearance holdout,
- distribution/calibration analysis,
- final sanity checks.

Do not casually fold them into training after viewing holdout results.

## 4. Camera-captured projected `synt_*` bank

These are valuable because the appearance passed through:

```text
rendered hole -> projector -> physical surface -> camera
```

They can be used to calibrate/construct training hole appearance.

Important anti-cheat rule:

- original source patches are centered,
- do not train a classifier that can solve the task from “object is at patch centre”.

## 5. Physical V2.16 candidate session

The existing 100-shot session is only one physical session.

Current 60/20/20 split is useful for experiments but remains **provisional**.

Rules:

- only development subset may be used for physical-domain fitting/augmentation,
- confirmation/holdout remain protected from gradient updates/model selection,
- final acceptance must use additional independent capture sessions.

## 6. Generated seed ranges

Current policy:

- generated training: `1..8,999,999`,
- frozen generated validation: `9,000,001..9,099,999`,
- frozen generated offline holdout: `99,000,001..99,099,999`.

The first current frozen validation run used `9000001..9000100`.

Never move those seeds into the training pool after inspecting their benchmark.

## 7. Media source split discipline

Media source/family identity, not individual frame, determines split.

For video/animation:

- an entire source belongs to one split,
- adjacent frames from the same clip must not leak across train/validation/holdout.

For images:

- exact and near-duplicate cross-split checks should remain enabled,
- source/license/provenance metadata should be recorded where known.

## 8. Simulator data is not acceptance data

Generated candidate packs must continue to contain an explicit marker equivalent to:

```text
physical_acceptance_data = false
```

Synthetic results can answer:

- whether an objective is learnable,
- whether code is deterministic,
- whether a model generalises across generated seeds/media,
- whether training infrastructure works.

They cannot answer:

- whether the detector is ready for normal physical gameplay.

## 9. Physical-domain bridge rules for V2.21

If V2.21 uses real camera frames as bases for generated worlds:

- use development/training capture sessions only,
- never use protected confirmation/holdout frames as generation bases,
- provenance must record the source physical session/frame,
- any temporal-noise model must be fitted only on allowed development data,
- generated samples derived from a physical source must inherit a group/family identity so variants cannot leak into both train and validation.

## 10. Model-selection hierarchy

Preferred hierarchy:

1. train data — gradients allowed,
2. generated validation — model selection allowed but never gradients,
3. generated offline holdout — report only,
4. physical development — controlled debugging/ablation,
5. physical confirmation — protected model-selection gate,
6. independent physical holdout sessions — final acceptance authority.

The current single physical session does not yet satisfy level 6.
