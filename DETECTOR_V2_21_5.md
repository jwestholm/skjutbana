# Detector V2.21.5 — Candidate-aligned listwise physical dense ranking

## Why this version exists

V2.21.4 proved that the broad physical PRE→POST temporal pool is no longer the main recall bottleneck:

- all 30 full-frame shots: dense pool oracle <=20 px = 90%, <=42 px = 100%;
- protected confirmation: <=20 px = 100%;
- protected holdout: <=20 px = 100%.

The V2.21.4 learned top-512 ranker nevertheless retained a <=20 px candidate on only 13.3% of all shots and 16.7% of DEVELOPMENT. That is too weak even on its own training domain.

The V2.21.4 training report also contains `forced_positive_jitter_px = [-4,-2,0,2,4]` and 492 positive samples. V2.21.5 removes that entire training semantic: the ranker may only learn from coordinates that the GT-free dense proposal engine actually emitted.

## Core invariant

**Train and inference see the same candidate type.**

1. Build GT-free full-frame temporal evidence.
2. Generate a broad dense candidate pool without GT.
3. Extract features from those exact candidates.
4. Only then use GT to label actual pool candidates for offline training/evaluation.
5. Never inject the GT coordinate or a GT-jitter coordinate into the candidate set.

## New ranking representation

Per candidate V2.21.5 uses:

- value, local contrast, local peak ratio and shot-relative percentile for eight temporal evidence maps;
- exact evidence-source membership/support from dense proposal construction;
- max/top-3/mean temporal percentile agreement;
- soft distance to existing CURRENT candidates;
- soft distance to V2.21.2 local temporal candidates;
- a small set of cross-source percentile interactions.

The CURRENT/V2.21.2 distances are context features only. They do not hard reject candidates and therefore do not remove the direct-proposal capability.

## Objective

Training is whole-shot/listwise. For each usable DEVELOPMENT shot:

- positives are **actual dense candidates <=20 px from GT**;
- negatives/hard negatives are other actual dense candidates from the same shot;
- target probability decays smoothly with distance inside the positive radius;
- stage 2 mines the current model's hardest false candidates.

No confirmation or holdout frame is opened by the fit path.

## DEVELOPMENT cross-fit gate

Before the final DEVELOPMENT fit, the script performs deterministic shot-level 3-fold cross-fitting using DEVELOPMENT only. This is deliberately reported before protected evaluation so a model that cannot rank its own physical development domain is rejected early.

## Live authority

None. V2.21.5 is offline/shadow only.
