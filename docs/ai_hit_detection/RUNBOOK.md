# Runbook — what to do next

## Current status

The V2.20.2 synthetic experiment is complete enough to make the next decision.

Do **not** spend time generating more identical synthetic candidate packs merely to increase sample count before V2.21 work exists.

Do **not** launch long V2.18 synthetic-only training.

Do **not** change live game hit authority.

## Next development sequence

### Step 1 — implement V2.21 physical pack audit

Create an inspector that prints keys/shapes/provenance for physical V2.16 packs and answers whether full recent-PRE/POST images are available.

Expected output should explicitly state something like:

```text
full_recent_pre_available: true/false
full_post_available: true/false
candidate_patches_available: true
GT_patches_available: true
```

This determines whether direct full-frame proposal work can reuse the current physical session.

### Step 2 — if necessary, add opt-in V2.21 full-frame capture

Only if old packs lack the required raw frames.

Requirements:

- recent PRE immediately before shot,
- multiple unique POST frames where practical,
- lossless/consistent format,
- shot/session provenance,
- disabled by default outside dedicated capture/testing,
- no game authority change.

### Step 3 — build domain-gap report

Run synthetic train/validation and physical development through the same feature extractor.

Do not train anything in this step.

Goal: identify why generated and physical candidates are separable/different.

### Step 4 — build direct proposal baseline

Run on full PRE/POST.

Primary metric: physical **oracle recall**, not Top-1.

Report rescued shots and candidate count.

### Step 5 — build physical-domain bridge

Use only allowed physical-development bases to generate new labelled camera-domain scenarios.

Train small challenger(s) and evaluate on protected confirmation plus frozen generated validation.

### Step 6 — only after V2.21 gates

Consider V2.22 overnight champion/challenger training.

## Physical testing philosophy

The project should minimize repeated real shooting, but not pretend physical data can disappear completely.

Use real captures for:

- reality calibration,
- domain-gap estimation,
- proposal acceptance,
- frozen acceptance sessions.

Use offline worlds for:

- large-scale diversity,
- hard-case generation,
- optimizer iterations,
- ablations.

## What result would justify moving to V2.22

At minimum:

1. direct-proposal union materially raises physical oracle above the existing 35–50%, preferably >=70% initially,
2. some ranking model trained with the physical-domain bridge shows measurable positive transfer on protected physical confirmation,
3. no train/validation leakage is found,
4. synthetic validation remains stable enough that the model is not simply overfitting one physical development board.
