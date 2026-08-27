# DETECTOR V2.20 – RGB media worlds + compact hole stamps

## Why V2.20 exists
V2.19 proved that a split-safe offline media bank can generate many labelled worlds, but visual QA exposed three problems:

1. saved outputs became grayscale,
2. static PRE/POST pairs drifted globally because camera jitter was re-sampled per frame,
3. new holes often looked like black/white streak residuals instead of compact bullet holes.

V2.20 keeps the good parts of V2.19 (deterministic seeding, media-bank indexing, family/split discipline, candidate-pack bridge) and fixes those realism issues.

## Core changes

### 1. RGB is now canonical
Observed scenario output remains RGB.
Grayscale is derived only as a compatibility view for existing replay/evidence code.

### 2. Shared camera state for static scenes
A scenario samples one camera state (gain, gamma, blur, black level and slight per-channel balance) and applies it to all PRE and POST frames.
Only small temporal noise remains per frame.
This keeps PRE/POST visually stable for static images while still allowing dynamic video/media motion when the source itself changes.

### 3. Compact hole stamps instead of pasted residuals
Historical `content/ai/holes/synt_*.png` patches are still used, but only to estimate plausible hole statistics:
- dark-core radius,
- dark-core depth,
- rim strength,
- rim width.

The final hole that is rendered into a new scenario is a compact procedural stamp with:
- dark core,
- slight torn-paper bright rim,
- mild irregularity,
- limited support area.

This removes source-context leakage and greatly reduces streak-like artefacts.

### 4. Deterministic scenario QA
Each scenario is rendered deterministically and checked before final acceptance.
For the final PRE/POST pair the generator measures:
- local absolute difference near GT,
- centre darkening,
- diff area,
- diff aspect ratio,
- static-scene global drift outside the hole ROI.

If the render fails QA, the hole rendering is retried deterministically (same scenario seed, new attempt sub-seed) up to the configured retry limit.

## Practical effect
V2.20 is intended as a better generator for synthetic training worlds before the next overnight runs.
It is still not a replacement for physical projector/camera acceptance data.
