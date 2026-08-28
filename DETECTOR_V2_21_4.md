# Detector V2.21.4 — Learned Physical-Domain Dense Temporal Ranker

## Why this version exists

V2.21.3 answered the hand-written-filter question: more temporal consensus rules did not improve the best V2.21.2 result.

The current physical full-frame reference point is:

| source | oracle <=20 px | oracle <=42 px |
|---|---:|---:|
| CURRENT | 26.67% | 90.00% |
| V2.21 global direct | 3.33% | 3.33% |
| V2.21.2 CURRENT + local | **63.33%** | **93.33%** |
| V2.21.3 final union | 46.67% | 93.33% |

V2.21.3 therefore remains rejected. V2.21.2 is the handcrafted baseline.

A particularly important diagnostic from V2.21.2 is that the new-hole location is often *not* a global 99th-percentile temporal point. On DEVELOPMENT, useful GT evidence was commonly around:

- blackhat gain ~90th percentile,
- top-hat gain ~87th percentile,
- persistent absolute change ~81st percentile,
- gradient gain ~78th percentile,
- persistent darkening ~75th percentile.

V2.21.3's masked-direct threshold of 99% was therefore intentionally precise but much too narrow for rescue recall.

## V2.21.4 strategy

### 1. Broad GT-free dense pool

Inference first creates a broad temporal pool inside the candidate-derived target mask. No GT is accepted by this function.

The default thresholds are intentionally lower:

- blackhat: 82%,
- top-hat: 82%,
- persistent abs: 72%,
- gradient gain: 70%,
- persistent dark/bright: 68%,
- fused: 64%,
- compact change: 55%.

This will produce many more candidates than a live detector should ultimately keep. That is deliberate: V2.21.4 first asks whether the evidence maps contain a usable candidate at all.

Long/saturated plateaus are spatially sampled instead of collapsed to one centroid. This is important because projector bands can be long, and collapsing one band to one point can remove the true local location before learning even gets a chance.

### 2. Storage-aware learned features

The ranker samples 21 point features from V2.21 temporal maps:

- base evidence channels,
- local peakness against a blurred neighbourhood,
- a few explicit cross-channel interactions.

No x/y coordinate is a model feature. The model therefore cannot learn "holes usually appear at this screen position".

The implementation avoids allocating 21 additional 4K float maps. Only one temporary Gaussian-blurred map is created at a time.

### 3. DEVELOPMENT-only pairwise learning

For each DEVELOPMENT shot:

- GT and a very small <=~6 px training-only jitter cloud provide positive evidence,
- natural dense-pool points <=10 px are also positive,
- points >=42 px are valid NEW-hole negatives,
- high-saliency hard negatives and random negatives are sampled,
- a linear pairwise logistic ranker learns `score(positive) > score(negative)`.

There are two stages:

1. initial hard-negative training from hand-written saliency,
2. model-mined hard-negative training from the first learned ranker.

Confirmation and holdout frames are not opened by the trainer.

### 4. Frozen benchmark

After saving the model, the benchmark evaluates:

- CURRENT,
- the frozen V2.21.2 union,
- the full broad dense pool,
- learned top-64 / top-128 / top-256 / top-512,
- CURRENT + learned top-512,
- V2.21.2 + learned top-512.

It also reports the rank of the first dense candidate within 20 px of GT.

This separates failure modes:

- **dense pool low** => proposal/evidence generation still fails,
- **dense pool high, learned rank poor** => ranker or physical training volume fails,
- **learned protected recall rises** => dense physical-domain learning is worth continuing.

## Leakage rules

Non-negotiable:

- `build_dense_pool_v2214()` has no GT input,
- `rank_dense_pool_v2214()` has no GT input,
- confirmation/holdout are not used for feature normalisation,
- confirmation/holdout are not used for hard-negative mining,
- confirmation/holdout are not used to choose top-K,
- model metadata records the exact development shot keys used during training,
- benchmark aborts if a protected shot key appears in model training metadata.

## Important limitation

The 30 direct-ready full-frame rounds currently come from one projector/camera session. The legacy 100-round session has no full frames. Consequently the 18/6/6 full-frame split is still provisional even though protected splits are respected algorithmically.

A good V2.21.4 result is evidence that the approach is useful, not final acceptance evidence. At least one more independent full-frame session is required before strong conclusions.

## No live authority

V2.21.4 does not change:

- `HitScanner` authority,
- live candidate ordering,
- game hit coordinates,
- game logic.

It is purely offline/shadow training and evaluation.
