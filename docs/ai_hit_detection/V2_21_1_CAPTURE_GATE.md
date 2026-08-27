# V2.21.1 capture gate

## Input result

V2.21 domain profile on 24,000 synthetic vs 23,040 physical-development
candidate rows produced **group domain AUC 1.0000**. Synthetic and
projector/camera groups are therefore perfectly separable by the current
feature representation.

The strongest shifts include:

- `known_hole_distance_scaled`: KS 0.988, |SMD| 5.641,
- `mean_absdiff`: KS 0.915,
- `p95_absdiff`: KS 0.885,
- `persistence`: KS 0.871, |SMD| 3.077,
- `center_absdiff`: KS 0.868,
- V2.17 embedding norm/mean/std: KS about 0.835–0.839.

Even the <=20px rows retain large temporal shifts. This explains why V2.18 can
reach ~98–100% Top-1 on unseen V2.20 worlds yet 0% Top-1 on the physical
candidate packs.

## Decision

Stop synthetic-only ranker optimization. Collect honest full-frame
projector/camera temporal evidence and measure candidate recall independently
of V1/V2.

## First capture

- background: white,
- rounds: 30,
- deterministic seed: 22101,
- online learning: frozen,
- full recent PRE: required,
- full POST: 2 frames,
- normal hit authority: unchanged/shadow only.

## Pass/fail interpretation

The first engineering target is `CURRENT + AI_DIRECT union oracle@20 >= 0.70`.

- >=0.70: continue with independent full-frame sessions and hard-background
  coverage; tune proposal recall before ranking.
- 0.50–0.70: inspect rescued cases and nearest direct proposal errors; improve
  proposal maps/NMS/registration.
- <=0.50 or no improvement over CURRENT: inspect PRE/POST capture timing and
  heatmaps before changing the learned ranker.

No result from this 30-shot session grants live authority.
