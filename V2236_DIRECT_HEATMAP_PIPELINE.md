# V2.23.6 Direct Heatmap Pipeline

## Decision

V2.23.6 stops treating the ~9,500-candidate dense pool as the primary learned decision space. V2.23.5 proved that registered evidence helps, but the global candidate reducer still discarded too many valid candidates.

The master-plan Source 4/direct-proposal stage is therefore activated now.

## Data contract

Each F2 framepack supplies GT-free PRE/POST camera frames. `direct_proposal_v221` produces registered, photometrically compensated physical maps. V2.23.6 normalises the same eight evidence families already used by the dense teacher and stores their spatial fields at 4x lower resolution.

The heatmap cache contains no GT-derived input feature. GT is stored separately for labels and metrics.

## Model contract

The model is fully convolutional over the registered evidence field:

- `linear_conv`: one learned 5x5 multi-channel spatial filter.
- `spatial_conv`: learned 5x5 multi-channel filters -> tanh -> learned 1x1 output combination.

Training uses pairwise positive-vs-negative spatial locations. Positive grid cells must be within 6.5 camera pixels of GT. Locations within 42 px but outside the positive radius are neutral and are not trained as negatives.

Hard-negative mining scores the complete training heatmap and mines false maxima outside 42 px. The hard-mined checkpoint is compared with the pre-mining checkpoint on engineering validation and is rejected automatically when it regresses.

## Metrics

The main outputs are direct spatial metrics rather than candidate rank:

- Top1 @5/@10/@20/@42
- Top3 @20/@42
- Top5 @20
- median / P90 / P95 Top1 error
- optional nearest-dense snap diagnostics

Deterministic baselines (`fused`, max-channel, mean-channel, physical mix) are measured with exactly the same peak/NMS policy.

## Split discipline

One substantial session:

- deterministic same-session engineering split
- learnability only
- no authority/generalisation claim

Two or more substantial sessions:

- newest session untouched as fresh domain
- model and direct policy selected on engineering validation first
- selected policy is then frozen before fresh-domain evaluation

## Authority

V2.23.6 is offline/shadow only. `eligible_for_live_authority=False` is hard-coded.
