# V2.23.2 — Proposal Learning + Domain Validation

V2.23.2 keeps the V2.22 runtime frozen and moves the AI work one stage earlier: a ranking model cannot learn the correct hit when the correct neighbourhood was never proposed.

The 2026-08-29 fresh F2 run under V2.23.1 showed that the captured live/V2.8 union remained only about one hundred candidates per shot and recovered the GT within 20 px in roughly one shot out of ten. The existing V2.23.1 research champion also failed to generalise to that new projector/camera session. This version therefore treats **proposal recall** and **ranking generalisation** as separate engineering gates.

## Pipeline

```text
F1/F2/manual labelled shot
        |
        +--> ordinary V2.23 candidate record (unchanged)
        |
        +--> V2.23.2 framepack
              - full recent PRE grayscale frame
              - up to three unique POST grayscale frames
              - timestamps
              - GT stored only as label metadata
              - current GT-free candidates
                         |
                         v
                 OFFLINE proposal lab
              direct V2.21 temporal maps
                    + V2.21.2 local
                    + V2.21.5 dense physical pool
                         |
                         v
                 proposal sidecar
              oracle @5/@10/@20/@42
              full GT-independent candidate union
                         |
                         v
                 unified V2.23 dataset
                         |
                 +-------+---------+
                 |                 |
             engineering      newest >=50-shot
             train/valid       F2 projector session
                 |             fresh-domain gate
                 v                 |
             challenger -----------+
                 |
          research promotion only
                 |
          live authority remains NO
```

## Full-frame framepacks

Framepacks live below:

`content/ai/training_v223/framepacks/<session>/`

Each labelled shot writes one JSON metadata file and one compressed NPZ. The NPZ stores one full PRE frame plus at most three distinct POST frames. This is intentionally richer and larger than the previous candidate-only record because proposal research needs the pixels that were lost when the live detector failed to nominate the real hit.

Ground truth is **not** an argument to direct/local/dense proposal generation. It is applied only after proposals exist to calculate oracle metrics and supervised labels.

## Offline high-recall proposal engine

V2.23.2 reuses the existing offline stack rather than inventing another detector:

- `direct_proposal_v221`
- `temporal_local_v2212`
- `physical_dense_v2215`

Dense proposal may create thousands of physical hypotheses per framepack. This is acceptable offline. Candidate duplicates from current/local/dense sources are merged while preserving dense physical evidence and source provenance.

Proposal sidecars are cached below:

`content/ai/training_v223/proposals_v2232/<session>/`

Re-running a completed session reuses these sidecars unless `--force` is supplied.

## Ranking dataset contract

Proposal-expanded candidates are merged into the existing unified record without using GT to decide retention. Only after the union is fixed are GT distance, relevance and oracle labels recalculated.

Model inputs remain physical/signal features only. GT distance, rank labels, current model scores and policy/bookkeeping leakage remain forbidden.

## Fresh-F2 domain gate

The newest F2/projector session containing at least 50 shots is excluded from fitting. It becomes a research domain gate.

A challenger must now have meaningful positive support and beat a reproducible reference baseline on both:

1. ordinary engineering validation; and
2. the fresh F2 projector/camera domain.

A pre-V2.23.2 champion has no fresh-domain metrics and is therefore quarantined automatically. This is deliberate.

The fresh-domain set is **not** the protected holdout and does not grant live authority. Protected holdout remains completely outside automatic model selection.

## Reference baseline

V2.23.1 could report zero baseline-eligible shots when imported candidates lacked explicit `baseline_rank`. V2.23.2 uses explicit rank when present and otherwise reproducibly sorts captured `baseline_score`. Promotion requires sufficient reference coverage on both validation and fresh F2.

## Training target placement

The old function named `center_bias` was actually uniform across the viewport. V2.23.2 makes it a soft centre prior:

- 75% Gaussian-like centre-biased targets;
- 25% uniform exploration.

Uniform, edge and corner sampling modes remain available as separate robustness tests. This sampling prior does not change the runtime hit detector or move a physical hit toward the centre.

## Authority

Everything added here is data collection, offline proposal analysis, model training and shadow evaluation. No V2.23.2 code grants game authority.
