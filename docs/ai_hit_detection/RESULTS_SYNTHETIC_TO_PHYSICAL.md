# Result: synthetic generalisation succeeds, physical transfer fails

**Date:** 2026-08-27

This experiment is the clearest decision point in the project so far.

## Experiment A — train on generated worlds

V2.18 listwise ranker was trained on V2.20.2-generated candidate groups from seeds `1..100`.

Approximate training-session result after 32 epochs:

| Split | Current Top-1 | V2.17 Top-1 | V2.18 Top-1 | V2.18 median rank |
|---|---:|---:|---:|---:|
| Development | 41.67% | 0% | 96.67% | 1 |
| Confirmation | 40% | 0% | 100% | 1 |
| Holdout | 30% | 5% | 95% | 1 |

This showed that the listwise model can strongly learn the generated ranking problem.

## Experiment B — frozen generated validation

A completely separate generated validation candidate set was compiled:

- seeds `9000001..9000100`,
- 100/100 groups saved,
- 100/100 groups had a candidate <=20 px and <=42 px,
- model weights remained frozen.

Result:

| Split | Current Top-1 | V2.17 Top-1 | V2.18 Top-1 | V2.18 median rank |
|---|---:|---:|---:|---:|
| Development | 80% | 0% | 98.33% | 1 |
| Confirmation | 65% | 5% | 100% | 1 |
| Holdout | 75% | 0% | 100% | 1 |

### Meaning

The model did **not** merely memorise seeds 1..100. It learned a rule that generalises extremely well inside the generated V2.20.2 domain.

However, the current detector itself is already far stronger on generated worlds than on physical data. That is a warning that the simulator is easier/cleaner than reality.

## Experiment C — same frozen model on real physical candidate packs

No retraining was performed. The generated-data-trained V2.18 ranker was applied to the existing V2.16 physical session.

### Ranked candidates, <=20 px

| Split | Current Top-1 | V2.17 Top-1 | V2.18 Top-1 | V2.18 median GT rank | Oracle |
|---|---:|---:|---:|---:|---:|
| Development | 6.67% | 0% | **0%** | 110 | 36.67% |
| Confirmation | 5% | 0% | **0%** | 125 | 25% |
| Holdout | 10% | 0% | **0%** | 104 | 45% |

### Raw + ranked union, <=20 px

| Split | Current Top-1 | V2.18 Top-1 | V2.18 median GT rank | Union oracle |
|---|---:|---:|---:|---:|
| Development | 6.67% | **0%** | 235 | 40% |
| Confirmation | 5% | **0%** | 287 | 35% |
| Holdout | 10% | **0%** | 185.5 | 50% |

### Offset/refinement observation

The offset head is not completely devoid of physical signal. For example, physical confirmation ranked refined-oracle rose from 25% to 40%, and confirmation union refined-oracle from 35% to 45% in this benchmark.

That is not enough to rescue ranking, but it suggests some local/geometric information transfers even though the score ordering does not.

## Interpretation

The key evidence is now:

```text
V2.20.2 train -> unseen V2.20.2 worlds    ~98-100% Top-1
V2.20.2 train -> physical V2.16 session   0% Top-1
```

This points to a strong **synthetic/physical domain gap**.

Likely contributors include:

- rendered hole statistics differ from real shot changes,
- real projector/camera temporal noise differs from the simulated camera transform,
- physical old-hole/paper/projector artefacts are not reproduced faithfully,
- V2.17 embeddings may rely on cues that are stable in the simulator but unstable in reality,
- synthetic candidate-generation behaviour is much easier than physical candidate generation,
- generated backgrounds/media may not reproduce the photometric structure of projected camera scenes,
- there may be a simulator-specific shortcut that has not yet been identified.

## Why ranking alone cannot solve the current physical problem

Physical union oracle within 20 px is only:

- 40% development,
- 35% confirmation,
- 50% holdout.

A perfect ranker cannot select the true hit when no candidate near the true hit exists.

Therefore the project now has two separate engineering goals:

1. **increase physical proposal recall**, and
2. **make ranking evidence transfer from training worlds to the camera domain**.

## Decision

- KEEP V2.18 listwise architecture as a research building block.
- KEEP V2.20.2 simulator for diversity and controlled tests.
- DO NOT run long synthetic-only optimizer loops yet.
- DO NOT grant live authority.
- BUILD V2.21 around physical-domain measurement/bridging and direct proposals.
