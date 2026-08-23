# Detector / AI V2.7 — hypothesis consolidation

## Why V2.7 exists

The locked V2.6 benchmark changed the bottleneck. At F2 evaluation the true
neighbourhood was present in the filtered pool for roughly **76.7%** of shots,
while the selected candidate was correct within 42 px only about **0.5%**.
Median GT rank was around **179**. The detector is therefore frozen for this
iteration; V2.7 works after V2.6 Shot Vault.

## Pipeline

```text
V2.6 detector + Shot Vault
        |
        v
noise filter
        |
        v
500-ish observations (often)
        |
        v
V2.7 micro-clustering
  - 16 px merge neighbourhood
  - bounded 30 px cluster diameter
  - robust weighted geometric-median centre
        |
        v
spatial coverage pool (max 120 hypotheses)
        |
        +--> deterministic hypothesis baseline
        |
        +--> Ranker V6 shadow / validated auto-gate
        |
        v
actual ranked output
```

Ground truth is **never** used to build a hypothesis or rank the current shot.
Synthetic GT is read only after current-shot ranking for validation and later
online learning.

## Hypotheses instead of raw candidates

Nearby V1/V2/tile/Vault observations are summarized into one hypothesis. The
hypothesis carries aggregate evidence such as member count, detector-source
diversity, Vault repeats, current-vs-carried fraction, cluster spread, temporal
signal and V2.4 patch priors. A robust centre is used instead of simply taking
the strongest detector peak.

A spatial coverage stage caps the rankable set at 120 hypotheses. Large macro
cells and multiple evidence heads are used so a noisy region cannot consume the
whole pool. This first release intentionally favours recall over aggressive
compression.

## Ranker V6

V6 is a small position-independent pairwise logistic model over hypothesis
features. It uses two supervised labels:

- strict positive: nearest real hypothesis <=12 px from synthetic GT;
- soft positive: nearest hypothesis <=20 px, down-weighted during training.

Hard negatives must be at least 55 px away. Prediction/validation happens
**before** the current shot is learned.

V6 cannot control selection merely because it has trained. A rolling validation
gate requires enough held-out eligible shots, a Top-1 advantage, a Top-3
advantage and a substantially better median rank. Even after the gate opens, the
current shot needs enough V6 score and margin. Until then the deterministic V2.7
hypothesis baseline remains authoritative.

## New diagnostics

Run:

```bash
python3 -m automation.detector_v27_analyze
```

It reports three distinct oracle stages at 10/20/42 px:

1. filtered V2.6 input;
2. all V2.7 clusters;
3. final spatial hypothesis pool.

This makes a loss attributable to one of four stages:

```text
filtered detector miss
clustering lost GT
spatial pool lost GT
ranking selected wrong
```

It also compares BASE, V6 shadow and ACTUAL median rank / Top-1 / Top-3, plus
V6 training/gate state.

## V2.7 development targets

The first goal is not to claim a final hit rate. It is to establish whether we
can shrink hundreds of observations to ~120 hypotheses **without materially
reducing 42 px recall**, then make the GT rank collapse from V2.6's ~179 toward
the top of that smaller set. Once that is measured on the locked seeds, the next
iteration can tune clustering and/or V6 independently instead of mixing detector
and ranking changes.
