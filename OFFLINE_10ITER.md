# Offline ranker experiment — 2026-09-07

Base: `ad03f4c`, branch `codex/offline-10iter`. This experiment changes no live
code, detector settings, model registry, calibration, candidate coordinates, or
tolerances. All learned models are isolated research artifacts. No promotion.

No primary accuracy improvement was achieved. **Iteration 10 is NOT approved for
live use.** Candidate availability is the largest observed bottleneck: candidates
within 20 px are absent for 91% of DEVELOPMENT and 94% of HOLDOUT shots. All data
in this experiment is projected F2 data and does not establish physical/live
performance.

## Data and frozen baseline

Use the three native 100-shot F2 sessions, without automatic proposal augmentation
or legacy-cache merging:

| Split | Session |
|---|---|
| TRAIN | 20260829_194140_ai_training_scene_38412b9c |
| DEVELOPMENT | 20260829_203625_ai_training_scene_e31e210c |
| PROTECTED FINAL HOLDOUT | 20260829_212658_ai_training_scene_67465571 |

The single-shot session is excluded. `splits.json` records every sample path,
identity, SHA-256, candidate fingerprint, and exclusion. Whole-session chronology
prevents adjacent shots crossing boundaries. Earlier splits are purged of exact
candidate duplicates or 8 px quantized coordinate-set Jaccard overlap >=0.9 with
later splits; none matched. This label-blind audit cannot certify visual uniqueness
because the older sessions lack full PRE/POST imagery. Same date, background and
camera limit independence. The holdout was unused for iteration tuning, but its
aggregate pool recall had been exposed in the prior evaluation-framework task.

Primary: Top-1 <=20 camera px divided by all DEVELOPMENT shots. Secondary metrics
are separately reported at 5/10/20/42 px: Top-1, Top-3, Top-10, full-pool oracle,
all-shot MRR (missing positives contribute zero), mean first-positive rank
conditional on presence, and per-shot ranks. Exact distance labels are recomputed
from saved coordinates; neither coordinates nor tolerances change.

The reference is stable descending saved baseline score on the complete original
pool. These records lack complete explicit ranks. This is an offline reference,
not a reconstruction of current live ranking or final emission. It reaches
Top1/Top3/Top10/Oracle @20 = 0/1/2/9 out of 100 on DEVELOPMENT.

All experiments reorder exactly the same candidates. Guardrails require identical
full-pool oracle and nondecreasing Top-10 coverage at every tolerance versus the
original baseline. Selection maximizes Top1@20, then MRR20, then Top3@20; exact ties
keep the incumbent. TRAIN alone fits scalers and models. Seed 2230 and single-thread
BLAS settings are fixed. Existing `train_rank_model` and `RankModelV223` are reused
without trainer/registry wrappers that can promote models. Only ten TRAIN shots
have any candidate within 20 px, so those ten supply supervised positive support.

## Results

Counts have denominator 100. Delta Top1@20 is zero for every trial.

| Iteration | Single hypothesis/change | Top1@20 | Top3@20 | Top10@20 | Oracle@20 | MRR20 | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | Saved-score reference | 0 | 1 | 2 | 9 | .008387 | Initial incumbent |
| 1 | Fit existing linear ranker | 0 | 0 | 1 | 9 | .003805 | Discard |
| 2 | MLP on same features | 0 | 0 | 1 | 9 | .003123 | Discard |
| 3 | Remove geometry/score | 0 | 0 | 1 | 9 | .005131 | Discard |
| 4 | Log evidence magnitudes | 0 | 0 | 1 | 9 | .005044 | Discard |
| 5 | Within-shot evidence percentiles | 0 | 0 | 1 | 9 | .005577 | Discard |
| 6 | Stronger evidence L2 | 0 | 0 | 2 | 9 | .004973 | Discard |
| 7 | Earlier evidence fit | 0 | 0 | 2 | 9 | .005056 | Discard |
| 8 | Evidence-only MLP | 0 | 0 | 1 | 9 | .004884 | Discard |
| 9 | Smaller evidence MLP | 0 | 0 | 2 | 9 | .005101 | Discard |
| 10 | Geometry-only linear ablation | 0 | 1 | 3 | 9 | .008768 | Keep by secondary tie-break |

Each trial records its parent, one changed configuration key, hypothesis, training
losses, hashes, original/previous/best comparisons, coverage checks, and conclusion.
Rejected models are retained only for audit and never become the incumbent. Trials
may branch from a rejected model to isolate an ablation; parent IDs are explicit.

Iteration 10 was frozen before loading holdout labels for scoring. Baseline and
challenger were each scored once; subsequent report generation only reads saved
results. No post-holdout tuning occurred.

| Holdout @20 | Baseline | Challenger |
|---|---:|---:|
| Top1 | 0/100 | 0/100 |
| Top3 | 0/100 | 0/100 |
| Top10 | 0/100 | 0/100 |
| Oracle | 6/100 | 6/100 |
| MRR (all shots) | .000930 | .001703 |
| Mean positive rank | 87.5 | 40.0 |

There is no demonstrated primary accuracy improvement. Positive ranks move upward
on holdout but never reach Top-10 at 20 px. At 42 px, holdout Top1 regresses from
1/100 to 0/100; MRR42 also regresses. The secondary MRR20 effect is not confined to
DEVELOPMENT, but these small sessions and ten comparisons do not establish a robust
accuracy gain. The largest remaining measurable failure is absent candidates:
91/100 DEVELOPMENT and 94/100 holdout at 20 px. Ranking cannot recover those misses.

The train/development shift is substantial: median positive-candidate area is
5.5 versus 169, and detector score is 0.66 versus 14. Even TRAIN Top1 was only
0–2/100 in the initial four fits. Limited feature signal and sparse positive support
are more credible next research targets than further blind hyperparameter tuning.
No confirmation, rescue, emission or physical/live accuracy is measured here.

## Artifacts and reproduction

All output is in ignored `evaluation_runs/offline_10iter_20260907/`:

- `splits.json`, `protocol.json`, `baseline_before_changes.json`, `environment.json`.
- `baseline/` and `iteration_01/` through `iteration_10/`: scorecards, configs,
  hypotheses, comparisons, model JSON/NPZ, hashes and training histories.
- `frozen_challenger.json`, `best_challenger/` (iteration 10 only).
- `holdout_baseline_scorecard.json`, `holdout_challenger_scorecard.json`.
- `final_report.json` (complete machine-readable report), `report.md`,
  `experiment_source.tar.gz`, `reproduce.sh`.

From the repository root, reproduce in a NEW directory (never overwrite a run):

```bash
bash evaluation_runs/offline_10iter_20260907/reproduce.sh \
  evaluation_runs/offline_10iter_reproduction
```

That script contains all ten exact trial commands, including parent configurations,
fixed seed/defaults, hypothesis and changed key. Preparation and first fit are:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m automation.offline_10iter_prepare --output evaluation_runs/new-experiment
python3 -m automation.offline_10iter baseline --output evaluation_runs/new-experiment
python3 -m automation.offline_10iter trial --output evaluation_runs/new-experiment \
  --iteration 1 --parent 0 \
  --hypothesis 'A TRAIN-fitted linear ranker can recover candidates lost by descending saved detector score.' \
  --change '{"method":"learned"}'
```

`finalize` requires all ten results and creates an exclusive holdout seal before
scoring. Once sealed, further trial calls fail. `offline_10iter_report` packages
saved results without accessing holdout captures. Frozen input hashes are checked
before loading. Source artifacts and base commit identify implementation; historical
producer detector settings/models/calibration remain unknown. NumPy/Python versions
and effective experimental model configuration are recorded separately.

```bash
python3 -m automation.offline_10iter_selftest
python3 -m automation.evaluation_selftest
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m automation.v2230_selftest
```
