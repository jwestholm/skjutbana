# Runbook — V2.21 next actions

## Current status

The synthetic-only question is answered:

- V2.18 can learn V2.20 generated worlds,
- it generalises to new generated seeds,
- the same frozen ranker fails on the projector/camera candidate session,
- existing projector/camera candidate oracle is too low for ranker-only work.

V2.21 therefore measures domain gap and starts direct full-frame proposals.

## 1. Install and selftest

```bash
python3 -m automation.offline_v221_selftest
```

Do not continue if this fails.

## 2. Audit the existing 100-shot projector/camera candidate session

```bash
python3 -m automation.physical_pack_v221_inspect \
  --root content/ai/candidate_shadow_v216
```

Expected for the old V2.16 seed-65432 session: candidate/GT patches are available, but full recent-PRE/full-POST evidence may be missing because old capture config had `save_full_frames=false`. The tool reports the truth instead of assuming.

## 3. Run candidate-domain gap profiling on existing data

```bash
python3 -m automation.domain_gap_v221 \
  --synthetic-root content/ai/candidate_synthetic_v220 \
  --physical-root content/ai/candidate_shadow_v216 \
  --synthetic-cache content/ai/reports/v218/v220_cache \
  --physical-cache content/ai/reports/v218/v220_physical_cache
```

Send back:

- terminal output, or
- preferably `content/ai/reports/v221/domain_gap_v221.json`.

Most important values:

- group domain-classifier AUC,
- top shifted temporal features,
- top shifted <=20px features.

AUC >=0.95 means the synthetic and camera domains are trivially separable to this representation and shortcut/domain mismatch work is mandatory.

## 4. Direct proposal benchmark on old packs

You may run:

```bash
python3 -m automation.direct_proposal_v221_benchmark \
  --root content/ai/candidate_shadow_v216
```

If old packs lack full frames the program should explicitly say so and produce no fake recall score.

## 5. If full frames are missing: collect a new dedicated projector/camera automation session

The V2.21 delta changes `content/ai/candidate_shadow_v216.json` so dedicated F2 automation capture now saves:

```text
full recent PRE : yes
full POST       : 2 frames
full reference PRE : no
```

This is storage-aware and shadow-only. It does not change the live scanner.

Start with a modest fresh session before another 100:

- 20–30 automated projected-hole rounds,
- at least one static/light background first,
- preferably a second contrasting background after baseline works.

After capture, run the audit and direct benchmark against the new capture root/session.

## 6. Direct-proposal gate

Primary target is not Top-1 yet. It is union oracle <=20 px:

```text
initial useful: >=70% confirmation + holdout
strong:         >=85%
pre-authority:  >=95% on multiple independent camera sessions
```

Only after proposal coverage materially rises should V2.21 proceed to physical-domain bridge ranker training.

## Do not do yet

- no 12-hour V2.18 synthetic-only training,
- no live AI authority,
- no using physical confirmation/holdout for gradient updates,
- no training on `shot_diag` overlay images as if they were raw frames.
