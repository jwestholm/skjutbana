# V2.23.1 — High-recall training pool and support-gated champion

## Why this delta exists

The first V2.23.0 100-shot F2 test proved the orchestration works: capture -> dataset -> challenger -> registry. It also exposed two correctness bugs in the learning boundary:

1. native V2.23 capture stored mostly the already truncated final candidate list, yielding only 10/100 proposal oracle within 20 camera pixels;
2. a validation split with zero usable <=20 candidates could still produce `promoted=True`.

The local audit also found 330 V2.16/V2.20 shot JSON files and about 1.9 GB of corresponding data, but V2.23.0 imported 0/330 because it treated the legacy format as JSON-only. The canonical repository contract is `CandidatePackV216.load(path)` (JSON + sibling NPZ).

## Changes

### Native F2/manual capture

After the normal V2.8 `rank_with_funnel` has run, V2.23.1 captures a GT-free union of:

- `_v28_all_hypotheses`
- `_v28_hypothesis_pool`
- `_v28_core_pool`
- `_v28_recall_baseline_pool`
- `_v28_actual_pool`
- normal runtime/latest/ranked/scanner candidates

Coordinates are deduplicated without consulting GT. GT is attached only after proposal creation for labels and diagnostics.

Each F2 shot now logs a concise line:

`[V2.23.1 POOL] shot=N union=... v28_all=... v28_recall=... nearest=... oracle20=... oracle42=...`

### Legacy import

`candidate_shadow_v216`, `candidate_synthetic_v220`, and `candidate_synthetic_v220_validation` are loaded using `CandidatePackV216.load(path)`, `pack.gt_xy` and `pack.candidates`.

- V2.16 split semantics reuse `_shot_split_keys_v217` when available.
- V2.20 engineering corpus is development.
- `candidate_synthetic_v220_validation` is protected holdout.
- forced-GT storage rows remain excluded.

Converted candidate-only records are cached under `content/ai/training_v223/cache/legacy_v2231/` to avoid rereading large patch/full-frame arrays every autonomous training round.

### Metrics and model gate

Dataset summary reports proposal oracle @5/@10/@20/@42 separately from model ranking.

Automatic challenger training requires:

- at least 8 development shots with an actual <=20 candidate;
- at least 12 validation shots;
- at least 5 validation shots with an actual <=20 candidate.

Research promotion additionally requires validation oracle20 rate >=20% and measurable improvement over baseline in conditional Top-1@20 or MRR@20.

These are engineering/research gates only. Protected holdout remains invisible to automatic selection and `eligible_for_live_authority` remains false.

### Existing V2.23.0 champion

A champion created before these support gates can remain on disk for audit, but `load_champion_model()` refuses to load it if the new gate fails. This quarantines the observed zero-oracle validation champion without deleting historical artifacts.

## Not in this delta

- No new live hit authority.
- No audio-trigger fix.
- No dense V2.21.5 full-frame recompilation in the live F2 hot path.
- No Hole-AI/NewHole-AI score fusion yet.
- No gameplay object-hit authority.

V2.21.5 dense proposal remains an important independent expert. Once the canonical legacy import and V2.8 high-recall native capture are verified, the next model step can deliberately fuse NewHole/Hole/dense evidence rather than mixing them into one ambiguous label.
