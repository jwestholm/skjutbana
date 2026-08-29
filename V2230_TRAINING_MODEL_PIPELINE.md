# V2.23.0 — Unified autonomous training/model pipeline

## Purpose

V2.23 deliberately freezes V2.22 runtime/perception work and moves effort to the actual learning problem:

> Given the actual candidate group produced for one shot, learn which candidate is the **current new physical hit**, improve from labelled F2/physical/offline data, and keep/reject challengers without allowing training data or holdouts to leak into live authority.

V2.23.0 does **not** change the selected live hit coordinate. Every V2.23 model is shadow/research only.

## Audit conclusion

The existing project already contains many useful pieces, but they solve different layers:

- `SimpleAIMemory` in `AIRuntime` is the learner F1/F2 currently updates online. It stores a bounded positive/negative memory and scores candidates by feature-space distance. It is useful continuity/debug infrastructure, not a protected champion/challenger ranker.
- V2.11/V9 is an offline physical-feature ranker and established the rule that policy/bookkeeping features must not be model input.
- V2.13–V2.15 Hole-AI asks whether a patch *looks like a physical hole*.
- V2.16 is the important bridge: F2 can save the **actual candidate group** with GT, candidate-centred before/after patches, provenance, and no fake recall from storage-forced GT rows.
- V2.17 correctly separates `HOLE` from `CURRENT NEW HOLE`; old holes are positive for static Hole-AI but negative for current-shot newness.
- V2.19 can generate new media/scenario worlds and compile them into the V2.16 candidate-pack family.
- V2.20 champion/challenger was the planned missing orchestration layer.
- V2.21.5 proved the dense physical proposal pool can have excellent recall, while its learned ranker still generalises poorly with too few independent shots. The proposal generator therefore remains a candidate source, not the V2.23 champion.

V2.23.0 implements the missing orchestration/data/model layer rather than replacing these components.

## New architecture

```text
                        labelled shot
                            |
            +---------------+----------------+
            |                                |
         F2 / F1                      manual physical GT
      projected/camera                   click on hit
            |                                |
            +---------------+----------------+
                            |
                   V2.23 native record
                actual candidates + GT
                            |
                            +<---- legacy V2.16 candidate packs
                            +<---- V2.20 generated candidate packs
                            |
                      unified dataset
                            |
                 session-safe split policy
                   /         |          \
             development  validation  protected holdout
                  |            |             |
                  |            |             +-- NOT used by autotrain selection
                  |
          train challenger portfolio
             linear listwise
             compact MLP listwise
                  |
             validation only
                  |
           research champion registry
                  |
            live SHADOW scoring
                  |
            NO authority in V2.23.0
```

## Native shot schema

Each native shot is append-only under:

```text
content/ai/training_v223/sessions/<session_id>/
    session.json
    index.jsonl
    shot_000001.json
    ...
```

A shot contains:

- stable session id and source kind (`f2_projected`, `f1_projected`, `physical_manual`, ...),
- GT camera XY and optional screen XY,
- background/sampling provenance,
- the **actual** candidate group,
- stable physical feature dict per candidate,
- baseline rank/score stored for comparison but **not** used as model input,
- GT distance/relevance labels stored only as labels,
- candidate pool oracle <=20px.

No V2.23 code inserts a candidate at GT. `capture_forced_gt_nearest` / diagnostic-forced rows from legacy packs are excluded from model input.

## Feature contract

V2.23.0 starts with a conservative scalar physical contract. Typical fields include detector score, area/radius/circularity, centre/local change, PRE→POST change, darkening/DoG/z-score, persistence, existed-before, temporal support, patch statistics and normalized position.

Explicitly forbidden as model input:

- GT distance / labels,
- current rank,
- `combined_score` / `ai_score`,
- `reason_*`, `rel_*`, `core_member`,
- storage-forced-GT flags.

The existing Hole-AI/NewHole-AI pixel models remain separate evidence sources. V2.23.0 audits them and preserves their semantics instead of silently treating their scores as physical truth. A later V2.23.x can add explicit model-evidence fusion once the exact installed artifacts and comparable candidate coverage are verified by `v2230_audit`.

## Models

V2.23.0 trains a small portfolio with no new ML dependency:

1. **Linear listwise ranker** — transparent baseline/challenger.
2. **Compact NumPy MLP listwise ranker** — captures feature interactions without requiring PyTorch/TensorFlow.

Training is per-shot/group, not independent candidate classification. Only shots where an actual <=20px candidate exists can teach candidate ordering. Proposal misses are reported as proposal misses; they are never hidden by injecting GT.

Model files are safe NPZ numeric arrays + JSON metadata (`allow_pickle=False`).

## Champion/challenger policy

- Challenger selection uses **validation only**.
- Protected holdout is listed but **never evaluated by the automatic selection loop**.
- With too few independent sessions, engineering validation may fall back to a deterministic shot split, but the result is marked `provisional`.
- V2.23.0 champions are always `research_shadow_champion`.
- `eligible_for_live_authority` is hard-coded false.
- The V2.22/global detector remains authority.

Primary selection objective:

1. conditional Top-1 <=20px when the pool actually contains GT,
2. MRR of the first <=20px candidate,
3. median positive rank.

Candidate-pool oracle is always reported separately. A ranking model cannot fix a proposal miss.

## F2 behavior

F2 keeps its current projected-camera training behavior. V2.23 wraps it without replacing the existing SimpleAIMemory path:

- every F2 shot is saved as a V2.23 candidate group before/independently of legacy click learning,
- when an F2 run finishes with >=10 saved groups, a short **background shadow challenger** run is scheduled,
- the game does not wait for this model and live authority is unchanged,
- the same dataset/model engine is available from CLI for guaranteed unattended/offline runs.

This makes F2 a data producer + quick self-training front end while preserving the older training path for comparison.

## Offline behavior

One portfolio:

```bash
python3 -m automation.v2230_train
```

Short plumbing run:

```bash
python3 -m automation.v2230_train --quick
```

Time-budgeted autonomous loop:

```bash
python3 -m automation.v2230_autotrain --hours 8
```

The autonomous loop repeatedly trains new seeded challenger portfolios and only updates the **research shadow champion** when validation improves. It never consumes protected holdout as a selection signal.

## Parked V2.22 TODOs

These are intentionally not fixed in V2.23.0 because they do not block model/data work:

- loading/mechanical sound may be accepted as an audio shot; future audio-proposal → physical-confirmation gate,
- remaining CV latency spikes,
- further local spatial-cluster cleanup,
- object-hit fast-path authority,
- final gameplay cursor/latency polish.

Re-open them only when they block data quality or after the learning pipeline has a credible model.
