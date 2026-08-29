# V2.23.0 documentation patch (append-only)

The included `automation.v2230_apply_docs` appends these sections idempotently to existing project documentation rather than overwriting files that may have changed on `dev`.

## CURRENT_STATE.md

### V2.23.0 — autonomous training/model pipeline

- V2.22.1–V2.22.6 runtime is frozen as the current perception/runtime foundation while model learning becomes the primary workstream.
- Known audio loading/mechanical false-trigger behavior and remaining runtime polish are parked TODOs unless they corrupt training data.
- Audit: F1/F2 currently updates `SimpleAIMemory`; V2.11/V9, V2.15 Hole-AI, V2.17 NewHole-AI and V2.21.5 physical-dense are separate research/shadow models rather than one champion system.
- V2.16 candidate packs are the main reusable bridge because they preserve actual detector candidates + GT + candidate patches/provenance. V2.19 generated worlds can compile compatible candidate packs.
- V2.23 introduces a native append-only shot-group schema, tolerant legacy pack importer, stable physical feature contract, listwise linear/compact-MLP challengers, validation-only research champion registry, F2 capture, and offline/time-budgeted autotrain.
- Protected holdout is never used by automatic model selection. V2.23.0 can promote only `research_shadow_champion`; `eligible_for_live_authority=false` is non-negotiable.

## HIT_DETECTION_PLAN.md

### V2.23 — unified self-learning layer

The active objective is no longer more runtime filtering. It is to learn the correct **current new hit** among the actual candidate group while preserving proposal recall as a separate metric.

Training sources:

1. V2.23 native F2 projected-camera groups,
2. V2.23 manual physical GT groups,
3. V2.16 historical candidate packs,
4. V2.19/V2.20 generated candidate packs,
5. later explicit evidence adapters for V2.15 Hole-AI, V2.17 NewHole-AI and other independent sources.

Model selection rules:

- split by whole physical session whenever independent sessions exist,
- provisional engineering split when they do not,
- never inject/force GT into model candidate pools,
- report proposal oracle separately from ranking accuracy,
- never use `reason_*`, `core_member`, GT distance, current rank or current model score as physical feature shortcuts,
- validation chooses challengers; protected holdout never drives autonomous selection,
- only research/shadow champion is allowed in V2.23.0.

Immediate data goal: accumulate several genuinely independent physical/projected sessions so candidate-ranker generalisation can be measured honestly. Continue autonomous generated-world work for scale, but unseen physical sessions decide eventual authority.

## AI_CONTEXT.md

### V2.23 model semantics

Keep three concepts separate:

- **Hole appearance:** `P(looks like a physical hole)`; old and new holes may both be positive.
- **New-hole evidence:** `P(changed into the current new hole at this shot)`; old holes are NOT-current negatives.
- **Candidate ranking:** given the actual candidate group, choose the current hit using physical/newness/model evidence without inventing a coordinate.

F2 is now both a legacy online-memory learner and a V2.23 data producer. V2.23 challenger training is independent of `SimpleAIMemory` and may run after F2 or from CLI. V2.23 shadow scores must never reorder/override live candidates in this release.

### Parked runtime TODO

- audio mechanical/loading false trigger -> future audio proposal / physical confirmation,
- remaining CV latency spikes,
- object-hit fast-path authority,
- final cursor/gameplay polish.
