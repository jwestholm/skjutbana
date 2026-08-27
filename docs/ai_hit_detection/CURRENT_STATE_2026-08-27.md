# Current state — 2026-08-27

## Goal

Reliable physical hit detection for the projector/camera shooting-range system, eventually >=95% on unseen physical mixed-background sessions.

The detector must identify the **new hole from the current shot**, not merely any dark mark or any hole-like patch.

## Hardware / runtime context

- projector-based target/game display,
- high-resolution USB camera with integrated microphone,
- audio peak triggers a shot event,
- camera frames are mapped to projector/game coordinates using calibration/homography,
- current live hit authority remains the existing scanner/game path; AI development is shadow/offline.

## Existing detector concepts

### Classical candidate generation

Current V1/V2 CV stack produces many candidate locations per shot. On difficult physical data it can generate hundreds of candidates but still omit the real new hole.

### Known holes

`HitScanner` already owns a known-hole registry and duplicate/rehit logic. Do not build another independent known-hole map.

Important semantics:

- known-hole state is incomplete,
- it is session-local,
- an unregistered old physical hole can still be present,
- proximity to a known hole is soft evidence only,
- hole-in-hole/re-hit must remain possible.

### V2.15 static Hole-AI

Learns “hole-like appearance”. Old and new holes are positive.

Useful as an evidence source, but it did not solve candidate ranking on the physical candidate pool.

### V2.17 NEW-hole AI

Uses before/after evidence to model the **current new hole** rather than generic hole appearance.

Patch-level AUC was strong, but pointwise candidate classification did not translate into same-shot ranking.

### V2.18 listwise ranker

Reuses the V2.17 representation and trains candidates in whole-shot groups with a listwise objective plus an offset/refinement head.

This is the first ranking formulation that clearly learned same-shot selection on development data.

## Current data inventory

### Diagnostic archive

`content/ai/shot_diag`

- about 39k files / about 13k shot groups,
- mostly white-background diagnostics,
- images contain diagnostic overlays/crosshairs/text,
- useful for human debugging,
- **not honest raw full-frame training data**.

### Camera-observed projected hole bank

`content/ai/holes/synt_*`

- about 15,134 synthetic projected hole patches,
- generated on projector -> physical surface -> camera,
- several background modes/sessions,
- valuable camera-domain hole appearance evidence.

Important: source patches are centered. Never allow the model to learn “hole = centre of patch”.

### Real hole holdout

`content/ai/holes/hole_*`

- 37 real hole patches in the measured archive at the time of V2.13–V2.15 work,
- kept out of ordinary training,
- used as small real/golden appearance holdout.

### Physical candidate session

V2.16 seed `65432`, white background:

- 100 labelled shot packs,
- ~38,423 candidates total,
- ~384 candidates/shot,
- one physical session only,
- 60/20/20 split is therefore **provisional**, not an independent-session acceptance split.

### V2.20.2 generated train worlds

- seeds `1..100`,
- 100 groups,
- 38,400 candidate rows after cap,
- designed for model development only.

### Frozen generated validation

- seeds `9000001..9000100`,
- 100 candidate groups,
- all 100 contained a candidate <=20 px and <=42 px,
- separate root/session,
- must remain frozen and never become training data.

The validation compile report also contains both `game` and `photo_or_image` scenarios and multiple challenge classes including dense old holes, near-edge, near-old-hole and one hole-in-hole case.

## Current strongest conclusions

1. **Listwise ranking is the right direction.** V2.18 can learn same-shot candidate selection.
2. **V2.20.2 synthetic worlds are internally learnable and generalise to unseen generated seeds.**
3. **Synthetic -> physical ranking transfer currently fails completely.**
4. **Physical proposal recall is still a hard ceiling.** Existing raw+ranked union oracle is only 35% confirmation / 50% holdout within 20 px.
5. **More synthetic-only epochs are not the next step.**
6. **Direct proposals and physical-domain bridging are now higher priority than further simulator cosmetics.**

## Current model/result snapshot

### V2.18 trained on generated seeds 1..100

Top-1 on its generated-session split:

- development: 96.67%
- confirmation: 100%
- holdout: 95%

### Same frozen model on unseen generated seeds 9000001..9000100

Top-1:

- development: 98.33%
- confirmation: 100%
- holdout: 100%

### Same frozen model on physical V2.16 session

Top-1:

- development: 0%
- confirmation: 0%
- holdout: 0%

Median GT ranks:

- development: 110
- confirmation: 125
- holdout: 104

This is the current central research result.

## V2.21 implementation added after the synthetic->camera transfer test

The next code delta now contains:

- explicit old-pack full-frame audit,
- synthetic-vs-camera candidate domain-gap profiler,
- direct full-frame PRE->POST proposal engine,
- direct proposal oracle/rescue benchmark,
- storage-aware full recent-PRE + two-POST capture for future dedicated F2 sessions,
- no live authority changes.

The next empirical decision is driven by the V2.21 audit/domain-gap report. If the old 100-shot packs lack full frames, a small fresh automated projector/camera session is required before honest physical direct-proposal recall can be measured.
