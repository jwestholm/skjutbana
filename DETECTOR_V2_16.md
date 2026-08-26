# Detector V2.16 — Candidate-Level Shadow Capture + Hard Negatives

## Why this version exists

V2.13–V2.15 answered an important but limited question: a small pixel model can
learn hole morphology from projector/camera `synt_*` patches and transfer to a
small real-hole holdout.  That does **not** yet prove it helps the shooting
range, because the real task is to choose the new hole among the candidates
produced by V1/V2.

V2.16 creates that missing bridge.

> One automation F2 run can now capture the *actual detector candidate pool*
> together with candidate-centred before/after camera patches and known GT.
> Once captured, Hole-AI/V9/temporal/fusion experiments and hard-negative mining
> can be repeated offline without the projector or camera.

This version remains capture/offline/shadow only.  It never changes the live hit
coordinate or candidate order.

## Scope and safety boundary

Only `AutomationAITrainingScene` is instrumented.  That scene is the subclass
used by externally automated F2 sessions.  Normal `AITrainingScene` and game
scenes are not changed by V2.16 candidate capture.

Capture failures are caught and logged; they are not allowed to break the real
F2 detector/training path.

## Candidate pack format

Default output:

```text
content/ai/candidate_shadow_v216/
  sessions/
    <timestamp>_seed<seed>_<background>/
      session.json
      index.jsonl
      shot_000001.json
      shot_000001.npz
      ...
```

Each shot stores:

- known GT screen + camera X/Y,
- actual ranked candidates and additional raw detector candidates,
- candidate provenance and original JSON-safe detector features,
- GT distance / <=10, <=20 and <=42 labels,
- one pre-shot 64x64 patch per candidate when available,
- up to three post-shot 64x64 patches per candidate,
- dedicated GT pre/post patches even if the detector missed GT,
- post-frame timestamps,
- optional full-frame pre/post imagery when explicitly enabled.

The default does **not** save 4K full frames because repeated 100-shot runs can
consume large amounts of disk.  Set `save_full_frames=true` in
`content/ai/candidate_shadow_v216.json` for a dedicated full-frame capture run.
Candidate patches are always enough for the V2.16 Hole-AI/hard-negative work.

### Storage cap is not allowed to fake recall

`max_candidates` defaults to 384.  If the GT-nearest source candidate lies
outside that storage cap, V2.16 can append it as
`capture_forced_gt_nearest=true`.  That row is **diagnostic only** and benchmark
code excludes it from live-pool/oracle metrics.  A storage convenience must
never turn a detector miss into a reported detector hit.

## Evidence scored on exactly the same candidate

Offline V2.16 attaches these shadow sources to each captured candidate:

1. **Current rank** — the actual V1/V2/funnel order from the captured run.
2. **Hole-AI V2.15** — mild, standard, fused probability, disagreement and
   offset refinement from the candidate-centred post patches.
3. **Temporal candidate evidence** — persistent before/after change in the same
   candidate crop.  This is a deliberately simple local companion to the
   full-frame V2.12 overlay, not a replacement for V2.12.
4. **V9 physical ranker** — when `content/ai/ranker_v9_offline.json` exists and
   can score the preserved candidate features.

The benchmark reports both:

- the current **ranked pool**, and
- the captured **raw + ranked union**.

This distinction is important.  If Hole-AI can rescue a raw V1/V2 candidate
that the current funnel removed, that is useful evidence for a later fusion
stage — but V2.16 still does not make it authoritative.

## Fusion discipline

V2.16 performs a small transparent convex search across:

- current-rank evidence,
- V9 physical evidence,
- Hole-AI evidence,
- temporal evidence.

Pure endpoints are included.  The optimizer is allowed to conclude that an
extra source adds no value.

With >=3 capture sessions, development/confirmation/holdout are split by whole
capture session.  With only one or two sessions V2.16 still produces an
engineering report, but labels the shot-level split **provisional** and hard
codes `eligible_for_live_authority=false`.

## Hard-negative mining — V2.17 semantic correction

Run:

```bash
python3 -m automation.candidate_v216_hardnegatives --export-images
```

The miner selects candidates far from the **current shot GT** (default >=55 px)
that nevertheless look attractive to one or more of current-rank, V9, Hole-AI
or temporal evidence.  These are excellent *NOT-NEW-hole* examples, but an
important V2.17 audit found that they are **not automatically static non-holes**.
A wrong candidate can be an old physical bullet hole.

Therefore the exporter now uses:

```text
new_hole_label   = 0
static_hole_label = null
label             = null
```

Old manifests created before this correction may contain generic `label=0`.
V2.17 deliberately ignores that field and uses only candidate identity/GT
semantics.  Do **not** feed the 800 exported candidates directly into V2.15
Hole-AI as negative hole examples.

V2.16 only exports candidate evidence. V2.17 adds the separate before/after
NEW-hole learner where old holes are semantically valid negatives.

## Commands

After installing V2.16:

```bash
python3 -m automation.candidate_v216_selftest
python3 -m automation.candidate_v216_verify
```

Before a capture there will naturally be zero candidate packs.

For the first useful data capture, run the normal game plus **one** automated
1x100 F2 session with a new seed.  V2.16 captures silently in parallel.
Afterwards, with projector/camera no longer needed:

```bash
python3 -m automation.candidate_v216_inspect
python3 -m automation.candidate_v216_benchmark
python3 -m automation.candidate_v216_hardnegatives --export-images
```

## What counts as success

The first 100-shot capture is an **external candidate-level experiment**, not a
final >=95% gate.  We want to learn:

- current V1/V2 oracle recall at <=20/42,
- current Top-1/Top-3,
- Hole-AI Top-1/Top-3 on the same pool,
- V9 Top-1/Top-3 when available,
- temporal-only performance,
- learned fusion performance,
- whether raw+ranked union contains GT that current ranked pool lost,
- whether Hole-AI offset refinement reduces localisation error,
- which wrong candidates become the strongest real hard negatives.

The V2.16 mined rows must **not** be blindly used to retrain static Hole-AI: a
wrong candidate can be an old real bullet hole.  The immediate next step is
V2.17 NEW-hole learning, where those same rows are semantically safe negatives
for the question "is this the hole that appeared at the current shot?" because
the model receives before+after evidence.  Static Hole-AI continues to treat
old and new physical holes as hole-like positives.

## Explicit non-goals

V2.16 does not:

- change game hit authority,
- re-order live candidates,
- override camera coordinates,
- claim the V2.15 `either` oracle is achievable,
- use game-object or spatial priors yet,
- replace V2.12 full-frame replay.

The goal is measurement discipline: turn one expensive physical/projected F2
run into a reusable offline candidate dataset.
