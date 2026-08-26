# Detector V2.17 — NEW-hole AI + Old-hole-safe Learning Semantics

## Why V2.17 exists

V2.16 produced the first real candidate-level capture:

- 100 projected/camera shots,
- 38,423 captured detector candidates,
- about 384 candidates/shot,
- 800 mined high-value wrong candidates,
- ranked-pool <=20px oracle recall only 25–45% across the provisional split,
- V9 helped ranking slightly,
- Hole-AI V2.15 and the first temporal heuristic did not yet deserve fusion weight.

The obvious reaction would be "retrain Hole-AI on the 800 negatives". That is
**wrong** because a wrong candidate can be an *old real bullet hole*. Static
Hole-AI asks "does this look like a hole?" and must be rewarded for recognizing
both old and new holes.

V2.17 therefore separates two different questions:

```text
Hole-AI V2.15:
    Does this PATCH look like a physical bullet hole?

NewHole-AI V2.17:
    Did a hole-like physical change APPEAR HERE at THIS shot?
```

That distinction is the core of this version.

## Label contract — non-negotiable

For static Hole-AI:

| candidate type | static Hole-AI label |
|---|---|
| current new hole | positive |
| old/pre-existing real hole | positive |
| confirmed non-hole | negative |
| uncertain | do not train |

For V2.17 NEW-hole AI:

| candidate type | NEW-hole label |
|---|---|
| current new hole | positive |
| old/pre-existing real hole | negative |
| non-hole | negative |
| uncertain far-from-current-GT candidate | still valid negative for **newness**, but not a static non-hole label |

Therefore **distance from current GT may never be used by itself to create a
static Hole-AI negative**. The V2.16 hard-negative exporter has been corrected:
it now writes `new_hole_label=0`, `static_hole_label=null`, and `label=null`.
Old manifests already exported by V2.16 are still safe for V2.17 because the
V2.17 loader ignores their old generic `label` field and uses only pack/row
identity.

## Existing `HitScanner.known_holes`

The current live scanner already has a known-hole registry and uses it as a
soft score penalty / re-hit mechanism. V2.17 does **not** create a second hole
registry.

Important limitation discovered during the audit:

- `known_holes` is reset when the AI training scene starts/exits,
- it therefore represents holes accepted by HitScanner during the current
  session,
- it is **not** a guaranteed inventory of physical holes that were already on
  the target before scene start.

V2.17 starts capturing a snapshot of this existing registry before each future
F2 round as extra provenance. It remains soft context only. The before/after
model is what protects us from repeatedly choosing a pre-existing physical hole
that is absent from the session registry.

Audit note: current `HitScanner` contains a documented soft penalty branch out
to `duplicate_radius_px * 1.5`, while `_is_near_known_hole()` currently returns
only holes within `duplicate_radius_px`; that outer branch is therefore
unreachable. V2.17 documents this but deliberately does **not** change live
candidate scoring while this work remains shadow/offline.

## Model input

For the same candidate coordinate, V2.17 consumes:

- pre-shot 64x64 camera patch,
- median/stack of post-shot camera patches,
- no game context,
- no GT at inference time,
- no known-hole list baked into the neural model.

### Important audit result: reference PRE vs true temporal PRE

The current `HitScanner._capture_pre_shot_snapshot()` normally uses the stable
`scene_reference_gray` when one exists. That is useful for the current classic
detector on a static target, but it is **not** a sufficiently current temporal
reference when game/video content moves. A moving sprite can otherwise look
like a gigantic "new" change.

V2.17 therefore extends future V2.16 candidate packs with an additional
`recent_pre_*` capture taken from `HitScanner.frame_history` shortly before the
audio peak. This is capture-only and does not change the live detector. The
V2.17 loader prefers this true recent-pre patch when available and falls back
to the legacy/reference pre patch for the already captured seed-65432 session.
That keeps the existing 100-shot dataset usable while making future dynamic
background captures semantically correct for NEW-hole learning.

The model uses six local before/after feature maps:

1. signed pre→post change,
2. absolute change,
3. darkening change,
4. local high-pass/residual change,
5. morphological black-hat gain,
6. gradient/edge gain,

plus eight physical scalar summaries including centre change, p95 change and
post-frame persistence.

The output is:

```text
P(NEW hole at this candidate)
+ dx/dy refinement to the new-hole centre
```

The model is intentionally small NumPy/OpenCV MLP infrastructure, consistent
with V2.13–V2.15. This is still a learning experiment, not the final network.

## Training data from V2.16

V2.17 can train immediately on the already captured seed-65432 session.

Positive examples:

- the dedicated GT before/post patch saved for every shot,
- optionally the nearest *actual live candidate* when it lies <=16px from GT.

Negative examples:

- the V2.16 mined hard candidates far from current GT (default >=55px).

Those negatives may include old real holes. That is **correct for NEW-hole
learning** and explicitly forbidden for static Hole-AI retraining.

The GT patch remains useful even when V1/V2 missed GT completely. This lets the
novelty model learn what the missing new change looked like and is a stepping
stone toward later direct AI proposals.

## Split discipline

V2.17 reuses V2.16's shot/session split semantics:

- >=3 capture sessions: whole-session development / confirmation / holdout,
- one/two sessions: shot-level split is marked **provisional**,
- `eligible_for_live_authority` is always false in V2.17.

The current 100-shot seed-65432 dataset is therefore valid for engineering and
proof-of-learning only, not for an authority claim.

## Commands

```bash
python3 -m automation.newhole_v217_selftest
python3 -m automation.newhole_v217_inspect
python3 -m automation.newhole_v217_train
python3 -m automation.newhole_v217_benchmark
python3 -m automation.newhole_v217_verify
```

The already exported V2.16 hard-negative manifest is used automatically:

```text
content/ai/reports/v216/hard_negatives/hard_negatives.jsonl
```

There is no need to rerun the projector/camera to test V2.17.

## What success means

V2.17 must answer:

1. Can a before/after learner distinguish the current new change from the real
   wrong candidates that fooled the detector?
2. Does candidate ranking by NEW-hole probability beat the V2.16 temporal
   heuristic and preferably begin to approach/beat V9 on the provisional
   candidate pool?
3. How many mined NOT-NEW examples are likely old-hole-like? This is reported
   diagnostically, not relabelled as non-hole.
4. Does offset refinement contain useful localisation signal?

Even a strong result does **not** grant live authority. The next gate is several
independent capture sessions and then champion/challenger offline learning.

## Why this is the right next step toward overnight learning

V2.17 creates a semantically correct self-learning target:

```text
capture expensive physical/projected shots once
        ↓
mine candidates that fooled the system
        ↓
train P(NEW hole | before, after, candidate)
        ↓
benchmark on frozen candidate packs
        ↓
keep/reject challenger
```

Once this improves on held-out candidate packs, the next version can automate
that loop for hours/days without projector or camera. That is the first true
"start AI learning and leave it overnight" stage.


## Observed V2.17 result on shooting PC (2026-08-26)

Using the first 100-shot seed-65432 V2.16 capture (one session, therefore provisional):

- development patch AUC: **0.924010**, recall 0.900000,
- confirmation patch AUC: **0.935817**, recall 0.923077,
- holdout patch AUC: **0.845982**, recall 0.821429,
- candidate Top-1 <=20px: **0%** on development/confirmation/holdout,
- candidate Top-3: 1.67% development, 0% confirmation/holdout,
- median candidate GT rank: 65 / 33 / 84,
- raw candidate oracle <=20px: 36.67% / 25% / 45%.

**Decision:** KEEP the learned before/after representation, but reject pointwise classification as the candidate-ranking objective. Strong AUC together with zero Top-1 is direct evidence that `P(NEW|patch)` is not enough to order hundreds of same-shot candidates. V2.18 therefore trains per-shot listwise ranking and offset refinement over the frozen V2.17 representation before any overnight optimizer is built.
