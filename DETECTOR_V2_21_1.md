# Detector V2.21.1 — Short Full-Frame Physical-Domain Capture

## Why this patch exists

V2.21 measured the domain gap between the V2.20 offline generator and the
projector/camera candidate data. The result was unambiguous:

- group domain AUC: **1.0000**,
- shortcut warning: **true**,
- largest shift: `known_hole_distance_scaled` (KS 0.988, |SMD| 5.641),
- large temporal shifts in `mean_absdiff`, `p95_absdiff`, `persistence` and
  `center_absdiff`,
- large V2.17 embedding distribution shifts.

Therefore another synthetic-only V2.18 training run is not the next step.
The next required evidence is honest full-frame projector/camera PRE+POST data
that can be searched by AI_DIRECT without first requiring a V1/V2 candidate.

## What V2.21.1 changes

### 1. Short capture control

`automation.v221_collect_fullframe` arms a one-shot control and starts the
existing `AutomationAITrainingScene`. The requested number of rounds becomes
the scene's natural `auto_target_iterations`, so a 30-round capture really
finishes as a 30-round run rather than starting a legacy 100-round run and
racing an external stop command.

Default first run:

```bash
python3 -m automation.v221_collect_fullframe white --shots 30 --seed 22101
```

### 2. Learning frozen during evidence capture

By default the short capture temporarily sets runtime `benchmark_mode=True`.
This prevents the online learner from changing while the evidence set is being
collected. The previous runtime value is restored at natural completion,
manual stop, or scene exit.

This is capture/evaluation work, not another training experiment.

### 3. One-shot/stale-safe control

`content/ai/v221_capture_control.json` is consumed when the automation scene
opens. It has an expiry time and is immediately disabled after consumption.
A crashed collector therefore cannot silently turn a later manual F2 run into
a short/frozen run.

### 4. True recent PRE is wired into the candidate recorder

The V2.21 recorder/config supports `full_recent_pre_frame`, but the scene must
actually supply the recent pre-shot frame. V2.21.1 explicitly packages that
wiring:

- derive the shot timestamp,
- select a camera frame before the audio peak from `HitScanner.frame_history`,
- save candidate recent-PRE patches,
- save the full recent-PRE frame,
- save two full POST frames.

The current synthetic F2 timing reveals the projected hole *after* the audio
peak, so the selected recent PRE precedes the new projected hole.

This remains capture-only and does not alter HitScanner's live reference or
candidate authority.

## First gate

After the 30-round white capture:

```bash
python3 -m automation.physical_pack_v221_inspect \
  --root content/ai/candidate_shadow_v216

python3 -m automation.direct_proposal_v221_benchmark \
  --root content/ai/candidate_shadow_v216
```

The benchmark ignores the old 100 patch-only packs for AI_DIRECT scoring and
uses only packs that contain honest full-frame PRE+POST evidence.

The updated CLI prints an `ALL FULL-FRAME SHOTS` section because a 30-shot
single-session split is provisional and the aggregate is the primary first
engineering signal.

Primary metric:

```text
CURRENT oracle@20
AI_DIRECT oracle@20
CURRENT + AI_DIRECT union oracle@20
rescued CURRENT misses @20
```

Initial engineering gate: **union oracle@20 >= 70%** on the new full-frame
capture. This is not a live-authority gate.

## What V2.21.1 deliberately does not do

- does not reorder live candidates,
- does not emit AI_DIRECT hits into games,
- does not retrain V2.18,
- does not use confirmation/holdout data for learning,
- does not treat `known_hole_distance_scaled` as reliable visual evidence,
- does not claim 30 rounds are sufficient for final validation.

If AI_DIRECT materially rescues current misses, the next step is multiple
independent full-frame sessions/backgrounds and tuning proposal recall. If it
does not, inspect direct heatmaps/nearest-proposal failures before touching the
ranker.
