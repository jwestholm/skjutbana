# Skjutbana V2.23.6 — Registered Evidence Direct Heatmap Localizer

Install this delta on top of V2.23.5.

## Why this version exists

V2.23.2 proved that the registered physical-evidence pipeline can place a dense candidate within 20 px of GT on roughly 95% of a fresh 100-shot F2 session. V2.23.3–V2.23.5 then tried several ways to rank ~9,500 candidates down to a useful Top-K. V2.23.5 improved the median positive rank and Top512 retention, but it still failed the explicit learnability gate.

V2.23.6 therefore advances the master plan to **direct AI proposal / heatmap localisation** instead of adding another global candidate ranker.

The dense pool is kept as a teacher/diagnostic/fallback. It is not thrown away.

## Architecture

```text
PRE + POST framepack
        ↓
registered / photometrically compensated physical evidence
        ↓
8 spatial evidence maps
        ↓
4x block downsample (full frame retained spatially)
        ↓
fully-convolutional heatmap localizer
        ↓
Top spatial peaks
        ↓
XY directly
```

Input channels are the same physical families used by the successful dense engine:

- blackhat_gain
- tophat_gain
- persistent_abs
- gradient_gain
- persistent_dark
- persistent_bright
- fused
- compact_change

GT is never used to construct these maps. GT is used only for engineering training labels and evaluation.

## Important changes versus V2.23.5

- No ~9,500-way global candidate ranking is required by the new path.
- Full spatial registered evidence is cached at stride 4.
- Linear-convolution and small nonlinear spatial-convolution models are tested.
- Hard-negative mining runs once, but the pre-mining checkpoint is retained and automatically wins if mining makes validation worse.
- Deterministic map baselines are measured too. If a simple physical-map policy beats the learned model, V2.23.6 reports that instead of forcing AI to be used just because it is AI.
- Optional nearest-dense snap is reported only as a diagnostic.
- Fresh-domain data is not touched until engineering model selection is finished.
- Live authority remains NO.

## Practical gates

The bootstrap test now asks whether the **direct localisation path** is useful, regardless of whether the best engineering policy is learned or deterministic:

- Top1 @20 >= 25%
- Top3 @20 >= 50%
- median Top1 XY error <= 100 px

The stricter learned-model gate additionally requires the learned model to beat the best deterministic validation baseline.

A later fresh-session research gate requires the validation-selected policy to achieve:

- Top1 @20 >= 35%
- Top3 @20 >= 60%
- median Top1 error <= 80 px

These are research/shadow gates only. They grant no live hit authority.

## Install/test

```bash
cd ~/skjutbana/skjutbana

python3 -m automation.v2236_selftest
python3 -m automation.v2236_verify_install
python3 -m automation.v2236_status
```

Prepare the direct-map cache from the 100 framepacks already captured:

```bash
python3 -m automation.v2236_prepare --session latest
```

Then run the learnability test:

```bash
python3 -m automation.v2236_train --quick --no-prepare
python3 -m automation.v2236_status
```

Do **not** capture a new F2 x100 before inspecting this result.

## What to send back

Send the complete `V2.23.6 TRAIN SUMMARY` and `v2236_status` output.

The most important values are:

- best deterministic baseline Top1@20 / Top3@20 / median error
- best heatmap model Top1@20 / Top3@20 / median/P95 error
- selected direct policy
- direct-path gate
- learned-model gate

## Path toward games

If this direct localisation path is clearly useful, the next step should not be another large offline detour. The next implementation should put the frozen direct policy into the existing shot resolver in shadow/advisory form, benchmark latency and fresh-session accuracy, and then make the hit-input contract stable enough that game development can continue independently while AI accuracy keeps improving behind it.
