# Detector V2.13 — Hole-AI Learning Proof

## Why V2.13 exists

V2.12 established the offline/replay architecture and the first independent temporal physical overlay. On the shooting computer we then discovered that the historical image stores have two very different meanings:

- `content/ai/shot_diag/` contains debug pictures with drawn crosshairs/text. Keep these for human troubleshooting; they are not clean AI training data.
- `content/ai/holes/` contains raw 128x128 camera patches plus JSON metadata. The current shooting-PC inventory is roughly 15,134 `synt_*` images and 37 real `hole_*` images.

The `synt_*` patches are valuable: the hole overlay was synthetic, but the displayed hole was photographed through the **real projector -> physical surface -> real camera** chain. They therefore carry the optics, exposure, focus and sensor characteristics of the installation.

V2.13 answers one narrow question before we build a larger network or give AI any live authority:

> Can a trainable pixel model learn useful hole morphology from the existing synthetic-camera archive, generalise to sessions/backgrounds it never trained on, and still recognise the 37 real holes it never saw during training?

If NO, we change the data/model before adding AI authority. If YES, Hole-AI becomes a justified new evidence source for V2.14.

## Critical anti-centre-bias design

Every `synt_*` and `hole_*` source image is centred around the known hole by archive construction. Feeding those source images directly to a binary classifier would be a bad experiment: the task would accidentally become "what normally appears in the centre of this archive format?".

V2.13 therefore **never uses the original 128x128 source crop directly as the class example**.

For each training example:

1. The source hole position is implicitly known at the source-image centre.
2. A **candidate point** is sampled at a random offset from the hole.
3. A smaller 64x64 crop is taken **around the candidate**, not around the hole.
4. Positive candidate points are randomly jittered up to 14 px from the hole.
5. Negative candidate points are 24..30 px away from the hole, using the same source image and the same preprocessing.
6. The network predicts both:
   - `P(candidate is a hole)`
   - `dx, dy` from candidate centre back to the hole.

This means the hole visibly moves around inside positive model inputs. The auxiliary offset task also forces localisation: a model that merely memorised a fixed centre cannot solve it.

There is an explicit `off_center_stress` evaluation where positive offsets are larger than normal training jitter.

## Model

V2.13 intentionally uses a **small one-hidden-layer neural network (MLP)** implemented in numpy. It requires no new ML framework beyond numpy/OpenCV already used by the project.

Input for each candidate crop:

- resized normalised grayscale pixels,
- local high-pass pixel channel.

Outputs:

- hole probability,
- X refinement offset,
- Y refinement offset.

This is not claimed to be the final architecture. The purpose is to establish whether raw pixels contain useful learnable information before introducing PyTorch/ONNX/CNN deployment complexity.

## Data separation

### Synthetic training data

`synt_*.png/json` only.

The split is by **whole session**, never random image. One physical session cannot appear in both train and synthetic test.

### Novel-background holdout

By default these backgrounds are excluded from all training:

- `black`
- `checker`
- `gray`
- `bubbles`

They produce a separate `novel_background_holdout` result. This asks whether the model learned hole morphology or only the dominant white/white-grid projection patterns.

### Real golden holdout

Every `hole_*.png` is excluded from:

- training,
- validation,
- threshold selection.

The 37 real holes therefore become the first real-domain transfer test. Since each raw real hole crop also contains genuine nearby background, V2.13 can make candidate-centred positive and local-negative examples from it without modifying the original file.

Do **not** interpret this small real test as final 95% live performance. It is a domain-transfer signal.

## Non-learning baseline

Every evaluation also measures a deliberately simple baseline:

> darkness/contrast at candidate centre versus a surrounding ring.

The threshold for both Hole-AI and the baseline is selected on synthetic validation only. Test/novel-background/real results then use those frozen thresholds.

This prevents us from calling a trivial "dark centre" detector AI progress.

## Expected outputs

Training writes only under the already-local AI reports area:

```text
content/ai/reports/v213/
    hole_patch_ai_v213.npz
    hole_v213_report.json
```

The JSON report contains:

- archive inventory,
- session assignments,
- background split,
- epoch learning curve,
- selected validation threshold,
- AI vs centre-contrast baseline,
- synthetic session test,
- novel-background holdout,
- real holdout,
- off-centre stress,
- offset-refinement error.

## Commands

From the repository root:

```bash
python3 -m automation.hole_v213_selftest
```

Then inspect the real archive:

```bash
python3 -m automation.hole_v213_inspect
```

Recommended first real training run:

```bash
python3 -m automation.hole_v213_train
```

For a quick plumbing run before using all 15k sources:

```bash
python3 -m automation.hole_v213_train \
  --epochs 3 \
  --max-train-assets 2000 \
  --max-eval-assets 500
```

Re-evaluate a saved model without retraining:

```bash
python3 -m automation.hole_v213_evaluate
```

Visualise the exact candidate-centred crops (including deliberately off-centre positives) after training:

```bash
python3 -m automation.hole_v213_visualize --kind synthetic --index 0
python3 -m automation.hole_v213_visualize --kind real --index 0
```

In the generated diagnostic crops the cyan cross is the **candidate centre**, the red circle is the true hole position for positive samples, and the magenta circle is the network's predicted refinement. This is a direct visual check that the network input is not simply the original centred 128x128 archive format.

## What constitutes a useful V2.13 result?

The most important signs are not training accuracy.

We want:

1. validation and held-out-session AUC/F1 clearly above the simple centre-contrast baseline,
2. no collapse on `off_center_stress`,
3. useful performance on backgrounds that were completely excluded from training,
4. encouraging **positive recall on the 37 real holes** at a threshold chosen without seeing those real holes,
5. offset-refinement error showing the network really tracks hole position inside the crop.

If synthetic metrics are excellent but real holdout is poor, that is a valuable result: it identifies a synthetic-to-real domain gap and tells us to improve the synthetic generator/data before granting AI any authority.

## What V2.13 does NOT do yet

- It does not modify live hit selection.
- It does not replace V1/V2.
- It does not claim full-frame recall.
- It does not use `shot_diag` as clean training data.
- It does not yet mine true full-frame V1/V2 false candidates, because historical raw full-frame candidate patches are not available in the discovered archive.
- It does not yet use before/after pixels in Hole-AI; temporal newness remains the V2.12 physical overlay's job.

## Next gate: V2.14

If V2.13 proves useful pixel learning, V2.14 should capture/compile **real candidate-centred false-positive patches** from V1/V2/overlay full-frame replay and retrain Hole-AI with those true hard negatives. Then Hole-AI can be evaluated as shadow evidence on the same candidate union used by the detector.

That is the step where we start measuring whether AI materially improves the actual hit-selection pipeline rather than only recognising hole patches.
