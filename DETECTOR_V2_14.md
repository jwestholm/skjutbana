# Detector V2.14 — Background-Generalising Hole-AI

## Why V2.14 exists

V2.13 answered the first important learning question: **yes, a small pixel model can learn useful hole structure from the existing `synt_*` bank and transfer to real `hole_*` patches.**

Observed full V2.13 run on the shooting PC:

| Evaluation | V2.13 |
|---|---:|
| validation AUC | 0.874595 |
| synthetic-test AUC | 0.837996 |
| strict novel-background AUC | **0.529693** |
| real-hole AUC | **0.924489** |
| real-hole recall | **0.918919 (34/37)** |
| off-centre AUC | 0.800802 |

That is a clear KEEP for pixel learning, but it exposes the next bottleneck: **the model still depends too strongly on the background domain**.

V2.14 therefore does **not** touch live hit authority. It attacks background generalisation offline first.

## What changed

### 1. Background-invariant model input

V2.13 used normalized grayscale plus a local high-pass channel. V2.14 intentionally removes the absolute-intensity channel and feeds four local physical maps:

1. **local residual** — pixel minus broad Gaussian background,
2. **DoG** — compact small-scale structure,
3. **morphological black-hat** — compact dark structure relative to local surroundings,
4. **gradient energy** — edge/rim evidence.

All channels are locally/robustly normalized. A large global brightness change should therefore affect the representation much less than in V2.13.

### 2. Background remix / domain randomization

During training only, candidate crops may be decomposed into:

```text
candidate patch = slow background + compact/local residual
```

The slow background is then partially replaced by a procedural background while the compact residual is retained. Additional variation includes:

- brightness,
- contrast,
- gamma,
- blur,
- camera noise,
- smooth shadow/illumination gradients,
- projected lines/rectangles,
- procedural flat/gradient/checker/grid/bubble/stripe/multiscale textures.

The objective is **not** photorealistic game rendering. The objective is to make background identity an unreliable shortcut while preserving local hole evidence.

### 3. Model selection cannot look at the strict holdouts

This is critical.

V2.14 still reserves:

- `black`
- `checker`
- `gray`
- `bubbles`

as the same strict novel-background holdout used by V2.13. The real `hole_*` set also remains fully held out.

The best epoch is selected using only:

```text
clean validation sessions
+
procedurally transformed validation sessions
```

The selection score is the geometric mean of clean-validation AUC and procedural-domain-stress AUC. This prevents a clean white-domain score from hiding poor robustness without contaminating the real or strict novel-background tests.

### 4. Centre-shortcut protection remains

The original `synt_*` source image has the hole in the centre. V2.14 still never treats that source centre as the model's semantic candidate location.

- positive candidate centres are jittered,
- negative candidate centres use the same source/preprocessing but sit away from GT,
- the auxiliary head predicts the X/Y offset from candidate centre to the real hole,
- off-centre stress remains an explicit evaluation.

### 5. Automatic comparison with V2.13

If this exists:

```text
content/ai/reports/v213/hole_v213_report.json
```

V2.14 writes direct AUC/recall deltas into its report.

## Profiles

Three training profiles exist:

- `mild`
- `standard` (default)
- `strong`

Start with `standard`. If the strict novel-background result is still poor, use the sweep tool. The sweep ranks profiles using **non-holdout validation only**; strict novel/real values are printed for information but are not allowed to choose the winner.

## First gate

V2.14 is worth taking into the next candidate-shadow integration step if all are true:

- strict novel-background AUC >= **0.70**,
- real-hole recall >= **0.85**,
- off-centre AUC >= **0.78**.

A pass means only:

> Hole-AI has become robust enough to test as an additional candidate evidence source.

It does **not** mean the full shooting system is at the >=95% final-hit goal.

## Commands

Selftest:

```bash
python3 -m automation.hole_v214_selftest
```

Full standard training:

```bash
python3 -m automation.hole_v214_train
```

Quick smoke run:

```bash
python3 -m automation.hole_v214_train --epochs 3 --max-train-assets 2000 --max-eval-assets 500
```

Visualise what domain randomization and the new feature channels do:

```bash
python3 -m automation.hole_v214_visualize --kind synthetic --index 0
```

If needed, run the profile sweep:

```bash
python3 -m automation.hole_v214_sweep
```

Outputs live below `content/ai/reports/` and remain offline-only.
