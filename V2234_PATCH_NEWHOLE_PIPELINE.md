# V2.23.4 — Patch NewHole Candidate Model

## Why this version exists

The V2.23.3 bootstrap gave an unambiguous result: the dense proposal pool still contained a <=20 px candidate in about 92% of validation shots, but the best tabular reducer retained only about 34.8% of those positives at Top512 and placed the first positive at median rank 939. The final ranker therefore received the correct location in too few shots to be the main bottleneck.

This is the pre-agreed decision point for changing model class instead of adding more scalar heuristics.

## Task decomposition

V2.23.4 keeps proposal and recognition separate:

```text
PRE/POST framepack
      ↓
V2.21.5 dense proposals (~9k–10k)
      ↓
candidate-centred image patches
      ↓
Patch NewHole model
      ↓
Top512 retention pool
      ↓
final listwise ranker
```

The image model answers:

> Did a compact, persistent, hole-like physical change appear at this candidate for this shot?

It does not answer whether a static patch merely resembles a hole. That V2.17 semantic distinction remains intact.

## Patch representation

Each dense candidate gets a 32×32 physical crop average-pooled to 16×16 with five channels:

1. PRE grayscale.
2. Mean POST grayscale across up to three post-shot frames.
3. Mean absolute PRE→POST change amplified ×8.
4. Mean signed PRE→POST change encoded around 128 and amplified ×4.
5. Fraction of POST frames whose local pixel changed by at least 2.5 gray levels.

Patches are stored as uint8 in compressed per-shot NPZ banks. Training converts them to bounded float inputs.

## Models

V2.23.4 compares:

- `patch_mlp`: dependency-free image baseline over the complete 5×16×16 patch.
- `tiny_cnn`: learned 3×3 convolution → ReLU → 2×2 average pooling → hidden layer → candidate score.

Both are trained with pairwise NEW-hole ranking loss inside each shot. The loss compares positive NEW-hole patches against hard and random NOT-NEW candidates from the same shot.

## Training-only GT anchors

The candidate pool is still judged honestly. If a proposal candidate is <=20 px from GT it is an ordinary positive. In addition, V2.23.4 extracts a GT-centred patch and four ±4 px jittered patches as training-only positives.

This uses GT exactly as supervised learning should: as a label source. Those anchors are never added to proposal candidates or evaluation pools. Therefore they cannot improve oracle recall by construction.

This also lets the image learner study the real NEW-hole appearance on the small minority of shots where dense proposal misses <=20 entirely.

## Hard negatives

For each training shot, V2.23.4 keeps a bounded set of >42 px negatives containing:

- high dense-score candidates,
- high V2.23.3 NEW-hole-heuristic candidates,
- random physical background candidates.

20–42 px remains neutral to avoid teaching the model that near-localisation errors are definitely NOT-NEW.

## Bootstrap gate

With one substantial dense F2 session, the target is a qualitative jump over V2.23.3:

- validation retention20@512 >= 0.80,
- validation retention20@128 >= 0.60,
- validation median positive rank20 <= 100.

This is a learnability gate only. Passing it does not create a research champion and never changes live authority.

## Fresh-domain gate

With >=2 substantial sessions, the newest F2 session is untouched during fitting and trial selection.

Research gate:

- fresh-domain retention20@512 >= 0.85,
- fresh-domain retention20@128 >= 0.55,
- fresh-domain median positive rank20 <= 150,
- final ranker conditional Top1@20 >= 0.10.

The fresh-domain session is evaluated only after the winning patch-model and final-ranker trials are selected from engineering validation.

## Authority

V2.23.4 remains `shadow-only` and stores `live_authority=false`. No V2.23.4 model can alter the current live hit coordinate.
