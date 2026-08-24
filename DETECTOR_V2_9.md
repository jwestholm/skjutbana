# Detector / AI V2.9 — offline ranking laboratory

## Why V2.9 exists

V2.8 established a much cleaner boundary between detection and ranking.

In the measured 100-shot V2.8 run:

- filtered oracle recall within 42 px: about 59%;
- micro-cluster oracle recall: about 58%;
- V2.8 recall-pool oracle recall: about 58%;
- selected correctly within 42 px: only about 2%.

The clustering/pool pipeline therefore preserved almost all available correct
neighbourhoods, while final ranking still chose the wrong hypothesis in the
large majority of shots where a correct neighbourhood existed.

That makes ranking the dominant remaining software problem.

## V2.9 design rule

**Do not change V2.8 detector/clustering/pooling while studying ranking.**

V2.9 wraps the already-completed V2.8 `rank_with_funnel` call and records the
constructed hypothesis pools after V2.8 has made its normal decision. It never
changes the returned result.

## Dataset

Every labelled synthetic shot writes one atomic JSON file:

`content/ai/ranking_v29/sessions/<session>/shot_XXXXXX.json`

The record contains:

- ground truth;
- all V2.8 micro-hypotheses;
- recall-pool/core membership;
- baseline, recall-baseline, V6 and actual ranks;
- candidate-to-GT distance (label only, never a runtime feature);
- normalized hypothesis features;
- within-shot relative feature percentiles;
- raw compact V2.x evidence fields;
- benchmark seed and detector metadata when available.

Camera X/Y are stored for labels/debugging but are deliberately excluded from
the V7 feature vector.

## Feature representation

V7 keeps the physical/evidence features from V6 and adds robust within-shot
relative features. This is important because projector/camera signal magnitude
can change between shots.

Relative values are inputs to a learned model, not an automatic rank. A model
is free to learn that a very strong `absdiff` percentile is either good or bad.

The representation also includes small interaction terms such as:

- support x current;
- support x source diversity;
- signal x patch;
- tile x current;
- agreement x current;
- fresh single-observation;
- low-signal current observation.

## Feature discrimination

`automation.ranker_v29_analyze` compares the nearest GT hypothesis with the
high-ranked wrong baseline hypothesis from the same shot.

It reports which features tend to be:

- higher for GT;
- lower for GT;
- approximately non-discriminative.

This gives us a direct answer to questions such as whether real holes are
actually weaker, fresher, smaller, less persistent, or less multi-detector
supported than projector artifacts.

## Offline experiment

`automation.ranker_v29_experiment` trains several linear pairwise variants.

Training labels:

- <= 12 px: strong positive, weight 1.00;
- <= 20 px: medium positive, weight 0.68;
- <= 42 px: weak neighbourhood positive, weight 0.24;
- negatives must be >= 55 px away from GT.

The weak 42-px label exists so a 100-shot dataset is not forced to discard most
shots, but it has deliberately low weight.

### Cross-validation

The split is by complete shot, never by individual candidate. All hypotheses
from one shot stay together.

This prevents candidate-level leakage, which would otherwise make the ranker
look much better than it really is.

Metrics are aggregated on held-out shots:

- Top-1;
- Top-3;
- Top-5;
- median GT rank;
- MRR;
- 20-px and 42-px conditional evaluation.

## Model variants

V2.9 compares multiple feature subsets rather than trusting one representation:

- all features;
- no baseline-score feature;
- primitive evidence;
- support-focused;
- signal-focused.

The best cross-validated model is then trained on the complete captured session
and saved as `content/ai/ranker_v7_offline.json`.

## Authority

V7 is **shadow-only in V2.9**.

Even if the offline report is excellent, V2.9 does not let V7 select the real
candidate. A later version may add an authority gate only after:

1. enough independent data exists;
2. held-out cross-validation clearly beats the V2.8 baseline;
3. a second live seed confirms the result;
4. the model improves Top-1/Top-3 without hiding detector recall failures.

## Development loop after V2.9

The intended loop is now:

1. one physical 100-shot collection run;
2. inspect dataset and feature separation;
3. run many offline experiments in seconds/minutes;
4. save best shadow model;
5. one different 100-shot live validation;
6. only then decide whether a 1000-shot collection run is worth the hour.

The expensive camera should generate data. It should no longer be required for
every change to ranker weights or feature subsets.
