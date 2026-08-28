# Detector V2.21.3 — Anchored Temporal Consensus + Target-Masked Direct Rescue

## Why V2.21.3 exists

The first honest V2.21 full-frame projector/camera session contains 30 rounds with:

- true recent PRE full frame,
- two POST full frames,
- exact synthetic/projected GT,
- current V1/V2 candidate packs,
- no learning during capture.

V2.21.2 showed:

- CURRENT oracle <=20 px: **26.67%**,
- CURRENT oracle <=42 px: **90.00%**,
- global AI_DIRECT oracle <=20/42 px: **3.33% / 3.33%**,
- LOCAL temporal oracle <=20 px: **53.33%**,
- CURRENT + LOCAL oracle <=20 px: **63.33%**,
- CURRENT + LOCAL oracle <=42 px: **93.33%**,
- LOCAL rescued **11/30** current misses at <=20 px,
- registration shift was essentially zero,
- median current <=42 offset was about dx=+0.9, dy=+0.5 with large MAD, so there is no useful single global calibration-offset fix.

The temporal maps do contain useful signal at GT:

- blackhat gain median GT percentile ~90%,
- top-hat gain ~87%,
- persistent absolute change ~83%,
- gradient gain ~78%.

The debug images also show the core failure mode: global maps are dominated by room/projector structure and broad horizontal bands, while useful hole evidence often exists locally near a current anchor.

## V2.21.3 strategy

### 1. Plateau-aware local maxima

V2.21.2 selected a small number of local-max *pixels* per source. Saturated lines/plateaus can contain many equal-valued pixels, so arbitrary points can win.

V2.21.3 instead:

1. thresholds a local temporal ROI,
2. finds local maxima,
3. collapses connected equal-valued plateaus/components,
4. rejects very large/elongated components,
5. derives a stable representative coordinate from the compact component.

### 2. Multi-source consensus

Local components from these maps are clustered spatially:

- `blackhat_gain`,
- `tophat_gain`,
- `persistent_abs`,
- `gradient_gain`,
- `persistent_dark`.

A point supported by several independent temporal sources is preferred over a one-map spike.

### 3. Candidate-derived target mask for global rescue

The first V2.21 global direct proposal run searched the entire 4K camera image. The room/cabinet edges therefore dominated the global top-N list.

V2.21.3 creates a conservative search mask from the *current candidate cloud only*:

- robust 1–99% candidate core,
- convex hull,
- configurable dilation margin.

This uses no GT. It is deliberately broad enough to allow a current miss to be rescued, while preventing unrelated room edges from owning the global proposal list.

### 4. Development-only profile selection

A small fixed profile sweep is evaluated on DEVELOPMENT only. The winning profile is frozen before confirmation/holdout metrics are reported.

The profile objective is lexicographic:

1. oracle <=20 px,
2. oracle <=10 px,
3. oracle <=42 px,
4. fewer candidates.

Confirmation and holdout are not used to select the profile.

## Important limitations

- The dataset still has only two physical/projector sessions, so the split remains provisional.
- The new full-frame session contributes only 30 direct-ready rounds; confirmation and holdout each contain only 6 full-frame rounds.
- V2.21.3 is offline/shadow-only.
- No live candidate order, live hit coordinate or game authority is changed.
- A better oracle is not enough; a later ranker/fusion step must still pick the correct candidate reliably.

## What result would be useful

The primary next question is whether evidence-backed consensus can move the candidate oracle materially above V2.21.2's 63.33% <=20 px without exploding candidate count.

A useful outcome is:

- DEVELOPMENT improves over V2.21.2,
- protected confirmation/holdout do not collapse,
- target-masked direct rescue adds at least some genuinely new <=20 px coverage,
- candidate count stays in a range that can later be ranked.

If this still stalls below roughly 70% <=20 px, the next step should be a learned physical-domain dense evidence model using the honest PRE/POST full-frame captures, not more synthetic-only V2.18 ranking.
