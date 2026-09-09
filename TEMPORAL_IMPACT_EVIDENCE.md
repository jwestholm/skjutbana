# Temporal impact evidence

This is offline, research-only analysis of the post-fix ten-shot physical
session. It does not alter live selection or historical validation.

The temporal extractor samples the same camera-space track location across the
recorded pre-history, pre-snapshot and post frames. For this first pass the
registration transform is held fixed at the saved local patch geometry; a new
full-frame registration estimate is intentionally not made for every frame.

## Result

For nine `nearest_eligible` examples and nine selected winners, the median
after-frame dark-residual persistence count was 0.5 for the physical candidates
and 0 for winners. Persistence had exploratory pairwise AUC 0.71; impact-onset
darkening AUC was 0.70. Before-frame dark stability was not useful (AUC 0.58),
and after-frame peak AUC was 0.64. These estimates are not independent
validation: the sample is tiny, includes one no-@42 nearest example, and the
same shots supply both classes.

The strongest physically interpretable signal so far is therefore:

```text
stable or localized onset near the event + persistence in subsequent frames
```

It is not sufficient as a live policy. Several true tracks have zero measured
persistence at their representative track coordinate, which may indicate
representative-coordinate drift or an imperfect temporal reference. That
uncertainty must be resolved before ranking research.

## False-winner observations

The seven wrong winners do not form one proven nuisance class. Most have a
transient or weak post darkening; shots 1, 2 and 6 have substantial peaks but
little or no persistence. Correct tracks in shots 4, 7 and 9 show the clearest
persistent residual. The remaining correct tracks are weak or transient,
which is evidence against a simple persistence threshold.

## Reproduction

```text
python3 -m automation.temporal_impact_research --root content/ai/physical_traces/session_20260908_194746_b38de674 --features evaluation_runs/registered_impact_20260909/physical_v2/physical_features.json --output evaluation_runs/registered_impact_20260909/temporal_v1
python3 -m automation.temporal_impact_selftest
```

No classifier or ranker is frozen. A larger physical diagnostic set remains
necessary before temporal evidence can be promoted.

## Observation-coordinate follow-up

The observation-level audit found only limited measurable drift in the saved
export: approximately 8–10 px in three examples, with most tracks exposing one
unique coordinate. Recomputing temporal features at those available
observation coordinates does not materially change the exploratory persistence
result. Confirmation-search best XY is not preserved for most tracks, so a
complete confirmation-coordinate comparison remains an observability gap.
