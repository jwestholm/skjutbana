# Registered impact evidence research

## Status

This is offline development analysis. It does not change live authority, the
frozen confirmation shadow, candidate generation, or historical physical
metrics.

## Physical evidence

The latest post-fix session (`session_20260908_194746_b38de674`) has nine
causally available candidates within 42 px, all nine reaching eligible tracks,
and two correct final selections. Exact saved PRE/current frames reproduce the
recorded local-confirmation values. Applying the same V2 registration and
robust ring photometric reference offline gives registration shifts below
0.35 px for the physical examples; registration therefore does not explain
the seven ranking failures in this session.

Registered absolute residual, signed darkening, center-vs-ring contrast,
compactness, concentration, entropy and connected-component features have
substantial overlap between the nine physical true tracks and the nine chosen
false winners. The best simple one-feature separations are weak (compactness
about AUC 0.61; darkening and concentration about 0.58–0.60). Physical image
evidence does not justify a new ranker yet.

The two correct V1 examples have strong localized residuals, but several wrong
FAST examples also have strong residuals. Conversely, some true tracks have
weak immediate residuals. This means the saved representative track point,
the exact PRE timing, or scene-specific nuisance structure can matter as much
as the scalar confirmation fields.

## Synthetic domain gap

The same registered features separate the existing synthetic true/false
examples much more strongly: true center-darkening and compactness medians are
approximately 28.6 and 56.4, versus 2.6 and 2.6 for synthetic false examples;
true concentration is about 0.92 versus 0.20. This large separation is absent
in the physical data. Existing synthetic success must therefore not be used as
evidence that a registered residual ranker will work physically.

The controlled reference intervention that replaces only the confirmation PRE
is not yet a valid frozen experiment: its proposal-input equality assertion
exposed nondeterminism in the current synthetic runner. It remains a tooling
follow-up, not a detector result.

## Interpretation

**Proven:** PRE coordinate mapping and event ownership fixes are in place;
FAST saturation disappeared; candidate generation and track survival are not
the remaining physical bottleneck; current confirmation measures local image
change, not proof that a new hole was caused by the current shot.

**Strong hypothesis:** final ranking needs registered, localized, polarity-aware
new-impact evidence, and low-contrast nuisance changes are a major failure
family.

**Not proven:** that registration, photometric normalization, any single
threshold, or FAST removal improves physical accuracy. No research ranker is
frozen or promoted.

## Reproducibility

```text
python3 -m automation.registered_impact_research --root content/ai/physical_traces/session_20260908_194746_b38de674 --output evaluation_runs/registered_impact_20260909/physical_v2
python3 -m automation.registered_synthetic_forensics --root content/ai/physical_traces/session_20260908_194746_b38de674 --runs evaluation_runs/overnight_tracks_20260908/development evaluation_runs/overnight_tracks_20260908/holdout evaluation_runs/overnight_tracks_20260908/stress_development_510 --output evaluation_runs/registered_impact_20260909/synthetic_v1
python3 -m automation.registered_impact_selftest
```

## Temporal follow-up

A temporal forensic pass now records before stability, event onset, post
persistence and residual peaks. Persistence and onset are more promising than
single-frame residuals on the small physical sample, but the evidence is not
strong enough to define a ranker and does not resolve representative-coordinate
or PRE-reference uncertainty.
