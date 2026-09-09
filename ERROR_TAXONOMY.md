# Physical ranking error taxonomy

The latest saved imagery supports only conservative labels:

- **Detector-score/provenance mismatch:** all seven failed tracks are proven
  to lose on raw `best_score` after eligibility.
- **Low or ambiguous physical residual:** several true tracks have weak scalar
  compact/dark/persistence evidence at their representative coordinate.
- **Persistent nuisance:** selected false winners can remain visible in later
  frames, so persistence alone does not identify impact.
- **Unclassified morphology:** the present traces do not preserve enough
  semantic patch labels to prove line, tape, old-hole, illumination, or
  texture identity for each failure.

No stronger per-shot visual taxonomy is claimed without direct image-level
causal labels. Future verifier work should preserve patch provenance and use
these categories as hard-negative metadata only when visually established.
