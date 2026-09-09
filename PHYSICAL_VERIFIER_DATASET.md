# Unified physical verifier dataset

The builder `automation/physical_patch_dataset.py` creates a compact offline
dataset from all trace sessions with a manual camera GT and readable
pre/post arrays. It stores PRE/POST-derived vectors, session/event identity,
coordinate origin, timestamps, label origin, and GT distance only as metadata.
GT is never passed to the verifier representation.

Current inventory:

- 5 patch-compatible sessions
- 48 labelled shots, 47 usable
- 47 positive examples
- 39 conservative selected-false negatives
- 86 total examples
- 2 sessions with complete eligible-pool replay

Negative examples are only recorded selected outcomes more than 42 px from the
manual GT. Top-5/top-20 hard negatives require complete pool exports and are
therefore available only for the two post-fix sessions; they are not invented
for legacy traces.

Session-held-out centroid experiments over the unified set show domain shift:
texture vectors are the strongest of the tested representations, but positive
ranks vary widely by held-out session. This is useful research evidence, not a
promotion result. The latest ten is not statistically large enough to train a
reliable verifier alone.

Future collection should target 30–50 additional labelled shots over at least
three sessions, balanced across low/high contrast, lines, dark regions,
old-hole proximity, nearby grouping and no-impact audio events.

The recommended future split is development on two sessions and one untouched
validation session. The current historical set is not sufficient to designate
an untouched validation session because only two sessions have complete
full-pool semantics and their visual domains differ.
