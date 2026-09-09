# Physical patch verifier information test

The offline verifier uses five source-independent registered patch features
(compactness, dark contrast, center darkening, concentration, entropy). For
each held-out shot, it computes the standardized distance to the positive
centroid learned from the other shots. No source, raw detector score, rank, or
ground-truth coordinate enters the feature vector.

The latest ten-shot result places the physical true track in the top ten for
only **1/10** held-out events. This is a deliberately small information test,
not a classifier claim: nine physical positives are insufficient for reliable
learning, and no model blob is stored. It shows that the current handcrafted
features do not contain a robust absolute physical-impact identity signal.

The next viable verifier requires either richer patch representation and
strict leave-session-out validation or additional labelled physical sessions.
Synthetic data may augment training but cannot substitute for physical hard
negatives because the domain gap is proven.

## Rich patch follow-up

Four source-independent representations were evaluated with leave-one-shot-out
nearest-positive-centroid replay. Raw difference, gradient and low-resolution
spatial representations recover at most 2–4 truths at @50. Compact texture
statistics recover 8/10 at @50, 5/10 at @20 and 3/10 at @10, but this remains
far below a promotion standard and is derived from only nine positives. No
representation is frozen.
