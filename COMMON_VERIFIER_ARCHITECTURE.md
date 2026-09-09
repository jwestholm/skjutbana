# Research common-verifier architecture

The research interface is implemented in
`src/engine/offline/common_verifier.py`. It is disabled by default and is not
called by the live selector. `CommonFrameContext` carries shared PRE/POST
frames, timestamps, a causal cutoff and camera origin. `extract_patch_context`
provides consistent PRE, POST, signed and absolute patches; the representation
helpers preserve raw difference, gradient, low-resolution PCA-like structure,
and compact texture statistics.

The offline path is:

```text
V1/V2/FAST/vault/bank/rescue proposals
  -> complete tracking and eligibility
  -> CommonFrameContext + source-independent patch extraction
  -> research verifier / rank replay
```

Ground truth is used only by the dataset builder to label positive and hard
negative examples. It never enters feature extraction or verifier inputs.

## Rich patch experiments

Leave-one-shot-out nearest-positive-centroid information tests on the latest
physical session produced:

| representation | @1 | @3 | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|---:|---:|
| raw signed difference | 0 | 0 | 0 | 0 | 1 | 2 |
| gradient magnitude | 0 | 0 | 1 | 1 | 2 | 2 |
| 8×8 low-resolution patch | 0 | 0 | 0 | 1 | 1 | 4 |
| texture statistics | 0 | 2 | 2 | 3 | 5 | 8 |

These are information tests, not trained production models. Rich spatial
structure does not yet generalize from nine positives and the available hard
negatives. Texture statistics are the least bad but remain weaker than a safe
final selector and are not physically validated.

## Decision

The architecture is worth keeping, but current data supports decision **B**:
the common-verifier architecture is right while the physical training set is
insufficient for a reliable rich verifier. The next physical dataset should
provide at least 30–50 labelled shots across contrast, lines, old holes,
nearby impacts and no-impact audio events, with complete causal frame context.

## Historical dataset extension

The compatibility inventory found five patch-compatible sessions and 47 usable
positive examples. Legacy sessions lack complete selector snapshots but are
valuable for patch classification. Unified session-held-out experiments show
texture representations are strongest yet unstable across backgrounds; richer
real data is still required before freezing a verifier.
