# Research common-verifier architecture

## 2026-09-10 evidence-preservation results

Existing track audit already retains actual associated observations. The bounded
OBSERVATIONS32 experiment preserves each eligible representative and adds at
most three observed alternatives per parent, 32 per event, with producer/cutoff
checks and separately recomputed offline readiness. It changes coordinates only
in an offline pool, never a live track or the meaning of local confirmation.
This restores D01 event 1's 2.892 px evidence, but common ranking puts it 12th.
Event 2's protected CURRENT hit is lost by all tested common variants. Event 4
can be recovered, while event 6 still loses. No new Track/BoardState class is
needed to study these alternatives.

Source-independent compact/darkening confirmation rankings both select 0/6 D01.
Centered PRE variability increases D01 GT retention from 40.06% to 55.84% but
changes none of its selected coordinates and regresses S01. All four existing
S01/S02 no-physical events are still accepted, including by explicit local
confirmation. Physical ranking needs joint causal evidence and a defensible
event-level NO_IMPACT alternative; a large local scalar is insufficient proof.

The new drivers are `automation/evidence_retention_research.py` and
`automation/evidence_channel_research.py`, using immutable output directories,
source hashes, existing primary fits and whole-session exclusions. D01/S02 never
enter fitting. They are offline experiments at recorded cutoffs, not new physical
validation or new canonical SHADOW models. Current authority/model status/hash
remain unchanged. Full errors/counts/runtime, exact hybrid-loss ledger and
comparator regressions: [D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md).

## D01 and clean-change evidence

The frozen early-reference common model selects 1/6 D01 at @5/10/20/42, recovering
event 4 while losing CURRENT's event 2. The frozen snapshot model selects 0/6.
These failures persist despite five existing @42-positive eligible pools.

An offline clean channel reuses `EvidenceContext`; local background, persistence,
PRE variance, crop-half motion and broad-flow variants are evaluated separately.
Fifteen source-independent patch map/noise/edge features are added only in a
separate logistic research comparison, fitted on historical primary sessions
with whole-session exclusions. S02/D01 do not enter fitting. No GT point is
injected into selection. Pure edge features recover no D01 shots; noise features
alone recover one. Actual camera structure differs from an exact projected edge
map, which cannot be reconstructed without missing capture calibration.

Best observed D01 selection is flow/augmented union 2/6, with 6/6 @20/@42 oracle.
It severely attenuates all six GT neighborhoods and regresses POST_FIX/S02 while
continuing to accept known false events. It is not a frozen promotion candidate.
The existing canonical challenger and CURRENT authority are unchanged.

Board/reference/hole/audio and coordinate ownership were inventoried before any
central-class proposal. See [PHYSICAL_BOARD_STATE_ARCHITECTURE.md](PHYSICAL_BOARD_STATE_ARCHITECTURE.md)
and [D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md) for evidence, runtime,
input hashes, comparator results and the next discriminating capture.

## Previous measured implementation — 2026-09-09

The implemented causal research path is now
`src/engine/offline/accuracy_verifier.py`, with explicit session extraction,
supervised logistic/forest experiments, pairwise ranking and visual no-impact
gates under `automation/accuracy_*`. It combines registered PRE/POST change,
onset, persistence, polarity, multiscale morphology, local normalization and
jitter stability. Every feature has a timestamp/coordinate/normalization/live
availability contract. Source identity, source scores, GT and absolute XY are
excluded from feature inputs. The 0/1 output can abstain.

Whole-session exclusions yield common-logistic primary @42 11/49 versus CURRENT
7/49. The earlier-reference variant selects 5/9 S01 and 3/10 S02 at all four
thresholds; S01 uses a held-out fit, while the S02 fit trains on all primary
sessions. Two hashed offline references are frozen, with S02 explicitly
development-used. They are not integrated with CURRENT or the canonical shadow.

Proposal expansion reaches S02 9/10 @42 oracle but only 3/10 selection. Visual
no-impact rejection trades false emissions for unacceptable physical rejection.
Classifier inference is inexpensive; the current Python evidence extraction is
too slow for demonstrated live parity. This is captured-time, single-decision
research, not full asynchronous emission replay. Further data needs concern
physical onset/newness and defensible negatives, not just more oracle candidates.

`physical_event_truth.py` uses an impact array and explicit SINGLE_IMPACT,
NO_PHYSICAL_SHOT, AMBIGUOUS, UNKNOWN and future MULTI_IMPACT states. Unconfirmed
S02 event-1 ambiguity is a caveat plus exclusion sensitivity, not a label edit.
The single-output evaluator excludes unresolved/multiple-impact truth explicitly.

The 100% target, 95% minimum, all positive/negative experiments, feature contract,
per-session measurements and frozen hashes are in
[ACCURACY_95_100_RESEARCH.md](ACCURACY_95_100_RESEARCH.md). S03 remains untouched.
The material below records the earlier research foundation.

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

## S01/S02 development evidence — 2026-09-09

S02 has ten physical shots and three human-confirmed nonphysical events, all
with complete causal decisions. The verified funnel is 5/10 @42 causal
candidates, all five tracked/locally confirmed/eligible, but only 1/10 correctly
selected. Four V1 truths lose to FAST at tied onset distance; all three
nonphysical events emit false hits. Both frozen shadows score 0/10 @42.
The verifier needs an explicit physical rejection capability in addition to
ordering eligible tracks. A ranking-only verifier cannot recover the five
events without @42 candidates in the existing pool.

Using the existing interface with each recorded decision cutoff produces ten
S02 positive patch examples, nine wrong-emission negatives and three no-impact
negatives. S01 adds nine causal positives and seven negatives, including one
no-impact example. S01's tenth physical label has usable images but no complete
terminal cutoff, so it is excluded from causal verifier examples. These counts
verify extraction compatibility; they do not measure learned separability or
generalization. The generic historical patch builder uses later frames and
must not be mistaken for this causal path.

The next experiment should score the complete eligible pool through shared
causal image evidence, with historical/S01 training and whole-session S02
development assessment. Include physical hard negatives and no-impact events;
keep source labels and GT out of verifier inputs. Report candidate availability,
conditional ranking, readiness/emissions and no-impact acceptance separately.
The existing interface does not by itself implement registration or an
abstention policy; those research choices must be explicit and frozen before
final evaluation. Do not tune proposal generation in the same experiment.

S03 remains untouched final validation data. No common verifier, canonical
challenger or frozen confirmation selector is promoted; CURRENT live authority
is unchanged. Exact results and reproduction are in
[S02_PHYSICAL_FINDINGS.md](S02_PHYSICAL_FINDINGS.md).
