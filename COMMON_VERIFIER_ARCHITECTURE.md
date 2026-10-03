# Research common-verifier architecture

## Batched shared query, same witness contract — 2026-09-28

`query_physical_witnesses` extends the existing common query with T×C signals,
one frame prefix and ordered candidate identity/provenance records. Bounds remain
T≤50 and C≤32. It validates shared prefix metadata once, applies the existing
batched temporal kernel, and invokes the same per-coordinate identity checks and
EvidenceBundle construction/revocation as the scalar interface. Bundles remain
detached: modifying one candidate's metadata cannot alter another's proof.

`CommonTemporalHistory.witness_queries` adapts the same frame owner. Existing
camera-local patch sampling and `local_contrast_batch` supply the same exact
signal values, with masks and reductions shared across coordinates. The replay
already maintains append-only signals; batching backfills only newly observed
coordinates. There is no new cache, ring, Candidate class or temporal engine.

Clocked and ordinal-only evidence share the unchanged amplitude arithmetic;
ordinal mode never receives invented timestamps. State names/cutoffs, exact XY,
frame/version identity, persistence/revocation and correlation semantics remain
unchanged. No producer evidence becomes an independent vote. All 37,449 measured
prefix bundles across 45 development events match with zero numerical tolerance;
upstream producer score drift is documented separately from this contract.

Measured shared-query speedup is 9.43x KOMP / 7.42x D02; complete core transition
mean/p95 remains 50.25/59.23 ms and 154.66/170.49 ms. Proposal generation dominates.
One-candidate clocked microbenchmarks can be slightly slower through batching;
no adaptive policy or semantic shortcut was added. CURRENT and live ownership
are unchanged. Before a shadow hook, retain camera identities through main-thread
grayscale ingestion, establish immutable worker ownership and explicit bounded
backpressure, and prove pre-trigger search/event association. Current single-worker
CV dispatch only schedules pending audio events; it is not already continuous
pre-trigger processing. D02 throughput remains insufficient.

## Shared witness contract implemented — 2026-09-28

This supersedes the implementation status of the design below, not its safety
limits. Existing `EvidenceBundle` gains an optional `witnesses` tuple; a
candidate-only bundle has no reference/fused map. Existing image constructors
remain compatible. `common_verifier.query_physical_witness` is a pure bounded
query (maximum 50 observed samples). `CommonTemporalHistory.witness_query`
adapts its existing clocked frame history. No new central class or runtime hook.

Each record contains exact camera XY, coordinate version, candidate/event key,
producer/reference provenance, frame IDs/ordinals/available camera sequences,
raw signal and frozen temporal measurements, receipt cutoff or explicit unknown
timing, proposition/family/dependency family, strength units, state and revocation.
PRE/onset indices refer into that explicit frame list. Ordinals are not sequences
or times. Exposure and producer-delivery time stay UNKNOWN when unrecorded.
Previous support can only be reused on the identical coordinate/version and an
unchanged prefix of frame metadata and measurements; missing/evicted history is
unavailable, never silently shortened. Owners retire records explicitly.

The named proposition is a unique persistent local step. `SUPPORTED` means its
frozen criterion passes; `CONTRADICTED` marks observed persistence reversion;
ambiguous/absent support is `UNKNOWN`. A later UNKNOWN retains the last revocation
reason. Physical hit/no-hit truth remains UNKNOWN in this measurement bundle.
D02:4 reproduces SUPPORTED then CONTRADICTED at ordinal 15, not permanent HIT.
`observation_hypotheses.retain_exact_physical_witnesses` contains the measured
24-retained/32-competing rule; pending proposals still deduplicate normally.

Minimum family mapping (only TEMPORAL_NEWNESS is newly attached by this query):

| Family | Existing measurements/owner | Interpretation |
|---|---|---|
| TEMPORAL_NEWNESS | clean_physical_change; chronological signal, onset, persistence/revocation | One same-XY temporal channel; criterion/support units, not probability. |
| LOCAL_PHYSICAL_CHANGE | accuracy_verifier / registered_impact / V252; registered residual, polarity, compact local response | Preserve reference, registration and sampled XY; share raw dependency identity with temporal evidence where applicable. |
| SPATIAL_SUPPORT | same_frame_support / source provenance / descriptive spatial experiment | Raw counts and unique correlated source families separately; no independent-vote claim. |
| SURFACE_CONTEXT | known-hole/history and recorded surface metadata | Holes/repairs are context; absent repair labels remain unknown, never vetoes. |
| MOTION_CONTEXT | existing registration shifts/residuals and recorded regional context | Soft context, with frame/reference identity; not proof of no impact. |
| STRUCTURE_CONTEXT | existing edge/ridge cleanup and local edge response | Context only; real holes can occur on printed lines. |

No speculative family extractor, fusion rule, cluster/edge/motion veto or larger
feature bank was added. Five correlated measurements do not constitute five
proofs. The spatial experiment remains inconclusive. The old EvidenceBundle
weighted overlay is unchanged and must not reinterpret witness states as votes.

All 45 prior reconstructions / 37,449 candidate prefixes reproduce exactly.
This exactness covers temporal measurements, coordinate/provenance order,
retention and decisions. Six regenerated producer proposals have saliency/score
drift up to 7.63e-6 / 1.22e-6, recorded separately; full producer dictionaries are
not bit-exact and the floating operation responsible was not isolated here.
Smallest next computational extension: reuse batched temporal arithmetic. The
chronological replay already appends one signal value per retained coordinate/new
frame and backfills bounded history once on candidate birth; preserve that work
instead of introducing another cache. Keep camera/scanner grayscale ownership and
metadata together; do not create separate OBJECT/HitScanner/verifier histories.
Current scanner history loses some camera identity fields, so provenance must be
delivered from the camera owner rather than inferred. No live scheduling claim:
D02 proposal work alone exceeds the approximately 80 ms observed frame interval.

Real-hole V3 is a separate supervised dataset path, not an inference family.
68 source events yield 23 qualified research pairs and 45 partial; extraction
uses labelled GT only for dataset patches. Hash-bound visual review is required
after deterministic checks. Donor/background session splitting precedes all
augmentation; static/new-damage/final-selection labels remain distinct. The small
builder exists, but composition realism and a target-specific training adapter
remain unqualified. No unattended trainer or new physical validation exists.

## Exact witnesses and a general physical evidence bundle — 2026-09-27

This section is an inventory/design decision, not a new runtime subsystem.
CURRENT and the canonical SHADOW model/status/hashes remain unchanged. The bounded
offline exact-witness experiment improves KOMP selected @5 from 3/6 to 4/6 with no
retention-induced tolerance regression across 45 paired development events.
No candidate XY, region cap or confirmation cap is changed in production.

### Existing owners to extend

| Existing component | Ownership and limits |
|---|---|
| `offline/evidence.EvidenceBundle` / `EvidenceOverlay` | Shared reference, physical maps and metadata already exist. Its weighted fused overlay is not independent corroboration or calibrated truth. |
| `accuracy_verifier.EvidenceContext` | Registered PRE/history/POST, ROI, origin, clocks and registration. Receipt < peak is a software ordering constraint, not proof of clean physical PRE. |
| `CommonFrameContext` / `CommonTemporalHistory` | Shared bounded raw history, identities/cutoff and source-independent same-XY queries. Existing temporal history requires finite receipt clocks; do not feed KOMP ordinals as timestamps. |
| `clean_physical_change` | Frozen disc/annulus onset, persistence, polarity/noise. Several outputs describe ONE repeated physical channel. |
| `RegisteredFrameV252`, `RegisteredMetricsV252`, V2225 confirmation | Reusable registered maps and local confirmation; neighborhood best-offset support must retain the sampled location instead of silently becoming exact-center proof. |
| `shot_track_v2226`, `observation_hypotheses` | Frame-unique observations and bounded exact-XY first/latest/confirmed witnesses. Same-frame support is producer agreement, not extra temporal observations. The offline helper already rejects future/foreign ownership. |
| `analysis_filters_v2222`, V2 edge weights, `registered_impact` | Ridge cleanup, soft edge context, residual component circularity/elongation already exist. No duplicate physical feature bank is needed. |
| V253 history, known holes, frozen HitRegions | Prior context/search ownership; never automatic vetoes or physical truth. |

Fragmentation is primarily delivery and identity: candidate dictionaries, track
representatives, confirmation offsets, map caches and offline contexts carry
parts of the same proof with different provenance. Changing representative XY
must not transfer onset or confirmation samples from another point. Repeated
shared debug maps lack producer identity and cannot fill this gap. General
registration, region motion and causal local support also have different input
references; identical names do not establish identical evidence semantics.

The smallest future consolidation is an optional **candidate-local result record**
from the existing common query, with references to the existing frame owner and
bounded observed-hypothesis slots. Use EvidenceBundle metadata/overlays and common
query results rather than a new GeneralVerifier class, another image ring or a
parallel BoardState. No global dense maps or producer counts become extra votes.

Proposed record contract (not implemented this pass):

- key: event/stream, exact full-camera XY and coordinate-version identity;
- proposition and family, `SUPPORTED / CONTRADICTED / UNKNOWN`, raw values and
  strength with units/criterion version (not an invented probability);
- actual PRE/onset/POST frame IDs, camera sequences when recorded, read/receipt
  time and availability/consumption bounds, causal cutoff and unknown fields;
- producer/registration/reference identities and shared dependency-family IDs;
- invalidation/revocation reason and last evaluated prefix; game context separate.

`SUPPORTED` means the named physical observation passed its criterion, not that
an impact has been proven. A later reversion can CONTRADICT that persistent-step
claim. Missing history, multiple onsets and unsupported evidence remain UNKNOWN;
none proves NO_PHYSICAL_HIT. D02:4 and S02:10 show why support must be revocable.
Exact supported coordinates may occupy multiple nearby slots, but still compete
within the same 24-coordinate bound. Pending candidates can deduplicate normally.

### Real-Hole Synthetic V3: source-first design, not a training launch

The explicit source manifest verifies 68 manually labelled impact events with
extractable PRE/POST/GT, including 60 across H10/H20/POST_FIX/S01/S02 and eight
new D02/KOMP impacts. Of these, 61 have usable recorded causal POST cutoffs;
KOMP's six and S01:11 do not. Current count supersedes the historical 47-example
compatibility lower bound, not its historical measurements. Extractability does
not certify clean PRE: selected PRE contamination requires donor qualification.
CAN's reviewed regions lack native exact GT labels and are not added to this count.

`content/ai/holes` contains 63 distinct static real-patch images across five date
labels, but only 24 distinct stored date/run/round keys and no paired PRE. Do not
add these as 63 newly verified independent temporal impacts. The 15,575 synthetic
PNGs and 101 projected framepacks are not real projectile donors. Existing V214
already remixes physical residuals with backgrounds, brightness/noise/blur and
projected edges. Existing V213 dataset code splits whole sessions, but currently
puts all real assets in holdout: explicit source-hole/session assignment must be
extended before generating V3 descendants, not bypassed afterward.

Keep three targets separate: static hole appearance (old holes can be positive),
new persistent damage (old unchanged holes negative/controls), and within-event
selection (actual observed candidates plus event-level UNCERTAIN). Wrong selected
coordinates are not automatically static non-holes. Synthetic performance is
never physical validation. Donor and background sessions must be assigned before
augmentation, with all descendants and coordinate jitters kept in that split.

Next data work is a small manifest-driven donor qualification pass: retain actual
clean-before/settled-after frame identities, inspect registration/illumination
residual contamination, store raw patches and damage mask/delta separately, and
quarantine ambiguous/multiple-damage labels. Apply existing augmentation to the
damage layer, not to an entire textured source patch. Bound geometry changes by
measured donor scales; preserve old holes and background state through temporal
compositions. Negative layers can reuse recorded no-damage controls and existing
noise/edge/motion operations, with labels tied to physical damage rather than a
visual residual. Do not fit a ranker or fabricate a clean mask from a dirty PRE.

Reuse `automation.ai_training_loop`'s bounded-run/seed/result pattern for a future
explicit offline mode; do not drive F2 projection or model activation for V3.
There is currently **no implemented V3 unattended command**. The focused report
specifies the proposed future CLI, split/manifest gates and separate metrics.
No synthesis, fitting, model registration or large run was started here.

Full evidence: `/data/skjutbana/evaluation_runs/exact_witness_20260927_01/REPORT.md`.

## Delivery semantics after scheduling audit — 2026-09-26

The frame/evidence owners and live path are unchanged. The offline scheduling
audit distinguishes four identities: image receipt time; producer/list delivery;
track coordinate version; and readiness/selection. Native association
`observed_at` witnesses that a candidate list already exists at consumption
start. It is not worker-finish time. For legacy reconstructed snapshots that
same field may be replay execution time; it must never authorize early work.
Original recorded scanner snapshots can supply conservative availability-by
bounds, not exact first-delivery times.

An updated track coordinate must not reuse local samples from its earlier XY
without exact semantic proof. The current cohort has 116 final XY revisions;
the fixed model charges every observed version, including superseded ones.
Candidate work can start before confirmation, but the final frozen method still
depends on the complete allowed prefix and final three POST observations. Its
earliest three-frame onset support timestamp is not final evidence completion.

One worker, prior measured costs and patch-work overlap produce only 1→2 of 33
conditional deadline successes after exact-coordinate availability is accounted
for. Actual camera pickup and CPU occupancy remain unknown; no live scheduling
feasibility or new delivery owner is established. No candidate-state prototype
or live hook was introduced. The next timing measurement should use saved-data
runtime replay with aligned publication/pickup, producer/consumer, coordinate
revision, confirmation and selector clocks. See
`/data/skjutbana/evaluation_runs/candidate_timing_20260925_01/REPORT.md`.

## Bounded causal temporal context, offline prototype — 2026-09-25

`CommonTemporalHistory` is a small append-only companion to `CommonFrameContext`,
not a new verifier or live BoardState. It accepts the existing camera owner's
uint8 grayscale, retaining at most 48 frames / 512 MiB. One owned immutable copy
per append makes replay ownership explicit. Exact required frame IDs and a
receipt cutoff select a query interval; missing/evicted frames are unavailable,
never a silently shortened PRE. Stream changes require a new history. Metadata
keeps receipt time, actual camera sequence when known and detached provenance;
historic trace serialization IDs are not upgraded to camera acquisition IDs.

The local query reuses `accuracy_verifier.patch` with the original 52→49 crop and
full-camera subpixel XY. `clean_physical_change.causal_local_evidence` batches the
existing disc/annulus signal and three-sample onset medians, preserving arithmetic
order, thresholds, persistence, ambiguity and all old/early/causal diagnostics.
No object membership, GT, audio offset or later frame enters the evidence. The
primitive's physical state remains UNCERTAIN; a research selector's coordinate
proposal is not independently validated damage truth.

Existing HitScanner grayscale history is the eventual reuse point. Its live
ScanportFrame currently drops camera sequence provenance; a future shared owner
must retain identity and immutable lifetimes. OBJECT, HitScanner and a future
verifier should query that same owner, not install separate competing histories.
This prototype is serial/replay only and is **not connected to live ingestion**.
Existing registered/blurred detector maps and audio-relative PRE caches have
different semantics and cannot substitute for raw same-XY observations.

All 36 events / 3,398 candidates match reference fields and decisions exactly.
Mean complete query is 79.2 ms, p95 119.3 ms; only 7/33 development deadlines fit.
The raw grayscale owner is cheap (mean 1.28 ms per append), but coordinate-local
sampling still costs frames × candidates. Dense arbitrary feature banks and
lossy spatial/temporal approximations were not introduced. CURRENT and canonical
SHADOW model authority/status are unchanged. Detailed measured constraints:
`/data/skjutbana/evaluation_runs/causal_pre_runtime_20260925_01/REPORT.md`.

## Fixed camera-space conjunction and abstention audit — 2026-09-25

`common_verifier.py` now also exposes two pure offline functions over the existing
`accuracy_verifier` signed-log feature contract. No replacement verifier class,
new extractor, fitted model or live caller was added. `coherent_change_evidence`
combines change/localization with per-pixel temporal consistency at the same
radius (2/4/8), then selects the best scale. Scales and correlated change
derivatives are not independent votes. Optional PRE variability softens strength
without vetoing holes, edges, texture or early onset. Reasons are logged per
candidate. `coherent_event_decision` optionally reports competing locations as
UNCERTAIN; absent evidence never establishes NO_PHYSICAL_HIT.

The unchanged CURRENT pool yields 9/35 @42 for the conjunction and 4/35 with PRE
context. Both lose protected events; both accept all four known negatives. A
fixed twofold distinct-competitor dominance rule abstains on all 39 events,
including all 35 real impacts. No evidence probability or no-hit threshold is
validated. These rejected functions remain research-only and disabled in live
authority. Existing common-logistic and PAIR_CLEAN controls reproduce 12/35 and
10/35 @42 but retain protected/tail/no-impact failures.

`automation.physical_selection_pass` consolidates the existing exact replay,
cached evidence and metrics into one event ledger. It checks current working-tree
BASE ordering/readiness against 39 snapshots; source captures are denied by the
existing read guard. Camera timestamp causality is distinct from unproved
feature delivery/live latency. Original D01 pixels remain unavailable, and
trusted capture-time hole/repair/projected-edge/seam context is not invented.
No S03 or game-object truth is used. Full measured results and model hashes:
`/data/skjutbana/evaluation_runs/physical_selection_20260925_001/REPORT.md`.

## Captured OBJECT evidence limitation — 2026-09-23

The captured CAN_OBJECT retry shows that two-frame `persistent_local_change`
means localized absolute residuals, not demonstrated new physical damage.
The false mini winner reverses signed inner change between the two frames;
the false beans winner brightens an existing projected rim. Six/five/four
objects simultaneously have confirmed inside-shape candidates. This is not
independent evidence from six/five/four damage events, and it is not a new
multi-object veto or signed-darkening requirement.

Reviewed visible damage features already exist in selected PRE, while frozen
camera geometry does not cover the reviewed gold/duck marks. Temporal reference
and physical mapping correctness must be established before interpreting these
candidate comparisons as a verifier model problem. Waiting and local registration
alone fail the three-shot gate. Post-reaction pixels cannot establish the outcome
of an adaptive verifier operating on an unchanged projection.

The new offline replay imports the live OBJECT implementation; it does not add
a common verifier, ranker or gameplay authority. Manual image-feature probes are
descriptive only and never enter selectable pools or physical coordinate GT.
Global EXACT_COORDINATE, CURRENT and the canonical SHADOW challenger are unchanged.

## Long ranking-family pass: redundancy is real, but removing it is insufficient

The frozen feature audit finds 27 exact duplicate pairs in all four sessions.
The 36 registered-residual columns have effective correlation rank ~1.50, and
36 PRE/context columns ~2.20. These are derivatives of shared camera frames, not
independent confirmations. PRE/context is the largest opposing family in ten
wrong-winner comparisons; removing one dominating family would reverse five
pair comparisons. That counterfactual does not establish a safe global selector.

Offline family normalization, contribution caps, mean/median rank aggregation,
consensus, Pareto ordering, two-stage prioritization and six-family pairwise
ranking all fail protected cases. One justified refinement collapses exact or
strongly correlated within-family columns before fitting on other whole sessions.
It retains the same 10/35 @42 and catastrophic/protected regressions. Family
ablations and training-session jackknife expose instability rather than a robust
subset. Direct physical-family ranks and a per-pixel temporal refinement also fail.

D01:5 exposes a separate precision issue: an actually observed coordinate within
20 px may have weak evidence at the ranker's much smaller patch scales. Cached
GT-only probes can diagnose this mismatch but must never enter selectable pools,
training pairs or physical accuracy. Physical identity/coordinate hypotheses
still need to remain separate; a new Track or BoardState subsystem is unjustified.

Whole-event NO_IMPACT analysis remains separate from conditional XY selection.
Supported component counts and crop-half motion disagreement have descriptive
signal, but thresholds transfer poorly between the only two negative-bearing
sessions. No production gate or experimental live ranker is prepared.

Full atlas, coefficients, correlations, all results, scopes and rejected variants:
`/data/skjutbana/evaluation_runs/ranking_families_20260914_112457/`.

## Follow-up: grouping evidence does not repair learned ordering

The new bounded observation helper is separate from live tracks and retains
actual coordinate witnesses with cutoff/ownership/local-proof provenance.
Reusing frozen PAIR_CORE/CLEAN models, grouped max, mean and logmeanexp track
evidence then within-track coordinate ranking still fails protected cases.
Keeping CURRENT track identity preserves all selected hit counts, while improving
D01 candidate oracle to 2/4/5/5. Mean/logmeanexp grouping selects 10/35 @42, with
the same protected D01:2/S01:1/3/POST_FIX:4 failures and 1,047 px S01 miss.

All new selected winning tracks have one ready coordinate hypothesis, so a
within-winning-track spread cannot separate positives from the four negatives.
Top-track margins overlap broadly (negative 0.295–1.730, physical 0.018–6.071).
No event acceptance gate is derived from this conditional ranking score.

Event 5's recovered 11.657 px raw witness now ranks 135/145 under PAIR_CLEAN.
This gives a concrete weak-evidence counterfactual beyond a generic quota loss.
Further uniform image-feature work needs original immutable pixels or D02;
current D01 paths were overwritten by newer runtime captures and cannot be paired
with the original labels. The September 10 frozen cache and model hashes remain
unchanged. Follow-up outputs: `D01_hypotheses_20260914_061840/final/` under
`/data/skjutbana/evaluation_runs/`.

## 2026-09-14 counterfactual ranking result

Track association identity should remain separate from the final physical XY
hypothesis. Existing observation history already supports bounded medoid,
confirmed-observation and multi-hypothesis experiments; no replacement Track
class is justified. Medoid/confirmed coordinates restore D01 event 1's 2.892 px
evidence, but fixed track ranking still selects the wrong track.

Four-session leave-one-session-out pointwise and pairwise fits use only D01,
S01, S02 and POST_FIX, never S03. The strongest new method, pairwise clean
evidence on the current pool, selects D01 1/1/1/1, S01 4/4/4/5, S02 2/2/2/2
and POST_FIX 2/2/2/2. It loses protected D01 event 2 and produces 1,047 px S01
and 549 px POST_FIX P95 errors. Adding history geometry does not change hit
counts. Removing static texture drops D01 to 0/0/0/0; adding simple change-point
features worsens every session aggregate. None is a promotion candidate.

Counterfactual comparisons support onset, persistence, inner/ring and registered
clean-change evidence for D01 events 1/4/6, but show incompatible positives:
D01 event 2 and S01 event 3 can contain strong PRE structure or pre-audio change.
These signals must be interpreted jointly and cannot become independent vetoes.
The four known NO_IMPACT events are not separable with a defensible held-session
threshold. Full artifacts are in
`/data/skjutbana/evaluation_runs/D01_counterfactual_20260910_152433/`.

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
