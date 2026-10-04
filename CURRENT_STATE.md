# Current project state

## Manual label/source lineage audit — 2026-10-04

S01:3 and S02:6 are both **LABEL_TO_SOURCE_LINEAGE_UNRESOLVED**. Runtime/session
identity, unchanged native labels and benchmark source hashes agree. S02's target
event also matches all128 files in its preserved September9 hash manifest.
S01:3 is physical shot3; S02:6 is physical shot5, because controls occupy runtime
IDs. Adjacent-event records provide no evidence of a label swap.

Historical annotation code defaults to the last POST but allows frame/view
changes; it saves neither the clicked frame/hash nor view/calibration identity.
The current calibration is later and cannot reconstruct that missing provenance.
Measured annotation-overlay shifts are at most0.084/0.068px, not an explanation
for the15.817/22.242px label-to-recorded-witness offsets. Neither label nor nearby
candidate is established as replacement truth. Both labels remain immutable.

The existing dataset module now accepts an optional hash-bound lineage reporting
sidecar. It never changes truth, model features, membership or canonical metrics;
unaudited events stay NOT_AUDITED. Frozen top3 union remains **32/33/34/36 of37**.
Excluding the two unresolved events gives32/33/33/34 of35 as a sensitivity check
only, **not a verified-lineage subset or accuracy gain**. No event has had its
clicked-source identity verified by this two-event audit, so that subset is N/A.

Next: independent human review of the two preserved forensic comparison sheets;
any label rebinding requires a separate explicit decision. No training or new
shooting is justified by this audit. Annotation source identity remains an
instrumentation gap; no runtime or annotation-writer behavior changed here.
Report: `/data/skjutbana/evaluation_runs/label_source_lineage_audit_20261004_01/REPORT.md`.

## Chronological raw-information audit — 2026-10-04

Frozen classic ∪ Vision top3 remains **32/33/34/36 of37 @5/@10/@20/@42**.
No training, benchmark relabelling, model promotion or CURRENT change. The audit
measured17 prespecified development events,624 saved frames and607 transitions:
five hard events, seven protected Vision saves and five deterministically selected
easy events. Native crop pixels, recorded receipt cutoffs and KOMP ordinal-only
limitations are preserved. Extra clocked POST was inspected only before decision.

There is no demonstrated common representation fix for the five failures:

- POST_FIX:10 has a near-GT **brightening** peak at raw33/NMS27 before `_unit`,
  error4.678358px. Clipping creates8,740 tied maxima and stable rank5671;
  uint8 rounding does not cause this loss. Its identity as new event damage is
  still ambiguous: polarity fluctuates and exact-GT onset is unsupported.
- S02:4's first transition has local-contrast rank8. The first3 reference spans
  the observed change; one earlier observation cannot certify clean stable PRE.
- S01:3 and S02:6 have persistent steps at **existing** candidate coordinates
  15.816977px and22.241914px from unchanged manual GT. Frozen Vision already ranks
  those neighborhoods2 and1. Native labels omit source-frame identity. This is an
  unresolved association, not evidence permitting automatic GT correction.
- S01:5 has weak/ambiguous new change at its labelled neighborhood. No hardware
  insufficiency or absent-projectile conclusion follows.

All five exact-GT locations remain unsupported by the existing onset rule;
this is UNKNOWN, not NO_PHYSICAL_HIT. Clipping also affects successful cases,
so amplitude loss alone does not explain the failure wall. A fixed GT-blind
three-observation persistent-change map recovers0/5 hard events in32 peaks and
only1/7 protected saves. Preserve this negative result. Float32 after clipping
cannot restore discarded amplitude; no hard top32 loss was caused by rounding.

**LONG_RUN_RECOMMENDATION: NO.** Next is a bounded existing label-to-source-frame
association audit for S01:3/S02:6, preserving original labels and benchmark.
Resolve lineage before further training. A future pre-clipping signed-amplitude
experiment is narrowly motivated by POST_FIX:10, but not yet qualified/promoted.

Report: `/data/skjutbana/evaluation_runs/chronological_information_audit_20261004_01/REPORT.md`.
New tools are diagnostic-only; shared preprocessing/runtime are unchanged. S03,
settings, traces, champions, prior exact-witness and Synthetic V3 results remain
untouched. This is source-information diagnosis, not selected/live accuracy.

## Vision representation / full-heatmap mining smoke — 2026-10-04

The frozen NMS-first reranker baseline was reproduced from its saved models:
all **45 event output coordinate lists match exactly**, with classic ∪ Vision
top3 **32/33/34/36 of 37** at @5/@10/@20/@42. Clarification: its Vision-only
32-peak pool covers **26/37 @5**; classic ∪ that pool covers **32/37**.

`automation.physical_ai_vision_repr` now adapts the existing lab runner, V2.23.6
convolution, registered-map cache and nested peak reranker. It supports bounded
TRAIN-only full-heatmap negative mining, separate warmup/refinement checkpoints,
native/context representations, session-safe scalers, immutable champions,
per-event ranks and failure walls. The primary pool stays32 and final budget3.
No Evidence Fusion, runtime detector, settings, game, or temporal-witness change.

The audit found that the winning base model already uses stride1 and a native
5×5 receptive field. Existing mining only reweighted its pre-sampled patch bank.
The new miner adds at most64 full-map false peaks per TRAIN event in one round.
The context variant retains all four native persistent channels and replaces
unused slots with their 9×9-smoothed context; the existing network then sees a
13px context field. It preserves native localization. Reranker physical-map
features keep their original semantics even when upstream representation changes.

Four predeclared, equal-seed/equal-epoch comparisons completed genuine nested
development CV on37 impacts/8 controls (80 Vision fits,20 reranker fits):

| Variant | Union pool @5 | Final union top3 @5/@10/@20/@42 |
|---|---|---|
| Frozen baseline | 32 | 32/33/34/36 |
| Persistent maps, extra epochs, no new mining | 31 | 30/31/32/34 |
| Persistent maps + mined peaks | 31 | 29/30/31/33 |
| All eight maps + mined peaks | 29 | 28/30/31/34 |
| Native persistent + context + mined peaks | 28 | 28/31/33/34 |

**Negative result:** zero new hard events enter the pool; every challenger loses
protected saves and is rejected. The frozen32 champion remains intact. Some
deep ranks improve (S02:6 raw10036→3608 with all maps), still far outside32.
This is not selected accuracy or physical validation. All controls still receive
three proposals; no rejection/emission claim follows. S03 was not accessed.

**NIGHT_RUN_RECOMMENDATION: NO** for this measured search space. Four trials do
not prove all representations impossible. Next is a bounded source-information
audit of the five failures: compare available chronological signed changes
before normalization against the first3/last3 cached maps. Determine whether
usable information was discarded or never captured before expanding training.
KOMP stays ordinal-only; none of these map policies establishes clean causal PRE.

Report: `/data/skjutbana/evaluation_runs/vision_representation_20261004_01/REPORT.md`.
Search input: `/data/skjutbana/physical_ai_lab/datasets/vision_representation_v1_20261004`.
Smoke: `/data/skjutbana/physical_ai_lab/vision_repr_smoke_20261004`.
The earlier physical exact-witness and Synthetic V3 results remain unchanged.

## Vision peak ranking / nested OOF reranker — 2026-10-03

The subsequent one-hour Vision search completed 219 trials. Its budget champions
are trial_000050 (top1), trial_000044 (top3), trial_000086 (top8). Best combined
classic ∪ Vision oracle @5 is **29/31/31 of 37** at budgets 1/3/8. Top3 has six
unique saves: KOMP:1/5 and S02:5/11/12/13. This is proposal coverage, not selected
accuracy. Eight controls are scored separately; S03 remains forbidden.

The six-event “local maximum below top8 or NMS” diagnosis has been resolved.
For top3 champion trial_000044, the best <=5px raw ranks are POST_FIX:3 **42**,
POST_FIX:10 **13503**, S01:3 **74155**, S01:5 **16035**, S02:4 **6580**, S02:6
**10036**. The geometrically nearest maxima are sometimes different, lower-ranked
points. Suppressors and full ranks are recorded, not inferred. All 36 audited
event/model combinations across the three champions reproduce their saved top8
XY exactly. Mere existence of a weak local maximum did not establish latent
37/37 coverage. Five failures are far below any reasonable small pool.

The existing lab now supports `peak-prepare/run/resume/status` through a focused
Vision-stage adapter. Maximum internal pool **32**, final proposals still **1/3/8**.
Two fixed orders were measured: raw top32 → rerank → NMS, and original NMS → first
32 survivors → rerank. These are different bounded pools from the same heatmap;
they are not claimed to be the same candidate set. Only the latter contains the
POST_FIX:3 witness, at NMS rank27. No cap search, coordinate refinement or snapping.

The reranker reuses V2.23's small MLP, scaler, event-balanced <=5/>42 label loss,
hard-negative miner and optimizer checkpoints. Its 24 allowlisted features are
heatmap/local-peak geometry plus means/inner-ring contrasts of the existing eight
registered maps. Raw XY, GT, game geometry, model/fold ID are not input features.
No new temporal/support/revocation measurements were invented. Correlated maps
remain correlated measurements. Controls require native NO_IMPACT truth; UNKNOWN
training rows are excluded.

Training is genuinely nested: for each of five outer splits, three inner Vision
models fit two outer-TRAIN sessions and predict the excluded TRAIN session.
Outer held/development sessions are absent from all 15 inner fits. Reranker
weights/scalers/mining see only these cross-fitted TRAIN peaks. Held predictions
use the original outer Vision model, also excluding held/development. This
implements nested reuse **within Vision**, not downstream Evidence Fusion.
Architecture selection remains DEVELOPMENT CV, not untouched validation.

Two predeclared MLP configurations differed only in NMS order. Results:

| Method | Combined top3 @5/@10/@20/@42 | Changes vs trial_000044 |
|---|---|---|
| Original | 31/32/33/35 | reference |
| Rerank before NMS | 30/32/33/35 | loses KOMP:1; rejected |
| NMS before rerank | **32/33/34/36** | gains POST_FIX:3; no losses |

POST_FIX:3 preserves exact XY **(2250,1321)**, error **4.262282px**, promoted
raw42 → NMS27 → reranked3. All six prior unique saves survive. Top3 wrong proposals
>5px drop 87→85, >42px 79→73; all events still receive exactly three proposals.
Controls still receive three proposals, not rejected hits. Combined top1 remains
29/37 @5; top8 becomes32/37. The rejected order promotes a KOMP:1 neighbor just
outside 5px, which then suppresses the good peak. NMS itself is not a proven bug.

The remaining five—POST_FIX:10, S01:3/5, S02:4/6—have no <=5px peak in either
32-point pool. The NMS-first pool's union oracle ceiling is **32/37**, already
reached by this smoke; its remaining opportunity is better top1/burden/stability,
not >32 coverage without different upstream proposals. No weak-physical-evidence
or low-contrast diagnosis is invented for those absent coordinates.

Artifacts: `/data/skjutbana/evaluation_runs/vision_peak_rerank_20261003_01/REPORT.md`;
nested cache `/data/skjutbana/physical_ai_lab/datasets/vision_peaks32_nested_20261003`;
authoritative smoke `/data/skjutbana/physical_ai_lab/vision_peak_smoke_final_20261003`.
The two configurations were repeated after a reporting-only fix distinguishing
protected regressions from already-uncovered top1 events; every metric matched.
20 new tests pass, plus 33 lab, 26 Vision and 15 common-verifier tests; relevant
V2.23 checks pass except the previously documented unchanged V2.23.1 fixture.
720 final proposals pass model/hash/held-lineage checks. CURRENT, settings, traces,
game assets, frozen classic bank, exact-witness behavior and S03 remain unchanged.

Next: one bounded local reranker search to check stability and top1/burden under
the save-preserving champion guard, then stop at this pool's measured ceiling.
Do not spend days searching for >32 with this frozen input. No long run here.

## Vision Proposal Lab V1 — 2026-10-03

The one-hour Evidence experiment is now analyzed, not extended. Of 2,096 trials,
only `trial_000934` reached raw Selected@5 **22/37** (23/24/25 at @10/20/42).
It saved S01:2 and S01:9 with no @5 regressions, but emitted five control hits and
five new >100px errors (KOMP:5, S01:5, S01:10, S02:5, S02:11). The exact comparator
rejected the new catastrophes first. Strict champion `trial_001115` remains
20/37. `trial_000187` won the UNKNOWN-count tie-break (22→21, adding an incorrect
88.008839px emission on S02:4 and worsening emitted error tails); `trial_001115`
then improved median error 3.929977→3.899767 px.
`failure_wall.best_after` reports the raw maximum, including rejected trials.
Twelve of fifteen common failures lack a <=5px coordinate in the fixed retained
bank; POST_FIX:2, POST_FIX:8 and S01:10 have one but are not selected correctly.
The classic ceiling remains **25/37 @5**, not a claim about every historical
unretained proposal. Further Evidence-only search is not the immediate priority.

`automation.physical_ai_lab vision-*` now reuses the same unattended lifecycle
for V2.23.5 registered maps / V2.23.6 convolution. Native camera-resolution maps
are cached once; small models search stride 1/2/4, channel subsets, loss and
training settings. Own learned peaks have frozen **top-1/top-3/top-8** budgets.
They never snap to GT, classic proposals or game geometry. Classic XYs are
immutable; exact duplicates are removed from union, nearby distinct points stay.
Five whole-session folds fit weights, normalization and hard-negative mining on
three TRAIN sessions only; the development and held session are excluded.
Every proposal carries model/data/code hashes, training/held sessions, rank,
uncalibrated logit, exact camera XY and source-frame provenance. UNKNOWN truth
never trains as a negative; operator-supported NO_IMPACT can. Nested Vision→
Evidence fusion remains disabled pending outer-fold-safe cross-fitting.

Two predeclared seeded OOF smoke trials completed (10.88 / 19.58 s core trial
time; second via resume). Trial 1 (stride 2) measured:

| Additional budget | Vision oracle @5/@10 | Combined @5/@10/@20/@42 | Unique @5 saves |
|---|---|---|---|
| 1 | 1/3 | 25/27/29/31 | none |
| 3 | 4/9 | 26/29/30/32 | KOMP:1 |
| 8 | 7/12 | 27/30/31/33 | KOMP:1, KOMP:5 |

Denominator is 37 physical impacts; eight controls are also evaluated. KOMP:1
has rank-2 XY (1897.5,1357.5), error **3.982516 px**; KOMP:5 has rank-7 XY
(1761.5,1487.5), error **3.270506 px**. That model trained on D02/S01/S02, never
KOMP or its reserved development session POST_FIX. These are independent learned
proposals, **not selected hits or live validation**. Top-8 also adds 289 proposals
farther than 5px from labelled GT across physical events (270 farther than 42px),
and eight proposals on every control. Confidence is not calibrated; no rejection
or physical support claim follows. No Evidence selector was retrained on them.
Two trials are not a Vision plateau. The first3/last3 history representation is
fixed and GT-free but not independently qualified clean PRE; KOMP clocks remain
unknown. Stride 1 is covered by implementation/tests, not this physical smoke.

Artifacts:
`/data/skjutbana/evaluation_runs/vision_proposal_lab_20261003_01/REPORT.md`,
native cache `/data/skjutbana/physical_ai_lab/datasets/vision_proposals_v1_20261003`,
smoke `/data/skjutbana/physical_ai_lab/vision_proposal_smoke_20261003`.
The report includes full per-session/budget results, lineage, failure wall and
commands. New Vision tests (26), existing lab tests (33), relevant V2.23.6 and
shared witness tests pass. An additional legacy V2.23.1 selftest fails its old
"good" promotion fixture: baseline coverage/fresh-domain fields now required by
the unchanged registry are absent. Test and registry match HEAD; neither was
weakened. CURRENT, settings, traces, S03 and all physical witness semantics stay
unchanged. Synthetic V3 status stays frozen.

Next: one local unattended Vision search, then inspect bounded unique saves and
common failures before spending more compute on Evidence selection:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m automation.physical_ai_lab vision-run --experiment physical_vision_v1_1h --dataset /data/skjutbana/physical_ai_lab/datasets/vision_proposals_v1_20261003 --hours 1 --seed 2361
python3 -m automation.physical_ai_lab vision-status --experiment physical_vision_v1_1h
```

## Autonomous offline physical AI lab foundation — 2026-10-03

This foundation preceded the Vision phase above. `automation.physical_ai_lab` builds a fixed cached
benchmark and trains the existing V2.23 NumPy MLP through a physical-evidence
adapter. This does not modify CURRENT, the frozen canonical challenger, settings,
proposal generation, exact witness retention, thresholds or runtime authority.

The fixed cache has **45 events / 1,056 exact retained coordinates** from KOMP
(6), D02 (6), S01 (10), S02 (13), POST_FIX (10): 37 labelled impacts and eight
operator-supported controls. No GT coordinates generate candidates or enter
features. Native labels, source hashes, missing measurements, raw witness
provenance and cutoff uncertainty are preserved. D01 pixels and S01:11's complete
cutoff remain unavailable. KOMP clocks remain unknown. The cache is a last-PRE
**offline reconstruction**, not exact live replay or physical validation.

Five fixed folds reserve one entire session for scoring, one for development
reporting and three for weight/scaler fitting. Repeated outer-session scores are
used for research model selection and therefore are NOT an untouched validation
estimate. S03 remains forbidden and unopened. Feature transforms/scalers never
fit the held or development sessions. Vision features are disabled: upstream
lineage checks exist, but nested OOF model generation/verification is incomplete.

The runner supports deterministic random configurations, epoch-level Adam/RNG
checkpoints, interruption/resume, exclusive ownership, immutable completed trials,
research champion history, model/source/dataset hashes, per-event predictions,
leaderboards, failure-wall summaries and queued full-retraining family ablations.
It reuses V2.23.5's stable hard-negative ordering on training sessions only.
No automatic model activation exists. Protected @42 regressions, new catastrophic
>100px errors and increased control emissions block research champion replacement.
Successive halving/evolution are deferred; this is bounded random search first.

Short Evidence NN smoke trials completed across all five folds. None displaced
the frozen exact-witness baseline: initial trials either lost Selected@5 or
introduced control emissions; new catastrophic errors also blocked acceptance.
These are plumbing checks, not accuracy gains.
The frozen baseline remains KOMP **4/6 @5/10/20/42**, D02 real **2/2 @5**, controls
**0/4 final support**, shot 6 **2.450390 px**, and later revocation of D02:4.
CURRENT KOMP remains 1/6. The full frozen cache selects 20/21/22/23 of 37 impacts,
with eight controls UNKNOWN (not validated no-impact rejections).
The fixed retained-bank oracle is **25/37 @5**: selector search alone cannot
reach 95% on this representation. Failure-wall reporting must distinguish absent
coordinates from ranking mistakes before spending more compute.

`automation.physical_ai_vision` adapts existing V2.23.5 registered evidence maps
and V2.23.6 spatial convolution to physical chronological history. Full-map peaks
are its own coordinates; no dense/target snapping is used. A seven-event map-cache
and two-epoch smoke exercised this path. Its first-three/last-three history
representation is not qualified clean PRE, its existing pair loss does not train
control abstention, and it is not a full benchmark Vision result. Raw-patch failure
history remains preserved; this adapter does not reuse raw-patch capacity alone.
Synthetic V3 qualification remains unchanged; no donor synthesis/training ran.

Commands (fresh experiment names; all data/models under `/data/skjutbana/`):

```bash
OPENBLAS_NUM_THREADS=1 python3 -m automation.physical_ai_lab run --experiment physical_ai_v1_12h --dataset /data/skjutbana/physical_ai_lab/datasets/witness_v1_20261003 --hours 12
OPENBLAS_NUM_THREADS=1 python3 -m automation.physical_ai_lab resume --experiment physical_ai_v1_12h
python3 -m automation.physical_ai_lab status --experiment physical_ai_v1_12h
python3 -m automation.physical_ai_lab queue-ablations --experiment physical_ai_v1_12h
```

No long job was launched. An interrupted epoch resumes exactly; a hard kill
conservatively charges time since its last durable heartbeat, including downtime.
`resume --hours N` explicitly adds budget. Mixed-code/data resume is refused.
Ablation queuing requires an actual learned research champion; inference-time
feature zeroing is not reported as contribution evidence.

NEXT: one local unattended Evidence NN development run, then inspect champion
rejections and common failure walls before adding model complexity. Proposal
reference-preprocessing reuse remains a later runtime optimization. Report:
`/data/skjutbana/evaluation_runs/physical_ai_lab_foundation_20261003_01/REPORT.md`.
Local HEAD was clean dev at bccf437; creating the requested branch was blocked by
read-only `.git`. Changes remain uncommitted; no dev/main commit was modified.

## Shared temporal batching measured, no authority change — 2026-09-28

The existing shared witness query now has a bounded batch form. It validates a
frame prefix once, reuses the existing vectorized onset arithmetic for clocked
and explicitly ordinal-only observations, and returns separate EvidenceBundles
with unchanged exact XY, provenance, support and revocation. Local contrast
sampling batches existing coordinates and reuses existing masks; retention remains
24 retained / 32 competing. No proposer, threshold, candidate/confirmation cap,
model, CURRENT emission path or live hook changed.

**All 45 events / 37,449 candidate prefixes match exactly**, including complete
witness values, state transitions, retained order and final selected coordinates.
No temporal floating tolerance was needed. Upstream regenerated producer floats
still vary slightly (score up to 1.83e-6); proposal XY/order and regional evidence
are unchanged. KOMP stays **4/6 @5/10/20/42**, D02 real **2/2 @5**, false controls
**0/4 final support**, protected shot 6 **2.450390 px**. D02:4 revokes at ordinal 15.
S01/S02/POST_FIX and all six older CURRENT-correct @42 cases survive. D01 pixels
and S01:11's complete cutoff remain unavailable. This is offline reconstruction.

Independent single-threaded measurements: KOMP full research event runtime
**6.56 → 2.58 s**, D02 **11.71 → 7.52 s**. Shared query work improves **9.43x /
7.42x** respectively; local extraction improves about 4.6x. Excluding frame I/O
and journaling, complete transition mean/p95 is **50.25/59.23 ms KOMP** and
**154.66/170.49 ms D02**. Proposals now consume **73.6% / 89.8%** of core time.
No proposal optimization is credited for incidental timing variation.

Batching shares computations, not physical identities or another image history.
The runner already appends signals incrementally and backfills only on coordinate
birth. The broad D02 path still cannot sustain its ~80 ms recorded frame spacing
on one worker; KOMP clocks and pre-trigger event/search-context ownership still
prevent a live-feasibility claim. NEXT: profile-preserving reuse of identical
reference high-pass/mask/resize preparation between the existing self-bias and
current-frame registration calls, gated on unchanged proposal XY/order and witness
results. Keep both correlations and the existing physical semantics.

Synthetic V3 is frozen: 68 donor events, 23 qualified research pairs, 45 partial,
0 unusable/unknown; extractor/small builder exist, unattended training is not ready.
No new shooting/training; spatial rules remain inconclusive and the native shutdown
SIGSEGV remains separate. Report and reproducible measurements:
`/data/skjutbana/evaluation_runs/batched_shared_temporal_20260928_01/REPORT.md`.

## Shared exact witness and qualified donor prototypes — 2026-09-28

**CURRENT remains unchanged.** `EvidenceBundle.witnesses`, the existing common
query/history, and `observation_hypotheses` now own the frozen exact-XY contract.
There is no live hook or new verifier class. Candidate/version/frame provenance,
raw local signal, receipt cutoff (or explicit ordinal-only uncertainty), support
and revocation remain attached to the exact observed coordinate. A changed point
cannot inherit its neighbor's proof. The bank remains 24 retained / 32 competing;
support does not survive later contrary measurements automatically.

All **45 development reconstructions / 37,449 candidate-prefix comparisons**
match the prior exact-witness experiment without numeric tolerance or decision
differences in the temporal contract, coordinate identities and retention.
Six regenerated proposals have tiny producer saliency/score differences
(maximum 7.63e-6 / 1.22e-6); these do not change proposal order, temporal evidence
or decisions. Whole producer dictionaries are not claimed byte-identical.
KOMP stays **4/6 @5/10/20/42**, D02 physical **2/2 @5**, controls
**0/4 final supported**. Protected KOMP:6 is 2.450390 px. D02:4 explicitly revokes
at ordinal 15. S01/S02/POST_FIX retain previous results; D01 pixels and S01:11's
complete cutoff remain unavailable. UNKNOWN is not a validated rejection.

Profiling disproves the suggestion that research I/O dominates: reference means
are 6.14 s KOMP / 11.44 s D02, principally temporal rescans and proposals.
The shared provenance wrapper costs more, not less. Existing batched onset
arithmetic matches 28,861 clocked candidate prefixes and costs 1.72 ms mean /
4.18 ms p95 per prefix batch, but this is an isolated diagnostic. Full D02
proposal generation alone costs roughly 141–147 ms per transition. No live
latency or incremental full-pipeline feasibility claim is made.

The 68 labelled physical source events are now qualified conservatively:
**23 QUALIFIED research donor pairs, 45 PARTIAL, 0 UNUSABLE, 0 UNKNOWN**.
Qualification combines recorded causal availability, local onset/persistence,
residual localization and hash-bound visual PRE/POST review. H10:10 is withheld
because clean pre-damage appearance is not established. KOMP remains PARTIAL
because clocks/cutoffs are missing. Masks are residual proxies, not mechanical
damage segmentation. Human labels are unchanged.

A small reproducible V3 builder produced 23 original pairs, 60 training-only
experimental residual composites and eight real negative pairs from ten control
events. Whole-session splitting occurs before augmentation (15/4/4 original
donors in train/validation/test). Rebuilding yields identical array-file hashes.
Additive and relative-response composition are unvalidated assumptions; no
training, model activation, physical validation or unattended V3 trainer exists.
Static appearance, new damage and final-selection targets remain separate.

NEXT: batch shared temporal queries in the existing bounded replay update, keep
its already append-only same-XY samples, require exact prefix parity, and measure
the remaining proposal bottleneck. Qualify composition realism before training. The spatial
cluster result stays inconclusive; native diagnostic SIGSEGV stays separate.
Report, hashes, tests and artifacts:
`/data/skjutbana/evaluation_runs/shared_witness_donors_20260927_01/REPORT.md`.

## Exact physical witnesses retained: bounded offline gain — 2026-09-27

**CURRENT is unchanged.** One prespecified retention intervention keeps every
currently supported exact observed XY through spatial deduplication, subject to
the same 24-coordinate / 32-competing bound. Unsupported proposals keep ordinary
3.5 px deduplication; nearby pending points cannot replace supported coordinates.
Support is recomputed on each prefix and can still be revoked. No averaging,
GT-selected point, threshold, proposal/confirmation cap or model changed.

Kompetensprov selected reconstruction improves **3/4/4/4 → 4/4/4/4 of six** at
@5/10/20/42. Leg's (1936,1535), 2.499026 px witness is retained and selected;
left plate improves 4.004992→2.081959 px and protected outside-right improves
3.944572→2.450390 px. Neck/outside-left remain UNCERTAIN. Four emitted hypotheses
remain supported through saved POST, checked retrospectively with unknown clocks.

D02 real impacts were added to the chronological experiment: **both original and
exact-witness retention select 2/2 @5/10/20/42**, errors 4.304369/2.415355 px.
That gain over CURRENT 0/2 belongs to earlier-transition search, not the retention
intervention. Four controls remain UNCERTAIN, zero final witnesses; D02:4 still
has its revocable temporary witness. This is not four validated rejections.

Paired 24-PRE-history reconstructions on 33 usable S01/S02/POST_FIX events produce
no tolerance-level regression from the retention change. Selected counts remain
S01 4/4/5/5 of 9, S02 4/5/5/6 of 10, POST_FIX 6/6/6/6 of 10; 12/29 physical events
remain UNCERTAIN. POST_FIX:4's error improves 3.454538→1.868840 px. All six available
CURRENT-correct @42 cases survive. Four older false controls have zero final
witnesses, but S02:10 has temporary support. D01 pixels remain unavailable and
S01:11 lacks a complete cutoff. This is a new last-PRE reconstruction, not the
older final-POST causal evaluation or exact live replay. Proposal XY/reference
lists match in all 45 paired events; physical validation is unchanged.

The spatial information experiment gives no general cluster rule: D02:3's
accurate retained point is isolated, while D02:4's wrong winner has five nearby
coordinates from two correlated producers. KOMP uncapped live ledgers are lost;
NMS already thins its measured pool. No isolation/edge/line/motion veto is added.

The real-source recount verifies **68 manual physical labels with extractable
PRE/POST/GT pairs** across seven explicit development sessions, 61 with recorded
causal POST cutoffs. This is donor availability, not clean-delta qualification.
Another 63 legacy static patches lack paired PRE/reliable unique event identities;
101 framepacks are projected synthetic data. None increases the 68 paired count.
No fitting or augmentation run was started. Real-Hole Synthetic V3 is PARTIAL:
source qualification, source-session splits and delta/mask provenance must precede
training. Existing EvidenceBundle/CommonFrameContext/CommonTemporalHistory and
exact-observation helpers are the reuse points; no new central class was added.

Exact-witness retention is justified for a bounded diagnostic/SHADOW adaptation,
not CURRENT promotion. The full chronological prototype remains slow (~5.96 s
KOMP, ~11.28 s D02 per event); only retention itself is cheap (~0.066 ms for 32).
NEXT: preserve these exact-XY witness identities and revocation in the existing
shared offline evidence query, with strict frame/coordinate-version provenance;
qualify real-hole donors separately before any V3 synthesis. No new shooting.
Report, measurements, source inventory/design, hashes and tests:
`/data/skjutbana/evaluation_runs/exact_witness_20260927_01/REPORT.md`.

## Chronological PRE search: measured offline gain, no promotion — 2026-09-27

**CURRENT is unchanged at 1/6. One frozen chronological reconstruction selects
3/6 @5 and 4/6 @10/@20/@42**, emitting four hypotheses and withholding neck/outside-left
as UNCERTAIN. It preserves protected shot 6 (3.944572 px versus CURRENT 4.558086).
Recovered selected errors for leg/left plate/right plate are 6.436268/4.004992/
4.884709 px. These are DEVELOPMENT offline results, not exact live replay.

The unchanged eight-proposal V2.25.1 path processes every saved PRE transition
against the preceding three observations. A 24-coordinate temporal bank reuses
the existing 3.5 px deduper without blending XY; pending observations expire after
three further frames, while supported coordinates can survive to the final PRE.
The frozen causal-support ordering selects only at that final PRE cutoff. No POST,
GT, target score, cap increase or confirmation-cap change chooses a winner.
KOMP has unknown frame timestamps: its support is ordinal-only. D02 uses recorded
receipt/sequence provenance and the existing gap guard. No clock is fabricated.

Four real shots have automatically discovered near-hole support. All four selected
witnesses remain supported through all 62 saved POSTs, checked only retrospectively.
Raw spatial oracle is 5/5/6/6 at @5/10/20/42, but includes chance nearby proposals
before damage; it is not physical selection. Neck's nearest generated coordinate
is 11.716 px away and unsupported; outside-left has a 3.335 px proposal but no
supported near-hole step. Coverage remains incomplete.

All four D02 false controls are UNCERTAIN at final cutoff (zero final supported
witnesses). However, D02:4 temporarily passes the same support test at PRE ordinals
13–14 and reverts at 15. It must not be accepted early or described as proof of
NO_PHYSICAL_HIT. D02:4/5 remain one shared right-edge action with unknown cough/knock
attribution. The prespecified any-negative-witness gate blocks automatic promotion.

The implementation target is **multiple mechanisms: reference timing plus exact
coordinate retention/evidence coverage**. Leg's observed (1936,1535), 2.499 px from
GT, is spatially deduplicated at ordinal 31 despite support. A postmortem of ALL
ever-supported coordinates shows this dropped point stays supported at final
cutoff (9.479 versus the selected point's 8.634). This is a concrete retention
opportunity, not a measured alternate selector gain. The first leg witness instead
becomes ambiguous at ordinal 49; do not bypass that ambiguity to claim success.

The prototype is bounded but slow: mean 134 ms/transition and 6.31 s/event on KOMP,
252 ms/transition and 11.86 s/event on D02 (includes research I/O/journaling).
Maximum retained/transient coordinates 24/32; observed peak process RSS 218 MiB.
No live/shadow hook is enabled. NEXT: one offline test of retaining a supported
exact observed witness through spatial representative replacement, under the same
24-slot bound and D02 controls; no tuning, new shooting or broader search.

Runtime shared diagnostic-map copying explains stale early blackhat artifacts;
a camera-free reproduction confirms new receipts can save an unchanged old map.
Worker detection starts with a separate empty debug dictionary. Missing map
producer identity is an instrumentation limitation, not evidence that detection
used old image pixels. This experiment uses raw frames, not those change maps.
Report/hashes/funnels/tests: `/data/skjutbana/evaluation_runs/chronological_pre_20260927_01/REPORT.md`.

## Kompetensprov selection diagnosis: PRE precedes software, not damage — 2026-09-27

**CURRENT remains 1/6 at @5/@10/@20/@42. No cap, selector, threshold or live
authority changed.** The recovered six-shot DEVELOPMENT run supports a bounded
reconstruction, not exact live replay. Every selected PRE snapshot is pixel-identical
to saved PRE-history ordinal 45. Reviewed new damage is already visible before
that snapshot on all six shots (observed transitions around ordinals 26/28/39/26/28/21).
These are saved ordinals, not camera sequences or times. Frozen local-step amplitude
criteria accept GT neighborhoods 3/4/6; 1/2/5 remain ambiguous. GT probes diagnose
the failure and never enter proposal generation or selection.

Reusing V2.25.1 on pixel-matched proposal frames, with explicitly reconstructed
last-three-PRE references, cap 8/16/32/64 gives proposal oracle @42 of 2/2/3/3
out of six; @5 stays 1/6. Conditional existing confirmation/novelty selection stays
1/6 for every cap. Missing original history, hybrid/bank/cleanup and confirmation
frame association prevent claims about exact live counterfactual ranks. Cap 64
is offline-only: the production setting clamps to 32. Raising retention alone
does not recover correct selections. Shot 3's recorded 16.998 px proposal already
survived the eight-proposal pool; its full live final rank is still unavailable.

An existing broader/global extractor reconstructs oracle @5/10/20/42 = 1/1/3/5,
but a single local+global union ordered by existing source-independent V2.25.2
physical quality selects 0/6 and loses protected shot 6. This is a negative
diagnostic, not a proposed policy. The global mask reconstructed from saved setup
is 80.58% versus the logged 79.6%; global results are not exact live FULL replay.
Both outside shots lie inside the actual margin-expanded local search mask;
outside-right success therefore did not require global rescue. Local acceptance
does preclude the fallback comparison, but joint selection did not fix it here.

Only shot 6 has a reconstructed candidate with a unique PRE step that also
persists through every saved POST. Shots 3/4 have clear GT-local steps but no
corresponding candidate-local step. A @20/@42 oracle can miss the small causal
evidence footprint. Four distant PRE-only step witnesses revert in POST, which
also cautions against early onset-only acceptance. POST analysis is retrospective:
the lost timestamps prohibit a decision-time availability claim.

NEXT is ONE offline, GT-blind test of the existing proposer on earlier saved
PRE-history transitions, before the changed surface becomes the reference.
Keep existing physical criteria fixed, test D02 negatives, and stop if false
witnesses are equally supported or selected XY does not improve. No new shooting,
ranker sweep or live promotion is justified. The native termination crash remains
a separate open bug. Original 937-file recovery inventory, six labels, settings,
game/assets/menu and physical-traces symlink are preserved; no S03 access.
Artifacts, exact assumptions, source/input hashes, per-stage tables and images:
`/data/skjutbana/evaluation_runs/kompetensprov_selection_20260927_01/REPORT.md`.

## Kompetensprov six-shot capture recovered in part — 2026-09-27

**Preserve this run; no replacement shooting is requested. CURRENT selected
1/6 at each of @5/@10/@20/@42 camera px.** The completed run
`kompetensprov_capture_20260926_215855` binds exactly to
`session_20260926_215856_KOMPETENSPROV_AK_906234cc`. Six manual camera labels and
all original files remain unchanged. Each event retains one PRE snapshot,
50 PRE-history frames and 62 POST frames; 245 working evidence maps also survive.
Exact emitted camera XY is recovered from the recorded game HitEvent JSON log.
Errors in operator order (neck, leg, left plate, right plate, outside left,
outside right) are 160.042, 56.514, 204.203, 159.069, 167.065, 4.558 px.
Only event 6 supports downstream MISS classification credit. All six used the
registered region lane with `rescue=0`; successful global rescue was not tested.

The kernel/core journal identifies the capture PID 919927 crashing with SIGSEGV
at 22:02:18 CEST, after all six events. Trace JSON was written only at Python
exit; a native crash bypassed that finalization. The truncated core lacks a
usable Python/native caller stack, so the underlying native fault remains
unresolved. Missing candidate ledgers, exact frame timestamps/sequences,
runtime geometry and writer coverage counters cannot be fabricated from NPY
indices or file mtimes. Recovery is PARTIAL, not a reconstructed producer trace.
Event 3 has a logged pre-confirmation proposal at 16.998 px; its final rank and
the complete candidate oracle remain unavailable. Region benefit versus global
selection is unmeasured; outside-left emission moved inside the image AABB but
still scored MISS, while outside-right was physically accurate.

The existing bounded trace writer now checkpoints terminal events when their
POST window closes and queued arrays settle, before app exit. Synthetic six-event
SIGSEGV regression: old code leaves zero trace files; fixed code preserves six
complete traces. Explicit normal shutdown, visible metadata errors, launcher
exit status and capture-only fault stacks supplement this. No detector authority,
ranker, game semantics, settings, labels or symlinks changed. The native crash
itself is not claimed fixed. `physical_test check` now inventories surviving
artifacts when metadata is missing without declaring the capture healthy.

Recovery report, read-only reproduction script, hashes, measurements and tests:
`/data/skjutbana/evaluation_runs/kompetensprov_recovery_20260927_01/REPORT.md`.
NEXT: resolve the native crash in a camera-free reproduction of diagnostic
termination before considering another capture. Diagnostic mode is intentionally
silent/static; ordinary Kompetensprov competition retains commands/audio.

## Kompetensprov AK direct physical contract and diagnostic ready — 2026-09-26

**Ready for a short DEVELOPMENT diagnostic, not proved physically reliable.**
The intentional Kompetensprov assets/menu are preserved. The game already supplied
one full-figure HitRegion to existing frozen V2.24/V2.25 search. It now uses the
stable API with mask identity/frozen image rectangle, classifies the final physical
point exactly, and records original camera XY/shot_id. Game-local TARGET_ASSIST,
virtual vertical displacement and outside-mask bullet-radius expansion no longer
turn an emitted miss into a game hit. Numeric zone scores are unchanged. Frozen
shot geometry is reused; no new GameObject/mapper/detector subsystem is needed.

The explicitly requested D02 pre-retention audit is complete: **3,446 recorded
coordinates, one supported onset**, the same event-3 track at 2.104643 px/rank 76.
Event 1 gains no supported witness; all four false events also have none. All
553 previous final-track onset records match exactly. Increasing retention of
this pool does not solve event 1 under the frozen method. Stop this line without
fitting, new coordinates or a wider threshold search. CURRENT remains 0/2 physical
selection and 0/4 negative rejection on D02; no new selector was promoted.

D02 displayed another target, so Kompetensprov selection benefit is unmeasured.
A fixed default-game geometry diagnostic excludes both real D02 candidates as
well as its wrong winners; imposing it cannot establish a physical improvement.
Existing global FULL rescue remains available, but it is conditional on the local
path not already accepting. The fresh diagnostic must measure actual inside hits,
boundary zones, outside misses and false attraction to the figure.

The existing process-only launcher now supports a fresh KOMPETENSPROV_AK binding:

```bash
KOMPETENSPROV_RUN="/data/skjutbana/evaluation_runs/kompetensprov_capture_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_collection launch-kompetensprov --output-dir "$KOMPETENSPROV_RUN"
```

Select Kompetensprov AK, position the figure in setup, then ENTER. This opt-in
mode is silent/static, suppresses global overlays and has no automatic scene
transition at 45 s; ENTER ends it. Wait five seconds before the first shot and
after the last, and at least six seconds between six independently labeled shots:
two clear inside, two near an internal zone boundary, two outside the silhouette
on the observed board. Do not seed truth from AUTO. Close normally; check/label
using the exact binding. No physical session was run during implementation.

Capture reuses bounded 4 s PRE/5 s POST, caps 240/300, maximum 16 runtime events.
The trace now copies matching PANG-time game/camera regions from the existing
registry instead of depending on an absent scanner field. Missing/stale context
stays null. CURRENT engine authority/settings/models remain unchanged. Normal
competition retains its 45-second clock; this diagnostic is not a competition.
Report, complete operator procedure, source/input hashes and passing checks:
`/data/skjutbana/evaluation_runs/kompetensprov_physical_20260926_01/REPORT.md`.
S03/settings/trace symlink untouched; no stage, commit, push or merge.

## Completed D02 physical capture analyzed — 2026-09-26

**CURRENT selected 0/2 physical hits at @5/@10/@20/@42, rejected 0/4
NO_PHYSICAL_SHOT events, and got 0/6 events correct.** Physical errors are
409.053 and 391.140 camera px. This is DEVELOPMENT evidence, not validation.
Explicit run: `tonight_D02/capture_20260926_162528`, bound to
`/data/skjutbana/physical_traces/session_20260926_162528_D02_4e1d6abd`.
Human-corrected truth is impacts 1 (right grid) and 3 (left white), sound control
2, the same right-edge cough/knock action producing 4/5, and left-edge control 6.
Attribution of cough versus knock between 4/5 remains unknown. Preserve the
protocol deviation: four false runtime events are three independent actions.

Both real damage appearances precede the audio peak by about 1.5 s and are already
present in old PRE at about -0.38 s. This is receipt-time evidence, not exposure
timing or a fixed offset. Event 3's 2.105 px candidate survives every stage but
PRE-based novelty reduces its score tenfold; after confirmation it ranks 76.
Removing that penalty alone would still lose to the winner. Event 1 loses a
9.670 px proposal at the legacy cap and retains an 11.867 px representative,
rank 33. That @20 coordinate misses the small causal damage signal: none of its
89 final tracks has supported onset, although the separate human-GT probe does.
Only event 3's accurate coordinate has supported onset among 553 final tracks.
All four false events pass explicit local confirmation and two-hit fallback
readiness; lack of onset is UNCERTAIN, not validated NO_PHYSICAL_HIT.

Extended capture succeeded: every event saved all 50 available PRE-history frames
and 62–63 POST frames, with no recorded sequence gaps, cap or writer losses.
Only 5–13 POST frames precede selection. The last 24 PRE frames already bracket
both onsets; longer PRE creates no additional supported witness here. Events
4/5 share 93 identical camera frames. Original event/geometry notes remain blank;
physical setup/scene invariance and complete worker delivery remain unverified.
All six CURRENT decisions reproduce; scalar/batch causal checks match 555 probes
exactly (553 original tracks plus two explicitly nonselectable GT diagnostics).

Report, immutable input/source hashes, complete funnels and evidence:
`/data/skjutbana/evaluation_runs/D02_forensics_20260926_01/REPORT.md`.
Next, separately scoped: audit the frozen causal method on the already recorded
pre-retention coordinates to locate any useful witness lost by quotas/novelty.
No new coordinates, fitting, live promotion or physical capture is authorized by
this documentation. CURRENT/settings/models remain unchanged; source traces,
trace symlink and S03 untouched. No staging, commit, push or merge.

## D02 capture infrastructure ready, no physical capture — 2026-09-26

The existing process-only capture launcher now supports the accepted D02 plan via
`python3 -m automation.physical_collection launch-d02 --plan ... --output-dir ...`.
It allocates a fresh explicit DEVELOPMENT binding/root, copies the unchanged plan
and operator notes, checks plan/settings identity, and reuses the CAN launcher
runner/startup guard. Saved AI settings and the physical_traces symlink are not
rewritten. CAN_OBJECT retains its existing foundation gate and behavior.

D02 requests 4 seconds PRE (maximum 240 native-history frames plus the original
PRE snapshot) and 5 seconds POST (maximum 300 frames), capped at 16 captured
events. Capture continues after terminal event removal without extending the
decision/selector ledger. The existing history and bounded writer queue are
reused. Coverage metadata distinguishes requested, available, attempted and saved
frames, missing boundaries/identities, sequence gaps, cap losses and writer losses.
Default captures retain the legacy 24/64 limits. No continuity or exposure-time
guarantee is invented. Camera/worker provenance limitations remain documented.

CURRENT authority, candidate generation, ranking and thresholds are unchanged.
This is capture infrastructure, not an accuracy improvement or physical validation.
Verification: 103 camera-free unit tests and 84 V2.22.3–6 runtime assertions pass;
affected Python compiles and git diff --check passes. Tests cover byte-identical
settings, symlink preservation, unique/claimed roots, CAN compatibility, bounded
native frame retention, post-terminal capture, missing coverage and frozen outcome.
The six-event package remains unbound/uncaptured here. Jimmy's exact launch and
preservation procedure is in PHYSICAL_CAPTURE_RUNBOOK.md. No S03, saved settings,
trace symlink, staging, commit, push or merge was touched by this work.

## Candidate delivery / scheduling checkpoint — 2026-09-26

**CURRENT unchanged; live feasibility remains unproved. No runtime prototype or
hook was added.** The timing audit covers 40 development event rows and 3,943
eligible coordinates, including six cache-only D01 events and incomplete S01:11.
Thirty-three events have the prior causal inputs and measured processing costs.

Observation is much earlier than usable software evidence: across 3,356 raw-data
coordinates, median first image observation is event+43 ms, while the median
exact-coordinate availability-by witness is +509 ms. Actual first delivery and
camera grayscale pickup remain unknown. S01/S02 have native association clocks;
POST_FIX's cached association clocks were generated during replay and are rejected
as delivery evidence. Original scanner snapshots supply availability-by bounds.
Tracking introduces 116 final coordinate revisions across 29 events; earlier
track timestamps cannot be reused as delivery times for those exact new XYs.

One serialized simulation moves patch work earlier, charges all 3,485 observed
coordinate versions, and retains the full unchanged causal final inference.
It uses prior measured costs, estimated per-patch shares, optimistic camera
delivery at receipt, and explicit detector/confirmation reservations. Once exact
coordinate availability is included, conditional deadline success is **1/33
batch → 2/33 scheduled**, both in S01. The prior 7/33 receipt-only result is
preserved; it assumed the complete candidate pool was already available.
Median slack changes **−77.08→−43.69 ms**; worst overrun **126.89→66.42 ms**.
Total modeled CPU increases 111.23→112.50 ms/event, including frame maintenance
and superseded coordinates. Mean latency after the last required input is
45.27 ms. This does not establish that every possible exact incremental onset
implementation is infeasible; it rejects promotion from this bounded model.

Primary live-timing classification is PROVENANCE_INSUFFICIENT for all events.
Within the conditional model, 29 misses wait on late exact-coordinate witnesses,
two on processing, two fit; seven cannot be simulated. Those are scenario
attributions, not proven historical causes. Of 23 nearest useful @42 coordinates,
22 are witnessed at least 74.7 ms before cutoff; S02:12 has only a 1.21 ms witness
margin. No ground truth changes scheduling or priority.

One frozen replay again matches all **36 events / 3,398 candidates exactly** and
preserves **5/5/6/6→11/11/11/11 of 29**, 12 causal proposals / 17 UNCERTAIN,
the same six gains, protected S01:3 loss, wrong S02:5 and four uncertain negatives.
84 tests, 39 CURRENT/readiness replays, compilation and diff checks pass.
Timelines, per-event bounds, CPU model, classifications, sources and limitations:
`/data/skjutbana/evaluation_runs/candidate_timing_20260925_01/REPORT.md`.

Next, separately scoped: instrument an existing-data runtime replay with aligned
frame pickup, producer finish/consumption, exact XY revision, confirmation and
selector clocks plus worker occupancy. No shooting is needed for that first
timing measurement. Historical missing delivery clocks cannot be recovered by
offsets or observation timestamps. No S03 access, settings/trace mutation,
staging, commit, push, merge or physical capture occurred. Stop this pass here.

## Causal PRE runtime feasibility checkpoint — 2026-09-25

**CURRENT remains authority; the optimized causal method still misses the live
budget.** The frozen reference reproduces all 36 available events (33 development
plus three historical Can) and 3,398 candidate records. Batched local reductions
and a bounded append-only grayscale history match every evidence field, PRE/onset
identity, ordering and decision exactly, with zero numeric tolerance. No fitting,
threshold change, new candidate or additional selector was introduced.

Reference inference remeasures at mean 299 / median 316 / p95 466 / worst 495 ms;
220 ms is candidate-loop onset search. Optimized inference is 41.6 / 41.3 / 63.7 /
65.3 ms. **Complete post-event query**, including patches and original diagnostic
components, is 79.2 / 77.0 / 119.3 / 128.4 ms. Only 7/33 events fit recorded slack,
versus 1/33 for reference inference alone. Continuous grayscale ownership costs
1.28 / 1.06 / 2.67 / 3.76 ms per frame; sensor delivery and concurrent live load
remain unmeasured. Work still scales with frames × candidates. Do not call this
live-feasible or claim that precomputing arbitrary change maps solved latency.

The companion `CommonTemporalHistory` reuses the common frame-context boundary
and existing camera-local sampler; the camera owner's grayscale is the minimum
reusable state. It is offline/replay only, bounded to 48 frames / 512 MiB, with
receipt/sequence provenance, immutable owned pixels and explicit unavailable
history. No live hook, new BoardState or detector authority exists. A separate
proved HitScanner performance bug converted the newest frame twice; reusing the
first conversion with an isolated copy preserves pixels and buffer ownership.

The fixed replay remains **5/5/6/6 → 11/11/11/11 of 29**, with only 12 proposed
physical emissions and 17 UNCERTAIN. All four negatives remain UNCERTAIN. Six
gains, protected S01:3 regression and wrong S02:5 tail remain unchanged. No live
promotion; infrastructure is suitable only for a future measured SHADOW study.
142 unit tests, 84 runtime checks and all 39 frozen CURRENT checks pass.
Report/profiles/equivalence/source hashes:
`/data/skjutbana/evaluation_runs/causal_pre_runtime_20260925_01/REPORT.md`.

Next engineering step, only in a separate pass: use existing candidate-observation
delivery records to establish when coordinates become available and whether
same-coordinate evidence can be updated before the final decision. This pass
stops at the measured bottleneck. No physical capture, S03 access, settings/trace
mutation, staging, commit, push or merge occurred. Prior negative results remain.

## Causal PRE / local onset, offline checkpoint — 2026-09-25

**Keep CURRENT physical selection unchanged.** One fixed camera-local onset/PRE
primitive improves some development evidence but fails promotion. The old Can
least-squares experiment reproduces all three exact proposals/brackets; its
search was frozen OBJECT context, not full-camera nomination. The new same-XY
primitive uses no object/GT input and finds two of those onsets, leaving Can:1
ambiguous. Receipt timestamps still do not establish sensor exposure or ADC time.

On S01/S02/POST_FIX's 29 measurable impacts, nearest eligible coordinates have
11 clear onsets, 14 ambiguous, two insufficient and two unsupported. All 11 clear
cases have contaminated old snapshots; eight improve over the prior selector's
earlier median reference. Six previously losing correct coordinates become
winners. One fixed selector gives **5/5/6/6 → 11/11/11/11 out of 29** at
@5/@10/@20/@42 camera px, but withholds 17 impacts and loses protected S01:3.
The four known negatives become UNCERTAIN, not validated NO_PHYSICAL_HIT.
Conditional P95 falls 463.951→232.160 px with reduced output coverage; S02:5
remains a wrong proposal. No live/deployed SHADOW selector qualifies.

Original D01 pixels remain unavailable and were not opened; its six results
cannot be remeasured. Full CURRENT remains **5/6/7/7 out of 35**; no challenger
35-event score is invented. S01:11 remains incomplete. All 39 cached CURRENT
selections, 56 original physical-feature witnesses and three historic Can result
dictionaries reproduce. New evidence inference averages 292 ms after patch
extraction; only 1/33 events has that much recorded frame-to-decision slack.
Live delivery/latency remains unproved. The previous selector negative result
and canonical SHADOW model status/hashes are preserved.

Separate proven telemetry fix: the installed camera fast-update wrapper omitted
`first_main_pickup_monotonic`. A two-line fix preserves the existing timestamp,
frame/cursor and detector behavior; its regression fails before and passes after.
No new demonstrated PRE/frame-association defect explains historical contamination.
133 unittest cases plus 60 runtime checks pass; source compilation/diff checks
pass. Report, exact provenance, every candidate/event and limitations:
`/data/skjutbana/evaluation_runs/causal_pre_onset_20260925_01/REPORT.md`.

Stop this hypothesis without tuning the protected failures. Future authorized
work needs immutable PRE/onset/POST with delivery/acquisition provenance and the
six-event impact/sound-only/no-damage-motion diagnostic. No physical capture,
D02/Can launch, S03 access, settings/trace mutation, commit/push/merge occurred.

## General physical selection, camera-space-only — 2026-09-25

**CURRENT unchanged; no promotion candidate.** The focused offline pass returns
to the 100% end-to-end physical-coordinate goal (95% minimum), independently of
the unresolved downstream calibration findings below. No calibration, gameplay,
candidate generation, new training, physical capture or S03 access occurred.

The original D01/S01/S02/POST_FIX hash-verified caches reproduce all **39 complete
CURRENT decisions**, including the current working-tree BASE selector and
readiness. There are 35 measured impacts, four known no-physical events and one
additional S01 impact with incomplete terminal persistence, not a measured miss.
Counts below are @5/@10/@20/@42 full camera pixels:

| Session | Physical measured | CURRENT selected | Causal oracle | Eligible oracle | Ranking losses @42 |
|---|---:|---|---|---|---:|
| D01 | 6 | 0/1/1/1 | 2/4/5/5 | 1/3/5/5 | 4 |
| S01 | 9 | 2/2/3/3 | 6/6/7/9 | 6/6/7/9 | 6 |
| S02 | 10 | 1/1/1/1 | 3/3/3/5 | 3/3/3/5 | 4 |
| POST_FIX | 10 | 2/2/2/2 | 7/7/9/9 | 6/6/9/9 | 7 |
| Pooled | 35 | 5/6/7/7 | 18/20/24/28 | 16/18/24/28 | 21 |

All 21 nearest useful losing tracks tie on onset distance and lose on raw
`best_score`. Seven events lack an eligible @42 coordinate. CURRENT accepts all
four known negatives and emits on all 35 measured impacts. Mean/median/P95 error
is 197.554/183.203/464.701 px, with 25 errors >100 px.

The strongest earlier frozen common-logistic control reproduces **12/35 at all
four radii**, with D01 1/6, S01 5/9, S02 3/10, POST_FIX 3/10. It still regresses
D01:2, S01:3 and POST_FIX:4 at @20/@42, accepts all negatives, and worsens pooled
P95 to 681.164 px. The existing four-session PAIR_CLEAN control repeats 10/35 @42
and its known 1,047 px error. Neither is newly fitted or independently validated.

Exactly three new fixed variants were tested on unchanged coordinates: same-scale
local-change/localization/temporal conjunction; soft PRE variability context;
and an uncalibrated distinct-competitor abstention rule. They select 9/35, 4/35,
and 0/35 @42 respectively. The first loses four protected cases and all D01
successes; the last returns UNCERTAIN on all 35 impacts and all four negatives.
No correct NO_PHYSICAL_HIT decision or validated no-hit threshold is established.
No real live implementation bug was found or fixed. The search stopped here.

Two pure offline helpers extend the existing Common Verifier; no central class
or live policy was added. One event ledger joins proposals, tracking, local
proof, eligibility, exact rank and causal evidence. Camera timestamps establish
image-time availability, not verifier delivery or live latency. Cached scoring
and explanations for the best existing control average 0.673 ms (P95 0.840 ms);
image extraction/live cost remains unmeasured. Original overwritten D01 paths
were not read. S01:11 and absent surface/repair/projected-edge context remain
explicitly unavailable. Seven relevant suites pass 104 unittest cases.

Reproduction, all event/session/rank/error comparisons, model hashes, tests and
integrity: `/data/skjutbana/evaluation_runs/physical_selection_20260925_001/REPORT.md`
and its `final/` package. Next useful evidence is the six-event camera-space
onset/no-impact diagnostic described there, only in a future authorized pass.
No D02/Can capture, settings/physical-trace change, commit, merge or push occurred.

## Calibration residual decision — 2026-09-24, matched setup

**The remaining blocker is model fit, not window placement.** The newer physical
run `geometry_retry2_20260924_212534` supersedes the placement-isolation request
below: setup comparison and stability both MATCH, measurement is valid, and the
client origin is (2135,95). X11 top decoration is measured at 37 px; the bounded
creation correction restored the requested client pixels. Status remains
**CALIBRATION_FIT_POOR**, never physical PASS.

- Offline analysis used only that run's hash-verified report and five saved
  frames. All 24 marker centers reproduce with the existing detector. New tool:
  `automation/calibration_residual_research.py`; results, sorted residual tables,
  vector fields, pixel montage, model holdouts and provenance are under
  `/data/skjutbana/evaluation_runs/calibration_residuals_20260924_01/`.
- **Structural limitation established for these observations:** a homography
  maps a straight rendered row to a straight camera line. The bottom row
  (IDs 3,6,17,7,2, left to right) requires a minimum enclosing line half-width of
  **4.724 camera px**. Therefore no homography can fit all its observed centers
  within 3 px, independently of optimizer/RANSAC settings. In the five fresh
  frames the bound remains **4.501–4.895 px** after standard subpixel refinement
  and projective diagonal-intersection centers. This proves inadequacy for the
  recorded correspondences, not a unique physical cause.
- The five severe residuals are exactly the rightmost layout column u=0.95:
  IDs 1,10,19,11,2, top to bottom. Mean error there is 15.449 px; mean observed
  minus predicted vector is (-13.861,-4.270) camera px. Other columns average
  1.16–2.53 px. The field is structured and repeatable, with bottom-row curvature;
  random corner jitter is insufficient (five-frame mean/max deviation from each
  marker's own mean is 0.212/0.583 px). Layout fractions are not surveyed Board XY.
- Correspondence IDs, rendered integer-blit centers, winding, convexity and
  orientation are consistent. Mean-corner vs diagonal-center displacement is
  at most 1.190 px; refinement plus diagonal center changes at most 1.309 px.
  Neither explains 13–18 px right-edge errors. Right marker edges are weaker,
  but no marker exclusion/edge blacklist or extraction change is justified.
- Fixed diagnostic fits on the stored 24 points, all errors in CAMERA pixels:

  | Model | Mean | Median | Max | Within 1/2/3 px | Leave-one-marker-out max |
  |---|---:|---:|---:|---|---:|
  | Saved CURRENT homography | 4.845 | 2.015 | 18.247 | 3/12/16 | — |
  | All-point camera→client LS, inverted | 3.592 | 3.554 | 7.632 | 2/4/7 | 10.922 |
  | All-point client→camera LS | 3.520 | 3.474 | 8.436 | 1/3/10 | 11.487 |
  | MAGSAC, existing 3-client-px threshold | 4.280 | 3.216 | 13.154 | 3/10/11 | 13.966 |
  | One quadratic residual diagnostic | 2.898 | 2.922 | 4.440 | 0/5/14 | 7.973 |

  None satisfies the 3-camera-pixel requirement. Spatial column holdouts also
  fail; the quadratic has no validated inverse and is not a runtime transform.
  Every homography retains a true mathematical inverse (~1e-12 client px).
  No diagnostic model was installed and no calibration was rewritten.
- **Separate inter-capture change:** fresh centers average (-8.560,-0.950) camera
  px from the saved calibration observations despite matching software setup.
  Keeping the saved model fixed gives fresh mean/max 11.972/26.974 px (0/120
  within 3). No candidate fixed model transfers within 3 px. Refitting only fresh
  frame 1 and testing frames 2–5 still gives maxima 7.679/8.337/12.478/4.494 px
  for the four diagnostics. No alignment offset was silently removed. The source
  of this between-capture shift cannot be determined from setup metadata alone.
- Existing code/history provide ArUco homography, camera orientation, HitInput,
  AnalysisGeometry/WorkingSpaceMap, viewport/crop and appearance references, but
  no intrinsic lens calibration or projector optical correction to reuse.
  Camera/projector distortion, surface nonplanarity and systematic detection
  bias cannot be separated by one projected layout on an unsurveyed surface.
  Pure global projective keystone alone cannot explain the noncollinear row.
- **Next, one no-shoot discrimination test when convenient:** substitute a rigid
  flat matte surface spanning the marker layout, keeping camera/projector/window
  fixed; retain the calibration and capture with the existing geometry tool.
  Compare row curvature, not PASS against the old fit. The runbook specifies the
  procedure and its limits. No immediate operator action or new shooting needed.
- This pass adds offline measurement and tests only. Runtime calibration,
  OBJECT/global detector, PRE, gameplay, settings and all existing artifacts
  remain unchanged. Physical calibration is still unverified at the required
  tolerance; exact physical accuracy has not improved by this analysis.
- Validation: 53 unittest cases across residual measurement, foundation,
  coordinate roundtrips, crop/demo mapping and trace safety, plus 13 V2.24 mapping
  checks passed. New Python files compile and `git diff --check` passes. These
  software checks do not constitute physical validation.

## Latest physical geometry failure — 2026-09-24 evening

The actual no-shoot run `no_shoot_20260924_184227/geometry` **FAILed**. This
supersedes the earlier statement that no foundation hardware run was available.
No new shooting, OBJECT ranking, PRE or global detector work occurred here.

- **Proven setup mismatch:** fresh calibration recorded SDL position (2135,95),
  diagnostic runtime (2135,132), stable throughout five frames. Passing the saved
  post-map position directly as a new `SDL_VIDEO_WINDOW_POS` creation request did
  not restore the same client placement. The normal app's existing automation
  sets an already-created SDL window; creation and mapped readback are distinct.
  Desktop coordinates remain setup provenance, not an added camera transform.
- Propagating the recorded (0,+37) through the existing inverse predicts +47.39
  camera Y at marker 16 (observed +47.21), and +37.85 at marker 17 (+37.13).
  Across 120 observations, descriptive residual mean falls 41.75 -> 5.16 camera
  px; max 53.71 -> 18.26. This is an explanation, NOT an installed mapping fix.
  Right-edge residuals remain, matching errors already present in the fresh fit.
- **Fit is not acceptable for the 3-camera-pixel foundation requirement.** The
  stored RANSAC fit has 17/24 inliers and mean 3.316 CLIENT px. In camera units,
  all-marker mean/max are 4.845/18.247 px, with only 16/24 within 3 px. The legacy
  calibration save ceiling is mean 25 CLIENT px; saving never certified a
  3 CAMERA px physical bound. Forward/inverse roundtrip is excellent (~0.00014
  client px). The earlier independent-inverse defect is corrected and unrelated
  to this new setup mismatch.
- RANSAC excludes the right edge and some bottom markers (IDs 1,2,3,7,10,11,19).
  A single diagnostic all-point least-squares fit still has max 7.63 camera px
  on the stored points and 8.19 on the new frame. No alternative fit was adopted.
  Projective diagonal-center vs corner-average displacement is at most 0.826
  camera px on the captured frame: it cannot explain the large right-edge error.
  Warped/nonplanar surface, camera/projector distortion and detection quality
  remain hypotheses; these images do not identify one uniquely.
- Diagnostic placement now waits for mapped readback and permits at most ONE
  recreated window using the measured creation-request/readback discrepancy.
  It retains env-before-set_mode and never reinstates Window.position assignment.
  The final client/viewport/display/orientation must match calibration provenance
  before camera startup. No hardcoded -37, camera offset, snapping or margin.
  Native X11 client origin and actual decoration extents are recorded when the
  installed read-only xwininfo/xprop tools are available. Historical reports lack
  extents, so the exact 37-pixel decoration thickness remains unmeasured there.
- Report classification now separates SETUP_MISMATCH, CALIBRATION_FIT_POOR,
  PHYSICAL_GEOMETRY_FAIL and PASS, in that priority. Every report retains fit
  quality even when setup mismatch is primary. A poor fit may be measured at a
  matching setup to diagnose residuals but cannot produce physical PASS.
  New calibration metadata records explicit client units, RANSAC inlier IDs and
  foundation fit quality; fitting and saved matrices are otherwise unchanged.
- Both operator fixes are preserved: no main-loop VideoCapture capability probe,
  and no diagnostic SDL Window.position setter. Regression tests protect them.
- One further NO-SHOOT geometry measurement is needed when convenient; none was
  executed during this pass. Use the single command at the top of the runbook;
  retain the current calibration for this placement-isolation test. No request
  for more shooting, no physical PASS and no detector accuracy improvement.
- Analysis/provenance: `/data/skjutbana/evaluation_runs/geometry_setup_20260924_01/`.
  Existing reports/settings/traces are untouched. Intentional gameplay changes
  remain intact. Eleven relevant suites passed (163 unittest cases + 35 checks).

## Foundation checkpoint — 2026-09-24 (no new physical shots)

**Another Can shooting test is not ready.** OBJECT and global ranking are unchanged.
The preserved September 15 outcome remains gold / beans / mini for human gold /
miss / duck; even shot 1's correct object award does not validate its physical XY.

- The complete mapping uses game-local coordinates + viewport origin -> Pygame
  client pixels -> existing inverse homography -> full oriented camera pixels ->
  AnalysisGeometry/WorkingSpaceMap crop. The reverse subtracts the viewport once.
  Can draws into that same viewport; content_rect/ingame offsets are not an extra
  transform on this game. The existing full-camera/crop correction remains intact.
- **Actual calibration bug fixed for NEW calibrations:** reverse RANSAC was fitted
  independently and was not the inverse of the forward homography. New fits use
  one forward matrix and its mathematical inverse. Marker correspondences now
  use the pixel centers of the actual integer marker blits. Saved calibration and
  both settings files were NOT rewritten or silently corrected on load.
- The saved August 29 calibration numerically reproduces the captured mapping,
  but has up to 3.881 client pixels of center round-trip error. Replacing only
  its inverse would shift these object centers by -2.20..+5.46 camera pixels in X
  and -2.90..+1.27 in Y: insufficient to explain the physical projection mismatch.
  Manual visible outline measurements across seven targets give median observed
  minus predicted center (+8.25,+14.88) camera pixels. Vertical errors range
  +11.64..+17.17. Edge uncertainty is about 3 camera pixels; small shapes, blur and
  intentional shape insets limit size/scale conclusions. No offset was installed.
- A title bar is visible in the camera capture. Calibration markers and game
  rendering both use CLIENT coordinates, so decoration is already implicit in a
  calibration made at that physical window pose. Moving the window/projector or
  changing client/display geometry invalidates the old mapping. Capture-time
  window pose is missing, so stale setup versus other physical mapping error
  remains unresolved. Desktop position is provenance, never physical XY truth.
- Audio public timestamps remain processing wall time minus the tail of a
  256-sample/16 kHz (16 ms) chunk. ffmpeg/PulseAudio/ALSA/pipe backlog is unknown.
  A real reader defect discarded partial pipe reads; bounded accumulation now
  preserves all bytes, including fragmented samples. This is not proof of the
  cause of the historical 0.2..1.7 second discrepancy. No trigger threshold or
  timing correction changed. Both base and installed V2226 paths carry metadata.
- Camera public timestamps remain wall-clock read completion, before orientation.
  Requested one-frame buffering is a driver request, not an exposure guarantee.
  New metadata records read start/end monotonic, stream/sequence, raw backend
  timing, publication/first pickup, shape/orientation and buffer request. Audio
  records receive/detection/dispatch monotonic, sample indices, trigger offset,
  receipt-relative sample-time estimates and explicit unknown device backlog.
  Physical traces retain these fields and runtime calibration/display identity.
- **One offline PRE experiment, no ranking sweep:** within merged OBJECT context,
  a two-segment localized temporal-step scan found strongest steps at camera
  (2273,1346), (1938,1350), (2430,1342). Proposed preceding frames are respectively
  -1744.742, -275.419, -1185.190 ms relative to recorded audio. No manual damage
  boxes, GT coordinates or desired object were inputs. Post-hoc comparison puts
  all three steps inside the previously reviewed new-mark boxes, at the existing
  appearance brackets. Costs were 158/132/123 ms for stored history processing.
  This supports event-specific PRE research, not a constant offset or oldest PRE.
  Live delivery, scene transitions, rapid shots and negative controls remain
  unverified; the live PRE policy is unchanged and cannot be declared safe.
- No-shoot `physical_foundation_check geometry` projects existing ArUco markers
  and validates existing HitInput predictions without fitting/saving. It requires
  distributed observations, <=3 camera-pixel residual by default and <=0.05
  client-pixel round trips. `sync` records four paired flash/original-tone events,
  bounded raw PCM and camera observations. Its receipt offset/jitter includes
  output latency and propagation; it is NOT device-time calibration or PRE proof.
  Hardware checks were not run here: no camera device is available in this runtime.
- `physical_collection launch-can` now requires a READY foundation report before
  binding or hardware startup. The current foundation checker explicitly returns
  BLOCKED while the live PRE policy remains unvalidated. Do not fabricate READY
  or bypass the guard with an older capture command. See the runbook's no-shoot
  steps; geometry, timing and PRE are separate prerequisites.
- Evidence and source/hash provenance:
  `/data/skjutbana/evaluation_runs/physical_foundations_20260924_01/`.
  Original CAN sources and prior forensic outputs are unchanged. S03 was not used.
  No new mapper, surface owner, detector, ranker or timing framework was created.

## CURRENT FACTS — 2026-09-23 captured CAN_OBJECT forensics

The September 15 captured retry **failed physical object identification: 1/3**.
The user's actual observations supersede the planned event order: gold-can hit,
physical miss between the two leftmost objects, duck hit. Recorded awards were
gold_can, beans_xxl, mini_can. All three were physical board impacts; the miss is
not a NO_PHYSICAL_SHOT label. No new physical run or corrected result is claimed.

- Inputs: `/data/skjutbana/evaluation_runs/can_retry_20260915_150509`, bound to
  `session_20260915_150509_CAN_OBJECT_b2eb4562`. Original OBJECT crops match their
  hashes AND same-timestamp full-frame crops exactly. All eight captured runtime
  source hashes match this implementation. Saved source inputs remain immutable.
- New `automation.object_physical_replay` reuses the live OBJECT verifier and
  reproduces all three original returned coordinate lists and scores exactly.
  It is clock-neutral evidence/selection replay, not delivery-latency validation.
  Frozen camera/game pairs recover a projective extension with held-out error
  below 0.00005 game pixels; this reproduces the logged mapping, not its physical
  correctness. Calibration identity/matrix and physical setup notes were not
  captured. The tool refuses mismatched crops, input escape and existing output.
- **PRE timing is a concrete blocker.** New dark image features consistent with
  the human-observed impacts are already present in every chosen OBJECT PRE.
  Reviewed feature boxes in full-camera XYWH are shot 1 `(2267,1343,8,6)`,
  shot 2 `(1932,1347,10,9)`, shot 3 `(2426,1336,9,12)`. These are forensic image
  regions, NOT independently labeled precise impact coordinates or proposals.
  Their descriptive half-step appearance brackets relative to recorded audio
  are respectively -1745..-1665, -275..-196 and -1185..-1104 ms. They are not a
  calibrated clock offset. Audio timestamps use PCM read completion minus at most
  one 16 ms chunk's trailing samples; camera timestamps use read completion.
  Neither establishes device exposure/acquisition time. Upstream delay versus
  other timing/association error cannot be isolated from these records.
- **Frozen mapping is another concrete blocker.** Overlays show camera AABBs
  above the photographed projected targets. None of the reviewed gold/duck
  damage-box pixels maps inside any frozen exact shape. Gold's frozen bottom is
  y=1331.58 while the reviewed mark starts at y=1343; duck's is y=1328.50 while
  its mark starts at y=1336. Existing duck render/union-shape software tests pass.
  This is not evidence for another duck-only shape enlargement. The capture
  cannot distinguish stale calibration from physical projection/setup mismatch;
  no guessed camera offset or collision margin was installed.
- Shot 1's correct object award uses `(2276,1314)`, score 11.369, on the can's
  projected upper rim rather than the reviewed new mark. Correct object identity
  remains the human result; it does not validate that candidate as damage.
  Shot 2's false beans point `(2571,1299)`, score 11.590, is a brightening rim:
  signed inner changes in the actual verifier are +9.00 and +8.62 gray levels.
  Shot 3's false mini point `(1844,1327)`, score 4.745, reverses signed change
  (-3.15 then +4.77) yet passes absolute-change persistence. Its duck point
  `(2446,1314)`, score 0.898, survives per-object evaluation but ranks 63 by
  physical evidence and loses public selection. It is on the upper contour,
  separate from the visible new body mark. This failure is not missing-object
  evaluation or loss of that coordinate before evaluation.
- Confirmed inside-shape support exists in **6/7, 5/6 and 4/5** active objects;
  supported points span about 729, 733 and 602 camera pixels. Two correlated
  absolute residuals do not establish new permanent physical damage. Multiple
  supported objects are descriptive ambiguity, not a new hard rejection rule.
- Local phase shifts at key points are subpixel, not a demonstrated panel-motion
  model. Surrounding-pixel registration and a first-order motion/illumination
  residual do not reliably remove the false winners while preserving real hits.
  Projected-edge/brightness variation is visible; mechanical motion versus
  projection/camera interaction remains unresolved. No edge or old-hole veto.
- **Waiting alone failed.** The next saved pre-reaction frame changes shot 1
  gold->mini at +214 ms, keeps shot 2 beans at +164 ms, and changes shot 3
  mini->toilet at +172 ms, then soda at +253 ms. Earlier-history PRE substitution
  alone also fails the required gold/miss/duck outcome. It is a labeled offline
  diagnostic bypassing live PRE-age policy, not a proposed production baseline.
  Original projected reactions begin at +290/+221/+261 ms; later frames through
  about +2.4 s are not clean counterfactual evidence for an unchanged projection.
  Saved frame spacing is approximately 80 ms, not an assumed 30 fps.
- **No live change, threshold, model or adaptive-latency mode was promoted.**
  CURRENT/global selection, OBJECT implementation, gameplay, audio, settings and
  the trace symlink are unchanged by this pass. Global exact-coordinate work
  stays paused; its target remains 100%, minimum 95%.
- Replay/temporal data, per-object CSV/JSON, onset montage and frozen-region
  overlay: `/data/skjutbana/evaluation_runs/can_settled_forensics_20260915_131800/`.
  Safety snapshot: `/data/skjutbana/backups/can_settled_forensics_20260915_131800/`.
  Fifty explicit software suites passed (374 unittest cases + 286 checks).
  An explicit physical replay invocation adds three matching recorded cases;
  those assertions reproduce the two failures, not fix or relabel them.

Next code work should establish capture-time mapping and audio/video acquisition
provenance before changing OBJECT evidence. Next physical activity should first
check projected alignment without shooting and measure audio/video timing with
an independently visible/audible event. Do not repeat the three-shot gameplay
test until those prerequisites are understood. See PHYSICAL_CAPTURE_RUNBOOK.

## CURRENT FACTS — 2026-09-15 first physical Can false negative

- Independent human observations at 20 m, SFX OFF: shots 1/2 physically missed
  all objects and correctly scored MISS; shot 3 physically hit the duck and
  scored MISS. This is a real gameplay false negative, not a successful trial.
  Capture correctly refused the populated old D01 root. No replayable pixels or
  pre-top-k OBJECT ledger from these shots are available; no physical regression
  percentage or corrected shot-3 result can be claimed.
- The logged V252/V253 duck candidate at camera (2422,1330) belonged to CURRENT's
  separate proposal/freshness/novelty path. OBJECT does not ingest that pool.
  Freshness can measure an offset neighborhood (default radius 3), whereas OBJECT
  requires exact-XY persistence in its own causal crops. Its presence in OBJECT,
  component-cap/confirmation/ranking fate and availability before OBJECT's cutoff
  are UNKNOWN. It is 1.2932 camera pixels below the supplied frozen AABB; the
  original collision would reject it even if returned. No recentering was guessed.
- Reproduced and fixed a general OBJECT starvation bug: public global top-three
  truncation happened before object evaluation. Merged crops now reserve two
  component slots per object within the existing 24-slot crop budget (bounded
  expansion to at most 128 for many merged objects). Each object retains at most
  three observed coordinates, including a physically confirmed exact-shape point
  when present. Frozen collision is evaluated before selecting public top-three;
  evidence scores/thresholds do not change. Nearby higher-scored noise cannot
  erase an inside observation during deduplication. One projectile still scores
  at most once, and mere shape membership cannot confirm physical change.
- Reproduced and fixed duck render/collision mismatch. Existing HitShapeSpec now
  supports a small explicit union of existing primitives. Duck uses its actual
  body/head ellipses and beak polygon for rendering and frozen collision/AABB.
  Its rendered pixels are unchanged in four checked sizes; visible interior
  outside collision falls from 1,145/7,144 to 0 at the 115x100 software fixture.
  No arbitrary margin, camera-coordinate snap or detector threshold was added.
  This defect is relevant to shot 3 but cannot be proved its sole cause.
- One structured shot record now includes each object's raw component reserve/
  cap decisions, observed candidates, exact-XY evidence/confirmation reason,
  local/global ranks, camera AABB/exact-shape membership and retention/drop reason.
  Bounded frame ledgers survive final top-three loss. Scoped PRE/POST crop inputs
  can be persisted with camera bounds/timestamps/hashes separately from global
  replay frames through the existing asynchronous recorder.
- Use `physical_collection launch-can` for the next **fresh captured three-shot**
  retry: can hit, duck hit, outside-object physical miss. It uses a process-only
  binding, SFX OFF and immediate OBJECT feedback, with later CURRENT diagnostics.
  No settings file is rewritten. Preflight and an atomic runtime claim reject
  wrong, populated or reused bindings before hardware/game startup. See runbook.
- Source/output backup and forensic artifacts are under
  `/data/skjutbana/backups/can_duck_forensics_20260915_123652/` and
  `/data/skjutbana/evaluation_runs/can_duck_forensics_20260915_123652/`.
  Existing research, CURRENT selection algorithms, settings and trace symlink
  remain preserved. No source physical session was opened; no retry was captured.
  Broad exact-coordinate research remains paused. 100% remains its target.
- Verification: all 49 suites pass (363 unittest cases + 286 explicit checks),
  including 43 OBJECT and 42 lifecycle/readiness cases. Python compilation and
  diff whitespace checks pass. These verify software semantics, not physical
  retention of the two uncaptured misses or a corrected duck hit.
- Ten-repetition generated 4K comparison: seven-object total crop CPU mean
  49.67 -> 55.21 ms, median 49.30 -> 54.61, P95 51.68 -> 57.63, maximum
  52.73 -> 57.96. Per-object selection/deduplication is 0.35 ms across both
  POSTs. Correct duck geometry changes processed area 255,444 -> 262,827 pixels
  (3.08% -> 3.17% of the fixture). New capture I/O and camera waiting are not
  included in that benchmark. Human-reported shot-3 latency remains the physical
  baseline: ~34.5 ms first frame, 101 ms first hypothesis, 265 ms decision,
  300 ms presentation. Retry measurements must establish any live change.

## CURRENT FACTS — 2026-09-15 OBJECT readiness hardening

- **Ready for a physical Can Shootout trial; still physically unverified.** No
  ranking model, physical threshold or production NO_IMPACT policy was tuned.
  CURRENT selector/ranking code is unchanged; the global exact-coordinate target
  remains 100%, with 95% only the minimum. No physical source session was read.
- Safety backup: `/data/skjutbana/backups/object_hardening_20260914_222437/` contains
  965 working files, untracked-work inventory and binary Git diffs. AI settings
  and physical-trace symlinks/data were excluded. HEAD is still
  `4cd6376eba9e4c38984e15ee6c6c4c1afa015148`; no staging/commit/push/merge.
- Fixed reproduced queued-peak timestamp misbinding, re-awarding an already
  destroyed target from a second frozen shot, and stale comparison/reset state.
  Render-time marker animation and wrapper overlays no longer bypass OBJECT's
  image freeze. PRE must follow the first presented static live game frame;
  missing safe PRE means UNCERTAIN. Current holes remain in PRE, not a blacklist.
- Fixed deadline enforcement under overload. A 25-ROI clutter fixture previously
  returned late HITs at ~586 ms CPU despite a 350 ms deadline. Now it returns
  UNCERTAIN in 30/30 repetitions: mean 351.27 ms, P95 352.60, max 353.74.
  Checks occur between operations; individual OpenCV calls are not preemptible.
  No truncated candidate pool becomes a late hit. Evidence thresholds did not change.
- **Speaker feedback is instrumented, not reliably suppressed.** There is no
  calibrated acoustic/loopback reference. Known output records cue/owner/shot,
  API start, nominal end and availability. Nearby raw events retain that context;
  two-second association is diagnostic only, with no added trigger holdoff.
  Existing audio thresholds and 80 ms cooldown remain unchanged.
  `SKJUTBANA_CAN_SFX=0` prevents Can SFX output for an initial baseline; defaults
  still play sound. Direct generated PCM at the saved 0.56 threshold triggers
  none; a fixed 3x input-amplitude example triggers once for each cue. Both are
  synthetic. Output sample amplitude cannot establish microphone feedback safety.
- Joined shot records now include frozen geometry, player/distance, full top-3
  provenance, exact collision, known-hole distance, frame/decision/presentation/
  SFX times, CURRENT XY/agreement and projected-image contamination flags. No
  collision or playback coincidence is detector truth. Comparison mode holds
  visuals through CURRENT; normal immediate feedback can change its later frames.
- **49 suites pass: 352 unittest cases + 286 explicit checks.** Focused coverage
  comprises 36 OBJECT tests and 42 readiness tests, including multi-shot/player
  sequences, reset/exit/deadline paths, 32 layouts in three orders, 80 randomized
  mapping cases, and 28 cue/sample-rate combinations. This is software verification.
- Updated generated 4K benchmark, 30 repetitions: seven targets at 20 m process
  255,444 pixels including context (3.08% of the fixture); total CV mean 50.45 ms,
  median 50.46, P95 52.94, max 56.82. Full component/distribution results including
  14/25-object, overlap, 25m/50m and clutter cases are under
  `/data/skjutbana/evaluation_runs/object_hardening_20260914_222437/benchmark_deadline/`.
  Camera waiting, acoustic latency and actual projector response are unmeasured.
- That run's refreshed `tonight_can_object/` package has blank human notes and
  current source/configuration hashes, but no binding/capture. D02 and previous
  research outputs remain preserved. Follow PHYSICAL_CAPTURE_RUNBOOK.

## CURRENT FACTS — 2026-09-14 OBJECT / Can Shootout implementation

- **Can Shootout now uses OBJECT as its normal physical input. Ready for a
  physical trial; no physical accuracy or live latency has yet been verified.**
  Start `python3 main.py`, select **Skjut burkar – 20 m** and wait for live play.
  Two players, five shots each, projected physical target dimensions, +/- 5 m,
  streaks and animations remain. Accepted duck/can/other object hits play cached
  original procedural sounds. Uncertain, missed and duplicate events play none.
- OBJECT scans merged frozen object camera ROIs with context, not the whole-board
  winner. It reuses the shot registry, HitInput/working transforms, recent camera
  ring, V2 registration and coordinate-local disc/ring evidence. A persistent
  local change in two distinct causal POST frames can return PHYSICAL_HIT and
  up to three actual observed XYs. Exact frozen GameObject shapes decide gameplay.
  A nearby change can be PHYSICAL_HIT but still a game miss. NO_PHYSICAL_HIT is
  scope-limited; unresolved evidence/missing PRE/geometry/budget returns UNCERTAIN.
- The latest safe PRE contains current holes. Known holes are context, never a
  veto; adjacent change and observable enlargement are covered in software tests.
  No OBJECT output updates persistent physical truth. A same-hole projectile
  leaving no observable change cannot be reliably established from these pixels.
- Old game-local shadow/center assist is removed from normal gameplay. The legacy
  demo mapping fix/tests are retained byte-for-byte and auto-install is disabled.
  CURRENT/global selection algorithms and metrics are unchanged. This capability
  makes no claim toward global 95%/100% physical accuracy.
- One `[OBJECT-SHOT]` record joins scoped status/hypotheses/exact collision/timing
  and the later ordinary CURRENT result. Traces carry separate OBJECT diagnostics.
  Normal immediate projected reactions may affect CURRENT's later input frames;
  use `SKJUTBANA_OBJECT_COMPARE=1` to hold game visuals until CURRENT completes
  for the paired capture. Sound remains immediate. No object intent becomes GT.
- Generated 3840×2160 benchmark, 20 repetitions: seven objects at 20 m search
  **214,180 nomination pixels (2.58% of the fixture frame)**; including surrounding
  context, 255,444 pixels (3.08%) are processed. Mean PRE preparation 12.48 ms,
  first/second POST processing 18.40/18.86 ms, total 49.73 ms (P95 52.24 ms).
  Fourteen overlapping regions search the same pixels. At 25 m: 166,769 pixels
  (2.01%), total mean 37.89 ms. These include scoped CV, not camera delivery or
  global CV. Synthetic affine geometry is not actual calibrated scan geometry.
- Software validation: 36 focused OBJECT tests; all 48 broad suites pass
  (310 unittest cases plus 286 explicit PASS checks). Regression results and
  reproducible benchmark/provenance live in
  `/data/skjutbana/evaluation_runs/object_capability_20260914_184342/`.
  Capture package `tonight_can_object/` is prepared only: no binding or physical
  results. Instructions are in PHYSICAL_CAPTURE_RUNBOOK. D02 remains separate.
- All previous research and negative findings remain. No approved physical
  dataset was reread, no overwritten D01 pixels were used, and S03 was untouched.

## CURRENT FACTS — 2026-09-14 long ranking-family research pass

- **CURRENT unchanged; no experimental detector prepared.** Fourteen new ranking
  variants plus the frozen control, three unchanged candidate pools, six family
  ablations and twelve training-session jackknife fits fail the promotion gate.
  100% exact physical correctness remains the target; 95% is the minimum.
- **Failure atlas at @42:** seven CURRENT successes; 21 failures with a correct
  eligible coordinate where onset ties and producer `best_score` decides; three
  useful plausible observations outside eligible representatives; four events
  without a ready plausible observation within 42 px. The full stage ledger still
  records tighter-radius representation/cap losses separately.
- **Redundancy measured, not guessed:** 27 exact duplicate column pairs recur in
  every session; 158 pairs have absolute global Spearman correlation >=0.95.
  Registered residuals have 36 columns but effective rank ~1.50; PRE/context has
  36 columns and effective rank ~2.20. Collapsing before fitting reduces 132 inputs
  to 117 exact-unique or 90–95 grouped inputs but still selects 10/35 @42, loses
  D01:2/S01:1/3/POST_FIX:4 and retains the 1,047 px S01 error.
- **Robust aggregation alone is insufficient:** normalized, capped, mean/median
  ranks, consensus, Pareto, two-stage and six-family pairwise structures all fail.
  A direct four-family physical score avoids learned texture penalties and scores
  9/35 @42 without >1,000 px errors, but loses four protected events and all D01.
- **Event 5's oracle needs careful interpretation:** its 11.657 px observation
  has weak r4 evidence (normalized late absolute change 0.474, persistence 0),
  while the nonselectable cached GT probe has 3.025 and 0.8. A coordinate within
  20 px is not proof that a small patch at that XY covers the actual damage.
  GT probes remain diagnostics only, never inference candidates or accuracy gains.
- **Temporal and motion context remain research:** S01:3's clear local step starts
  ~273 ms before audio. Step-shape and per-pixel persistence refinements do not
  select safely. Crop-half residual survival favors the good point in only 14/25
  nonidentical comparisons; old image halves are not calibrated physical panels.
- **NO_IMPACT remains unsolved:** negatives tend to have more supported spatial
  components and larger crop-half disagreement, but held-session descriptive
  boundaries either accept negatives or reject real shots. No live gate is fitted.
- **D02 workflow tightened:** saved-configuration inspection, exact plan/binding
  checks and read-only postflight camera/crop metadata export are implemented.
  The six-event package is refreshed; human notes are blank and no binding/capture
  exists. The can game, exports, menu and prior mapping fix remain unchanged.
- **Verification:** 41 suites pass: 274 unittest cases plus 168 explicit checks.
  Research tools deny physical-trace access and writes to baseline caches. Outputs:
  `/data/skjutbana/evaluation_runs/ranking_families_20260914_112457/`.
  `review/failure_atlas.html`, `review/all_metrics.csv` and `review/all_results.json`
  contain the human reader, all 208 summary rows and 2,028 event rows. Use
  `context_complete/`; the earlier context output used a broader raw population
  and was corrected with regression coverage.

## CURRENT FACTS — 2026-09-14 observation hypotheses / D02 preparation

- **100% physical correctness remains the target; 95% is the minimum. No physical
  selector is promoted.** New comparisons use only the original September 9 D01
  data preserved in the hash-verified September 10 counterfactual cache, together
  with cached S01/S02/POST_FIX. No current D01 pixels enter these experiments.
- **D01 provenance problem explained:** 584 arrays and six trace JSONs were
  replaced by September 11 runtime captures in the September 9 directory, while
  original labels stayed unchanged. Recorded capture times/root/commit identify
  the new runtime inputs; the precise launch command is unknown. The writer's
  reused shot directories and replace-on-write behavior explain the overwrite.
  Current D01 paths are invalid for the old labels. No files were restored/deleted.
- **Capture safety implemented:** a recorder refuses a populated capture root or
  an already claimed shot directory; later event IDs cannot append after refusal.
  A fresh binding is required after application restart. Collection start also
  refuses an existing output binding/settings backup. These affect capture only.
- **Small offline observation helper implemented:** existing audit history feeds
  current/first-alternative/best-confirmed/latest observed XY nominations, at most
  32 extras per event. Exact XY witnesses retain frame, delivery, producer and
  local proof; confirmation never transfers between coordinates. D01 oracle is
  2/4/5/5, but fixed CURRENT identity changes no selected count. POST_FIX's old
  reconstructed associations have post-cutoff delivery timestamps, so no extra
  hypothesis is claimed causally available there.
- **Event 5 recovered within a fixed budget:** its 11.657 px raw point (legacy
  rank 276) is never examined by the old 48-slot queues. It is physical queue
  rank 201 (51 consumed), weak queue rank 65 (17 consumed), and clean queue
  ranks 221/232/222 (26/29/29 consumed). A fixed raw-contour maximin coverage
  reserve retains it third, yielding D01 oracle 1/3/6/6. It then ranks 135/145;
  selected accuracy drops to 0/6. Coverage is research-only.
- **Joint identity/XY ranking remains blocked:** mean/logmeanexp grouping of the
  existing held-session PAIR_CLEAN model yields D01 1/1/1/1, S01 4/4/4/5, S02
  2/2/2/2, POST_FIX 2/2/2/2. It still loses D01:2, S01:1/3 and POST_FIX:4, with a
  1,047 px S01 miss. NO_IMPACT margins overlap physical events; no gate is fitted.
- **Intentional game work preserved.** The current can game actually uses
  region-first OBJECT_DEMO gameplay (newer than its former same-shot assist).
  A real demo mapping bug was fixed: frozen full-camera regions now use the
  existing V2.24.4 camera→crop/scale map before reading local difference arrays.
  Game rules/canonical candidate output are unchanged by this fix. Software
  tests establish mapping correctness, not physical accuracy of OBJECT_DEMO.
- **D02 prepared, not captured:** two physical shots plus four controls, execution
  order 3→1→2→4→5→6. A provenance sidecar requires human board/seam/display notes
  and records saved geometry/configuration/source identity. See
  [PHYSICAL_CAPTURE_RUNBOOK.md](PHYSICAL_CAPTURE_RUNBOOK.md) for exact commands.
- **Verification:** 40 suites pass, 252 unittest cases plus 168 explicit checks.
  Repeated research selections/oracles/ranks match. Outputs:
  `/data/skjutbana/evaluation_runs/D01_hypotheses_20260914_061840/` (`final/` is
  authoritative; the initial run preceded a numerical tie-ordering fix).

## CURRENT FACTS — 2026-09-14 offline counterfactual pass

- **No live change is promoted.** CURRENT remains D01 0/1/1/1, S01 2/2/3/3,
  S02 1/1/1/1 and POST_FIX 2/2/2/2 at @5/@10/@20/@42. The new best
  four-session-held-out pairwise selector reaches 1/1/1/1, 4/4/4/5,
  2/2/2/2 and 2/2/2/2 respectively, but destroys D01 event 2 and creates
  severe S01/POST_FIX outliers. It fails promotion.
- **Track identity and impact-coordinate evidence are demonstrably different.**
  An actual-observation medoid or confirmed-observation representative restores
  D01 oracle from 1/3/5/5 to 2/4/5/5 and preserves event 1's 2.892 px point.
  Keeping the existing winning track identity still leaves the selected result
  wrong; representative replacement alone changes no selected hit count.
- **Bounded late reserves improve recall selectively, not selection.** Adding at
  most 48 spatially diverse candidates keeps total pools below 200. A clean-family
  reserve recovers D01 event 3 at 7.328 px; physical/weak reserves recover event 5
  only to 23.718/35.283 px, not its known 11.657 px deeper rescue. D01 oracles are
  2/5/5/5 (clean), 1/4/5/6 (physical) and 1/4/5/6 (weak). Comparator oracle gains
  exist, but held-session selectors do not use them safely.
- **Ranking evidence is contradictory across sessions.** D01 events 1/4/6 have
  stronger localized onset, persistence and clean-change support than CURRENT's
  wrong winners. D01 event 2 and S01 event 3 show why simple rules fail: valid
  impacts can also have high PRE texture/change or an onset before the recorded
  audio boundary. Removing static texture features or adding a change-point
  feature makes held-session selection worse.
- **NO_IMPACT remains unresolved.** Four known negatives overlap positives in
  every inspected evidence family. Observation-history locality is a promising
  descriptive feature (AUC 0.824 on this tiny GT-referenced comparison), but the
  held-session gate either accepts all three S02 negatives or rejects genuine
  events. No rejection rule is installed.
- Complete immutable outputs are under
  `/data/skjutbana/evaluation_runs/D01_counterfactual_20260910_152433/`.
  The ledger covers all 35 complete physical events and four negatives. The broad
  suite passes 37 modules: 232 unittest cases plus 168 explicit checks. D02 was
  not created; S03 was not opened or inspected by the research drivers.

## CURRENT FACTS — 2026-09-10 evidence preservation continuation

- **100% correct physical output is the target; 95% is the minimum. Neither is
  achieved.** The D01 oracle/selection gap remains open. CURRENT, canonical
  SHADOW model/hash, live quotas, association and ranking are unchanged.
- **Exact event-3 cause identified:** legacy rank-171 contour at 7.328 px has
  merged rank 307 / legacy-pool rank 174. V2 reserve 50 plus 150 additional legacy
  selections fill the hybrid 200 cap. It is removed by `hybrid_capacity_exhausted`,
  with no spatial merge/replacement. Opt-in tracing now records input identity,
  geometry operations, quota outcome and output rank without changing candidates.
- **Event 1 evidence preserved offline:** reusing existing association history
  adds bounded alternative XY. Research-pool oracle becomes 2/4/5/5 from 1/3/5/5;
  the 2.892 px observation is available at rank 12, but selection still fails.
  The fixed 150+50 spatial/size retention trial and 32-contour rescue do not
  recover event 5. No cap increase or unproven live retention rule was installed.
- **Protected event 2 remains a veto on promotion:** CURRENT 9.146 px becomes
  197.077 px under the common verifier. Observation alternatives select the
  same coordinates as that verifier on D01/S01/S02/POST_FIX. D01 selected tuples
  remain CURRENT 0/1/1/1 versus research 1/1/1/1; common ranking also loses
  S01:3 and POST_FIX:4 despite aggregate gains. Earlier best flow remains 2/6
  and remains rejected. All per-event regressions and tails are reported.
- **Independent evidence trials rejected:** centered PRE retains 55.84% GT mass
  versus 40.06%, but no D01 selection changes and S01 regresses 3/9→2/9. Simple
  compact/darkening confirmation rankings both select 0/6 D01. Motion-only 3×3
  crop classification scores 0/6 D01, below majority 3/6; no cross-session motion
  prior is supported. Missing capture calibration prevents physical Board/seam
  claims. Broad flow remains research-only.
- **NO_IMPACT remains unsolved:** S01:4 and S02:2/8/10 are accepted by all tested
  methods, including with explicit local confirmation. No timing rejection,
  synthetic negatives, relabeling, persistent self-training or gameplay priors.
- **Inventory complete for this pass:** current/history mapping, surface/hole
  state, audio visualization and GameObjects are documented. The audio waveform
  calibration scene still exists; no removed physical weapon-profile system
  was found to revive. Reuse it for deterministic profiling if later justified.
  The smallest Board consolidation is a view/revision contract over existing
  owners. No new central class, fullscreen mode or AI game SDK was created.
- **Verification:** exact recorded-input CURRENT replay matches all 39 complete
  decisions; repeated offline results are identical. Five new coordinate tests
  cover existing homographies, crop/scale, explicit Board fixtures, content rect,
  orientation and four-corner bounds. All 36 suites pass: 214 unittest cases plus
  168 explicit checks. 200 additional merge differential comparisons match HEAD.
  Finalization again agrees with all six D01 labels. All 4,954 input hashes match;
  no inspected-session additions; settings/symlink preserved. One event-6 upstream
  reconstruction is only 199/200 XY/score exact, and is labeled partial.
- **Next capture recommendation revised to six diagnostic events:** two physical
  controls (low contrast, printed-line/near-hole), two sound-only (isolated/after
  shot), two observed no-damage panel-motion controls. Existing D01 positives
  reduce the need for duplicate positive pilot shots; no claim of adequate
  statistical validation. First save actual calibration/bounds/seam provenance.
  D02 was not created or captured. S03 was not opened, inspected or modified.

Full event/rank/accuracy/runtime tables and reproduction:
[D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md).
Inventory: [PHYSICAL_BOARD_STATE_ARCHITECTURE.md](PHYSICAL_BOARD_STATE_ARCHITECTURE.md).
Outputs: `/data/skjutbana/evaluation_runs/D01_evidence_20260910_081143/`.

## Preserved D01 completion facts — 2026-09-09

- **Target: 100% correct physical output; minimum 95%. Neither is achieved.**
  CURRENT and the canonical SHADOW challenger remain unchanged.
- **D01 finalized and verified:** six captured events, six precise physical
  labels, no collected no-impact events. Binding-based finalization agrees with
  the preserved manual manifest and prior report. All six quality fields PASS.
  Exact recorded-input replay matches 6/6 tracks/emissions and repeats identically.
  All 765 session files and eight original ancillary inputs retain their hashes.
- **D01 accuracy @5/10/20/42:** CURRENT **0/1/1/1 of 6**; causal proposal oracle
  **2/4/5/5**; eligible track-XY oracle **1/3/5/5**. Frozen early common verifier
  **1/1/1/1**, recovering event 4 while losing CURRENT's event 2. Contour rescue
  raises union oracle to **2/5/6/6** but still selects only **1/6**.
- **Proven losses:** event 5's useful legacy contours rank 228/276/279 and are
  lost at the 200-candidate cap. Event 1's 2.892 px proposal becomes same-frame
  support for a 12.835 px representative. Event 3 loses a 7.328 px legacy retained
  point before hybrid cleanup; that union's exact loss is not instrumented.
  Four oracle-positive events lose final ranking.
- **Clean-change research:** 71 causal contexts, 65 physical plus six existing
  no-physical. PRE variability/persistence removes 67.91% of D01 raw change mass,
  retains 40.06% of GT-local mass and improves mean S/B 2.969→3.788. Selection is
  still 1/6. Flow plus augmented verification reaches the best observed D01
  **2/2/2/2 of 6**, but severely attenuates all six GT neighborhoods, regresses
  comparison sessions and accepts known no-impact events. No promotion.
- **Architecture evidence:** isolated edge features do not recover D01; the
  motion-only 3×3 ROI proxy ties a majority baseline (3/6). Exact projected-edge,
  board/seam and Board Space attribution are unavailable because calibration is
  absent. Existing board/reference/hole/coordinate/audio-waveform foundations
  were inventoried; no new central live class was added.
- **Workflow fixed:** `physical_collection finalize --binding ...` builds the
  aggregate manifest from labels/assignments, checks fresh quality and hashes,
  and requires explicit ordinal mapping or `--in-capture-order`. Preview,
  historical `--labels`, recovery and external assignments are supported;
  existing outputs and ambiguous mappings are refused.
- **Preserved target/plan:** Testtavla and its Bilder menu entry are unchanged.
  The unbound plan is copied exactly to `research/physical_capture_plans/D01.json`;
  runtime bindings/reports stay local.
- **Next:** review D01, then eight D02 diagnostic events: four physical shots
  across weak-flat and edge/near-hole pairs, four observed sound/motion controls.
  Save geometry/calibration and an independent event log. D02 was not created.
  S03 was not opened, inspected, evaluated, tuned against or modified.

Full results, tests and reproduction: [D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md).
Outputs: `/data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/`.

## Preserved S02/common-verifier facts — earlier 2026-09-09 pass

- **Accuracy objective:** 100% correct physical output; 95% is only the minimum
  acceptable level. No practical success or live promotion has been achieved.
- **Completed common-verifier research:** 65 causal decision contexts from five
  explicitly allowed development sessions (59 physical, six no-physical), with
  whole-session training exclusions and frozen baselines. Registered multiscale
  patch logistic selection reaches **5/5/5/5 of 9 on held-out S01** and
  **3/3/3/3 of 10 on S02 development**, at @5/10/20/42. These are different fitted
  models under the same method. Primary held-out @42 improves 7/49 → 11/49;
  including development-used S02, 8/59 → 14/59. This is offline selection at
  recorded cutoffs, not new live-compatible end-to-end physical validation.
- **Separate failures remain:** bounded earlier-reference proposals raise S02's
  oracle to 3/3/5/9 but union selection stays 3/3/3/3. A spatial null averages
  7.082/10 @42, so broad coverage is not proof of physical proposal quality.
  Common logistic falsely selects all six known no-physical events. A visual
  gate rejecting all three S02 false events also rejects five physical shots.
  Broad-negative, forest, guarded-PRE, pairwise and contour-rescue trials are
  measured and documented; none is a promotion candidate.
- **Truth caveat:** S02 event 1 may have produced two visible changes, unconfirmed.
  Its native label is unchanged. Exclusion sensitivity is reported. A minimal
  evaluation contract supports NO_PHYSICAL_SHOT, single-impact, UNKNOWN/AMBIGUOUS
  and an extensible impact array; no generalized live event engine was added.
- **Trace integrity fix:** cleanup now preserves copied upstream contour/filter
  evidence and input/after-novelty/after-ridge/retained boundaries with coordinate
  provenance. Previously overwritten per-coordinate rejection evidence cannot
  be recovered retrospectively. Candidate scores, ordering and live authority
  remain unchanged. S03 has not been opened, inspected, evaluated or modified.
- **Frozen offline references and results:**
  `/data/skjutbana/evaluation_runs/accuracy_95_100_20260909_174137/`.
  See [ACCURACY_95_100_RESEARCH.md](ACCURACY_95_100_RESEARCH.md) for exact metrics,
  per-shot S02 proposal forensics, hashes, limits, tests and reproduction.
- **Latest completed collection: S02** at
  `session_20260909_162957_S02_b900192c`, bound by
  `evaluation_runs/S02_retry_binding.json`. Ten precise physical labels map to
  events 1,3,4,5,6,7,9,11,12,13; human-confirmed nonphysical events are 2,8,10.
  Their S/unresolved files remain untouched; the external mapping, label
  manifest, quality and finalization reports are in
  `/data/skjutbana/evaluation_runs/S02_finalization_20260909_170523/`.
- **Measured S02:** causal candidate recall is **3/10 @5/@10/@20, 5/10 @42**.
  CURRENT selected/emitted accuracy is **1/10 at all four radii**; mean
  187.926471, median 163.723005, P95 426.563857 px. All ten physical events
  emitted, and **3/3 nonphysical events also emitted false hits**. Both frozen
  shadows are 0/10 @42. This is DEVELOPMENT physical evidence; provenance does
  not establish independent validation of a new selector.
- **Verified S02 funnel:** 10 physical → 5 causal @42 → 5 tracked → 5 locally
  confirmed → 5 eligible → 1 selected. The four correct losing tracks rank
  71,101,110,46; V1 loses to FAST at tied onset distance. The other five events
  lack causal @42 proposals, so a verifier alone has a 5/10 ceiling on the
  existing coordinates. Existing FAST-exclusion replay stays 1/10; no tuning.
- **Quality:** S02 trace/frame/label/replay-readiness/patch/temporal PASS; exact
  recorded-input replay matches all 13 selected tracks and recorded emissions.
  All 1,664 S02 files and the locally modified settings file are preserved.
  This is not regenerated detector replay. No S03 data was opened.
- **S01 comparison:** ten physical labels and nonphysical event 4; nine physical
  decisions are complete, with causal oracle 9/9 @42 and CURRENT 3/9. Event 11
  has images/GT but incomplete terminal persistence, so its accuracy is
  unavailable. The previous finalization is preserved. S01 remains useful for
  research; do not describe its tenth outcome as a measured timeout or miss.
- **Workflow:** `physical_collection reset-labels --binding ...` previews an
  exact allowlist; `--apply` verifies an external label backup before removing
  only ground-truth/status/assignment files. Tests exercise evidence preservation
  and refusals. `physical_test check/label/classify/evaluate --binding ...` now
  targets the same captured session. Postflight checks actual artifacts and
  explicit PASS values, rejects incomplete mapping, and scopes single-session
  inspection without reading sibling sessions.
- **Next evidence:** a small diagnostic capture identifying each newly visible
  change and its physical onset, plus explicit no-impact controls; details and
  reasons are in the accuracy report. Existing traces cannot recover overwritten
  rejection ledgers or resolve unconfirmed physical ambiguity. The completed
  offline experiments are frozen for comparison. S03 stays reserved; CURRENT,
  canonical SHADOW challenger and confirmation hash remain unchanged.

Full metrics, evidence limits, hashes and reproduction:
[S02_PHYSICAL_FINDINGS.md](S02_PHYSICAL_FINDINGS.md).

## Previous post-fix comparator and research foundations (historical)

- **PROVEN physical validation:** post-PRE session
  `session_20260908_194746_b38de674` has 10 physical shots, 10 audio events,
  zero false events. Causal oracle is 7/10 @5/@10 and 9/10 @20/@42.
  CURRENT remains **2/10**, mean 171.10, median 135.75, p95 464.70 px.
- **PROVEN reconstructed funnel:** 10 physical -> 9 causal proposals @42 ->
  9 tracked -> 9 locally confirmed -> 9 eligible -> 2 correctly selected.
  All seven losses occur at final ranking, at ranks 9–59, outside the debug
  top eight. Complete input replay matches all ten winners and 136 saved
  track checkpoints plus 85 tracking counters. This is recorded-input
  reconstruction, not regenerated
  physical detection or a new physical validation.
- **PROVEN instrumentation:** complete active pools, eligibility/rejection
  reasons, association ledgers and compact source histories are captured.
  CURRENT_EXACT_REPLAY raises on mismatched track ids, coordinates or ranks.
- **PROVEN prior fixes:** PRE camera-plane mapping is fixed; its historical
  residual median was 192.385 -> 1.538. FAST saturation disappeared. Pending
  confirmation boundaries and producer rejection gates are implemented.
  The additional producer-transport gap is fixed: the actual worker result
  list consumed by tracking/confirmation now carries the event tag too.
  Tracks also reject association with a known different producer, preventing
  cross-event XY/hit/best-score history from surviving a later tag change.
  Pending-event regressions cover both 6→7 and 14→15 patterns and delayed
  valid pre-boundary delivery. Fresh physical validation is still required.
- **PROVEN bottleneck:** surviving correct tracks lose final ranking. FAST
  wins all seven oracle-positive failures, but complete-pool FAST exclusion
  still gives 2/10. No slot cap or historic-best-over-current score advantage
  explains these seven losses. Appearance/confirmation discrimination remains
  the selection problem to isolate.
- Previous 20-shot independent validation remains 1/20 @42 for CURRENT and
  both shadows. Audio events 7 and 15 remain NO_PHYSICAL_SHOT. Its causal
  oracle remains 12/20 @42. Historical results and inputs are preserved.
- **RESEARCH_ONLY, rejected for promotion:** fixed local-contrast ranking is
  0/10 on physical development. Synthetic development is 85/90 vs CURRENT
  81/90 @42; protected holdout ties at 141/150 and worsens mean error. The
  510-event development stress run has 450 impacts plus 60 no-impact events.
  CURRENT is 406/450 @42, with 30/60 forced no-impact events becoming ready.
  No AI, canonical challenger, or frozen CONFIRMATION_SELECTION_SHADOW is
  promoted or changed. Frozen confirmation hash:
  `123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f`.
- **Next step:** collect 20 deliberate physical diagnostic shots with the
  complete tracing and corrected producer transport/association; verify exact replay and
  capture latency first. Include light/dark, low-contrast and nearby repeated
  impacts. Keep the existing ranking. The next offline hypothesis should
  examine registered, polarity-aware localized novelty/confirmation; no
  research ranking is ready for a physical promotion trial.
- **Latest offline evidence:** registered PRE→POST residual features reproduce
  saved confirmation values, but physical true tracks and false winners remain
  weakly separable; synthetic separability is much stronger and therefore has
  a clear domain gap. No registered-impact ranker is frozen or promoted.

See [OVERNIGHT_TRACK_RESEARCH.md](OVERNIGHT_TRACK_RESEARCH.md) and
[CAUSAL_CANDIDATE_AUDIT.md](CAUSAL_CANDIDATE_AUDIT.md). Checkpoints below are
historical and do not supersede CURRENT FACTS.

<!-- V2.24.0 GAME_HIT_CONTEXT -->
## V2.24.0 checkpoint

Game-hit context foundation is available. Existing games remain valid without
changes. Future games may return `HitRegion` AABBs in game-local coordinates;
shot-critical snapshotting and game->camera transformation are prepared for the
next local-physical-search stage. No local object-aware hit authority is enabled
in V2.24.0.

<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->
## V2.24.1 checkpoint — object-aware local physical search

V2.24.0 has been physically smoke-tested on the shooting PC: its selftests pass
and the application starts. V2.24.1 now uses frozen camera HitRegions to
constrain the first V2.22.5 FAST physical proposal search. Overlapping regions
are merged after a camera-space safety margin. Existing V2.22.5 PRE->POST local
confirmation remains the physical gate and its FULL-RESCUE remains global.

Games with no HitRegions retain the existing global detector behaviour. No new
AI or game-context hit authority is enabled. Next planned checkpoint is V2.24.2
with a dedicated game-context verification scene before V2.25.0 introduces the
small shared GameObject/Hittable/Breakable/ObjectManager layer.

<!-- V2.24.2 GAME_CONTEXT_TESTSCENE -->
## V2.24.2 checkpoint — dedicated HitRegion testscene

V2.24.1 installed and started successfully after V2.24.0. No physical V2.24.1
shot series was required before proceeding because V2.24.2 provides the scene
needed to exercise HitRegions intentionally.

The new scene is available under Games after running
`python3 -m automation.v242_prepare`. It exposes target/no-shoot/moving/overlap/
edge/outside-region cases plus an EMPTY-regions global-fallback mode. The scene
logs returned HitEvent XY and shows the frozen shot-time region geometry in
cyan for direct visual comparison.

No V2.25 object system or new AI authority is enabled yet. The next decision is
based on physical V2.24.2 results: fix the V2.24 bridge if required, otherwise
proceed to V2.25.0 GameObject/HittableObject/BreakableObject/ObjectManager.

<!-- V2.24.3 LOCAL_ROI_ALIGNMENT -->
## V2.24.3 checkpoint — physical V2.24.2 findings and correction

A physical V2.24.2 run on 2026-09-04 produced nine camera shots. HitRegion
snapshots reported seven game and seven camera regions, the moving target was
classified successfully in two shots, EMPTY/GLOBAL correctly produced
`objects=0`, and shot-critical timing remained healthy. However, no
`V2.24.1 LOCAL-SEARCH` line appeared in the run. Most region shots instead
logged `outside_detector_roi`, and two early frames bypassed the V2.24.1
extractor through CandidateGenerator V2's `waiting_post_peak` path.

V2.24.3 therefore corrects ROI coordinate alignment and moves object-local
search to the whole live HitScanner ROI. It also changes zero local overlap from
a silent global first pass to an explicit V2.22.5 FULL-RESCUE transition.

Next action: run `Hit Context Test (V2.24.3)` with the short physical matrix in
`V243_TEST_PLAN.md`. Do not start V2.25.0 until normal region shots visibly use
the local ROI path and returned XY matches the physical holes reasonably.

<!-- V2.24.3 LOCAL_ROI_INTEGRATION_FIX -->
## V2.24.3 checkpoint — local ROI integration fix

The first V2.24.2 physical test verified game->camera snapshots, moving-target
classification, overlap/no-shoot semantics and EMPTY/global compatibility, but
it also exposed that the intended V2.24.1 local-first detector was frequently
bypassed. Most shots logged `outside_detector_roi` and no V2.24.1 LOCAL-SEARCH.

V2.24.3 fixes the implicit viewport-local content-rect origin, refreshes HitInput
calibration before the shot snapshot and applies the frozen object mask at
HitScanner ROI level. FULL-RESCUE remains global. The V2.24.2 testscene is
retained and upgraded with faster movement plus frozen/current motion distance.

Next action: repeat the short physical Hit Context Test matrix. Only after clean
local-ROI results should development proceed to V2.25.0 reusable objects.

<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->
## V2.24.4 checkpoint — detector working-space ROI mapping

The V2.24.3 physical run isolated the remaining integration bug: HitRegions
were present and transformed, but every region-enabled shot logged
`region=0.0% overlap=0`. V2.22.1 intentionally executes detection on a smaller
crop-local image and restores full camera coordinates afterwards; V2.24.3 fed
full-camera AABBs directly into that local mask.

V2.24.4 maps full-camera HitRegions to the active analysis working space using
`AnalysisGeometryV2221` crop origin and actual work/crop scale. The testscene
and logs now expose full-frame size, crop rectangle, work size, scale and an
example camera->work region mapping. V2.25.0 remains gated on one clean physical
acceptance run of this bridge.

<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->
## V2.25.0 checkpoint — reusable game objects

V2.24.4 is accepted as the game-context bridge. V2.25 introduces a stable
`src.engine.game_objects` package with exact shapes, entity/part identity,
gameplay projectile penetration, layered durability, event/reaction handling,
effect requests and ObjectManager resolution against PANG-time snapshots.

A shot-id bridge fixes the prior `event_shot=None` limitation without replacing
HitInput or changing detector authority. Continuous object motion during the
shot-critical wait remains a later checkpoint.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 checkpoint — region-balanced physical hit proposals

V2.25.0 GameObject `shot_id` and frozen collision were physically verified. Detector
XY remained the blocker: five widely separated real shots were selected in one small
camera area. V2.25.1 adds balanced per-region proposal/confirmation on top of the
accepted V2.24.4 working-space mapping. Physical acceptance is pending a repeat of
the five-object test.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 checkpoint — registered PRE→POST authority

V2.25.0 shot-id/frozen GameObject resolution is physically verified. V2.25.1 region
proposal balancing is physically verified but did not correct final XY because local
authority still leaked through early/legacy candidates. V2.25.2 closes that authority
boundary. Physical five-shot acceptance is pending.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 checkpoint — cross-thread novelty authority

V2.25.0 shot_id/frozen GameObject resolution is physically verified. V2.25.1 per-region
proposal works. V2.25.2 registered evidence runs physically but its readiness bridge
was instance-local and its freshness gate remained too permissive. V2.25.3 fixes both
at the authority layer; five-shot physical acceptance is pending.

<!-- V2.25.3-r3 FULL_FILE_DELIVERY -->
## V2.25.3-r3 – full-file delivery

Packaging-only correction after the V2.25.3 runtime work. Future delivery for this development line uses complete replacement files only: no prepare/apply scripts and no menu/settings mutation helpers. `content/menu.json` is shipped as a complete schema version 1 file with the diagnostic games already present. Central configuration files are not replaced unless the version actually requires a source change.


## Evaluation foundation — measurement only

A versioned stage-observation scorecard, JSON/terminal CLI, provenance manifests,
and measurement selftests are available; see [EVALUATION.md](EVALUATION.md).
Historical V2.23 framepacks can measure saved candidate-pool recall at separately
reported 5/10/20/42 camera-pixel tolerances. They do not record complete stage
boundaries, effective producer settings/models/calibration, or final emissions.
Unavailable stages remain explicit. No detector behavior, parameters, models or
ranking changed. Full live-path replay and V2.25.3 physical acceptance remain
pending; projected F2 data is not physical validation.

The initial existing-data run covered 101 projected captures (100 F2, one single
projected): saved-pool recall was 1/101 @5 px, 2/101 @10 px, 6/101 @20 px and
26/101 @42 px. POST coverage was 93 with one frame, six with two and two with
three. These are historical snapshot measurements, not V2.25.3 detector results.

## Offline ten-iteration ranker research

Completed ten isolated native-F2 ranking trials on chronological 100-shot TRAIN,
100-shot DEVELOPMENT and 100-shot protected final holdout sessions. No live code,
settings, model registry or detector thresholds changed. Iteration 10 (geometry-only
linear ranker) won only the development MRR20 tie-break: Top1@20 stayed 0/100.
Holdout Top1/Top3/Top10@20 all remained 0/100; MRR20 improved, while Top1@42 regressed
1/100 to 0/100. No primary accuracy improvement or promotion is justified. Candidate
coverage remains the dominant limitation (6/100 holdout @20). See
[OFFLINE_10ITER.md](OFFLINE_10ITER.md); models and full JSON results are isolated under
`evaluation_runs/offline_10iter_20260907/`. V2.25.3 physical acceptance remains pending.

Iteration 10 is NOT approved for live use. All experiment data is projected F2
data and does not establish physical/live performance.

## Overnight physical audit and isolated challenger — 2026-09-07

On `codex/overnight-ai`, the biathlon audit assigns five holes to event groups
`{1,2},3,4,5,6`; event 2 is the likely redundant trigger 100.66 ms after event 1.
Original labels and traces remain untouched; ordinal-to-event mapping is external.
The first pair cannot be acoustically disambiguated without missing audio logs.
The second physical shot (event 3) has explicit local-confirmation evidence near
human GT but loses selection to the first hole. Legacy `state=confirmed` means
emitted; the saved top-eight track view is not a complete confirmation pool.

A frozen, SHADOW-only adapter consolidates the existing V2.23 linear listwise model.
Six historical conditional-ranking trials used chronological whole sessions;
stronger regularization gives only a small development MRR20 tie-break gain,
with Top1@20 unchanged (1/19 oracle-positive). Reused holdout Top1@20 remains 0/6.
No model/config is promoted to live authority. See `OVERNIGHT_AI_REPORT.md`
for the completed session report and caveats when available.

Measurement improvements transport worker pipeline snapshots, capture the actual
local-confirmation output and deterministic emission-boundary selection, and attach
per-event audio thresholds/cooldown evidence. Canonical scoring runs only when
finalizing the trace, with explicit retained-pool and post-decision semantics.
`automation.physical_test` supplies start/check/label/classify/evaluate/stop helpers.
All generated evaluations remain under ignored `evaluation_runs/`.

### Frozen confirmation selection shadow

The accepted development replay formula is now frozen in
`src/engine/ai/confirmation_selection_shadow.py` as
`CONFIRMATION_SELECTION_SHADOW` with status `PHYSICAL_REPLAY_CHALLENGER` and a
configuration hash. Physical traces record it beside `CURRENT_DETERMINISTIC` and
`CANONICAL_AI_SHADOW`; it cannot alter emitted coordinates. The physical-test
evaluator reports overall and conditional selector metrics plus retained-pool oracle
availability. The 10-shot replay remains development evidence; the next labelled
session is the independent validation dataset.

### Detector score root-cause audit

`automation.physical_score_audit` now decomposes the V2 score formula and reports
candidate-source distributions, per-shot selector comparisons, and track-score
evidence without changing runtime behavior. The 10-shot development trace shows
FAST V2.22.5 candidates saturated near 35–38 while genuine V2.6 vault candidates
are commonly below 3; the next research question is source-balanced ordering.
The frozen confirmation selector remains unchanged and shadow-only.

### Async track timestamp correction

A reproduced installer-composition regression let V2.22.6 overwrite V2.22.4's
camera-frame timestamp handoff and its “worker waiting is not a negative frame”
rule. The wrappers now share one timestamp-consumption helper; older ready results
also use the currently installed tracking method. Two failing-before/passing-after
regressions cover delayed-result association and waiting-frame aging. This changes
runtime timing semantics, not detector/audio thresholds or AI authority. Physical
acceptance remains pending; historical traces are not rewritten or claimed fixed.

## Temporal impact evidence — 2026-09-09

- **RESEARCH_ONLY:** temporal forensic extraction over the latest physical frames finds exploratory pairwise AUC about 0.71 for post-impact dark-residual persistence and 0.70 for impact-onset darkening. Before-frame stability alone is weak (about 0.58). The sample is small and not a validation result.
- **NOT PHYSICALLY VALIDATED:** no temporal ranker or synthetic-model change is frozen or promoted. Some true tracks have weak persistence at their saved representative coordinate, so coordinate drift/reference timing remain unresolved.

- **Observation-coordinate audit:** available association histories show only
  limited measurable drift (about 8–10 px in three examples); most physical
  tracks expose one coordinate. Confirmation-search best XY is not preserved
  for most rows, so observation-level temporal ranking remains unvalidated.

- **Final-ranking audit:** live eligible ordering is exactly
  `(onset_distance_to_peak, -track.best_score)`. For every seven @42-positive
  failure, onset is tied and the lower true-track `best_score` loses. Selected
  false-winner coordinates do not recur within 40 px across this ten-shot
  session; no recurrence penalty is justified.

- **Score comparability audit:** post-PRE FAST saturation is gone, but eligible
  source distributions remain materially different (FAST median 13.41, V1
  median 5.06). The seven losses are still decided by raw `best_score`; source
  normalization and common-evidence replays remain research-only.

- **Common-evidence research:** registered compactness is available for all
  973 eligible tracks and moves several truths upward (@10: 5/10, @20: 7/10),
  but no static common feature selects the seven failed events correctly.
  Temporal common evidence is only reconstructed for labelled pairs, so no
  common final ranker is ready to freeze.

- **Full-pool temporal replay:** causal temporal features are reconstructable
  for all 973 eligible tracks, with 4–8 POST frames before each saved live
  decision. Persistence reaches @5=3, @10=5, @20=7, but still selects the
  false winner in all seven failed pairwise events. No temporal ranker is ready.

- **Local-motion audit:** bounded local shifts are tiny for both true and false
  tracks (false median ~0.047 px), motion-explained residual fraction is near
  zero, and local alignment does not reverse the seven failures. Edge-shift
  residual is not supported as the dominant cause.

- **Long-horizon diagnostic:** post-decision frames extend to ~2.4 s for most
  events and next-event PRE exists for 9/10, but permanence does not separate
  the seven failures. Next-PRE center-dark ranks give only @5=3/@10=3; no
  wait-for-settle policy is justified.

- **Architecture review:** proposal scores are currently allowed to decide the
  final hit, although they are source-specific discovery scores. Static,
  temporal, motion and permanence replays do not provide a safe replacement;
  the highest-value next change is a research-only common verifier interface
  with richer physical patch hard negatives.

- **Common verifier implementation:** research-only `CommonFrameContext`,
  shared causal patch extraction and four rich source-independent information
  tests are implemented. Best texture-statistic replay reaches @10=3/@20=5;
  no verifier is physically validated or promoted. Current decision is that
  the architecture is sound but the physical training set is insufficient.

- **Historical physical mining:** 6 sessions / 61 traces were found. Five
  sessions are patch-compatible, yielding 47 usable positives and 39
  conservative selected-false negatives across 86 examples. Only the two
  post-fix sessions support complete final-rank replay. Session-held-out rich
  patch tests show domain shift; no verifier is ready to freeze.

- **Capture engineering:** domain audit identifies edge/gradient density as the
  clearest session shift. New tooling generates balanced manifests, scores
  trace quality, and rebuilds the research dataset. Recommended future data is
  3 independent sessions × 10 shots, with the third held out untouched.

- **Capture workflow hardened:** plan/session binding, validation-training guard,
  trace-quality scoring, finalization checks, and a practical operator runbook
  are implemented. Session C is explicitly `VALIDATION_UNTOUCHED`; no-impact
  events are separate from the ten physical shots.

- **Adversarial workflow hardening:** preflight now refuses reused bindings and
  reports actionable readiness; `physical_collection finalize` provides the
  guarded postflight entry point. Session metadata includes a deterministic
  plan identity, and validation tuning guards cover train/tune/fit/config
  selection operations.

## Physical S01 recovery

The 2026-09-09 S01 run was written into the historical biathlon5 root because the old settings trace root remained active. A separate recovered copy was created at `content/ai/physical_traces/session_20260909_S01_recovered_20260909`; it contains 11 timestamped runtime events and complete PRE/POST frame references. It has since been manually labeled: ten physical shots and nonphysical event 4. Event 11 has incomplete terminal persistence, so only nine physical decisions support final-selection metrics. The contaminated historical directory is preserved and must not be used as a clean session. `physical_collection start` reserves a unique root, configures runtime settings, records a byte-for-byte settings backup, and preflight verifies the actual configured root.
