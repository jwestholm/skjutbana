# Physical accuracy research — 2026-09-09

## 2026-09-10 continuation

The D01 evidence pass has now traced event 3 to its exact hybrid quota cap and
added observational diagnostics with regression coverage. Bounded observation
history restores event 1's 2.892 px coordinate: D01 pool oracle 1/3/5/5→2/4/5/5,
but selection stays 1/1/1/1 under the frozen common method and loses CURRENT's
event 2. Current physical outputs remain 0/1/1/1. The earlier flow 2/2/2/2 result
is still the best observed D01 selection, still rejected for its counterevidence.

Fixed 150+50 retention and 32-contour rescue fail event 5; simple confirmation
rankings select 0/6. Centered PRE preserves more D01 GT mass with no selection
gain and an S01 regression. Motion-only crop classification does not generalize.
NO_IMPACT remains unresolved. Repeated exact CURRENT replay covers all 39
complete D01/S01/S02/POST_FIX decisions. No algorithm is promoted and no S03 data
is accessed. The existing model references below are unchanged.

Detailed four-session tables, all per-event failures, current/history inventory,
36 passing suites and revised **six-event diagnostic D02 recommendation** are in
[D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md). No D02 was created/captured.
New immutable outputs: `/data/skjutbana/evaluation_runs/D01_evidence_20260910_081143/`.
The September 9 eight-event suggestion below is historical and superseded.

## Subsequent D01 pass

D01 is finalized: six physical events, all quality PASS, 6/6 exact recorded-input
replay and unchanged hashes. CURRENT @5/10/20/42 is 0/1/1/1; eligible oracle
1/3/5/5; frozen early common verification 1/1/1/1. Bounded contour rescue reaches
2/5/6/6 oracle but still selects 1/6. PRE-variability/persistence suppresses
67.91% of raw change mass with 40.06% GT-local retention; final selection remains
1/6. Best tested flow/augmented union selects 2/6 at all radii, with strong GT
attenuation, comparator regressions and unchanged no-impact failure. No promotion.

The original references/results below remain preserved. The extended cohort
contains 71 causal contexts (65 physical, six existing no-physical); D01 supplies
no new negative events and is never used to fit the tested models. New results,
per-event tables, architecture inventory, workflow regression fix, source plan,
test target and eight-event D02 recommendation are documented in
[D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md).
All outputs are under `/data/skjutbana/evaluation_runs/D01_accuracy_20260909_201752/`.
S03 was not opened, inspected, evaluated, tuned against or modified.

## Preserved earlier S02 research

**100% correct physical output is the target; 95% is the minimum acceptable
outcome. Neither has been achieved. CURRENT live authority is unchanged.**

This pass produced frozen **OFFLINE RESEARCH REFERENCES**, not a deployable
detector. The strongest useful finding is that common registered patch evidence
can improve actual offline selection, while broad proposal recall alone remains
misleading. No new end-to-end physical improvement has been validated.

All outputs, models, input hashes, failed attempts and reports are preserved at:

`/data/skjutbana/evaluation_runs/accuracy_95_100_20260909_174137/`

## Baseline and ground-truth integrity

The prior S02 workflow/evaluation changes were inspected and reproduced before
new research. A local commit was attempted using an explicit 15-file allowlist,
but `.git/index.lock` could not be created: `.git` is read-only. Nothing was
staged or committed. Exact copies of those 15 files are in
`completed_work_snapshot/`; settings and the trace symlink were excluded.

All thresholds below are **@5 / @10 / @20 / @42 camera pixels**, in that order.
The project's existing @42 acceptance boundary is preserved, including the
42.273624 px S02 miss. P95 uses the existing nearest-rank convention. Candidate
availability is diagnostic, never substituted for selected/emitted correctness.

| Physical baseline | Observable physical events | Causal candidate hits | CURRENT selected/emitted hits | Mean px | Median px | P95 px |
|---|---:|---|---|---:|---:|---:|
| S01 | 9 of 10 intended | 6/6/7/9 | 2/2/3/3 | 204.108808 | 205.897500 | 463.950857 |
| Previous post-fix comparator | 10 | 7/7/9/9 | 2/2/2/2 | 171.099411 | 135.748659 | 464.701006 |
| S02 | 10 | 3/3/3/5 | 1/1/1/1 | 187.926471 | 163.723005 | 426.563857 |

S01 event 11 lacks a complete terminal outcome; it is unavailable, not a measured
miss. S01's one no-physical event and all three S02 no-physical events emitted
false hits. Both existing frozen shadows remain 0/10 @42 on S02. Exact
recorded-input CURRENT replay matches all **13/13 S02** decisions and emissions.
S02 trace, frames, labels with external assignments, replay readiness, patch and
temporal checks pass. Native labels, trace files and settings remain unchanged.

S02's physical events are **1,3,4,5,6,7,9,11,12,13**; no-physical events are
**2,8,10**, approximately 1.49–1.50 seconds after preceding physical events.
This timing is not used as a rejection rule. The user subsequently reported that
event 1 might have produced two visible changes. That is **unconfirmed**. Its
single stored label is preserved; research also reports exclusion sensitivity.

S02 is explicitly **DEVELOPMENT_USED**, never independent validation of these
research choices. **S03 was not opened, inspected, evaluated, tuned against or
modified.** Tools use a five-session allowlist and reject S03 paths before reads.

## Cohort, causal contract and minimum event schema

Primary training/research sessions are H10 (`20260908_144822_d6713dec`), H20
(`20260908_163626_9b47fac6`), POST_FIX (`20260908_194746_b38de674`) and S01.
H20 events 7 and 15 are human-supported no-physical events. The cohort has
**59 physical examples and six no-physical events**, across 65 complete decision
contexts; the primary subset has 49 physical and three no-physical events.
H20 events 6 and 14 have recorded selector inputs but unavailable final emitted
coordinates. Historical comparisons involving them are **selected-point**
comparisons; the missing emissions are never filled with another event's output.

Each primary session is held out of model fitting in turn. The S02 model is
trained on all four primary sessions. Thus S01's held-out result and S02's result
come from different fitted models under the same method. S01 is training data
for the final S02 research reference, not independent validation of that model.

`physical_event_truth.py` provides an evaluation-only `physical-event-truth-1`
contract with `impacts: [...]`, camera coordinates, reasons and caveats. It
supports `SINGLE_IMPACT`, `NO_PHYSICAL_SHOT`, `AMBIGUOUS`, `UNKNOWN`, and a future
`MULTI_IMPACT` state. Unknown/ambiguous/multiple-impact cases are explicitly
excluded from the current single-output metric, never silently flattened.
Native capture labels are not migrated or rewritten. A discharge without a
defensible visible impact can remain unknown with a reason.

Common verifier feature provenance:

| Family | Source and normalization | Timestamp / causal availability | Coordinates | Live availability / source independence |
|---|---|---|---|---|
| Registered signed/absolute change, polarity | Recorded grayscale PRE/POST; global bounded phase-correlation translation; per-frame ring median and patch noise normalization | PRE before audio peak; POST at/before recorded decision and before next event | Full camera XY translated once by recorded crop origin | Frame-buffer inputs are possible live; independent of detector/source scores |
| Onset, persistence, consistency | Same patches across PRE history and up to ten causal POST frames; normalized inner-versus-ring contrasts | Explicit PRE/POST timestamps saved per event; no later POST used | Shared 49×49 patch, radii 2/4/8/16 | Capture-time causality proven; worker delivery parity is not proven |
| Morphology, scale and jitter | Residual mass offset/spread/elongation, dark/bright balance, local texture normalization, ±1 px jitter stability | Same causal frame context | Same geometry for every proposal | No GT, source identity, detector score or absolute XY is a feature |
| Event no-impact evidence | Median/90th/99th percentiles of nine common image features over bounded image proposals | Same decision cutoff | Event aggregate of camera patches | No audio interval/amplitude, old-hole distance, or winning source used |

There are 72 fixed patch features, signed-log compressed. The linear model uses
training-only mean/scale, fixed L2 regularization and a deterministic optimizer.
The nonlinear comparison is an OpenCV 96-tree depth-six forest. Training balances
sessions/events/classes; GT-centered patches supply supervised positives only.
Injected GT points are **never eligible for evaluation selection or oracle
recall**. Expanded proposals, including negatives, are generated without labels.

The research selector ranks the complete eligible track pool, applies the real
track readiness predicate and can return zero or one point. Expanded proposals
have explicit repeated-frame support. This is **one decision at its recorded
cutoff**, not a replay of the whole asynchronous resolver, known-hole remapping
or eventual timeout. An unready alternative at that cutoff is not evidence of
an eventual live false rejection. No trial below had an unready S02 winner.

## Experiments and decisions

All experiments were offline, causal in captured image time, and used the same
physical labels. The initial four hypotheses are preserved in
`experiment_protocol.json`. Later interventions followed observed failures,
with fixed rules applied to whole sessions, without shot-specific tuning.

| Hypothesis / intervention | Result | Decision |
|---|---|---|
| H1: learn combinations of registered multiscale, onset, polarity, morphology and jitter evidence on current eligible coordinates | Snapshot logistic: primary held-out 10/10/10/11 of 49; S02 3/3/3/3 of 10. Forest: primary 3/3/3/5; S02 2/2/2/2 on current coordinates, and expansion degrades it | Keep logistic as frozen offline reference; no promotion |
| H2: bounded two-polarity, three-scale persistent-change maxima, spatial quotas, 256 cap, same frozen verifier | Snapshot union S02 oracle 3/3/3/6; selected 3/3/3/3. Expanded-only selects 0/10 | Reject as an accuracy solution; keep reproducible proposal diagnostic |
| H3: fixed earlier PRE median, −0.8 to −0.4 s, keeping the same features/model families | Early-reference union S02 oracle 3/3/5/9; selected 3/3/3/3. On current tracks, S01 held-out 5/5/5/5 of 9 and S02 3/3/3/3 | Keep second frozen offline reference; no live PRE timing change |
| H4: threshold the best patch score using primary out-of-fold physical scores | Logistic rejects 0/3 S02 false events, with 0/10 physical rejections | Reject as sufficient no-impact verification |
| H5: retrain with broader expanded negatives, retaining session holdouts | Snapshot logistic current S02 falls to 2/10; early-reference union is also 2/10. No robust no-impact gain | Reject |
| H6: also restrict the PRE noise estimator to the same earlier interval | S02 union oracle 3/3/6/9, but logistic selected 1/1/1/1; primary current 8/8/8/9 | Reject; late PRE variation contains useful discrimination in this dataset |
| H7: within-event pairwise training, then two rounds of top-32 hard-negative mining | Current S02 remains 3/10; the tested threshold rejects 1/3 false events with no physical rejection. Primary current hard-mined 8/8/9/11 of 49 | Keep negative/limited result; no reliable superiority to common logistic. Threshold was calibrated on the union pool, so current-pool transfer is diagnostic |
| H8: separate visual-only event gate from common patch distributions | Balanced gate rejects 1/3 S02 false events and 4/10 physical events. Primary-calibrated alternative rejects 3/3 false events and 5/10 physical events | Reject: physical false rejections are unacceptable |
| H9: rescue bounded raw contour locations before later destructive filtering | S02 union oracle 3/4/4/7; selected 3/3/3/3. Up to 256 rescue coordinates; unavailable/stale masks excluded explicitly | Reject as an accuracy solution; preserve the audit and trace fix |

Full per-session thresholds, errors, TopK, counts and no-impact metrics for every
trial are in the corresponding `verifier_*/results.json`,
`pairwise_history_early/results.json` and `no_impact_history_early/results.json`.
The guarded-reference, broad-negative, forest and contour trials are not hidden
behind the best S02 result. No synthetic example is used as physical validation.

## Frozen early-reference method: actual offline selection

| Session | Physical n | CURRENT selected @5/10/20/42 | Common verifier selected @5/10/20/42 | Eligible oracle @5/10/20/42 | Conditional @42 | Mean / median / P95 error px |
|---|---:|---|---|---|---|---|
| H10 held out | 10 | 1/1/1/1 | 2/2/2/2 | 6/7/7/9 | 2/9 | 236.142612 / 106.967203 / 817.171870 |
| H20 held out | 20 | 1/1/1/1 | 0/0/0/1 | 10/10/10/12 | 1/12 | 409.021967 / 449.150063 / 609.787627 |
| POST_FIX held out | 10 | 2/2/2/2 | 3/3/3/3 | 6/6/9/9 | 3/9 | 195.744312 / 165.108462 / 490.276392 |
| S01 held out | 9 | 2/2/3/3 | 5/5/5/5 | 6/6/7/9 | 5/9 | 162.296926 / 4.300208 / 463.950857 |
| S02 development challenge | 10 | 1/1/1/1 | 3/3/3/3 | 3/3/3/5 | 3/5 | 143.010361 / 126.152024 / 385.398431 |

The POST_FIX eligible track representatives have 6 rather than 7 @5/@10 hits;
the original causal proposal oracle remains 7/7/9/9. These are different stages,
not a changed baseline. The unchanged CURRENT selectors reproduce exactly.

Primary held-out selection improves from **7/49 to 11/49 @42**. Adding S02 as
development evidence gives **8/59 to 14/59**, or **13.56% to 23.73%**. This remains
76.27 percentage points below the 100% target and 71.27 below the 95% minimum.
Across all 65 known events, the offline correct-result count is 14/65
(21.54%), including zero correct no-impact rejections. The H20 tail error
worsens substantially, so this is not a general production improvement. Snapshot-reference logistic has the same aggregate @42 count and
lower primary mean error; both references are retained with explicit identity.

The primary session bootstrap gives an observed +8.163265 percentage-point
gain and a descriptive 95% interval of +1.428571 to +18.918919 points. Including
development-used S02 gives +10.169492 points, interval +3.750000 to +19.148936.
Only four/five sessions and development model selection underpin these intervals;
they do not establish independent physical generalization.

S01 common-verifier Top1/3/5/10/20/50 @42: **5/7/8/8/8/9**.
S02: **3/3/3/3/3/3**. The five S02 oracle-positive events have new correct-track
ranks **1,1,1,98,56** for events 1,3,9,11,12. CURRENT ranks the four lost positive
tracks at approximately 71,101,110,46. The new model recovers events 3 and 9;
it still fails events 11 and 12 and all five original proposal misses.

Excluding possibly ambiguous S02 event 1 leaves CURRENT **0/9**, common verifier
**2/9** at every threshold, eligible oracle **2/2/2/4**. The new mean/median/P95
are **158.804144 / 147.354973 / 385.398431** px. This is sensitivity analysis,
not a relabeling or a replacement headline result.

Frozen manifests: `frozen_snapshot.json` and `frozen_history_early.json`.
Early-reference parameter SHA-256:
`3cd60866944f6de82500e149aec18bc0ca04f3cece014fa9ed1be62dea4db546`.
Both produce the same 13 S02 selected-point decisions with
SHA-256 `be3439eb0d0ef8fd1de781e9b08ec2912105d142f2a42fe6ffdb125095a7ee87`.
Repeated hash-checked replay is deterministic. These files are not live manifests.

## Proposal ceilings and burden

| Session | Current eligible oracle | Earlier-reference expanded-only oracle | Union oracle | Union selected | Mean / maximum union candidates |
|---|---|---|---|---|---|
| H10 | 6/7/7/9 | 4/4/5/9 | 7/8/8/10 | 2/2/2/2 | 290.9 / 325 |
| H20 | 10/10/10/12 | 6/10/11/15 | 12/14/14/19 | 0/0/0/0 | 328.0 / 360 |
| POST_FIX | 6/6/9/9 | 4/5/8/9 | 7/7/10/10 | 1/1/1/1 | 285.3 / 300 |
| S01 | 6/6/7/9 | 3/5/7/9 | 7/7/9/9 | 4/4/4/4 | 257.0 / 300 |
| S02 | 3/3/3/5 | 2/3/5/9 | 3/3/5/9 | 3/3/3/3 | 315.153846 / 335 |

S02 expansion averages 184.384615 extra candidates (maximum 196), versus
130.769231 eligible current tracks. The method is bounded by 256, not every
pixel. However, a count-matched uniform spatial null on the actual ROI/labels
averages **0.148/0.660/2.348/7.082** oracle hits on S02; its @42 95% range is
4–10. Thus **9/10 @42 alone does not prove projectile-specific proposal quality**.
This null is a coverage diagnostic, not synthetic validation truth.

One S01 benchmark: context loading/registration **747.054 ms**, 256-budget
proposal extraction **1091.670 ms**, features for 97 current tracks **293.951 ms**,
linear prediction **0.131 ms**, process peak RSS **527.5 MiB**. S02 full union
extraction averages context **704.083 ms**, proposals **900.137 ms**, features
**925.623 ms**. These unoptimized Python/disk timings do not meet live latency
requirements. Classifier inference is cheap; image evidence extraction is not.

Original S02 ceiling is 5/10 @42. The best tested expanded ceiling is 9/10, so
even perfect selection over that pool remains below 95%. The tested combination
is 90% proposal recall × 33.33% conditional selection = **30% actual selection**.
Both subsystems must approach 100%; even 98% × 98% is only 96.04%.

## Five S02 proposal misses: what is and is not observable

All five GT points are inside the saved detector ROI and correctly translated
camera crop. Available V2 registration shifts are below 0.03 px, nowhere near
the 42 px error scale. The values below use radius-four disks at the unchanged
label; a weak difference is not proof of no physical discharge or a wrong label.

| Event / physical shot | Nearest retained px | Saved PRE delta max | Nearest raw contour px (area) | Nearest broad geometric contour px | Earlier-reference rescue nearest px | Supported diagnosis |
|---|---:|---:|---|---:|---:|---|
| 4 / 3 | 47.883879 | 1 | 22.334945 (0) | 47.883879 | 46.397 | Only degenerate nearby raw contour; no area≥2 contour within 42. No measured ROI/crop loss; weak local change |
| 5 / 4 | 76.150322 | 0 | 11.327185 (0) | 21.519467 | 28.848 | A nearby nondegenerate raw contour exists but disappears before retained output; exact filtering/cap stage unavailable |
| 6 / 5 | 52.636643 | 1 | 40.333213 (0) | 42.570731 | 39.666 | Degenerate nearest raw contour; no broad geometric contour within 42. Retained pool has no FAST candidate; V2 telemetry absent |
| 7 / 6 | 51.421461 | 0 | 15.996072 (0) | 51.421461 | 20.516 | Nearby raw contours are degenerate; no area≥2 contour within 42. Weak local change, not a proven later ranking loss |
| 13 / 10 | 42.273624 | 4 | 21.740674 (4) | 21.740674 | 28.249 | A nearby valid contour exists but is lost before retention; exact patch/cap/cleanup cause unavailable |

Nearest retained FAST points for events 4/5/7/13 are respectively
118.401/114.585/83.520/45.089 px away. Raw FAST and discarded V2 proposals were
not saved, so it is impossible to say whether a correct point existed before
their own caps. Event 6's recorded source frame is +4.324 ms after the trigger;
this is consistent with the V2 minimum-age gate, but unavailable configuration
and telemetry prevent declaring that the uniquely proven cause.

Legacy generated/kept counts are 322/200, 324/200, 267/200, 296/200, 335/200.
Subsequent ridge removals are 38,25,52,41,49; stale-known removals are 0,0,2,2,1.
Those counts prove filtering and caps occurred. They **do not identify which
discarded coordinate was removed by which rule**. In particular, old-hole
suppression is excluded for event 5 by its zero stale-known count, but not
uniquely established for event 13. Cross-shot novelty, raw FAST retention and
exact onset remain partly unidentifiable from these saved artifacts.

The raw contour audit reconstructs geometry from saved causal masks, not missing
scores or rejection histories. `S02_miss_chronology.png` additionally shows
earlier/later images for visual forensics only; the later frames are not used in
features. Some changes appear in PRE image history, and some labeled positions
have little visible change in the recorded window. This does not authorize
silently revising labels, assuming one visible hole per discharge, or installing
a universal earlier-PRE timing offset. A rapid real shot must remain possible.

**Real trace bug fixed:** V2.22.2 cleanup replaced the upstream contour/filter
ledger. Capture now preserves a copied upstream ledger with crop provenance and
camera-space input/after-novelty/after-ridge/retained snapshots. An early upstream
return cannot borrow the previous frame's ledger. The complete hybrid RAW pool
remains explicitly unavailable where not instrumented. No candidate score,
ordering, eligibility predicate or live selector was changed by this fix.

## No-impact accounting

| S02 method | Correct physical @42 | Any physical selection | True negatives | False emissions | Physical false rejections |
|---|---:|---:|---:|---:|---:|
| CURRENT | 1/10 | 10/10 | 0/3 | 3/3 | 0/10 |
| Frozen common logistic | 3/10 | 10/10 | 0/3 | 3/3 | 0/10 |
| Pairwise current-pool threshold diagnostic | 3/10 | 10/10 | 1/3 | 2/3 | 0/10 |
| Visual event gate, balanced | 2/10 | 6/10 | 1/3 | 2/3 | 4/10 |
| Visual event gate, primary-calibrated | 2/10 | 5/10 | 3/3 | 0/3 | 5/10 |

The common method also falsely selects all three primary no-physical events.
Rejections remain in the physical accuracy denominator. Mean/median/P95 after
rejection describe only emitted points and must be read with the rejection count.
The visual gate's apparent false-emission improvement is bought with too many
real-shot rejections. Three primary negative events cannot establish robust
visual rejection across conditions. None of these gates is promoted.

## Reproduction

Use new output paths. Existing baselines must not be overwritten.

```bash
export OPENBLAS_NUM_THREADS=1
RUN=$(mktemp -d /data/skjutbana/evaluation_runs/accuracy_repeat_XXXXXXXX)
BASE=/data/skjutbana/evaluation_runs/S02_finalization_20260909_170523
python3 "$BASE/analyze_sessions.py" --repo "$PWD" --output "$RUN/baseline"
for REF in snapshot history_early history_guarded; do
  python3 -m automation.accuracy_physical_dataset --reference "$REF" --output "$RUN/dataset_$REF"
  python3 -m automation.accuracy_verifier_research --dataset "$RUN/dataset_$REF" --output "$RUN/verifier_$REF"
done
python3 -m automation.accuracy_verifier_research --dataset "$RUN/dataset_snapshot" --train-pool union --output "$RUN/verifier_broad_negatives"
python3 -m automation.accuracy_verifier_research --dataset "$RUN/dataset_history_early" --train-pool union --output "$RUN/verifier_early_broad"
python3 -m automation.accuracy_pairwise_research --dataset "$RUN/dataset_history_early" --output "$RUN/pairwise_history_early"
python3 -m automation.accuracy_no_impact_research --dataset "$RUN/dataset_history_early" --output "$RUN/no_impact_history_early"
python3 -m automation.accuracy_contour_rescue --source "$RUN/dataset_history_early" --output "$RUN/dataset_contour_rescue"
python3 -m automation.accuracy_verifier_research --dataset "$RUN/dataset_contour_rescue" --output "$RUN/verifier_contour_rescue"
python3 -m automation.accuracy_proposal_forensics --output "$RUN/proposal_forensics"
python3 -m automation.accuracy_research_analysis --root "$RUN" --output "$RUN/research_analysis"
FROZEN=/data/skjutbana/evaluation_runs/accuracy_95_100_20260909_174137
python3 -m automation.accuracy_frozen_replay --manifest "$FROZEN/frozen_history_early.json" --output "$RUN/frozen_replay"
```

## Next evidence and promotion boundary

The pass stops with a reproducible, modestly stronger **frozen research method**,
fully characterized as far below practical success. It does not stop at an
untested first hypothesis. Ranking, proposal and no-impact interventions were
measured separately; failed variants are retained as research records.

The smallest useful new physical capture is a **12-event diagnostic session:
six deliberate discharges and six explicitly observed no-impact events**.
Cover light/dark/printed-edge regions, a nearby pair, and a taped/repeated region;
include both isolated false transients and ones following real shots. Record a
continuous view spanning PRE/onset/POST and an independent discharge/event log,
with each newly visible change identified immediately. Preserve exposure/frame,
audio-event and worker-delivery timestamps and the new rejection ledger. Label
unresolvable changes UNKNOWN/AMBIGUOUS, rather than inventing one impact per
trigger. This targets missing causal/newness information and negative diversity;
it is **not** a 95% validation exercise. S03 remains reserved.

This capture is useful because overwritten per-coordinate rejection evidence,
unconfirmed multiple-change truth and independent physical onset cannot be
recovered from the existing traces. Do not use it to tune around specific S02
coordinates. Continue common evidence research against the frozen references,
with whole-session splits and explicit no-impact accounting. Do not deploy an
earlier reference, expanded pool or rejection gate from this pass.

Once a complete frozen candidate approaches 100%, validate on substantially more
independent physical data: at least 50, preferably 100+ diverse shots plus
no-impact events. With zero failures, 59 independent trials are the minimum for
a one-sided exact 95% lower bound above 95% (`0.05**(1/59) > 0.95`); this does
not replace scenario diversity or separate no-impact tests. A ten-shot S03 alone
cannot establish practical success. Freeze before its eventual one-time use.

## Verification and preserved files

Final verification: **176 unittest cases and 104 explicit checks across 30 suites, all PASS**. The initial preserved-work run also reproduced 99 passing tests across 11 suites. Changed/new Python files compile (22 files), and `git diff --check` passes. S02 finalization recheck is FINALIZED with all six quality fields PASS. CURRENT recorded-input replay remains S01 10/10, S02 13/13 and POST_FIX 10/10. Two frozen-verifier repeats produce identical 13-event output hashes.

| Exact suite (`python3 -m ...`) | Cases/checks | Result |
|---|---:|---|
| `automation.evaluation_selftest` | 13 tests | PASS |
| `automation.physical_label_reset_selftest` | 15 tests | PASS |
| `automation.physical_postflight_selftest` | 12 tests | PASS |
| `automation.physical_collection_selftest` | 9 tests | PASS |
| `automation.physical_workflow_adversarial_selftest` | 5 tests | PASS |
| `automation.physical_capture_tools_selftest` | 2 tests | PASS |
| `automation.physical_trace_label_selftest` | 6 tests | PASS |
| `automation.physical_trace_selftest` | 4 tests | PASS |
| `automation.track_audit_selftest` | 15 tests | PASS |
| `automation.overnight_selftest` | 15 tests | PASS |
| `automation.common_verifier_selftest` | 3 tests | PASS |
| `automation.physical_trace_stress_selftest` | 3 tests | PASS |
| `automation.async_track_timing_selftest` | 3 tests | PASS |
| `automation.causal_candidate_selftest` | 10 tests | PASS |
| `automation.candidate_root_cause_selftest` | 2 tests | PASS |
| `automation.physical_dataset_selftest` | 2 tests | PASS |
| `automation.registered_impact_selftest` | 12 tests | PASS |
| `automation.temporal_impact_selftest` | 5 tests | PASS |
| `automation.physical_patch_verifier_selftest` | 2 tests | PASS |
| `automation.full_pool_temporal_selftest` | 3 tests | PASS |
| `automation.common_impact_ranking_selftest` | 2 tests | PASS |
| `automation.accuracy_research_selftest` | 22 tests | PASS |
| `automation.proposal_trace_selftest` | 4 tests | PASS |
| `automation.score_comparability_selftest` | 2 tests | PASS |
| `automation.pre_spatial_mapping_selftest` | 5 tests | PASS |
| `automation.v2221_selftest` | 14 explicit checks | PASS |
| `automation.v2222_selftest` | 23 explicit checks | PASS |
| `automation.v2224_selftest` | 24 explicit checks | PASS |
| `automation.v2225_selftest` | 21 explicit checks | PASS |
| `automation.v2226_selftest` | 22 explicit checks | PASS |

Detailed logs and exit codes are in `tests/`; compilation inputs are in `compilation.json`. Tests cover label-only reset safety, immutable evidence, event truth/denominators, no GT injection, S03 refusal, causal cutoffs, coordinate translation, registration, bounded proposals, unknown rescue availability, model determinism, frozen-hash tampering, trace transport, cleanup boundaries and adversarial workflows. The new cleanup regression first exposed a test assertion assuming candidate order; it was corrected to identify the candidate by coordinates, with all four regression cases passing.

The preservation audit verifies all **1,664 S02 files**, every extracted development trace/label hash, the original settings hash, and the intended trace symlink. No trace session was deleted. Git is still on `codex/score-root-cause`; `.git` was read-only. No commit, push, merge or dev/main modification occurred. `content/ai/settings.json` and `content/ai/physical_traces` remain pre-existing local changes and are excluded from staging instructions.

Changed/new files (including the preserved earlier S02 work):

- `ACCURACY_95_100_RESEARCH.md`
- `COMMON_VERIFIER_ARCHITECTURE.md`
- `CURRENT_STATE.md`
- `PHYSICAL_CAPTURE_RUNBOOK.md`
- `PHYSICAL_COLLECTION_PLAN.md`
- `S02_PHYSICAL_FINDINGS.md`
- `SCORE_ROOT_CAUSE.md`
- `automation/accuracy_contour_rescue.py`
- `automation/accuracy_frozen_replay.py`
- `automation/accuracy_no_impact_research.py`
- `automation/accuracy_pairwise_research.py`
- `automation/accuracy_physical_dataset.py`
- `automation/accuracy_proposal_forensics.py`
- `automation/accuracy_research_analysis.py`
- `automation/accuracy_research_selftest.py`
- `automation/accuracy_verifier_research.py`
- `automation/physical_collection.py`
- `automation/physical_collection_selftest.py`
- `automation/physical_finalize.py`
- `automation/physical_label_reset.py`
- `automation/physical_label_reset_selftest.py`
- `automation/physical_postflight_selftest.py`
- `automation/physical_test.py`
- `automation/physical_trace_quality.py`
- `automation/physical_workflow_adversarial_selftest.py`
- `automation/proposal_trace_selftest.py`
- `src/engine/camera/hit_scanner_v2222.py`
- `src/engine/offline/accuracy_verifier.py`
- `src/engine/offline/physical_event_truth.py`

To create a local commit from the reviewed final tree outside the read-only Git sandbox, the exact allowlist commands are also saved in `commit_reviewed_work.sh` under the research output directory. They exclude settings and the physical trace symlink:

```bash
if [ "$(git branch --show-current)" != "codex/score-root-cause" ]; then
  echo "Refusing commit: wrong branch." >&2
  exit 1
fi
if [ -n "$(git diff --cached --name-only)" ]; then
  echo "Refusing commit: inspect existing staged changes first." >&2
  exit 1
fi
git add -- \
  ACCURACY_95_100_RESEARCH.md \
  COMMON_VERIFIER_ARCHITECTURE.md \
  CURRENT_STATE.md \
  PHYSICAL_CAPTURE_RUNBOOK.md \
  PHYSICAL_COLLECTION_PLAN.md \
  S02_PHYSICAL_FINDINGS.md \
  SCORE_ROOT_CAUSE.md \
  automation/accuracy_contour_rescue.py \
  automation/accuracy_frozen_replay.py \
  automation/accuracy_no_impact_research.py \
  automation/accuracy_pairwise_research.py \
  automation/accuracy_physical_dataset.py \
  automation/accuracy_proposal_forensics.py \
  automation/accuracy_research_analysis.py \
  automation/accuracy_research_selftest.py \
  automation/accuracy_verifier_research.py \
  automation/physical_collection.py \
  automation/physical_collection_selftest.py \
  automation/physical_finalize.py \
  automation/physical_label_reset.py \
  automation/physical_label_reset_selftest.py \
  automation/physical_postflight_selftest.py \
  automation/physical_test.py \
  automation/physical_trace_quality.py \
  automation/physical_workflow_adversarial_selftest.py \
  automation/proposal_trace_selftest.py \
  src/engine/camera/hit_scanner_v2222.py \
  src/engine/offline/accuracy_verifier.py \
  src/engine/offline/physical_event_truth.py
git diff --cached --check
git commit -m 'Preserve S02 evaluation and offline physical accuracy research'
```
