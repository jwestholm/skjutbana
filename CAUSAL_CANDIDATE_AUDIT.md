# Causal candidate audit — 2026-09-08

## Evidence status and immutable validation

**PROVEN** means supported by source plus recorded/reproduced evidence.
**STRONG HYPOTHESIS** means repeated evidence without complete causal isolation.
**SPECULATION** means plausible but unproven. This distinction applies throughout.

Dataset: `content/ai/physical_traces/session_20260908_163626_9b47fac6`.
Original evaluation: `evaluation_runs/physical_20260908_172039_ae9d026f`.
Branch: `codex/score-root-cause`. Capture source commit:
`e7f9f029de22ae0e741634b67fbcc93d122e9054`.

The shooter fired exactly 20 shots, deliberately at dart sectors 1 through 20
in order. Manual clicks are precise. Audio generated 22 events. Events 7 and 15
are independently classified `NO_PHYSICAL_SHOT`, respectively 1.4839297 and
1.4904709 seconds after the preceding real shot. Event mapping is 1–6 → physical
1–6; 8–14 → physical 7–13; 16–22 → physical 14–20. These facts are not relabelled
by this audit. All original traces, clicks, assignments and evaluations remain
unchanged. Subsequent experiments use this dataset as DEVELOPMENT data.

The original `physical_comparison.json` says **INDEPENDENT_PHYSICAL_VALIDATION**:
selector hash matched, source commit matched, `labels_before_selection=[]`,
canonical manifest matched, physical `event_count=20`.

Frozen confirmation hash:
`123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f`.

| Frozen selector | Top1 @42 | Mean px | Median px | P95 px | Errors >100 px |
|---|---:|---:|---:|---:|---:|
| CURRENT_DETERMINISTIC | 1/20 (5%) | 509.362 | 475.003 | 876.595 | 19 |
| CONFIRMATION_SELECTION_SHADOW | 1/20 (5%) | 415.914 | 418.574 | 667.261 | 19 |
| CANONICAL_AI_SHADOW | 1/20 (5%) | 557.925 | 583.582 | 850.488 | 19 |

The confirmation shadow reduces geographical error but does not improve Top1.
It is not promoted. Its source/configuration is unchanged. The canonical model
remains `SHADOW` / `OFFLINE_CHALLENGER`, identity transform, manifest SHA256
`63a366d984c56e3fe65cdc62efd5db7cc9e74d3cd3a6307068b0b849ddd02b86`;
model.json SHA256 `2078c398f141e8f7f63e52849697f163552165878602510e17408b5e9ce1678e`;
model.npz SHA256 `1c58d31d44ec46d7db5f7aa3281ed7c7314c61c1f36f1c3c8475d4282bb10305`.

## What was read and measured

The audit parsed every trace, stage, pipeline pool, confirmation output, debug
track, decision, outcome, frame entry and audio diagnostic for all 22 events.
It read the original evaluation trace, comparison, metrics, false-event and health
files, setup/assignments, and all 460 runtime-log lines. It loaded and checked
all **2,737 NPY artifacts** (full arrays for shape, dtype, range and finiteness),
including PRE/POST frames and evidence maps. There are 1,668 scanner observations,
1,563 belonging to physical-shot traces. All 22 traces report complete persistence;
health reports healthy. Completeness does not imply causal purity or accuracy.

RAW is unavailable. FILTERED/RETAINED snapshots and local-confirmation output are
available; repeated pipeline copies are not independent generator runs. The log
shows 22 CV results, 23 local-confirmation rounds (event 12 has two), 16 FAST
extractions (15 physical plus false event 15), and no FULL-RESCUE execution.
`v2_rescue_temporal` means FAST temporal rescue proposals, not FULL-RESCUE.
All events emitted, including both false audio events. The original final outcome
loses emitted XY for events 6 and 14 after later debug overwrite; their decision
snapshots and HIT log lines still exist. This explains why generic historical
emission metrics differ from the complete deterministic-selector metrics.

Reproducible new output, never replacing a baseline:

```bash
python3 -m automation.causal_candidate_audit \
  --root content/ai/physical_traces/session_20260908_163626_9b47fac6 \
  --output evaluation_runs/causal_audit_<new-id>
python3 -m automation.causal_temporal_research \
  --root content/ai/physical_traces/session_20260908_163626_9b47fac6 \
  --output evaluation_runs/causal_temporal_<new-id>
```

Actual audit output: `evaluation_runs/causal_audit_20260908/full_v1/`:
`audit.json`, `event_01.json` … `event_22.json`, `artifacts.json`, `timeline.md`.
Each candidate feature version has classification, first observed time and all
origin references. Every stage has a timeline and registration/reference fields.
`supplement.json` in the parent directory records original evaluation hashes,
observation counts and per-winner statistics. Research outputs are separately
versioned; final experiment is `temporal_final_v4/`.

## Causal cutoff and candidate availability

**PROVEN:** `bootstrap.wrapped_emit()` calls `capture_decision()` immediately
before the AI/emission hook. `decision_input.timestamp` is a wall-clock delivery
boundary, whereas a candidate's `timestamp` is a camera evidence timestamp.
A worker can deliver an old frame after the decision; frame age alone is not proof
of availability. Advisory AI does not override these decisions.

`src/engine/offline/causal_candidates.py` defines:

- `CAUSALLY_AVAILABLE`: explicitly in the decision pool/winning track, or in an
  event-owned scanner observation before the boundary. Historical synchronous
  confirmation outputs are also provable when shot id and frame time equal the
  winning track's captured last-seen frame. That confirmation precedes tracking
  and resolution in the same scanner call. It is not inferred from track state.
- `POST_DECISION`: same-event evidence first recorded after the boundary without
  proof it was already consumed; or evidence timestamp beyond the boundary.
- `CROSS_EVENT_OR_FUTURE`: another event owns the pool, or its evidence frame is
  at/after the next audio peak. This takes precedence over a misleading timestamp
  on a copied decision pool. Copies of an earlier proven candidate retain their
  earlier availability; new feature/timestamp versions are not backdated.
- `UNKNOWN`: missing boundary/delivery/ownership evidence prevents proof. A saved
  coordinate alone does not establish when its later features were available.

Full candidate feature versions are deduplicated exactly; these counts are not
unique holes. Tracks retain their own coordinates, distinct from proposal XY.
Legacy debug tracks are only the top eight. `state=confirmed` means emitted,
not that the track passed V2.22.5 local confirmation.

| Oracle (20 physical shots) | @5 | @10 | @20 | @42 |
|---|---:|---:|---:|---:|
| **Causal available** | **10/20 (50%)** | **10/20 (50%)** | **10/20 (50%)** | **12/20 (60%)** |
| Decision retained pool alone | 10/20 | 10/20 | 10/20 | 12/20 |
| Posthoc all observed | 11/20 | 12/20 | 12/20 | 14/20 |
| Historical last-observed retained | 11/20 | 12/20 | 12/20 | 14/20 |
| Post-decision-only feature versions | 0/20 | 0/20 | 0/20 | 0/20 |
| Cross-event-only feature versions | 1/20 | 2/20 | 2/20 | 2/20 |

These sets need not be disjoint in space. A zero post-decision-only oracle does
not mean there were no repeated observations after selection. **The original
selector comparison already used the decision pool and reported 12/20 @42.**
The misleading 14/20 comes from `last_observed_retained`, not an error requiring
rewriting the independent selector result. Generic `metrics.json` stage pools
also differ from the unowned last-observed candidate list.

Of physical observations, **1,136/1,563 (72.7%)** occur after the decision.
27 observations occur at/after the next audio event; five nonempty candidate
observations have the other event's owner. 428 saved POST frames have timestamps
beyond their original decision; 23 are at/after the next event.
Physical candidate-feature-version totals are 14,744 causal, 188 post-decision,
805 cross-event and zero unknown. Thus 993/15,737 (6.3%) versions lack causal
availability. Annotation/track feature variants explain why version counts
exceed the 3,646 decision-retained proposals; do not call them independent holes.

## Exact timelines

Decision +s is wall-clock capture boundary. Candidate/track frame time can be
substantially earlier. `Last` and `All posthoc` coincide on nearest distance.
|Event|Physical|Decision +s|Causal nearest px|Last retained px|Posthoc all px|CURRENT px|Source|
|---|---|---|---|---|---|---|---|
|1|1|0.434|88.340|88.340|88.340|154.976|V26_VAULT|
|2|2|0.770|3.642|3.642|3.642|447.053|FAST_V2225|
|3|3|0.664|2.409|2.409|2.409|770.712|FAST_V2225|
|4|4|0.755|65.153|65.153|65.153|496.851|FAST_V2225|
|5|5|0.831|3.239|3.239|3.239|729.576|FAST_V2225|
|6|6|0.820|44.126|5.442|5.442|321.707|FAST_V2225|
|7|FALSE|0.548|—|—|—|—|V26_VAULT|
|8|7|0.663|4.151|4.151|4.151|756.750|FAST_V2225|
|9|8|0.698|2.769|2.769|2.769|834.152|FAST_V2225|
|10|9|0.760|3.896|3.896|3.896|860.426|FAST_V2225|
|11|10|0.471|1.719|1.719|1.719|1.719|V26_VAULT|
|12|11|0.486|32.598|32.598|32.598|192.084|V26_VAULT|
|13|12|0.407|43.502|43.502|43.502|379.676|V26_VAULT|
|14|13|0.672|178.177|4.183|4.183|319.020|FAST_V2225|
|15|FALSE|1.170|—|—|—|—|FAST_V2225|
|16|14|0.726|69.054|69.054|69.054|882.737|FAST_V2225|
|17|15|0.709|107.794|107.794|107.794|396.795|FAST_V2225|
|18|16|0.708|77.324|77.324|77.324|876.595|FAST_V2225|
|19|17|0.695|2.796|2.796|2.796|559.266|FAST_V2225|
|20|18|0.784|0.740|0.740|0.740|453.156|FAST_V2225|
|21|19|0.480|40.167|40.167|40.167|121.862|V26_VAULT|
|22|20|0.678|2.724|2.724|2.724|632.129|FAST_V2225|

Event 6: peak `1788885626.4964387`; winning confirmation frame
`1788885627.174298` (+0.677859 s); decision `1788885627.3162303` (+0.819792 s).
Event 7 peak `1788885627.9803684`. The 5.441649 px retained candidate has frame
`1788885627.982724` (+1.486285 s), 2.356 ms after event 7. Its pool owner is 7.
The actual causal nearest proposal is **44.126051 px**, outside 42.

Event 14: peak `1788885752.6620774`; winning confirmation frame
`1788885753.2575214` (+0.595444 s); decision `1788885753.3337026` (+0.671625 s).
Event 15 peak `1788885754.1525483`. The 4.182798 px candidate has frame
`1788885754.43405` (+1.771973 s), 281.502 ms after event 15. Its pool owner is 15.
The causal nearest proposal is **178.177061 px**.

Events 7 and 15 themselves decide at `1788885628.5280342` and
`1788885755.3230295`. Both emit false hits, without physical GT. Their local
confirmation does not establish a real gunshot. These independent event records
are included in the audit but excluded from 20-shot accuracy denominators.

## Ownership audit: observed contamination and a separate runtime defect

**PROVEN A — diagnostic contamination in this dataset.**
`PhysicalTraceRecorder.observe_scanner()` loops over every event still in
`scanner.audio_events`, including matched events. For each it copies shared
`last_candidates`, tracks/debug maps and any history frames newer than its peak.
It gates `pipeline` and confirmation by shot id but historically did not gate
`candidates`. `finish()` marked a flag without stopping observation and overwrote
outcome repeatedly. `_prune_finished_events()` retains matched events until
`max(1.5, event_timeout_s + 0.5)`; recorded timeout is 2 s, hence about 2.5 s.
The recorder retains active trace objects until flush/shutdown, which persists
and finalizes them. POST capture has a 64-frame per-trace limit.

That lifetime overlaps the 1.48–1.49 s false-event gaps. It explains both extra
oracle positives and event 6's final `runtime_event_debug.shot_id=7` (similarly
14→15). Later map copies also lack a synchronized producer-frame identity;
use them cautiously. The audit reproduces reference features from actual source
frames instead of assuming each map's observation time is its source time.

**PROVEN C — legitimate asynchronous delivery exists.**
`shot_async_v2224` clones scanner event/history containers, copies input gray,
and dispatches immutable jobs carrying shot id, peak and frame time to a single
CV worker. Results preserve those IDs. `take_ready_for_active()` accepts all
pending event IDs, drops terminal results and orders by frame time. `apply_result()`
sets the shared scanner pool and its producer shot id. The installed V2.22.6
tracking wrapper consumes the worker's camera timestamp rather than harvest time.
Queued local/worker state is cleared for terminal events.

**PROVEN: no later-event evidence influenced these 22 completed decisions.**
Every captured decision precedes its next audio peak; every decision-pool frame
precedes that boundary. `_resolve_audio_events()` skips terminal events. Trace
append operations do not feed candidates back to the detector. Thus the 6→7 and
14→15 examples cannot retroactively change live emission. No observed shared
vault/bank crossing is established by those examples.

**PROVEN B/D — a separate pending-event correctness gap remains in runtime.**
It is unsafe to generalize the previous paragraph to all runtime situations.
`LocalConfirmManagerV2225.active_waiting()` checks only pending id, phase and
minimum frame gap; it has no next-audio boundary. The oldest waiting event can
consume the current gray frame even after a newer event arrives. Shared tracks
have no event ownership; `_best_track_for_event()` uses first-seen onset within
−0.08 to +1.5 s and score, not last-evidence time or producer event. An old track
can receive later observations, remain onset-eligible and become ready.

The source characterization test creates an event-6 candidate at +0.040 s,
a next peak at +1.4839297 s, then a new localized change at +1.4862853 s.
The actual local manager returns event 6; actual local confirmation passes;
actual frame-unique tracking and readiness permit the old event to select it.
No external hit is emitted by this test. This is a **runtime correctness bug
reproduced in isolation**, not evidence that it happened in this physical set.

V2 models, candidate banks and V2.6 vaults are keyed by shot id. V2 caches are
removed during resolved-shot diagnostics/finalization/reset. Vault default age
is 2.2 s, up to six retained shot states; fusion reads the current shot's state.
These are shared mutable detector objects but the single worker serializes jobs;
state sharing alone is not proof of a future-vault leak. Track association and
local confirmation have the concrete demonstrated missing boundary.

Per the instruction to leave live authority unchanged, this audit does not
install a runtime boundary or detector correction. **The pending-event boundary
fix is the highest-priority next engineering step**, ahead of ranker tuning or
physical promotion. It must retain delayed pre-boundary worker results while
excluding post-boundary evidence from earlier events, including track updates.

## Exact `pre_shot_change` semantics and the spatial reference defect

**PROVEN:** this field is producer-dependent and is neither an existed-before
probability nor a count of motion before the gunshot.

For FAST/V2, `_candidate_features()` computes the radius-2 (13 pixel) mean of
`abs(reference_work - current_norm)`. Both are 8-bit grayscale images (0–255),
so the mean has 0–255 intensity units. Current is lightly blurred, phase-registers
to PRE when accepted, and has a median exposure offset clamped to ±15 applied.
PRE is the latest up-to-three frames in [peak−0.32 s, peak−0.006 s], median3 if
stable (latest if unstable; fallback background when unavailable), then blurred.
FAST copies `center_change` into **both** `pre_shot_change` and the misleading
`center_darkening`. Actual directional darkening is separately `v2_darkening`.
A high value means a strong absolute residual between the supplied images. It
can be brightening, misregistration or the wrong image region, not a new hole.

For legacy V1/vault-origin candidates, `_verify_patch()` measures a mean of the
PRE subtract-delta over an adaptive center disk (the configured diff mode matters).
It uses +0.25*pre_shot_change in one score branch. Vault metadata identifies
preservation, not necessarily a V2 origin; unflagged V1 proposals also receive
vault tags. AI training feature builders sometimes divide their own patch delta
by 64. Those are different features; do not interpret them as the trace's raw
FAST 0–255 intensity units.

FAST's base score includes 0.23*center_change, clips to [3.6,35], and can receive
later hybrid/bank/confirmation support. V2.22.2 novelty cleanup removes or demotes
small PRE residuals and preserves strong residuals near known holes/ridges. Thus
an invalidly high residual is rewarded and can bypass cleanup. Local confirmation
sorts by score and then this field. It does not interpret high values as evidence
that something existed before the gunshot.

**PROVEN reference-plane bug:** `hit_scanner_v2221.patched_detect()` crops current,
scene/surface references and masks, and shifts known holes. It does **not** crop
`scanner.frame_history`. V2 `_collect_pre_frames()` slices that full-camera history
with a crop-local bbox. For this session crop origin is (1521,972), V2 bbox is
(40,31,1446,702). PRE is therefore taken from camera x=40..1446, y=31..702,
while current comes from x=1561..2967, y=1003..1674. No later wrapper corrects
this collector. The worker clones preserve the full-frame history arrays.

Reconstruction from saved frames gives **zero error for all 681 physical FAST
proposal `pre_shot_change` values**, across all 15 FAST physical events. Including
false event 15 gives 731/731 exact matches. Fixing only PRE's spatial origin in
the offline calculation changes median residual **192.384613 → 1.538462**.
These are residual measurements at frozen coordinates, not accuracy improvement.

Recorded registration estimates compare unrelated regions: e.g. event 6
(−205.49,+45.55), event 14 similarly implausible. They fail the shift/response
acceptance gate, leaving unregistered unrelated images to be subtracted.
Correct-plane reconstruction gives near-zero shifts (all magnitudes <0.1 px)
and responses approximately 1.037–1.084 on FAST events, with zero exposure offset.
Thus treating the original huge shift estimates as physical camera movement is
incorrect. **This bug is a concrete upstream cause of FAST saturation and the
cross-source score gap; normalization would not repair the evidence.**

Local confirmation uses a different reference: the shot-local immutable
V2.22.3 recent PRE snapshot, chosen near peak−0.35 s and no later than peak−0.08 s
(fallback scene reference if no safe ring frame). Saved snapshots here are about
0.317–0.392 s before audio. It compares full-camera PRE and current directly,
without registration or exposure normalization, searches ±4 px for largest
3×3-smoothed absolute residual, then computes radius-2 center and radius-4..8
ring. Acceptance is:

```text
center_abs >= 1.65 AND
(compact >= 0.45 OR peak_abs >= 3.4 OR darkening >= 1.3)
```

Darkening is optional. Six wrong winners have zero confirmation darkening.
Small brightness/noise changes or movement of an existing edge can pass; static
unchanged pixels fail, but local confirmation is not proof of new bullet-hole
appearance. Across physical rounds 3,618/3,835 proposals pass (94.3%, including
event 12's repeated round). All 19 false winners pass.

Object-context V2.25.2/V2.25.3 registered freshness/novelty authority is bypassed
when no camera HitRegions exist (`best_track_v252/v253` delegate to earlier
selectors). Global target mode retains ordinary V2.22.2 cleanup and known-hole
logic, but does not acquire the object-region freshness guarantee. No permanent
coordinate blacklist was added; neighboring/re-hit changes remain valid research
cases.

## All false winners and positive control

The clusters overlap; they do not sum to 19:

| Evidence/category | Count among 19 wrong | Status |
|---|---:|---|
| FAST winner, saturated base score, high PSC >=100 | 15 (78.9%) | PROVEN; same 15 events |
| Wrong-plane FAST residual reproduced | 15 (78.9%) | PROVEN |
| FAST temporal-rescue flag | 14 (73.7%) | PROVEN; overlaps FAST |
| Local-confirmation false positive relative to precise GT | 19 (100%) | PROVEN |
| Zero local-confirmation darkening | 6 (31.6%) | PROVEN |
| V1-origin/vault-tagged winner (no FAST) | 4 (21.1%) | PROVEN: events 1,12,13,21 |
| Two unique frame hits | 18 (94.7%) | PROVEN; event 12 has three |
| No causal proposal @42 | 8 (42.1%) | PROVEN |
| Causal @42 proposal loses selection | 11 (57.9%) | PROVEN; nine FAST wins, two vault wins |
| Winner influenced by a later audio event in this dataset | 0 (0%) | PROVEN from decision chronology |

The 15 FAST winners lie in a narrow bright part of the image (candidate x about
2839–2899). PRE/POST patches show mostly smooth already-present background rather
than new holes. **STRONG HYPOTHESIS:** small illumination/sensor/codec variations
supply their permissive second-frame confirmation. The spatial-reference error
is proven; the exact physical source of each 2–9-unit confirmation change is not.

Visual comparison of all 22 snapshot/confirmation pairs supports persistent
scene structure for the four non-FAST wrong winners: radial edge at event 1,
text/graphics at 13 and 21, and an existing hole-like mark at 12. This is
**STRONG HYPOTHESIS: 4/19 (21.1%)**, with old-hole identity at event 12 unverified.
No defensible tape count or proven old-hole count can be assigned. The human
observation that old structures attract candidates is compatible with these
images but does not imply that every false FAST peak is an old hole.

For the 20 decision pools, FAST has 681 proposals, median 35, 622/681 >=35;
vault-tagged candidates number 2,965, median 1.928534, none >=35. All nearest
causal GT candidates are vault-tagged. Nine of 11 causal oracle-positive losses
are cross-source FAST-over-vault losses, but sole causation by scale is not
claimed: wrong reference, permissive confirmation and history also operate.

**Positive control: event 11, physical dart shot 10.** GT is
(2546.314623,1394.713822); CURRENT is (2546.444458,1393.0), error **1.718733 px**.
Its first CV frame is +0.012512 s: below V2's 0.015 s minimum event age, so V2
returns legacy candidates (`waiting_post_peak`). No FAST saturated competitor
enters that frame. This is a V1-origin proposal with V2.6 vault metadata, not a
banked multi-frame V2 discovery.

It is position 1/181, source percentile 1.0, raw retained score 27.519551,
PSC 24.166667, contrast 22.352564, area 39.5. Local confirmation at +0.374225 s
has center_abs 8.384615, ring_abs 3.585526, compact 4.799089, peak_abs 17,
darkening 2.076923. Confirmation raises score to 32.519551. Track 1428 has two
unique frames, age 0.361713 s, same-frame support 1, vault hits 1, carried=0.
The decision is +0.470931 s. Its score, same onset and readiness win the ordinary
selector; both frozen shadows also select it.

**PROVEN:** success coincides with avoiding the broken FAST branch and a strong
legacy candidate at GT. **Not proven:** that this generalizes, or that its
confirmation alone uniquely detects newly created material. Four other physical
shots also skip FAST and fail. PRE images often already contain hole-like
appearance near GT. Sensor exposure/camera latency versus software timestamps
has not been physically synchronized; no labels are invalidated by that fact.

## Research-only temporal experiments

No live selection, thresholds, AI settings, frozen formula, or authority was
changed. Experiments are offline candidate evaluation, **not live-path-equivalent
replay and not new physical validation**. Candidate coordinates and membership
are frozen to the causal decision pool; GT is used only after ranking for error.

Final experiment uses correctly positioned immediate PRE stack and the exact
local-confirmation frame consumed before the decision, registration/photometry
from V2, then one fixed score: radius-2 residual mean minus radius-4..7 ring mean.
Two polarity variants use absolute residual or positive PRE−POST darkening.
No weights were searched. Nearby new changes remain legal; unchanged structures
have zero temporal score. Every physical shot has the required source frames.

| RESEARCH_ONLY variant | @5/@10/@20/@42 | Mean px | Median px | P95 px |
|---|---:|---:|---:|---:|
| REGISTERED_ABS_COMPACT | 1/20 at each | 313.276 | 312.656 | 497.640 |
| REGISTERED_DARK_COMPACT | 1/20 at each | 346.184 | 362.414 | 608.179 |

Neither improves Top1. Neither is frozen or promoted. P95 here uses NumPy's
linear percentile; original validation's P95 numbers above are preserved using
its original implementation, not silently recalculated.

Earlier preserved measurement passes used the first proposal frame: V1-only
shots were initially unavailable (15 physical evaluated; absolute 2/15, dark
1/15); expanding geometry handling covered 19 (absolute 2/19, dark 1/19).
Event 12's proposal timestamp is actually −4.667 ms and is stored as PRE history,
not POST. The final pass keeps that source fact, uses its real later confirmation
frame for temporal ranking and covers all 20. These are timing/coverage variants,
not hidden weight tuning. Source-reference reproduction remains exact throughout.

The result identifies a repairable input-coordinate defect but does not show
that correcting it and reranking existing proposals solves candidate generation.
A full detector replay with corrected reference geometry is still required;
changing the reference can change proposals, thresholds, saliency and recall.

## Diagnostic changes and remaining work

New shared causal analysis supports both export and physical-session audit through
explicit `causal_candidate_audit_v1` fields. Export adds
`causally_available_candidates_v1` and names historical stage semantics. Historical
raw/filtered/retained/selected metrics are retained, not silently redefined.
The label UI's C overlay defaults to causal candidates; T adds later evidence:
green causal, orange post-decision, magenta cross-event, grey unknown. Both camera
and target-space views use the same classifications; all session events define
boundaries even when already labelled or skipped. Overlay results are cached
outside the rendering loop. Frame navigation remains available for human labels.

New captures retain pool owner and local-confirmation output in the decision
snapshot, record next-audio boundaries, label observation phases, and preserve
first terminal outcome. Traces intentionally retain diagnostic follow-up frames;
the fix is explicit semantics and immutable terminal results, not deleting data.
Existing traces are never rewritten.

Highest-priority engineering: add a tested next-event evidence boundary to local
confirmation and track consumption for pending events, preserving delayed older
frames. Then, as a separate detector hypothesis, correct V2 PRE history's spatial
mapping and run an isolated full-pipeline replay. Only then prepare a new frozen
shadow for another physical test. Neither runtime defect was silently repaired
under this measurement/research task's prohibition on live-authority changes.

Next physical experiment: use a fresh documented target and a camera-visible,
audio-synchronized timing marker to establish exposure-to-audio timing; save
immediate PRE, proposal and confirmation frames plus producer/consumer event IDs.
Repeat deliberate 1–20 shots, include a few planned neighboring impacts and
separate controlled nonphysical audio triggers. Exercise a deliberately delayed
pending decision across a later trigger in a controlled diagnostic test before
physical acceptance. Freeze the candidate version before collecting labels.
A clear unchanged-PRE scene and true near-rehit delta are required positive and
negative controls. Do not solve novelty with a coordinate blacklist.

Remaining uncertainties: exact sensor exposure latency, origin of small
confirmation residuals, whether particular visible marks are old physical holes
or projected/printed structures, and corrected full-pipeline recall/accuracy.
The present evidence does not validate a new detector or selector for live use.

## Engineering fixes after acceptance

Commit `5d45527` isolates pending-event ownership. Local confirmation now stops
at the next audio peak, and delayed worker results are accepted only when their
own evidence frame predates that peak. Focused regressions cover both false-event
timing patterns, late delivery, terminal immutability, pending/new-event ordering,
and normal delayed single-shot processing.

The V2 PRE collector now translates crop-local detector bboxes to the
full-camera frame-history plane exactly once, including fallback background
slicing. Corrected offline source-frame replay
(`evaluation_runs/causal_audit_20260908/temporal_postfix_v6`) covers 731 FAST
proposal versions: median PRE residual 192.385→1.538, median and p90 corrected
diagnostic score 3.6, and 0 saturated corrected scores. The 15 physical FAST
false-winner score artifacts collapse in this replay. This is recorded-source
replay, not regenerated live candidate selection or physical validation; temporal
selection remains 1/20 @42. Cross-source ordering, stale structures and
permissive confirmation remain separate bottlenecks.

## Per-winner feature evidence

Track score includes confirmation/history; PSC is the producer residual described above.

|Event|Track score|PSC|Local abs|Local dark|Track age s|Causal nearest score|Causal nearest px|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|18.767|7.385|8.308|2.154|0.319|7.874|88.340|
|2|38.759|192.615|7.692|7.692|0.574|1.630|3.642|
|3|37.997|227.154|6.769|6.769|0.558|16.757|2.409|
|4|38.675|187.692|3.615|3.615|0.651|9.446|65.153|
|5|37.630|212.923|5.000|0.000|0.678|2.207|3.239|
|6|38.266|209.923|6.077|0.000|0.640|0.883|44.126|
|8|38.079|192.923|5.692|5.692|0.560|14.418|4.151|
|9|37.274|190.846|3.923|0.000|0.583|16.380|2.769|
|10|37.064|190.538|2.538|0.000|0.571|1.895|3.896|
|11|32.520|24.167|8.385|2.077|0.362|27.520|1.719|
|12|27.123|6.000|15.769|15.769|0.401|0.808|32.598|
|13|19.015|6.417|9.077|0.077|0.248|7.872|43.502|
|14|37.212|190.154|4.846|4.846|0.571|1.117|178.177|
|16|39.409|194.385|9.000|9.000|0.565|10.642|69.054|
|17|37.069|209.462|3.462|0.000|0.561|1.076|107.794|
|18|38.745|190.538|7.615|7.615|0.571|1.212|77.324|
|19|38.694|190.846|8.385|8.385|0.569|1.220|2.796|
|20|37.330|189.538|4.231|0.000|0.685|1.442|0.740|
|21|19.222|12.667|7.615|5.385|0.371|0.859|40.167|
|22|37.176|199.308|3.923|3.923|0.481|4.198|2.724|

## Verification and repository state

Passed after the diagnostic changes:

```text
python3 -m automation.overnight_selftest             15 tests
python3 -m automation.evaluation_selftest            13 tests
python3 -m automation.physical_trace_selftest         4 tests
python3 -m automation.async_track_timing_selftest     3 tests
python3 -m automation.causal_candidate_selftest       9 tests
python3 -m automation.physical_trace_label_selftest   6 tests
python3 -m automation.physical_trace_target_selftest  4 tests
```

All eight changed/new Python files compile; `git diff --check` passes.
New tests cover exact 6→7 and 14→15 timestamps, late worker delivery, missing
evidence, same-XY future feature versions, synchronous confirmation proof,
terminal-outcome immutability, additive export semantics, nearby new changes,
and characterizations of both still-unfixed runtime defects. Characterization
success is not a claim that either detector defect has been repaired.

The full new export reproduces all six historical stage fields
(raw/filtered/retained/confirmed/selected/emitted) exactly on all 20 original
physical rows, while adding causal metrics. All 22 trace hashes and all six
original evaluation-file hashes match the audit's preserved digests.
The label helper/coordinate tests pass; no new physical or interactive label
acceptance session was conducted.

No commit, push, merge, history rewrite or branch switch was performed. Source,
tests and documentation remain reviewable in the working tree on
`codex/score-root-cause`. The pre-existing `content/ai/settings.json` modification
was left alone. Generated reports and images stay in ignored `evaluation_runs/`;
no physical trace, settings or generated evaluation artifact was staged.

## Overnight follow-up — 2026-09-09

The above independent validation is preserved unchanged. The later post-PRE
independent validation (`session_20260908_194746_b38de674`, evaluation
`physical_20260908_195522_ab2304cd`) reports 7/10 causal availability @5/@10 and
9/10 @20/@42, CURRENT 2/10, mean 171.10, median 135.75, p95 464.70 px. The PRE
mapping and ownership fixes are committed; FAST saturation is absent.

A full recorded-input track reconstruction now resolves the latest nine-shot
oracle-positive funnel as **9 generated/consumed → 9 tracked → 9 locally
confirmed → 9 eligible → 2 selected**. Earlier claims of missing track fate were
limited to direct inspection of the top-eight export. Full proposal and local
confirmation lists allow independent verification of every winner, 136 saved
track states and 85 tracking counters. Original traces and metrics were not edited.

The same method verifies 22/22 prior dart audio winners and all ten earlier
physical winners. The prior physical-only causal @42 oracle remains 12/20; all
12 survive into eligible locally confirmed tracks. Events 7/15 remain false.

Complete future captures now include association ledgers and the active decision
pool with eligibility, current/best scores, ownership, source history and rank.
Exact replay fails on mismatches. A newly isolated transport defect was fixed:
producer identity is now added to `result.candidates`, the list actually consumed
by tracking, rather than only the diagnostic copy. Pending-event rejection and
valid late pre-boundary delivery have passing integration regressions.

The new ranking experiment remains RESEARCH_ONLY, fails the physical development
set and provides no synthetic holdout @42 improvement. No AI/frozen confirmation
selector is promoted. See [OVERNIGHT_TRACK_RESEARCH.md](OVERNIGHT_TRACK_RESEARCH.md).

The final ownership regression additionally proved that last-candidate tagging
alone cannot protect track history under late older-frame delivery after a newer
producer observation. V2.22.6 now prevents association across known different
producer events. This is a component-level correctness correction, not an
explanation of the latest seven physical ranking losses or a new physical
validation. See the measured before/after episode in `OVERNIGHT_TRACK_RESEARCH.md`.
