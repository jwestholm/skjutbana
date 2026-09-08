# Overnight AI report — 2026-09-07/08

## 1. Executive summary

The physical evidence points to two different bottlenecks. Smoke2 proves that a
near-perfect retained candidate can lose final selection (6.09 px and 4.84 px
nearest retained versus 241.75 px and 620.41 px emitted). Biathlon5 repeats that
pattern for event 3: a 1.54 px candidate had local-confirmation evidence and two
frame observations, while the emitted point was 67.73 px away. The last two real
biathlon holes were not within 20 px of the retained pool, so ranking could not
have repaired those proposal misses.

The sixth runtime event is classified as `NO_PHYSICAL_SHOT` in an external mapping:
event 2 is 100.66 ms after event 1, shares the same PRE snapshot, and the image
progression shows five holes across event groups `{1,2},3,4,5,6`. This is the
strongest available classification, not an audio-causal proof. The traces do not
contain raw audio waveforms or per-chunk trigger logs, so echo versus duplicate
dispatch cannot be distinguished retrospectively.

One canonical challenger now wraps the existing V2.23 linear listwise model in a
hash-pinned SHADOW adapter. It exposes complete ranking, Top-N, feature values and
an uncalibrated score margin, but has no live-application API. No model was
promoted to authority.

## 2. Physical-session findings

The mapping used for biathlon5 is:

| Runtime event | Physical state | Label mapping | Key result |
|---|---|---|---|
| 1 | precise | label 1 | nearest retained 1.71 px; timed out |
| 2 | `NO_PHYSICAL_SHOT` | none | 100.66 ms after event 1; duplicate-trigger candidate |
| 3 | precise | label 2 | nearest retained 1.54 px; emitted error 67.73 px |
| 4 | precise | label 3 | nearest retained 7.03 px; emitted error 212.84 px |
| 5 | precise | label 4 | nearest retained 23.65 px; emitted error 115.11 px |
| 6 | precise | label 5 | nearest retained 35.89 px; emitted error 459.63 px |

The mapping is stored in `evaluation_runs/overnight_20260907/biathlon_mapping.json`;
the original labels and traces were not changed. Smoke2 remains two approximate
shots with nearest retained distances 6.09 px and 4.84 px and emitted errors
241.75 px and 620.41 px.

## 3. Sixth/nonphysical event and trigger investigation

Event 2 is the separately classified false event. Events 1 and 2 have PRE frames
from the same camera instant and are only 100.66 ms apart. The hole image does not
show a sixth hole; the five manually labeled holes align with events 1, 3, 4, 5
and 6 after temporal progression. Event 2 timed out and emitted nothing.

The audio implementation has an 80 ms cooldown. New traces now attach the raw
event's peak, RMS, threshold, dynamic threshold, crest threshold, previous peak
timestamp and event interval, and V2.22.6 near-miss telemetry records rejected
transients. Those fields were not present in this older capture, so the cause is
classified as a likely duplicate/second threshold crossing, with acoustic echo,
duplicate dispatch and cooldown interaction unresolved.

## 4. Stage-by-stage failure analysis

The old physical producer did not persist complete RAW and FILTERED pools. The
audit therefore reports those stages as unavailable instead of inferring them
from later lists. The retained pool was observed for all six biathlon events and
both smoke2 events. In biathlon, three of five real shots had a retained candidate
within 20 px; two had no retained candidate within 20 px. The false event is not
included in hit denominators.

Evidence-supported categories are:

* `RAW_MISS`, `FILTER_LOSS`, and `RETENTION_LOSS`: unresolved for these traces.
* `SELECTION_LOSS`: event 1 timed out despite a 1.71 px retained candidate; event
  3 selected a track 67.73 px away despite a 1.54 px locally confirmed candidate;
  smoke2 supplies two additional retained-versus-emitted examples.
* `EMISSION_MAPPING_LOSS`: not established. The wrong points are already present
  in detector tracks before coordinate mapping, and no mapping-only evidence was
  recorded.

The saved top-eight track view is a debug view, not a complete confirmation pool.
The new trace schema transports worker pipeline output and the actual local
confirmation output, freezes deterministic selection at the emission boundary,
and evaluates the challenger only after the decision with explicit pool semantics.

## 5. Current AI architecture

The live chain is documented in [AI_STACK_AUDIT.md](AI_STACK_AUDIT.md). In brief,
audio opens the shot window; V2.22.1/.2 crop and clean the physical proposal path;
V2.22.4 runs asynchronous CV; V2.22.5 performs local PRE→POST confirmation and a
bounded FULL rescue; V2.22.6 tracks distinct camera frames; V2.25 context/freshness/
novelty gates apply only when the required frozen object context exists. AIRuntime
and resolver/ranker systems run in advisory or shadow modes on the current path.

V6–V9, pairwise rankers, dense/reducer, patch, registered-evidence and heatmap
systems remain available as historical or shadow research. Their incompatible
schemas and protocols are not silently treated as one installed model. Existing
registries mark the V2.23.4–V2.23.6 research gates false and live authority false;
V7/V8 recommendations also fail their saved quality gates. V9 reports historical
gains but `enough_data=false` and authority remains false.

## 6. Canonical challenger design

`src/engine/ai/canonical_challenger.py` loads a manifest with `mode=SHADOW`, model
hashes, feature transform and status. It uses the shared physical feature extractor,
returns a stable full order, Top-N, per-candidate features/scores, score margin and
manifest hash, and never mutates candidates or emits a point. The selected artifact
is `evaluation_runs/overnight_20260907/ranking/challenger.json`, status
`OFFLINE_CHALLENGER`. Future physical traces store deterministic and challenger
results side by side without granting challenger authority.

## 7. Offline benchmark definition

`automation/conditional_ranking.py` uses `content/ai/ranking_v29/ranking_dataset.jsonl`,
whole-session chronological splits, exact candidate-fingerprint purging, and a
frozen reused holdout. Metrics are conditional: only shots with a candidate within
the requested radius enter conditional Top-1/3/10, MRR and mean positive rank;
oracle availability and excluded counts remain separate. The benchmark is
projected historical data, not physical validation or live-path-equivalent replay.

## 8–10. Experiments and metrics

Six hypothesis-led linear trials were run: physical identity, signed-log,
within-shot percentile, geometry-only, evidence-only and stronger regularization.
The regularized trial was retained with guardrails because it improved the MRR20
tie-break while preserving conditional Top-10 guardrails. Development has 200
shots and 19 oracle-positive shots at 20 px. Baseline conditional Top-1/MRR/Top-3
at 20 px are 5.26% / 0.0995 / 10.53%; challenger is 5.26% / 0.1029 / 5.26%.
Thus Top-1 did not improve and Top-3 regressed.

On the reused 100-shot holdout, oracle availability at 20 px is 6/100. Baseline
and challenger both have conditional Top-1/Top-3/Top-10 of 0/6; MRR is 0.0352
versus 0.0366. At 42 px, challenger is 4% / 8% / 28% for Top-1/3/10 versus
baseline 0% / 0% / 12%, but this is a reused historical holdout and does not
justify promotion. Model bytes were rebuilt and matched the frozen hashes in
`evaluation_runs/overnight_20260907/reproducibility/`.

## 11. Retrospective physical shadow comparison

The challenger was run on the frozen retained pool after each recorded decision.
This is explicitly post-decision retained-pool analysis, not a live replay and not
physical validation. On biathlon real events, it selected the close candidate for
events 1 and 3, selected a farther point for event 4, tied the deterministic point
for event 5, and selected a point 192.0 px from event 6's GT versus deterministic
459.6 px. Smoke2 challenger distances were 328.8 px and 777.4 px versus
deterministic 241.8 px and 620.4 px. The sample is too small for an accuracy claim.

An oracle component probe also found the nearest retained candidate passes the
default local confirmation calculation on the first two saved POST frames for all
five real biathlon shots. This uses selected saved frames and the nearest candidate
by GT; it is a diagnostic oracle, not an exact reconstruction of scheduling or
candidate order.

## 12. Negative results

No experiment improved primary conditional Top-1@20. Percentile, geometry and
evidence-only variants failed guardrails or objective comparison. Existing V7/V8
artifacts fail their own saved recommendations. Dense/reducer and direct heatmap
research has useful projected oracle evidence but no live or physical validation.
Audio cause for event 2 remains unresolved. The physical traces also show very
large registration shifts/weak responses in several events, but this is diagnostic
evidence only and was not tuned overnight.

## 13. Commits created

* `32aa298` — physical audit, conditional benchmark, canonical shadow adapter,
  trace transport, physical workflow and tests.
* `fb7334e` — preserve asynchronous detector frame timestamps through frame-unique
  tracking; includes regression tests for delayed results and waiting frames.

The earlier evaluation-contract commits were already present and left intact.

## 14. Tests

Passed: evaluation selftest; physical trace, target, label and stress selftests;
V2.22.4, V2.22.5, V2.22.6, V2.22 resolver, V2.25.2 and V2.25.3 selftests;
10-case overnight selftest; three async timestamp regressions; Python compilation
of new and modified modules. The tests are synthetic or integration-level unless
explicitly described as physical analysis.

## 15–17. Bottleneck and evidence boundaries

The immediate demonstrated bottleneck is final candidate/track selection when a
good candidate is retained, with proposal availability also limiting the last two
biathlon shots. Physical evidence demonstrates candidate presence, local-confirm
diagnostics, wrong-track selection, timing and false-event classification. It does
not demonstrate broad detector accuracy, audio root cause, RAW/FILTERED recall,
or AI improvement. Offline metrics are projected historical evidence only; the
retrospective challenger result is shadow analysis only. Nothing is
`PHYSICAL_SHADOW_VALIDATED` or `ELIGIBLE_FOR_AUTHORITY`.

## 18–19. Next engineering step and exact workflow

The next step is a physical series with preserved runtime logs and explicit shot
count, exercising rapid pairs and separated shots after the timestamp correction.
Use the helper documented in [PHYSICAL_TEST_WORKFLOW.md](PHYSICAL_TEST_WORKFLOW.md):

```bash
python3 -m automation.physical_test start
python3 -m automation.physical_test check
python3 -m automation.physical_test label
python3 -m automation.physical_test classify --shot-id N --no-physical-shot --reason "verified from timing/image progression"
python3 -m automation.physical_test evaluate
python3 -m automation.physical_test stop
```

Classification is optional per event and must be based on physical evidence; do
not invent a sixth hole. Close the launched application before `stop` so traces
flush. The helper refuses settings conflicts, creates unique output directories,
checks trace artifacts and keeps false events outside hit denominators.

## 20. Git status

The intended final worktree state is the two local commits plus the intentionally
modified `content/ai/settings.json`. During report generation, source/doc changes
are staged and committed locally; generated `evaluation_runs/` and physical traces
remain ignored. No push, merge, dev/main modification or settings commit was made.

OVERNIGHT_WORK_COMPLETE
