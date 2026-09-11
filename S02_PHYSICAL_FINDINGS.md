# S02 physical findings — 2026-09-09

The baseline below is preserved. Subsequent registered-verifier, proposal and
no-impact research is in [ACCURACY_95_100_RESEARCH.md](ACCURACY_95_100_RESEARCH.md).
S02 is now explicitly **DEVELOPMENT_USED**. The user reported an unconfirmed
possibility that event 1 produced two visible physical changes. Its native label
has not been changed; the later report includes exclusion sensitivity. No S03
data was opened or used.

S02 is finalized as **DEVELOPMENT** evidence. CURRENT selected/emitted accuracy
is **1/10 @42**, with **5/10 causal candidate recall**. All five corresponding
tracks survive tracking, local confirmation and eligibility; four lose final
ranking. The other five physical events have no causal candidate within 42 px.
All three human-confirmed nonphysical audio events emitted false hits.
No detector, ranking, settings, model or live authority change is justified.

## Inputs and label validation

- Binding: `evaluation_runs/S02_retry_binding.json`.
- Session: `content/ai/physical_traces/session_20260909_162957_S02_b900192c`
  (resolved under `/data/skjutbana/physical_traces`).
- Capture commit: `4a9e6df6330b8f06d641344927db35a273eae711`, dirty working
  tree recorded by the producer. Binding source commit is `unknown` and no
  `test_setup.json` freezes validation provenance. Do not call this an
  independent physical validation of a new selector.
- Physical shots 1–10 map to runtime events **1,3,4,5,6,7,9,11,12,13**.
  Human/timing evidence explicitly classifies **2,8,10** as
  `NO_PHYSICAL_SHOT`. They occur 1.496043, 1.492994 and 1.493442 seconds after
  the preceding physical event; timing alone is not the classification proof.
- Ten native labels are precise camera-space manual clicks, with matching
  event IDs, finite coordinates within the captured 3840×2160 frame, and
  attachment timestamps after capture. The three native S markers say
  `unresolved`; **S does not encode a nonphysical event**.
- Native labels, status files and all captured evidence remain untouched.
  An external assignment file records the user's confirmed mapping. The
  finalized manifest preserves every runtime event ID.

Artifacts are isolated in:

```text
/data/skjutbana/evaluation_runs/S02_finalization_20260909_170523/
  S02_input_sha256.json          # all 1,664 input files, including frames/maps
  S02_assignments.json           # explicit external human event mapping
  S02_labels.json                # derived plan mapping, native GT preserved
  S02_quality.json
  S02_finalized.json
  S02_evaluation/                # health, export, selector comparison, metrics
  S02_track_replay/audit.json    # complete track survival + fixed old replays
  S02_evidence_details.json      # source distributions and ownership checks
  comparison/                   # S01/S02/post-fix metrics and causal patches
  analyze_sessions.py           # read-only reproduction, explicit three roots
  integrity_verified.json
  selftests_final.log
```

## Quality and evidence limits

S02 trace, frame, label, full-replay readiness, patch and temporal checks all
PASS, with no reported problems. Actual frame/map headers were opened and
checked against recorded metadata. `CURRENT_EXACT_REPLAY` matches all **13/13**
complete snapshots, predicates, ordering and selected track IDs/coordinates;
the selections also match all recorded emissions. This is exact replay of
recorded selector inputs, not regenerated detection from images.

There are 6–10 causal POST frames per physical event (6–16 across all events).
All 1,700 eligible tracks across the 13 decisions have the expected known
producer and causal ownership. Physical-event pools contain 1,309 eligible
tracks, of which 1,293 have explicit local-confirmation proof. Eligibility and
emitted `state=confirmed` are not substitutes for that proof. The five correct
candidate tracks specifically have local-confirmation proof.

No known ownership violation appears in these decision pools. The three false
events start after their preceding decisions have completed, so they do not
exercise the adversarial still-pending/delayed-delivery cases. Their emissions
demonstrate insufficient physical rejection, not proof of renewed cross-event
transport contamination.

S01's prior finalization is preserved. A stricter recheck reports labels/frame/
patch/temporal PASS, trace FAIL and full-replay readiness WARN: event 11 has
`incomplete_persistence`, images and a valid label, but no usable final decision.
Ten other events replay exactly, including nonphysical event 4. Thus S01 has
**nine evaluable physical decisions**, not ten measured outcomes. The tenth
shot is unavailable, not an inferred detector miss or timeout.

## Exact comparison

Radii are original camera pixels. Each four-number cell lists **@5/@10/@20/@42**.
The oracle is candidate availability at the causal decision boundary, never
selected accuracy. The wider causal-observation audit agrees with retained-pool
recall in all three sessions.

| Session | Physical / audio events | Observable physical decisions | Causal oracle hits | CURRENT selected/emitted hits | False emissions / nonphysical events |
|---|---:|---:|---|---|---:|
| S01 recovered | 10 / 11 | 9 (one unavailable) | 6/9, 6/9, 7/9, 9/9 | 2/9, 2/9, 3/9, 3/9 | 1/1 |
| S02 | 10 / 13 | 10 | 3/10, 3/10, 3/10, 5/10 | 1/10, 1/10, 1/10, 1/10 | 3/3 |
| Previous post-fix `session_20260908_194746_b38de674` | 10 / 10 | 10 | 7/10, 7/10, 9/10, 9/10 | 2/10, 2/10, 2/10, 2/10 | 0/0 (not tested) |

S01 has three verified successes out of ten physical shots plus one unknown
outcome; 3/9 is its observed accuracy, not a complete-session 30% estimate.
S02 conditional CURRENT accuracy given an oracle-positive event is **1/5 = 20%**
at 42 px; corresponding values are 3/9 for S01 and 2/9 for the comparator.

| CURRENT error metric | S01 (9) | S02 (10) | Previous post-fix (10) |
|---|---:|---:|---:|
| Mean px | 204.108808 | 187.926471 | 171.099411 |
| Median px | 205.897500 | 163.723005 | 135.748659 |
| P95 px | 463.950857 | 426.563857 | 464.701006 |
| Errors >100 px | 5 | 7 | 8 |
| Physical-event detector latency mean ms | 956.271092 | 627.121186 | 565.249968 |
| Physical-event detector latency median ms | 589.156389 | 650.671363 | 560.307860 |
| Physical-event detector latency P95 ms | 4199.182749 | 733.578444 | 707.928896 |

P95 uses the nearest-rank empirical definition. These small, different sessions
are descriptive comparisons; lower mean error is not evidence of improved
accuracy. S02 has zero physical misses/timeouts in the recorded trace: all ten
physical events emitted, nine at the wrong position.

| S02 selector | Hits @5/@10/@20/@42 | Mean px | Median px | P95 px | >100 px |
|---|---|---:|---:|---:|---:|
| CURRENT (recorded live) | 1/1/1/1 of 10 | 187.926471 | 163.723005 | 426.563857 | 7 |
| Frozen confirmation shadow | 0/0/0/0 of 10 | 357.993474 | 338.052369 | 555.635885 | 10 |
| Canonical AI shadow | 0/0/0/0 of 10 | 297.312633 | 302.400493 | 563.423484 | 9 |

For comparison, confirmation/canonical shadows reach respectively 2/9 and 2/9
@42 on complete S01 events, and both 0/10 on the previous post-fix session.
Shadow results are frozen retrospective scoring of recorded pools, not physical
emissions or proof of live-path-equivalent replacement behavior. S02 lacks a
captured canonical result, so its canonical row is explicitly recomputed offline.

Confirmation configuration remains
`123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f`.
Canonical remains **SHADOW / OFFLINE_CHALLENGER**; manifest SHA256 is
`63a366d984c56e3fe65cdc62efd5db7cc9e74d3cd3a6307068b0b849ddd02b86`,
model JSON SHA256 is
`2078c398f141e8f7f63e52849697f163552165878602510e17408b5e9ce1678e`,
and model NPZ SHA256 is
`1c58d31d44ec46d7db5f7aa3281ed7c7314c61c1f36f1c3c8475d4282bb10305`.

## Event-level funnel and ranking evidence

| Physical shot | Runtime event | Nearest causal candidate px | CURRENT emitted error px | Nearest correct-track rank @42 | Loss |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.866312 | 0.866312 | 1 | Correct |
| 2 | 3 | 3.449521 | 426.563857 | 71 | Final ranking |
| 3 | 4 | 47.883879 | 138.971956 | — | No causal candidate @42 |
| 4 | 5 | 76.150322 | 188.474054 | — | No causal candidate @42 |
| 5 | 6 | 52.636643 | 320.971500 | — | No causal candidate @42 |
| 6 | 7 | 51.421461 | 340.959769 | — | No causal candidate @42 |
| 7 | 9 | 4.231041 | 106.426094 | 101 | Final ranking |
| 8 | 11 | 37.866144 | 218.645023 | 110 | Final ranking |
| 9 | 12 | 32.669363 | 63.935551 | 46 | Final ranking |
| 10 | 13 | 42.273624 | 73.450592 | — | No causal candidate @42 |

The verified @42 funnel is **10 physical → 5 causal proposals → 5 tracked →
5 locally confirmed → 5 eligible → 1 selected**. Event 12's track representative
is 34.119474 px from GT after association, still within 42. The event-13 proposal
miss is close to the existing cutoff; no radius was relaxed to count it.

The four oracle-positive losers are V1 tracks, all displaced by FAST winners.
Their onset-distance keys tie the winners exactly, so raw `best_score` decides
the losses. Across physical eligible pools FAST median best score is 17.75
(357 tracks) versus V1 7.186860 (831); 121 event-6 tracks have UNKNOWN proposal
source but known producer ownership. FAST maximum best score is 31.628668;
this is not the historical FAST ceiling-saturation failure. No physical track
has historical best score above its current candidate score.

Existing fixed research-only alternatives were replayed without tuning:
complete-pool FAST exclusion and non-FAST preference each remain **1/10 @42**,
mean 231.779312 px. The existing signed-local-contrast ranking is **0/10**,
mean 217.623279 px. Neither supports promotion or a source ban.

## Common-verifier implication and next experiment

S02 reinforces final-ranking failure **conditional on candidate availability**.
It also exposes a second limit: a verifier confined to these existing eligible
coordinates has an observed @42 ceiling of 5/10. It cannot recover the five
missing proposals by improving their order. Audit those missing locations as a
separate proposal/localization question; do not tune the generator together
with a ranking hypothesis.

Causal shared-context extraction succeeds for all ten S02 GT patches and twelve
negative examples: nine wrong emissions plus three nonphysical emissions.
Combined with the nine causally reconstructable S01 events this adds nineteen
positive and nineteen negative examples (four no-impact negatives). These are
patch compatibility checks and research inputs, not classifier measurements.
S01 event 11 remains usable for offline image diagnostics but cannot enter a
decision-causal verifier test without the missing cutoff.

The next bounded experiment is one **source-independent causal physical-patch
verifier**, trained on historical development/S01 and assessed on S02 as an
entire development session. Score all eligible tracks through the same image
context, include hard negatives and no-impact events, and test rejection as
well as ranking. Keep GT/source identity out of verifier inputs. Report oracle
recall, conditional ranking, actual replayed readiness/emission behavior and
no-impact false acceptance separately. Lock the configuration before evaluating
S03 once; S03 has not been opened or used here. No new live authority is enabled.

## Reproduction and checks

From the repository root, preserve the run above and choose new output paths:

```bash
RUN=/data/skjutbana/evaluation_runs/S02_finalization_20260909_170523
RECHECK=$(mktemp -d /data/skjutbana/evaluation_runs/S02_recheck_XXXXXXXX)
python3 -m automation.physical_trace_quality --session-root content/ai/physical_traces/session_20260909_162957_S02_b900192c --mapping "$RUN/S02_assignments.json" --output "$RECHECK/quality.json"
python3 -m automation.physical_collection finalize --plan evaluation_runs/physical_capture_plan.json --session S02 --labels "$RUN/S02_labels.json" --quality "$RECHECK/quality.json" --output "$RECHECK/finalized.json"
python3 -m automation.physical_test evaluate --binding evaluation_runs/S02_retry_binding.json --mapping "$RUN/S02_assignments.json" --output "$RECHECK/evaluation"
python3 -m automation.track_survival_replay --root content/ai/physical_traces/session_20260909_162957_S02_b900192c --exclude-events 2 8 10 --output "$RECHECK/tracks"
python3 "$RUN/analyze_sessions.py" --repo "$PWD" --output "$RECHECK/comparison"
```

The comparison script uses only S01, S02 and the named post-fix comparator.
No broad inventory or rebuild was run. Measurement/workflow selftests passed
99 tests across label reset, postflight, collection, adversarial workflow,
capture-plan, labeling, evaluation, trace recording, track audit, overnight
and common-verifier modules. Compilation and `git diff --check` also passed.
S02's reset command was previewed only; actual resets were tested exclusively
on disposable synthetic sessions. No commit, push or merge was performed.
