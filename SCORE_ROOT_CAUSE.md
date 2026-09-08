# Detector score root-cause audit

## Executive summary

The 10-shot physical development trace shows a repeatable score-scale and
provenance problem. Genuine holes are usually carried by the V2.6 vault with
low candidate scores (0.57–2.63 in the closest examples), while many wrong
winners are FAST V2.22.5 proposals saturated at the V2 score ceiling (35.0;
track `best_score` commonly 36–39). Shot 6 is different: its genuine candidate
is also a FAST proposal, has score 36.5, and is retained at position 1, so all
three selectors choose it. This explains the selection failure without proving
that the image score formula is intrinsically incorrect.

The frozen `CONFIRMATION_SELECTION_SHADOW` was not changed. The audit is
read-only and uses the existing labelled session as development evidence.

## Exact detector score path

`CandidateGeneratorV2._candidate_features` in
`src/engine/camera/candidate_generator_v2.py` computes:

```text
raw = 0.16*v2_saliency
    + 0.23*center_change
    + 0.15*local_contrast
    + 0.11*dog_value
    + 0.06*min(zscore, 25)
score = clip(raw, 3.6, 35.0)
```

`center_change` comes from the radius-2 absolute-difference patch mean;
`local_contrast` is center change minus the radius-4..7 ring mean;
`dog_value` is the local DoG/blackhat response; `zscore` is the local
absolute-difference value divided by the estimated noise. The score is not
normalised by proposal source or shot percentile.

`_apply_known_hole_penalty` then multiplies the clipped score by 0.15, 0.4 or
0.7 near a known hole. Hybrid V1/V2 agreement adds 1.5. Candidate-bank carried
entries replace their score with `best_score + repeat_bonus + 0.35`, where the
repeat bonus is capped at 3.4. The live track selector then ranks eligible
tracks by onset distance first and `-track.best_score` second. Track history can
therefore produce a `best_score` greater than the current candidate's score.

The FAST extractor uses the same feature formula but only retains a bounded
sparse peak set and marks `v2225_fast_extract`. V2.6 vault candidates preserve
older candidate features and provenance, but their numeric score remains on the
same nominal field without source calibration.

## Physical score decomposition

The audit command generated JSON and CSV for every retained candidate. The
closest physical candidates had candidate scores:

```text
shot 1  2.63    shot 2  1.63    shot 5  1.68    shot 7  1.72
shot 8  0.69    shot 10 1.28    shot 6 36.50
```

Those low-score candidates are V2.6 vault records. Wrong deterministic winners
in shots 1, 4, 5, 7, 8, 9 and 10 are FAST V2.22.5 records at score 35.0; their
track best scores are approximately 36.6–38.9. Shot 6's genuine FAST candidate
has score 36.5 and track best score 41.5. The shadow winner is often closer
because local confirmation fields distinguish the low-score vault candidate,
but this is a retrospective development result and remains frozen shadow-only.

The audit also records candidate bank hits, vault hits, unique-frame support,
timestamps, source flags, all available local-confirmation values, and matching
track snapshots. The saved top-eight track view does not contain every retained
candidate, so missing track identity is reported as unavailable rather than
inferred.

## Candidate-source score distributions

From the trustworthy retained candidate flags in this session:

| Source | Count | Median | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| FAST_V2225 | 245 | 35.0 | 35.0 | 35.0 | 36.5 |
| V26_VAULT | 954 | 5.69 | 7.33 | 8.73 | 12.45 |

The distributions are sharply separated. This is evidence that the global
numeric ordering is not comparable across proposal provenance in practice.
It does not identify whether the underlying image evidence or the carry/merge
policy is the first cause. The score ceiling and bank replacement rules are
the concrete mechanisms that amplify the mismatch.

The exact FAST saturation cause is the shared V2 feature formula's final
`clip(raw, 3.6, 35.0)`. FAST and full V2 use the same feature computation, but
FAST keeps sparse peaks selected from a different temporal/saliency path. In
this trace every FAST candidate at the raw ceiling is recorded as 35.0; the
only observed physical-positive FAST candidate is shot 6 (36.5 after later
carry/merge evidence). The nine other nearest physical-positive candidates
are V2.6 vault records.

| Source | Candidates | Median | P90 | P99 | Max | Saturated (>=35) | Physical-positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| FAST_V2225 | 245 | 35.0 | 35.0 | 35.0 | 36.5 | 244 | 1 |
| V26_VAULT | 954 | 5.69 | 7.33 | 8.73 | 12.45 | 0 | 9 |

No V1, standalone rescue, registered, or unclassified legacy candidates were
present in this retained pool. Rescue peaks use a temporal map
`absdiff*(1+0.55*clip(zscore,0,6))+0.35*max(dog,0)` before candidate conversion;
bank/vault carries replace score with
`best_score + min(3.4,0.85*(hits-1)) + 0.35`. These paths do not document a
common calibrated probability or likelihood meaning, so higher score is not
proven semantically comparable across sources.

## Correct-candidate versus wrong-winner comparison

The nearest correct candidates in shots 1, 2, 5, 7, 8 and 10 are low-score
V2.6 vault candidates buried at positions 69, 101, 86, 85, 87 and 81. The
deterministic winners are mostly high-score FAST candidates near the front of
the list. Shot 6 is the positive control: its true candidate is FAST, has the
highest score, and is retained at position 1. The canonical AI follows the
same score/provenance ordering and does not fix the mismatch. The frozen
confirmation shadow improves retrospective selection on this development set,
but it is not physical validation.

Among the eight newest-session selection losses where a retained candidate was
within 42 px, six are strongly attributable to cross-source scale mismatch:
the GT candidate is V2.6 vault, the deterministic winner is FAST, and the
winner score is at least 30 while the GT score is below 10. Shots 2 and 3 are
ambiguous same-source/history cases. The other two shots have no retained
candidate within 42 px and cannot be classified as selection-scale losses.
This supports “6/8 oracle-positive losses”, rather than claiming all ten
failures are explained by calibration.

The older smoke2 and biathlon5 traces show compatible low-score vault and
high-score FAST patterns, but their deterministic selections are not preserved
in a uniform selector schema. They are a generality check only, not validation.

## Replay normalization experiments

`automation/score_normalization_research.py` runs fixed deterministic
RESEARCH_ONLY selectors over frozen retained pools: within-source percentile,
source median/MAD robust-z, and confirmation-first percentile.

| Selector | Accuracy @42 | Mean px | Median px | P95 px | >100 px |
|---|---:|---:|---:|---:|---:|
| Current deterministic | 0.10 | 582.59 | 675.99 | 1029.60 | 9 |
| Canonical AI shadow | 0.10 | 602.52 | 698.51 | 1063.83 | 9 |
| Confirmation selection shadow (frozen) | 0.20 | 96.32 | 73.40 | 262.64 | 3 |
| Within-source percentile | 0.10 | 608.81 | 702.37 | 1063.83 | 9 |
| Source robust-z | 0.00 | 408.30 | 347.56 | 795.02 | 10 |
| Confirmation-first percentile | 0.10 | 608.81 | 702.37 | 1063.83 | 9 |

The fixed normalizers do not improve the development set. Source-scale mismatch
is a real mechanism, but normalization alone cannot identify the new hole among
same-source vault artifacts. No `SCORE_NORMALIZATION_SHADOW` is frozen.

## Diagnostic tooling added

Run:

```bash
python3 -m automation.physical_score_audit \
  --root content/ai/physical_traces/session_20260908_144822_d6713dec \
  --comparison evaluation_runs/physical_20260908_145817_9cdbf1e1/selector_shadow_recheck2/physical_comparison.json \
  --output evaluation_runs/physical_20260908_145817_9cdbf1e1/score_audit_<id>
```

It writes `score_audit.json`, `score_audit.md`, and `candidates.csv` without
altering traces. Visual images are intentionally not produced because the
recorded traces do not guarantee a synchronized candidate-to-crop mapping.

## Validation-workflow improvements

`automation.physical_test start` now snapshots the session git commit,
settings, canonical challenger manifest hash, and frozen shadow hash.
`evaluate` reports `INDEPENDENT_PHYSICAL_VALIDATION` only when the shadow hash,
source commit, manifest, and label timing checks pass. Sessions without setup
metadata remain explicitly `DEVELOPMENT`; mismatches are `INVALIDATED`.
Selector metrics retain oracle availability separately from ranking accuracy.

Next validation commands:

```bash
python3 -m automation.physical_test start
# fire exactly the planned shots
python3 -m automation.physical_test check
python3 -m automation.physical_test label
python3 -m automation.physical_test evaluate
```

Do not edit the frozen selector or label shots before runtime selections are
complete. No selector is promoted automatically.

## Research-only conclusions

No live detector or selector experiment was run. The evidence supports a future
research hypothesis—source-balanced or within-source score normalisation before
global ordering—but this session does not tune or install it. A safe experiment
must be replay-only, use a new hypothesis name, and report development metrics
against the frozen baseline before any physical consideration.

## Negative results and limitations

The trace does not retain every raw generator intermediate, so source classes
are flag-based. Track snapshots expose only a debug subset, so persistence for a
candidate absent from that view cannot be inferred. The 10-shot dataset is too
small for calibration or physical claims. No visual diagnostic is generated for
the same synchronization reason.

## Tests and commits

Passed:

```bash
python3 -m automation.overnight_selftest
python3 -m automation.evaluation_selftest
python3 -m automation.physical_trace_selftest
python3 -m automation.async_track_timing_selftest
python3 -m py_compile automation/physical_score_audit.py automation/physical_test.py
git diff --check
```

Earlier checkpoint: `a4c5a12 Add frozen confirmation selection shadow diagnostics`.
This branch adds the score audit and validation metadata in a separate local
commit. `content/ai/settings.json` remains intentionally uncommitted.

## Current status and recommended next test

Work remains on `codex/score-root-cause`. The only intentional working-tree
change outside this work is `content/ai/settings.json`. The next useful step is
an independent physical session using the commands above, followed by the
automatic three-selector comparison and score audit. Treat the result as
validation only if the report says `INDEPENDENT_PHYSICAL_VALIDATION`.

## Independent 20-shot validation and causal follow-up — 2026-09-08

The earlier next-validation recommendation above is now fulfilled by
`session_20260908_163626_9b47fac6`, evaluated in
`physical_20260908_172039_ae9d026f`. Its **INDEPENDENT_PHYSICAL_VALIDATION** result
is preserved unchanged: CURRENT, frozen confirmation shadow and canonical AI
shadow each achieve **1/20 @42 (5%)**. Their mean errors are respectively
509.362, 415.914 and 557.925 px. Lower geographical error did not improve Top1;
no confirmation-shadow promotion is justified. Events 7 and 15 are nonphysical
triggers; the remaining events map to deliberate dart shots 1–20 in order.

See [CAUSAL_CANDIDATE_AUDIT.md](CAUSAL_CANDIDATE_AUDIT.md) for the complete source,
all-event timeline, measurements, limitations and reproduction commands.
Causal oracle is **10/20 @5/@10/@20 and 12/20 @42**. Last-observed retained
oracle appears to be 11/20 @5 and 12/20 @10/@20, 14/20 @42. The extra successes
at events 6 and 14 are actually later-event candidates, recorded after their
live decisions. The original selector comparison already used decision pools
and correctly reported 12/20 @42; its validation is not rewritten.

**PROVEN upstream score cause:** the crop wrapper supplies crop-local current
images to V2 but leaves PRE frame history in full-camera coordinates. V2 slices
that history with a crop-local bbox, comparing unrelated image regions. Offline
source-frame reconstruction matches all **681 physical FAST proposal PSC values
exactly**. Correcting only the PRE spatial origin reduces median residual from
192.384613 to 1.538462 at those frozen coordinates. This is an intensity-residual
measurement, not an accuracy claim. The source-scale mismatch is real, but a
reference-image defect precedes it; normalization cannot repair invalid evidence.

Fifteen of 19 wrong winners (78.9%) are FAST with PSC >=100 and saturated base
scores; all 19 pass permissive local confirmation, six with zero darkening.
Four wrong winners have legacy/vault provenance. The sole correct shot is event
11 / physical shot 10: V2 returns `waiting_post_peak`, the genuine legacy proposal
ranks first, and no broken FAST proposal competes. Its error is 1.718733 px.

The observed 6→7 and 14→15 contamination affects diagnostics, not their completed
live decisions. **A separate runtime correctness gap is reproduced:** an earlier
still-pending event can use local-confirmation evidence after the next audio
peak, and shared tracks have no next-event evidence guard. This is the highest
priority next fix. No live authority code was changed by this measurement audit.

Two fixed RESEARCH_ONLY temporal rankings using correctly positioned immediate
PRE and the causal confirmation frame each achieve 1/20 @42; means are 313.276
and 346.184 px. These offline candidate tests do not improve Top1 and are not
live-path-equivalent replay. They neither replace the frozen confirmation shadow
nor constitute physical validation. Repair pending-event evidence ownership,
then test spatial-reference correction in full detector replay as a separate
hypothesis, before another frozen physical evaluation.

Pending-event ownership is fixed in commit `5d45527`: local confirmation now
stops at the next audio peak, while delayed worker results remain valid when
their captured evidence frame predates that boundary. Physical validation remains
pending. The V2 PRE spatial mapping correction is implemented in the working
tree and translates crop-local bboxes to full-camera frame-history coordinates
exactly once. Offline replay of 731 recorded FAST proposal versions reduces
median PRE residual 192.385→1.538 and removes all corrected score saturation
(median and p90 corrected score 3.6; 0 saturated). This proves the input-plane
bug naturally caused the observed saturation in replay; regenerated candidate
recall and live accuracy remain unvalidated.
