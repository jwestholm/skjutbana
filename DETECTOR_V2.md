# Detector V2.1 — high-recall camera candidate pipeline

## Purpose

Detector V2.1 is an experimental high-recall path that runs in parallel with
the existing `HitScanner` detector.

The legacy detector remains the baseline. V2 candidates are merged with the
legacy candidates before the existing tracking / AI pipeline. If V2 fails on a
frame, the legacy result is returned unchanged.

Runtime rollback:

```json
{
  "enabled": false
}
```

in:

`content/ai/detector_v2.json`

The configuration is hot-reloaded.

---

## Baseline that motivated V2.1

The first real V2 benchmark used 1000 synthetic white-background shots observed
through the physical projector/camera setup.

At 42 px candidate recall:

```text
legacy V1 : 655 / 1000 = 65.5 %
V2 frame  : 716 / 1000 = 71.6 %
merged    : 751 / 1000 = 75.1 %
```

V2 recovered useful holes that V1 missed, but two important bottlenecks were
visible:

1. Many misses still had strong signal near synthetic ground truth but no V2
   peak.
2. The F2 training report was far lower (~11 % Found) than the best/ever
   detector recall (~75 %), indicating that candidates seen in one camera frame
   may disappear before the exact later evaluation snapshot.

V2.1 is designed specifically around those two observations.

---

## V2.1 changes

### 1. Independent temporal rescue

The original V2 temporal rescue was still indirectly gated by the composite
saliency threshold.

That could reject a true temporal signal when:

- the soft projector-artifact prior lowered composite saliency,
- the pre-existing edge prior lowered composite saliency,
- nearby projected texture won the composite local competition.

V2.1 builds an independent temporal response from:

```text
absdiff
+ z-score
+ multiscale temporal contrast
```

and computes a separate robust MAD-based threshold for it.

This gives a small, real camera change a second way to nominate a candidate
without simply lowering every detector threshold globally.

Important configuration:

```json
{
  "rescue_temporal_robust_sigma": 2.7,
  "rescue_temporal_min_score": 6.0
}
```

---

### 2. Peak centre refinement

Candidate peaks are refined in a small bounded patch using temporal evidence.

This matters because a synthetic hole can contain both a dark core and a bright
rim. The strongest individual response can therefore be on the rim rather than
at the physical centre.

Refinement is bounded to a few pixels so a projected edge cannot drag a
candidate across the image.

---

### 3. Per-shot candidate bank

This is the largest structural change.

The existing AI runtime deliberately replaces its candidate snapshot on every
camera frame. That protects later shots from stale data, but it also means:

```text
frame N:   true hole candidate exists
frame N+1: candidate is weaker / absent
...
F2 evaluates later snapshot
=> the earlier valid candidate is gone
```

V2.1 has a bank scoped to the current `shot_id`.

It does **not** preserve arbitrary one-frame candidates.

A bank entry can be carried only after repeated evidence on distinct camera
frames.

Key safeguards:

- a bank entry can match at most one candidate per camera frame;
- weak rescue candidates require a consecutive observation streak;
- unconfirmed entries expire after a short interval;
- an absent unconfirmed candidate is never emitted;
- confirmed candidates may remain through the F2 evaluation window;
- carried candidates have their own hard output budget;
- the bank is cleared when the shot resolves / scanner resets.

This was stress-tested because a first naive bank could slowly accumulate random
noise peaks. The current design keeps no-hole output bounded instead of growing
toward the candidate limit.

Relevant configuration:

```json
{
  "candidate_bank_enabled": true,
  "candidate_bank_merge_radius_px": 4.0,
  "candidate_bank_unconfirmed_max_age_s": 0.12,
  "candidate_bank_confirm_min_span_s": 0.02,
  "candidate_bank_primary_carry_min_hits": 2,
  "candidate_bank_rescue_carry_min_hits": 3,
  "candidate_bank_rescue_min_hits": 3,
  "candidate_bank_carried_limit": 40,
  "candidate_bank_max_age_s": 1.35
}
```

---

## Full V2 pipeline

```text
existing V1 candidate generator
               |
               +--------------------------------------+
                                                      |
camera frames                                         |
    |                                                 |
recent pre-shot stack                                 |
    |                                                 |
robust pre reference + temporal noise                 |
    |                                                 |
small global registration                             |
    |                                                 |
photometric normalization                             |
    |                                                 |
absdiff / darkening / per-pixel z-score               |
    |                                                 |
multiscale local contrast                             |
    |                                                 |
soft artifact + edge priors                           |
    |                                                 |
persistent change map                                 |
    |                                                 |
    +--> composite saliency local maxima              |
    |                                                 |
    +--> lower-gate saliency rescue                   |
    |                                                 |
    +--> independent temporal rescue                  |
                    |                                 |
             peak refinement                          |
                    |                                 |
           spatial quota + NMS                        |
                    |                                 |
              frame V2 candidates                    |
                    |                                 |
        confirmed per-shot candidate bank            |
                    |                                 |
              banked V2 candidates ------------------+
                              |
                    hybrid V1/V2 merge
                              |
                       HitScanner / AI
```

---

## Machine-readable diagnostics

File:

```text
content/ai/detector_v2/shot_diagnostics.jsonl
```

Each synthetic labelled shot can contain four different detector views:

```text
legacy       best/ever V1 candidate during the shot
v2_frame     best/ever unbanked V2 frame candidate
v2           best/ever V2 candidate after candidate bank
merged       best/ever hybrid candidate
```

V2.1 also records the **actual F2 evaluation snapshot**:

```text
evaluation_funnel.raw
evaluation_funnel.filtered
evaluation_funnel.ranked
evaluation_funnel.selected
```

and detector provenance in that exact raw snapshot:

```text
raw_v1_nearest_px
raw_v2_nearest_px
raw_v2_bank_confirmed_nearest_px
raw_v2_bank_carried_nearest_px
raw_v2_bank_carried_count
```

This distinction is critical.

`BEST/EVER candidate recall` answers:

> Did the camera/detector see the hole at any useful time?

`ACTUAL F2 EVALUATION recall` answers:

> Was that candidate still available when the training evaluator actually
> looked?

---

## Analyse a run

After a new game process has run synthetic training:

```bash
python3 -m automation.detector_v2_analyze
```

By default the analyzer selects the newest Detector V2 runtime session.

It writes:

```text
content/ai/detector_v2/latest_summary.json
```

and prints:

- V1 / V2-frame / V2-bank / merged recall at 10, 20 and 42 px;
- number of shots recovered by the candidate bank;
- actual F2 raw → filter → rank → selected funnel;
- V1/V2/banked provenance at the F2 snapshot;
- where ground truth was lost;
- median signal exactly around synthetic ground truth;
- miss/recovery classifications by background.

Historical analysis:

```bash
python3 -m automation.detector_v2_analyze --all
```

One explicit runtime:

```bash
python3 -m automation.detector_v2_analyze --session SESSION_ID
```

---

## Recommended next benchmark

Start with 10 complete runs, not another 100-run marathon:

```bash
python3 -m automation.ai_training_loop 1 10
```

Then:

```bash
python3 -m automation.detector_v2_analyze
```

For the first V2.1 test, compare these numbers in this order:

1. `v2 frame` vs old V2 frame recall — did temporal rescue improve candidate
   generation?
2. `v2 bank` vs `v2 frame` — how many shots were saved by temporal candidate
   preservation?
3. `merged BEST/EVER` vs `ACTUAL F2 raw` — is the large snapshot-loss gap
   shrinking?
4. `raw -> filtered -> ranked -> selected` — where is the next bottleneck?

Do not judge V2.1 only by AI Top-1. Candidate generation and AI ranking are
different stages.

---

## Internal smoke tests before packaging

The update was checked with synthetic image and candidate-bank tests.

Observed properties:

- weak temporal signal suppressed below composite saliency can still be nominated
  by the independent temporal rescue;
- a candidate observed on several consecutive frames remains available after it
  disappears from later frames;
- a weak rescue candidate is not carried after only one/two frames;
- random single-frame candidate streams remain close to current-frame output
  rather than growing toward the bank output limit;
- full no-hole synthetic camera sequences remained bounded instead of ramping
  toward ~150 carried candidates;
- source files compile and the JSON configuration matches all default keys.

These are software smoke tests, not substitutes for the physical
projector/camera benchmark.

---

## Important invariant for future AI work

Candidate generation should optimise primarily for recall.

A true hole that never reaches the candidate list cannot be recovered by
ranking.

But high recall must not be achieved by unbounded temporal accumulation.
Therefore:

```text
permissive CURRENT candidate generation
              +
strict evidence requirement for CARRYING a candidate
              +
later filtering/ranking
```

is the intended V2.1 architecture.
