# FAST selection audit (post PRE and ownership fixes)

## Dataset and result

Session `session_20260908_194746_b38de674` contains ten deliberate physical
shots and ten audio events, with no false events. Evaluation
`physical_20260908_195522_ab2304cd` is an `INDEPENDENT_PHYSICAL_VALIDATION`.
Causal candidate availability is 7/10 at 5 and 10 px, and 9/10 at 20 and
42 px. CURRENT is 2/10 at every tolerance (mean 171.10 px, median 135.75 px,
p95 464.70 px, eight errors over 100 px).

## True candidates versus live winners

The nearest causally available candidate for shots 1,2,3,4,6,7,8,9,10 was
respectively 2.24, 3.80, 3.22, 3.29, 2.60, 3.49, 10.08, 4.15 and 11.49 px
from the physical hole. Their retained ranks were 67,67,48,2,63,57,68,1,17.
Sources were V26/V1 for shots 1,2,3,4,6,7,9 and FAST/V2 for 8 and 10.
Only shots 4 and 9 were selected; both winners were V1 and within 4.2 px.
The other seven winners were FAST (distances 107.39–464.70 px), despite
local confirmation. The nearest true candidates were generally low-score
vault candidates (1.18–6.28) and were buried below score-ordered FAST tracks;
shot 8/10 true FAST proposals were also eligible but lost to earlier/higher
scoring FAST tracks. Track ids for the selected winners were 5, 98, 197, 276,
486, 580, 660, 765 and 871; each had two hits on two unique frames. The saved
top-eight debug snapshots do not retain a track id for every nearest true
candidate, so unavailable fields are reported as unavailable rather than
inferred.

## FAST authority path

`shot_async_v2224` dispatches one detector worker. `shot_fast_v2225` emits the
first FAST proposal and seeds local confirmation; V1/vault proposals arrive
through the same retained stream. `shot_track_v2226` creates score-ordered
tracks and records one temporal hit per frame. `HitScanner._best_track_for_event`
filters by onset, then chooses the highest `best_score`. There is no explicit
FAST bonus, but early FAST proposals create tracks first and their score field
has different semantics from vault/V1 scores. This is a strong hypothesis for
the selection bottleneck; it is not a universal invalidity claim.

## Research-only ablations

Using recorded top-eight track snapshots (therefore not a complete live replay):

| policy | evaluated | @5/@10/@20/@42 | mean / median / p95 | >100 px |
|---|---:|---:|---:|---:|
| current semantics | 10 | 2/2/2/2 | 171.10 / 144.30 / 464.70 | 8 |
| exclude FAST | 5 | 2/2/2/2 | 71.52 / 80.48 / 142.47 | 2 |
| prefer non-FAST, FAST fallback | 10 | 2/2/2/2 | 135.36 / 142.47 / 378.92 | 7 |

These are development diagnostics only; no live authority or shadow selector
was changed. The result supports source-aware selection as the next experiment.

The follow-up track-survival audit found that only the two successful true
candidates are visible in the saved eight-track snapshots. The other seven are
in the retained causal candidate pool but have no corresponding exported track;
this is an export limitation, not proof of runtime eviction. See
`TRACK_SURVIVAL_AUDIT.md`.

## Historical comparison and confirmation

FAST-selected success was 0/16 in the prior 20-shot session, 0/7 here, and
1/8 in the earlier 10-shot session; non-FAST was 1/6, 2/3 and 0/2 respectively.
Biathlon5/smoke2 lack comparable causal source fields. FAST saturation is gone
after the PRE fix (retained FAST median 9.91, maximum 16.03; zero saturated),
so saturation and final-authority ordering are separate issues. Local
confirmation currently verifies local change/contrast, not causality of a new
bullet hole; confirm_darkening can be zero (shots 7 and 10) while a FAST winner
remains eligible. This is a semantic gap, not yet a confirmed implementation
bug.

## Recommendation

Keep FAST as proposal/recovery evidence and run a source-aware, research-only
selector replay with complete candidate pools. Require a new physical series
before any promotion. Do not use ground-truth coordinates or tune weights.
