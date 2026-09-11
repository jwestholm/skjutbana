# Long-horizon physical impact evidence

This is an explicitly future-aware offline diagnostic. It never enters live
selection or exact replay.

## Frame availability

The latest session stores post-decision frames for events 1–9, with total
post horizons of approximately 2.4–2.5 seconds for most events. Events 4, 5
and 7 have only about 0.7–0.8 seconds. The next event's pre-snapshot is
available for the first nine events and provides a stable-state diagnostic;
event 10 has no following event.

## Permanence result

Long-horizon next-event PRE residuals do not reliably favor physical truths.
For the seven failed events, false winners remain more persistent than the true
track in shots 1 and 6, while true persistence wins in shots 2, 3 and 7. The
remaining failed cases are mixed. Ranking all eligible tracks by next-event
center darkening gives true ranks `[22,64,29,4,23,51,3,22,5,None]` for shots
1–10: @5=3, @10=3, @20=3, @50=7.

At the approximately one-second post horizon, coverage is 682/973 rows and
true ranks are `[17,49,48,None,None,27,None,39,4,25]`, with @5=1 and @10=1.
Waiting longer therefore does not improve the known final-ranking problem in
this session.

Known impacts do remain measurable in several later frames, but the same is
true of selected nuisance winners. Persistence is not equivalent to physical
causation here.

## Interpretation

**PROVEN:** the trace contains useful long-horizon imagery and next-event
pre-snapshots for nine events; future-aware permanence was evaluated separately
from causal evidence.

**NOT SUPPORTED:** long-horizon permanence alone is not a reliable final
ranking signal and does not justify adding a wait-for-settle live policy.

**RESEARCH_ONLY:** stable-before/stable-after deltas remain useful diagnostics,
but no stable-state ranker or `NO_VALID_IMPACT` policy is frozen.

The remaining information gap is semantic identity of the changed structure,
not simply whether a change persists. Candidate provenance, morphology and
physical hole appearance need richer evidence than scalar persistence.

Reproduction:

```text
python3 -m automation.long_horizon_impact --root content/ai/physical_traces/session_20260908_194746_b38de674 --features evaluation_runs/registered_impact_20260909/physical_v2/physical_features.json --output evaluation_runs/registered_impact_20260909/long_horizon.json
```
