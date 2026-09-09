# Future physical collection plan

## Recommended design

Use **3 independent sessions × 10 shots = 30 shots**. Hold the application
and detector commit fixed, but vary realistic scene state between sessions:

- Session A: 4 light/flat or low-texture shots, 3 low-contrast shots, 3
  nearby/repeated impacts.
- Session B: 4 printed-line/edge-heavy shots, 3 dark-region shots, 3 old-hole
  proximity or grouping shots.
- Session C: 3 textured shots, 3 mixed light/dark shots, 2 low-contrast shots,
  2 deliberate no-impact audio events.

Fire intended sectors in a documented order, but label only after the trace is
finalized. Keep Session C untouched until a research verifier configuration is
frozen on Sessions A/B and historical development data.

This design targets the measured domain gap: the prior 22-event session has
the highest edge density, while independent low-contrast, dark, texture and
line conditions are sparse. Ten shots per session is enough to expose session
shift without making one session dominate training.

## Budgets

- **Minimum:** 2 sessions × 8 shots (16): tests whether a verifier transfers
  across one new scene shift; weak for final validation.
- **Recommended:** 3 × 10 (30): supports development on two sessions and one
  untouched validation session.
- **Ideal:** 4 × 10 (40): three development/robustness sessions plus one
  untouched validation session, including four no-impact events.

At the observed cadence, 10 shots take roughly 10–15 minutes including reloads;
manual labeling should take about 1–2 minutes per event with the assisted
workflow, so the recommended plan is roughly 45–75 minutes total.

## Required trace fields

Every event must retain shared PRE and causal POST frame references, decision
cutoff, registration metadata, complete eligible pool, consumed observation
ledger, confirmation input/best XY, and ownership tags. Labels must be recorded
after finalization and never fed into detector state.

## Commands

Generate a manifest:

```text
python3 -m automation.physical_capture_plan --sessions 3 --shots-per-session 10 --output evaluation_runs/physical_capture_plan.json
```

After capture and labeling, rebuild diagnostics:

```text
python3 -m automation.rebuild_physical_research --output evaluation_runs/physical_research_rebuild_<id>
```

Freeze any verifier on development sessions only, then evaluate Session C once
as untouched physical validation. Do not recycle Session C into training.
