# Future physical collection plan

## Current accuracy objective and next evidence — 2026-09-09

Target **100% actual correct physical output**, with **95% minimum acceptance**.
The completed common-verifier pass is documented in
[ACCURACY_95_100_RESEARCH.md](ACCURACY_95_100_RESEARCH.md). It tested registered
patch combinations, bounded proposal expansion, earlier/guarded PRE references,
broader negatives, pairwise hard-negative training, visual no-impact gating and
contour rescue. No live authority changed. S01/S02 results are development
evidence; S03 was not opened or used and remains reserved final validation.

The new offline method selects 5/9 held-out S01 and 3/10 S02 at @5/10/20/42.
Expanded S02 oracle reaches 9/10 @42 but actual selection remains 3/10; the
common verifier also emits for 3/3 known false S02 events. A rejection setting
that rejects all three false events rejects five real shots. More oracle points
alone cannot meet the physical objective.

The smallest next diagnostic capture is **six discharges plus six known
no-impact events**, under a fresh development binding. Include light, dark,
printed edges and nearby/repeated/taped regions. Preserve continuous PRE/onset/
POST imagery and an independent discharge log, identifying each visible change
immediately. Include isolated no-impact events and events following real shots;
do not encode a minimum-time rejection rule. Keep UNKNOWN/AMBIGUOUS truth where
newness is not defensible. S02 event 1's possible second change is unconfirmed
and its original label remains unchanged.

This small capture targets irrecoverable overwritten rejection ledgers,
uncertain physical onset/newness and sparse defensible negatives. It is not a
95% validation claim. Once a complete candidate is frozen near the target,
require 50+ and preferably 100+ varied independent physical shots, with separate
no-impact and nearby/repeated checks. S03's ten-shot plan alone is insufficient
to establish >=95% reliability. The original collection design below remains
historical development guidance, not the final acceptance sample size.

## Recommended design

Use **3 independent sessions × 10 shots = 30 shots**. Hold the application
and detector commit fixed, but vary realistic scene state between sessions:

- Session A: 4 light/flat or low-texture shots, 3 low-contrast shots, 3
  nearby/repeated impacts.
- Session B: 4 printed-line/edge-heavy shots, 3 dark-region shots, 3 old-hole
  proximity or grouping shots.
- Session C: 3 textured shots, 3 mixed light/dark shots, 4 low-contrast shots,
  plus 2 deliberate no-impact audio events recorded separately.

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

After capture and labeling, rebuild diagnostics only through an explicit
development-session view that excludes S03:

```text
python3 -m automation.rebuild_physical_research --root /data/skjutbana/development_sessions_<id> --output /data/skjutbana/evaluation_runs/physical_research_rebuild_<id>
```

Freeze any verifier on development sessions only, then evaluate Session C once
as untouched physical validation. Do not recycle Session C into training.
The existing generic patch builder includes later frames; its output is an
offline information dataset, not a causal verifier replay. Future verifier
experiments must use each event's recorded decision cutoff and eligible pool.

## Workflow safeguards

Use `physical_collection start` to bind a session manifest and
`physical_trace_quality` before finalization. The validation guard refuses
training/tuning on S03. See [PHYSICAL_CAPTURE_RUNBOOK.md](PHYSICAL_CAPTURE_RUNBOOK.md).
Use `physical_collection reset-labels --binding ...` to preview a label-only
reset; `--apply` archives and removes only explicit label/assignment artifacts.
It never deletes captured evidence. S means unresolved in the labeler; classify
human-confirmed nonphysical events explicitly.

## Completed development sessions — 2026-09-09

S01 contains ten physical labels plus nonphysical event 4. Nine physical events
have complete decisions: oracle @42 is 9/9 and CURRENT 3/9. Event 11 retains
images and GT but lacks a complete terminal decision; its accuracy is unknown.
S02 contains ten physical labels plus nonphysical events 2,8,10, with all 13
decisions complete. Oracle @42 is 5/10 and CURRENT 1/10; all three nonphysical
events emitted false hits. See [S02_PHYSICAL_FINDINGS.md](S02_PHYSICAL_FINDINGS.md).
Keep these as development evidence and leave S03 untouched until a verifier is
frozen. No ranking or live authority change follows from these results.
