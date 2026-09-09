# Physical capture runbook

The plan means **10 real physical shots per session, plus any deliberate
no-impact audio events recorded separately**. No-impact events are never
silently counted as physical shots.

## Before shooting

```bash
python3 -m automation.physical_capture_plan --sessions 3 --shots-per-session 10 --output evaluation_runs/physical_capture_plan.json
python3 -m automation.physical_collection start --plan evaluation_runs/physical_capture_plan.json --session S01 --output evaluation_runs/S01_binding.json
python3 -m automation.physical_test start --prepare-only
```

Confirm the displayed session class and intended category sequence. The
detector never receives intended coordinates.

## Session A / B

Start the application using the existing project command, fire the displayed
10-shot sequence, and run a non-destructive health check after 3–5 shots:

```bash
python3 -m automation.physical_test check
```

Close the application normally so traces flush. Then run the quality report:

```bash
python3 -m automation.physical_trace_quality --output evaluation_runs/S01_quality.json
```

Label each event with the existing `physical_test label` workflow. Mark false
audio events `NO_PHYSICAL`, and use `AMBIGUOUS` or `SKIP` rather than guessing.
Finalize only when every event is resolved.

## Session C — untouched validation

Bind `S03` explicitly. Do not train, tune, fit, select a configuration, or
inspect it while developing. Only run postflight, label consistency and the
final frozen evaluation path.

```bash
python3 -m automation.physical_collection start --plan evaluation_runs/physical_capture_plan.json --session S03 --output evaluation_runs/S03_binding.json
```

## Rebuild and final validation

```bash
python3 -m automation.rebuild_physical_research --output evaluation_runs/physical_research_rebuild_<id>
```

The quality report must be PASS/WARN with no missing frame artifacts before
the dataset is used. Validation data is evaluation-only; the guard refuses
training and tuning operations.

## Recovery

- False audio event: record it explicitly as `NO_PHYSICAL`; do not renumber
  later runtime events.
- Wrong label: leave the trace immutable, correct the label manifest, and rerun
  consistency checks.
- Crash/incomplete session: keep it marked incomplete; do not claim ten shots.
- Reused session id: bind a fresh unique trace root and refuse overwrite.
- Missing artifact: stop and repair capture workflow before continuing.

## Hardened one-command flow

Before each session, run the preflight (it refuses reused bindings and reports
NOT READY with reasons):

```bash
python3 -m automation.physical_collection preflight --plan evaluation_runs/physical_capture_plan.json --session S01
```

Bind/start with the existing prepare-only capture lifecycle. After labeling and
quality output, finalize through the guarded wrapper:

```bash
python3 -m automation.physical_collection finalize \
  --plan evaluation_runs/physical_capture_plan.json \
  --session S01 \
  --labels evaluation_runs/S01_labels.json \
  --quality evaluation_runs/S01_quality.json \
  --output evaluation_runs/S01_finalized.json
```

An interrupted or crashed session remains incomplete and must not be silently
resumed. Bind a new session unless the trace health report proves the original
session was fully flushed.
