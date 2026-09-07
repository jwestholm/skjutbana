# Physical trace capture

This instrumentation records observations around the existing V2.25.x scanner;
it does not change thresholds, ranking, candidate selection, confirmation,
rescue, or emission behavior. Capture is disabled by default. A writer thread
performs disk I/O; if its queue is full or writing fails, the detector continues
and records a structured error. Trace JSON is finalized after queued artifacts
are persisted, so it never advertises a frame or evidence file that failed to
write. Each trace includes expected/persisted PRE, POST, and evidence counts,
queue drops, writer errors, and a `trace_complete` flag.

The asynchronous queue is bounded to 128 jobs and a 1 GiB payload budget; the
writer uses multiple workers, while detector-side capture remains nonblocking.
Shutdown performs a bounded flush and reports any incomplete persistence.

## Enable for one physical session

Edit the ignored local file `content/ai/settings.json` and add:

```json
{
  "physical_trace_capture_enabled": true,
  "physical_trace_root": "content/ai/physical_traces/session_20260907"
}
```

Preserve existing settings when adding these keys. Start the game normally from
the repository root:

```bash
python3 main.py
```

Before shooting, verify the application starts and tracing is enabled in the
runtime log if that status is exposed. During the session, shoot normally and
keep the camera/audio path unchanged. Each audio shot receives a unique scanner
`shot_id`; the recorder stores the pre-shot snapshot/history, every observed POST
frame through event completion/timeout, selected debug maps when available,
candidate observations, tracks, thresholds/window telemetry, event state and
outcome. Frames and maps are written under `content/ai/physical_traces/` and are
ignored by Git.

After shooting, disable capture by setting `physical_trace_capture_enabled` back
to `false` (or removing the key) and restart the game. Capture remains disabled
by default and is never enabled by source code.

The complete workflow is: (1) enable tracing in local settings, (2) start the
game, (3) shoot normally, (4) stop the game, (5) run the label command, (6)
click and accept each visible hole, (7) export the traces, (8) run the
evaluation command, and (9) disable tracing again before the next normal run.

## Label physical ground truth by clicking

The recorder never invents labels. After stopping the game, run one interactive
labeling session. It opens the best captured POST frame for each unlabeled shot;
the click is stored in original full-camera coordinates and is never snapped to a
detector candidate:

```bash
python3 -m automation.physical_trace_label \
  --root content/ai/physical_traces/session_20260907
```

When `content/settings.json` contains `camera_calibration.homography`, the
default view is a perspective-corrected target/projector view with enhanced
physical PRE→POST evidence overlaid. Use `--calibration PATH` to select another
saved calibration, or `--target-width`/`--target-height` to change the target
canvas size.

Controls:

- Left click: mark the exact hole location.
- Enter or Space: accept the click and save `ground_truth.json`.
- R: clear the click and retry.
- S: mark the shot unresolved and continue.
- Left/Right arrows: previous/next shot.
- Up/Down arrows: previous/next captured POST frame.
- `1`: target/projector view; `2`: enhanced physical difference; `3`: raw POST;
  `4`: raw PRE.
- C: toggle optional yellow detector-candidate overlays (off by default).
- Q or Escape: quit; rerun the same command to resume.

The tool prints total, labeled, skipped/unresolved, and remaining counts. To
relabel existing human annotations, use `--include-labeled`; to revisit skipped
shots, use `--include-skipped`. A skipped shot is recorded separately in
`ground_truth_status.json` and is not treated as detector output.

The target view uses only saved camera frames and calibration geometry. It does
not use resolver results, selected candidates, game hit markers, or any other
detector output as ground truth. If a session has frame files but lost its
`trace.json`, the labeler can reconstruct display-only frame metadata; it does
not invent detector stages or outcomes.

For scripted or headless attachment, the existing command remains available:

```python
from src.engine.physical_trace import get_physical_trace_recorder
get_physical_trace_recorder().attach_ground_truth(shot_id=1, camera_xy=(1234.5, 678.0))
```

Equivalent command-line form:

```bash
python3 -m automation.physical_trace_attach_gt \
  --root content/ai/physical_traces/session_20260907 \
  --shot-id 1 --camera-x 1234.5 --camera-y 678.0
```

The label is stored separately as `ground_truth.json`, with `space: "camera"` and
`label_source: "manual_verified"`. Attach labels only after the physical session.

## Export and evaluate

Export self-contained traces to the existing evaluation schema:

```bash
python3 -m automation.physical_trace_export \
  --root content/ai/physical_traces/session_20260907 \
  --output content/ai/physical_traces/session_20260907/evaluation_trace.json
python3 -m automation.evaluate_pipeline \
  --trace content/ai/physical_traces/session_20260907/evaluation_trace.json \
  --dataset-id physical-session-20260907 \
  --output evaluation_runs/physical-session-20260907
```

The export preserves missing stages as unavailable; it does not infer raw,
filtered, confirmation or rescue lineage from later candidates. A real physical
session is required to verify timing, frame completeness, calibration context,
and writer performance. No physical capture has been run or validated yet.
