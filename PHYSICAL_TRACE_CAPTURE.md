# Physical trace capture

This instrumentation records observations around the existing V2.25.x scanner;
it does not change thresholds, ranking, candidate selection, confirmation,
rescue, or emission behavior. Capture is disabled by default. A writer thread
performs disk I/O; if its queue is full or writing fails, the detector continues
and records an error count.

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

## Attach physical ground truth

The recorder never invents labels. After independently verifying a hole in camera
coordinates, attach it to a shot directory:

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
