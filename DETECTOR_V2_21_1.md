# V2.21.1 – Controlled full-frame projector/camera capture

## Why this patch exists

V2.21 produced three decisive measurements:

- the historical 100 V2.16 packs contain no honest full recent-PRE/full POST frames,
- the synthetic-vs-camera feature distributions are trivially separable (group domain AUC 1.0000),
- therefore a full-frame AI_DIRECT benchmark cannot be fabricated from the old patch-only archive.

The next experiment must collect **new projector/camera-domain packs with honest full-frame temporal evidence**.

The previous automation entry point always used the scene default of 100 F2 iterations.  V2.21.1 adds an explicit iteration override so we can first collect 10 frames as a storage/plumbing smoke test, then 30-shot sessions for direct-proposal measurement.

## Code changes

### `automation/autostart_ai_training.py`
Adds:

```text
--iterations N
```

The external automation sends a structured `startAITraining` request containing the requested run length.

Examples:

```bash
python3 -m automation.autostart_ai_training white --iterations 10
python3 -m automation.autostart_ai_training checker_anim --iterations 30
```

### `app.py`
`startAITraining` now accepts either the old form:

```text
[background]
```

or the new forms:

```text
[background, iterations]
{"background": ..., "iterations": ...}
```

The requested count is validated to `1..10000` and assigned to the new automation scene **before** `on_enter()` runs.  Old callers remain compatible and still receive the scene default of 100.

### `src/engine/scenes/automation_ai_training.py`
The `aiTraining.started` event now also exposes `target_iterations`, making it visible in automation logs/diagnostics.

## What does not change

- No live hit-selection authority changes.
- V1/V2 remains untouched.
- V2.18 ranking authority remains untouched.
- No model is retrained.
- The new full-frame data is capture/evaluation evidence only until its quality is measured.

## Experiment gate

First collect 10 shots.  Continue to 30-shot multi-background sessions only if the audit confirms honest full-frame recent PRE + POST data.

The next decision metric is **AI_DIRECT union oracle recall** on those new projector/camera-domain packs.
