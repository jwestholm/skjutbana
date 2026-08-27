# Detector V2.21.1 — controlled full-frame projector/camera capture

## Purpose

V2.21 established two facts:

1. historical V2.16 candidate packs are patch-only and cannot honestly benchmark full-frame direct proposals;
2. V2.20 synthetic data and physical projector/camera data are trivially separable (`domain AUC = 1.0`), so more synthetic-only ranking training is not the next useful step.

V2.21.1 makes the next physical capture experiment small and controllable by allowing the automation runner to request an explicit number of F2 iterations.

## Correct application path

The executable entry point remains `main.py`. It imports the application class with:

```python
from src.engine.app import App
```

Therefore automation command handling belongs in `src/engine/app.py`.
The first V2.21.1 package accidentally created `./app.py`; that root file is not part of the intended repository layout and must be removed.

## Changes

- `automation.autostart_ai_training` accepts `--iterations N`.
- `startAITraining` accepts an optional iteration count.
- `AutomationAITrainingScene.auto_target_iterations` is set before scene activation.
- Values outside 1..10000 are rejected.
- Existing callers that only pass a background still use the scene default.
- No live detector authority changes.
- `main.py` is unchanged.

## First experiment

Run only ten white-background rounds first:

```bash
python3 -m automation.autostart_ai_training white --iterations 10
```

Then verify V2.21 full-frame capture:

```bash
python3 -m automation.physical_pack_v221_inspect \
  --root content/ai/candidate_shadow_v216
```

If new packs are direct-ready, measure:

```bash
python3 -m automation.direct_proposal_v221_benchmark \
  --root content/ai/candidate_shadow_v216
```

The first goal is candidate recall/oracle improvement from AI direct proposals, not Top-1 ranking.
