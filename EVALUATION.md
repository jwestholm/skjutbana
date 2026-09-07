# Evaluation contract v1.0

This foundation measures observations, without executing or changing detector policy.
Run from the repository root with Python 3.10+; the framepack adapter also needs
NumPy. No training, parameter sweep, model selection, or hardware access occurs.

```bash
python3 -m automation.evaluation_selftest
python3 -m automation.evaluate_pipeline \
  --framepacks content/ai/training_v223/framepacks \
  --dataset-id v223-existing-framepacks \
  --output evaluation_runs/v223-saved-pool-baseline
python3 -m automation.evaluate_pipeline \
  --trace /path/to/observations.json --dataset-id physical-session-id \
  --output evaluation_runs/physical-session-id
```

Output directories must be new. Each run also archives its Python/Markdown source
as source.tar.gz. Preserve report.json, summary.txt, source.tar.gz,
and all hashed input artifacts together for a frozen baseline. Generated runs are
ignored by Git; copy them to durable storage before cleaning a workspace. Reports
include the full normalized observations, per-shot outcomes, and input hashes.
Reruns have new provenance timestamps; scorecards for identical observations and
Top-K are deterministic. Never compare mixed domains or different dataset hashes
as though they were the same experiment.

## Three evidence levels

A. `offline_candidate`: candidate-only analysis, including historical saved pools.
The framepack adapter reuses `training_v223.framepack` discovery/loading and the
existing offline distance/tolerance functions. It reads the original pool without
ranking, truncation, deduplication, or proposal-sidecar augmentation. It does not
call the detector. `saved_pool` is a separate metric, not raw recall or live Top-K.

B. `live_path_replay`: reserved for a producer that executes FAST, filtering,
retention, registered/temporal confirmation, global FULL rescue, winner selection,
and final emission with recorded timing, context, and state. No such producer is
implemented here. The existing `LiveHybridReplayDetector` calls the real candidate
wrapper, but substitutes full-image ROI, background and timestamps, resets state,
and returns the last POST candidate list. Its name does not establish live parity.
Existing `replay.py`, proposal benchmarks and ReplayMetrics remain useful for A.

C. `physical_validation`: real shots with independently labelled physical camera
locations and complete emission observation windows. F2 projected dots are not
physical holes. V2.25.3 physical acceptance remains pending.

Existing training records normalize features and can coerce missing coordinates;
the new scorecard instead rejects malformed coordinates. Existing AI diagnostics
and training capture contain candidate snapshots, not a complete stage trace.
They cannot reconstruct missing confirmation or emission decisions. A future
producer should attach the contract below to those capture facilities rather than
build a second detector or infer stages from scores.

## One labelled shot

Identity is `(session_id, shot_id)`, unique within a report. All coordinates must
be canonical full-camera pixels, explicitly `coordinate_space: "camera"`. Convert
crop/working, screen, or game coordinates using the shot-time transform before
export. Never divide tolerances by an assumed resolution scale. Ground truth is an
independent physical location, never a selected candidate or object center.

For every tolerance **5, 10, 20, 42 px**, independently, correct means Euclidean
camera distance **<= tolerance**. Every metric includes its evaluated denominator
and unavailable count; missing labels do not count as failures. These tolerances
are evaluation thresholds, not detector settings.

The trace is a JSON object:

```json
{
  "schema_version": "1.0",
  "mode": "offline_candidate",
  "producer_runtime": {
    "effective_detector_settings": null,
    "models": null,
    "calibration": null
  },
  "limitations": ["Illustrative contract, not a measured physical shot"],
  "shots": [{
    "session_id": "example", "shot_id": "1",
    "source_kind": "synthetic_contract_example",
    "coordinate_space": "camera",
    "ground_truth": {"camera_x": 100, "camera_y": 200},
    "raw": null, "filtered": null, "retained": null,
    "confirmed": null, "selected": null, "emitted": null,
    "ranked": null, "rescue_used": null, "latency_ms": null
  }]
}
```

Each stage is either a **complete observed list** of `{camera_x, camera_y}` or
`null`/absent for unavailable. `[]` means observed and empty. Partial/truncated
snapshots must not be passed as complete stages. Extra candidate/shot metadata
is preserved. Producers should include candidate IDs, frame timestamps, branch
(FAST/rescue), decision reasons, and capture completeness for later diagnostics.

- `raw`: all proposals before candidate filtering, including rescue proposals if
  rescue ran. Store per-branch detail as metadata; do not silently omit rescue.
- `filtered`: candidates after filtering, before bounded retention.
- `retained`: candidates actually admitted to the confirmation budget (live K).
- `confirmed`: candidates that passed the physical/temporal authority checks.
- `selected`: zero or one winning candidate after selection/ranking.
- `emitted`: all final camera XY emissions attributed to this shot over a closed
  observation window, including delayed and duplicate events.
- `ranked`: complete final ordered candidate list at an explicitly documented
  ranking boundary. Top-1/3/K use this order, never a score-based reconstruction.
  CLI `--top-k` is an analysis slice; it does not change live retention K.

First loss is assigned only after a fully observed successful prefix. Missing
upstream data yields `not_evaluated`, even if a later stage is known to fail.
Independent stage counts use their own denominators and need not form a monotonic
funnel when evidence is missing or branches recover. First-loss counts describe
location availability, not candidate lineage or causal proof. Trace producers must
aggregate FAST/rescue consistently; candidate IDs and branch traces are needed to
explain an individual rejection. Do not subtract aggregate counts with unequal
coverage to estimate stage losses.

At each tolerance, off-target emissions are false emissions. Correct emissions
beyond the first are duplicates. These counts are disjoint; multiple wrong outputs
are false emissions, not additional correct duplicates. Counts require both GT and
a complete emitted list. Background false-positive rate needs separately labelled
no-shot windows and exposure duration; that metric is not implemented in v1.

`rescue_used` is an observed boolean (null if unknown). `latency_ms` is elapsed
monotonic time from audio PANG/peak to first final emission; record null for no
emission or missing clocks. Report mean, median, nearest-rank p95 and coverage.
Never substitute offline processing duration or timestamp spacing. Capture timeout
and end-of-window metadata so missing hits and censored latency remain auditable.

## Provenance and reproducibility

Every run records evaluator commit, branch, dirty flag/status, Python version,
UTC timestamp, tool/schema version, shot count, all four tolerances, Top-K, dataset
ID and per-input SHA-256 hashes, plus a source manifest/hash including untracked
Python/Markdown files. The report records the observation producer runtime
separately: effective merged detector settings, all loaded model identifiers and
SHA-256 hashes, and calibration identifier, matrices, camera dimensions, crop and
coordinate transforms. Snapshot actual runtime values, including defaults and
local overrides; current local settings cannot reconstruct historical settings.

Unknown producer settings/models/calibration are explicit nulls. The historical
framepacks lack this provenance: their scorecards are reproducible from the saved
observations, but their detector execution is **not** reproducible. No current
model files are loaded or represented as historical models. For future traces,
producer_runtime should also include producer commit/dirty source hash, dependency
versions, backend/thread configuration, RNG seeds where relevant, and state reset
policy. Keep referenced model/calibration artifacts with the capture. A hash alone
does not recover missing bytes. The CLI requires `validation_evidence` for B/C;
this is a producer declaration, not automated parity certification.

## What is needed next

Capture real, independently labelled physical shots with full-resolution PRE
history and enough timestamped POST frames through confirmation, rescue and the
emission timeout. Include shot/audio timestamps, ROI/crop geometry, calibration,
frozen HitRegions, worker/main readiness, known-hole and cross-shot novelty state,
model/settings snapshots, selection and emission events, and no-shot windows.
Keep sequence order and session boundaries for cross-shot behavior. Instrument
existing diagnostic boundaries without changing authority. Then implement a
hardware-input adapter around the actual live scheduler/state machine and verify
its stage/event trace against a real capture before claiming B parity. Finally
run the physical acceptance matrix for V2.25.3; saved F2 recall cannot replace it.
