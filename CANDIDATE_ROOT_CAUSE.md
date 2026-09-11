# Candidate root-cause diagnosis

This is read-only offline diagnostic work on the DEVELOPMENT session. It does
not execute or modify the live detector, tune parameters, train models, or
promote an artifact.

Run it with:

```bash
python3 -m automation.candidate_root_cause \
  --session content/ai/training_v223/sessions/20260829_203625_ai_training_scene_e31e210c \
  --output evaluation_runs/candidate_root_cause_20260907
```

The report reads the saved native candidate records and computes oracle coverage,
nearest-candidate distances, and offsets. It checks for matching framepacks, but
the DEVELOPMENT session has none. Consequently ROI membership, coordinate
consistency against pixels, PRE→POST signal, darkening/registered evidence,
pre-filter peaks, threshold rejection, suppression, and candidate limits are all
reported **UNAVAILABLE**. The tool never turns an absent pixel trace into a
candidate-generation failure.

The DEVELOPMENT session contains 100 projected F2 shots. Its saved candidate
pool has no candidate within 20 px for 91 shots. This establishes that the saved
pool is sparse, but it does not establish why the current V2.25.x live detector
would fail: the capture lacks the temporal images and runtime telemetry required
to distinguish scene/coordinate mismatch from generation, filtering, threshold,
or confirmation loss. The correct conclusion is therefore **F2 capture mismatch
or missing temporal data dominates**, and this dataset is insufficient for
candidate-generator optimization. The `TEMPORAL_DATA_INSUFFICIENT` label does
**not** mean that the live detector has a temporal failure. It means the stored
dataset cannot determine the actual failure cause. The old F2 dataset must not
be used for further candidate-generator optimization.

Nearest-candidate offsets are descriptive only. They are not evidence of a
systematic detector offset without verified coordinate/image correspondence.
Likewise, a nominal camera coordinate being finite does not prove it lies inside
the valid ROI.

No diagnostic images are generated: there are no matching DEVELOPMENT pixels from
which a truthful ground-truth/ROI/evidence overlay could be made. A small image
sample becomes appropriate only after linked physical or replay captures exist.

Required next capture/runtime instrumentation: shot ID and audio/PANG timestamps;
full-resolution PRE history; every POST frame through confirmation and rescue
timeout; calibration generation, camera dimensions, crop/working geometry and
valid ROI; raw response maps/peaks before thresholds; filtering rejection reasons;
candidate-limit/quota counters; registered readiness and confirmation outcomes;
rescue branch; selected and emitted coordinates with monotonic latency. Preserve
the same independent ground-truth coordinate and all settings/model hashes.

No physical or live-performance conclusion can be made from this result. The
evaluated data is projected F2 data, and it does not contain the evidence needed
to connect these saved-pool misses to current V2.25.x physical behavior.
