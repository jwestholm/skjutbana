# DETECTOR V2.21 — Domain Gap + Direct Proposal Foundation

## Why this version exists

The decisive experiment on 2026-08-27 showed:

- V2.18 trained on V2.20 generated seeds `1..100` reached ~98–100% Top-1 on frozen unseen generated seeds `9000001..9000100`.
- The exact same frozen model reached 0% Top-1 on the existing projector/camera V2.16 candidate session.
- Existing physical/projector-camera candidate union oracle is only about 35% confirmation / 50% holdout within 20 px.

Therefore V2.21 does **not** train longer. It introduces the tools required to measure and attack:

1. synthetic -> camera domain gap,
2. missing-candidate / proposal-recall ceiling.

Live hit authority and candidate ordering are unchanged.

---

## Implemented in this delta

### 1. Physical candidate-pack audit

`src/engine/offline/physical_pack_audit_v221.py`

`automation/physical_pack_v221_inspect.py`

The audit answers explicitly:

- whether GT patches exist,
- whether true recent-PRE candidate patches exist,
- whether full reference PRE exists,
- whether full recent PRE exists,
- whether full POST stack exists,
- whether an honest direct full-frame benchmark can be run,
- candidate oracle without forced GT-nearest diagnostic rows,
- split-by-split readiness.

It never uses `shot_diag` as raw truth.

### 2. Candidate-domain gap profiler

`src/engine/offline/domain_gap_v221.py`

`automation/domain_gap_v221.py`

It compares the same V2.17/V2.18 representation between generated and projector/camera candidates:

- V2.17 probability,
- offset magnitude,
- eight temporal scalar features,
- embedding norm/mean/std,
- known-hole-distance diagnostic,
- current-rank fraction.

For each feature it records:

- robust quantiles,
- standardized mean difference,
- KS statistic,
- quantile-Wasserstein distance.

It also trains a tiny **group-level domain classifier for diagnostics only**. A very high AUC means generated and camera groups are trivially separable and is a shortcut/domain-gap warning, not a hit model.

### 3. Direct full-frame proposal engine

`src/engine/offline/direct_proposal_v221.py`

The engine searches PRE -> POST directly, without accepting GT as an input. Evidence sources:

- persistent absolute change,
- persistent darkening,
- persistent brightening,
- new black-hat structure,
- new top-hat structure,
- compact local change,
- gradient gain.

The sources create independent local maxima, then a permissive recall-oriented fused/NMS candidate pool.

V2.21 direct proposals are **offline/shadow-only**.

### 4. Direct-proposal benchmark

`src/engine/offline/direct_proposal_benchmark_v221.py`

`automation/direct_proposal_v221_benchmark.py`

For packs containing full-frame evidence, report:

- current V1/V2 oracle @5/10/20/42,
- AI_DIRECT oracle @5/10/20/42,
- union oracle,
- proposals/shot,
- rescued current misses,
- runtime.

Ground truth is used only *after* proposal generation for scoring.

Forced GT-nearest diagnostic candidates are excluded from the current pool.

### 5. Storage-aware full-frame shadow capture

`CandidateCaptureConfigV216` has two new backward-compatible controls:

- `save_full_reference_pre`
- `save_full_recent_pre`

The V2.21 config enables:

```text
save_full_frames       = true
save_full_reference_pre= false
save_full_recent_pre   = true
full_frame_post_count  = 2
```

This avoids saving the less useful large legacy/reference PRE frame while preserving:

- true recent PRE immediately before the shot,
- two full POST frames.

The capture remains automation/shadow data only. Normal game authority does not change.

### 6. V2.18 benchmark report-path fix

`automation/newhole_v218_benchmark.py` now prints the actual report path beside the supplied `--model`, matching where the benchmark already writes the file.

---

## Initial gates

V2.21 direct proposal development should first target physical/projector-camera union oracle <=20 px:

- initial useful: >=70% confirmation and holdout,
- strong: >=85%,
- pre-authority: >=95% across multiple independent physical sessions.

Do not optimize ranker Top-1 aggressively while oracle is still around 35–50%.

---

## Immediate run order

After overlaying the delta:

```bash
python3 -m automation.offline_v221_selftest
```

Then audit the existing physical/projector-camera packs:

```bash
python3 -m automation.physical_pack_v221_inspect \
  --root content/ai/candidate_shadow_v216
```

Then profile the already-known synthetic -> camera domain gap:

```bash
python3 -m automation.domain_gap_v221 \
  --synthetic-root content/ai/candidate_synthetic_v220 \
  --physical-root content/ai/candidate_shadow_v216 \
  --synthetic-cache content/ai/reports/v218/v220_cache \
  --physical-cache content/ai/reports/v218/v220_physical_cache
```

If the old packs have no full frames, the direct benchmark will refuse to invent a score. The updated shadow-capture config is then ready for a new automation/projector-camera capture session.
