# Detector V2.12 — Offline Replay & Temporal Evidence Foundation

V2.12 is the first version whose main job is to remove hardware from the everyday detector-development loop.

It **does not replace** V2.11/V9 and **does not change live hit authority**. Instead it adds an offline laboratory around the **current live V1→V2 candidate detector** plus one independent direct-image evidence source.

See `HIT_DETECTION_PLAN.md` for the current long-term plan and TODO gates.

## What is new

### 1. Read-only historical archive inspection

`python3 -m automation.offline_archive_inspect <archive-root>`

The inspector:

- recursively inventories images,
- pairs common `pre/before/reference` and `post/after` names,
- ignores `diff/delta` images as replay inputs,
- reads sidecar JSON when available,
- extracts camera-space GT from several historical key conventions,
- estimates rough background class,
- reports ambiguous/unclassified files instead of guessing,
- recognises standalone labelled hole patches for future Hole-AI,
- writes a JSONL manifest without modifying source data.

The first run is intentionally diagnostic. Historical naming may differ; if pairing is poor, extend the importer rather than rename the archive.

### 2. Current live V1→V2 detector runs without camera/projector

`LiveHybridReplayDetector` instantiates the real `HitScanner` and installs the same V1→V2 compatibility wrapper used live. Only hardware/environment inputs are adapted: a replay pre-shot background and a full-frame replay ROI. Candidate generation itself remains the current live code.

`ReplayScannerContext` is also retained as a lower-level adapter for focused CandidateGeneratorV2 experiments:

- pre-shot frame history,
- pending audio event and peak time,
- ROI mask,
- known-hole lookup,
- candidate limit,
- optional ground truth.

This is an adapter around the live detector, not a copied offline detector.

Offline replay freezes the current detector config values for the run but forcibly disables diagnostic file writes. The historical archive remains read-only.

### 3. First independent direct-image overlay

Source 1 is `physical_fusion_v212`, built from separate before/after maps:

- `absdiff`: persistent absolute pixel change,
- `darkening`: persistent dark change,
- `persistent_zscore`: change relative to pre-shot temporal noise,
- `temporal_consensus`: fraction of post frames agreeing that a pixel changed,
- `local_contrast`: small local persistent change above broad background change.

Post frames are optionally translation-registered and median exposure-offset compensated before differencing.

The key hypothesis is simple:

> A new bullet hole is a physical change that appears after the microphone shot time and persists at approximately the same camera location.

The fused heatmap emits **permissive proposals**. It is not an authoritative ranker.

### 4. Multi-source recall union

V2.12 reports each physical component independently, plus three main benchmark views:

- `current_detector` (the current live V1→V2 hybrid),
- `current_detector_v1` provenance,
- `current_detector_v2` provenance,
- `current_detector_agreement`,
- every component (`overlay_temporal_consensus`, `overlay_persistent_zscore`, `overlay_local_contrast`, `overlay_darkening`, `overlay_absdiff`),
- fused new `overlay`,
- `union` of the current live detector + fused overlay.

This matters because we want to add/keep/reject one evidence idea at a time instead of only seeing a blended result.

Candidates keep `evidence_sources` provenance. The most important new diagnostic is:

`overlay_rescues_current_detector`

which counts labelled shots where the new direct-image source contains GT but current live V1→V2 detector does not.

This is the first concrete step toward the eventual evidence-fusion architecture.

### 5. Evidence visualisation

For one manifest shot:

```bash
python3 -m automation.offline_v212_visualize \
  --manifest content/ai/offline/v212/archive_manifest.jsonl \
  --root /path/to/archive \
  --index 0
```

It writes reference/post images, every component heatmap, fused heatmap, candidate/GT overlay and a compressed `.npz` containing the numeric maps.

## Deliberately NOT in V2.12

- no live authority change,
- no V9 replacement,
- no game-object prior,
- no spatial centre prior,
- no projector-frame prior,
- no neural Hole-AI yet,
- no claim of real accuracy before the shooting-PC archive has been replayed.

Adding all of those at once would make it impossible to know what actually improved detection.

## First real test sequence

### A. Verify install/API compatibility

```bash
python3 -m automation.offline_v212_selftest --require-v2
```

### B. Inspect only a small part of the archive first

```bash
python3 -m automation.offline_archive_inspect /PATH/TO/ARCHIVE --limit 200
```

Read:

- paired shots,
- labelled shots,
- ambiguous groups,
- unclassified filenames,
- standalone labelled images,
- background estimate.

If pairing is poor, stop there and adjust the archive importer to the actual historical naming convention.

### C. Visualise several examples

```bash
python3 -m automation.offline_v212_visualize \
  --manifest content/ai/offline/v212/archive_manifest.jsonl \
  --root /PATH/TO/ARCHIVE \
  --index 0
```

Do this for easy and difficult shots before tuning anything.

### D. Run the first frozen comparison

```bash
python3 -m automation.offline_v212_replay \
  --manifest content/ai/offline/v212/archive_manifest.jsonl \
  --root /PATH/TO/ARCHIVE \
  --max-shots 200
```

The first question is **not Top-1**. It is:

1. How much real recall does the current live V1→V2 detector have?
2. How much does the overlay have?
3. Does the union improve recall?
4. How many current-detector misses does the overlay rescue?
5. How many shots neither source can propose?

If the answer to 3/4 is positive, scale to the full labelled archive and only then tune the overlay weights/thresholds.

### E. Compile candidates for later fast learning

For a bounded development set, `--include-candidates` stores candidate evidence in the benchmark JSON. This lets later ranking/fusion experiments run repeatedly without re-reading/reprocessing the full images.

## Current success gate

V2.12 succeeds when:

- the actual archive can be converted to reproducible `ShotCase`s,
- the current live V1→V2 detector can execute through the replay adapter inside the real repo,
- real baseline metrics are stable across repeated runs,
- we can quantify whether temporal direct-image evidence is complementary to V2.

It is acceptable if the first overlay is rejected by the data. The infrastructure and a clean negative result are still progress. What is not acceptable from V2.12 onward is tuning against 100 physical projector shots without first running the same idea offline.
