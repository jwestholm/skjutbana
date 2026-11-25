# Claude handoff — F2 synthetic-shot work (2026-11-25)

Claude: continue from the current local `dev` branch and read this handoff plus
the new F2 section at the top of `CURRENT_STATE.md`. Do not treat the latest
130-shot run as a valid test of black-background flash capture, and do not start
the 20-run learning comparison until the two instrumentation issues below are
fixed and a new frozen baseline passes its image/data checks.

## Changes in this commit

- Added an F2 `pastel_images` background by applying a light pastel wash to the
  existing photos.
- Made headless F2 rounds single-shot and paced them for the camera: no burst
  queue, 3 seconds between iterations, and a post-reveal gate requiring at least
  12 camera frames and 1 second of settling.
- Added robust multi-frame marker validation during calibration.
- Added F2 black-background precision-flash policy and a hook that redraws the
  labelled synthetic hole over the flash frame.
- Added late camera-ring capture for F2 evidence; black-background evidence
  prefers frames from the actual flash window. Candidate packs now record the
  per-shot background and capture-source metadata.
- Raised the V2.21 capture-control limit so a 130-shot matrix is not clamped to
  91 shots.
- Made the automation runner reusable with a shared event listener, optional
  window positioning, and optional per-iteration progress output.
- Added selftests for these behaviors and corrected README/CURRENT_STATE notes.

## Tests run

All passed on 2026-11-25:

- `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -m automation.ai_training_content_selftest` — 18 tests.
- `python3 -m automation.candidate_v216_selftest` — 9 tests.
- `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -m automation.precision_flash_selftest` — 19 tests.
- `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -m automation.object_highlight_selftest` — 12 tests.
- `python3 -m py_compile` on all changed Python files; `git diff --check`.

These are software selftests, not physical-accuracy validation.

## Latest F2 run and limitations

Frozen baseline artifacts:

- Results: `/data/skjutbana/evaluation_runs/projected_f2_runs/20261124_155027_white_1runs`
- V2.16 packs: `content/ai/candidate_shadow_v216/sessions/20261124_155030_seed20261124_white`
- Manifest: 130 iterations, 13 backgrounds × 10, `mixed_co2_airgun`, exclusions
  `aim8a` and `aim10`, learning frozen.
- Saved packs: 130; capture errors: 0.
- Report: found 24/130 (18.5%), Top-1 2/130 (1.5%), Top-3 4/130 (3.1%), AI
  correct 2/130 (1.5%). Funnel reports raw-hotspot proximity 23.1%, post-filter
  GT survival 18.5%, Top-K inclusion 3.8%.
- Active `content/ai/memory.json` still matches the preserved pre-run baseline
  SHA-256 `7ed21dca1f8a5a200e317dca2ee14f83e3e7be8736112742863322f102798813`.

Two concrete problems block a learning comparison:

1. All ten black rounds say `post_frame_source=camera_ring_after_projector_settle`,
   not `precision_flash_window`. App startup logged precision flash disabled;
   black GT patches contain values only in the 0–1 range. Find why the F2
   automation path does not activate the precision flash. Then run a short,
   frozen black-background check and inspect actual saved camera frames for a
   visible white flash and dark synthetic hole before collecting a large run.
2. The report says `candidate_count_raw=0` in all 130 rounds while runtime logs
   and V2.16 packs contain candidates. Trace which producer feeds this field
   (`ai_training.py` diagnostics/report) and repair/validate the metric before
   relying on candidate-stage statistics.

Do not claim accuracy improvement from these data. Keep the online memory frozen
until a valid matched baseline, training run, and frozen post-run comparison
exist. Projected synthetic performance is not physical-shot accuracy.

## Git / safety

This work is committed locally to `dev`; do not push. Do not stage generated
candidate packs, physical traces, `content/ai/settings.json`, or local
evaluation artifacts. The working tree should be clean after this commit.
