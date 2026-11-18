# Bench: offline verification on the live shot path

Recorded replay calls the shared `shot_path_step` with recorded inputs, a
virtual clock and a virtual executor. `python3 -m automation.bench_fidelity_selftest`
compares the installed scanner, camera and audio methods with game startup.
The worker thread name also controls which branches execute (see Replay and compare).

Evidence levels are never mixed:
`SYNTHETIC_LIVE_PATH` (generated frames/audio), `RECORDED_SOURCE_LIVE_PATH`
(recorded camera bytes + PCM, incl. projected dots), and physical shots with
independent ground truth. Projected dots are optics tests, not paper damage.

## Before a shooting session (no shooting needed)

```bash
python3 -m automation.bench board-snapshot             # white-projection board image + hole wear heatmap (game not running)
python3 -m automation.bench board-snapshot --compare /data/skjutbana/board_snapshots/<earlier>   # + new blobs (review aid)
python3 -m automation.bench calibration-check          # validate calibration, never writes
python3 -m automation.bench calibration-check --save   # recalibrate (backup -> /data/skjutbana/backups); also fits the lens model
python3 -m automation.bench calibration-check --distortion           # also report the radial lens model (held-out markers)
python3 -m automation.bench calibration-check --distortion --save    # same as --save, with the lens report

# --save always saves the fresh plain calibration; the lens model (camera_distortion) is
# included only if it beats the plain homography on held-out markers. Every calibration
# save (also the menu "Kalibrera viewport via kamera") backs up the old settings first and
# refuses to overwrite if the backup fails; the menu path fits the lens model in the
# background and adds it on the main thread when it wins.
```

The optional lens model (`camera_distortion` in the calibration, `src/engine/lens_model.py`)
is used only by exact camera↔screen mappers (`hit_input.camera_to_screen/screen_to_camera`,
bench truth, optical lab). Top-level `homography` stays the plain fit, so crop/ROI code is
unchanged; absent field = today's behaviour. On the saved 24 markers, held-out error went
from median 3.4 mm / max 12.2 mm (homography) to 1.4 mm / 4.4 mm (with lens model).

The game window must not be running (one process owns the camera).

## Record a session that can be replayed forever

```bash
SKJUTBANA_SOURCE_RECORD=1 ./start.sh          # play/shoot normally; raw MJPG + PCM are recorded
```

Sessions go to `/data/skjutbana/source_sessions/<UTC>_<id>/` (~14 GB/hour;
recording refuses to start below 4 GB free and stops cleanly at 2 GB).
Add `ground_truth.jsonl` afterwards (one line per shot:
`{"wall_ts": <shot time>, "camera_x": .., "camera_y": ..}` or
`{"wall_ts": .., "kind": "no_impact"}` for claps/false triggers).

Projected test holes (real optics, no shooting):

```bash
python3 -m automation.bench optical-holes --background gray --shots 12 --hole-mm 3
python3 -m automation.bench optical-holes --background moving --shots 12
```

Precision flash (engine `src/engine/precision_flash.py`, opt-in for `hit_precision="exact"`):

```bash
python3 -m automation.bench optical-holes --background dark --shots 12 --flash
python3 -m automation.bench optical-holes --background moving --shots 12 --flash --hole-style varied
python3 -m automation.bench flash-score SESSION_DIR [SESSION_DIR ...]
python3 -m automation.bench flash-replay SESSION_DIR [...] --output DIR --set min_darkening=0.04
```

`flash-replay` re-runs only the image analysis on the recorded flash frames with
changed `FlashConfig` fields (flash timing is what was projected).

## Replay and compare

Recorded replay advances a virtual clock using recorded input timestamps. It
freezes `time.time`, `time.monotonic` and `time.perf_counter` at the modelled
start while each worker job runs. This makes event ordering repeatable across
machines. The internal `stages_ms` measurements read that frozen clock.

`replay SESSION --output NEW_DIR --wall-clock-profile` additionally writes
`wall_clock_profile.json`. Each submitted job has host monotonic timestamps
in nanoseconds. Subtract start from finish for elapsed host time, including waits.
Submit to start includes any input decoding and other replay-loop work before
the virtual executor runs that job. A null start or finish means that endpoint
was never observed. The timestamps are local to the machine and clock domain.
The `thread` field is sampled inside the job. A `MainThread` job
misses the runtime branches guarded by the `shot-cv-v2224` thread prefix.
Matching the prefix alone does not establish full parity with the live runtime.

Run `python3 -m automation.bench latency-report REPLAY_DIR` on a profiled replay.
It prints each submitted job's thread and the elapsed milliseconds from submission to start,
start to finish, and submission to finish. Each sum is computed from
integer nanoseconds and carries its observed/submitted count. `UNKNOWN` marks
missing endpoints. Overlapping intervals contribute separately to the sums.
The sidecar contains executor timestamps only. Attribution to stages, emissions
or individual shots requires evidence outside these rows.

### Inspect a PRE image

`python3 -m automation.bench.image_context earlier.png pre.png --grid 4 6`
prints a JSON grid for explicitly supplied images in chronological order.
Each cell contains its pixel bounds (exclusive right and bottom), mean gray
level and mean Sobel gradient magnitude in the last image. With two or more
frames, `ctx_motion` is the mean within-cell temporal standard deviation
of individual pixels. A single image yields `null` and the reason
`single_frame`. The reported elapsed cost covers the in-memory calculation.
The offline trainer excludes `ctx_brightness`, `ctx_texture` and `ctx_motion`
from feature discovery when they appear in a supplied candidate export.

### Run comparisons

```bash
python3 -m automation.bench replay SESSION_DIR --output /tmp/r1 [--pulse-offset-s 0.025]
SKJUTBANA_MIN_POST_AGE_S=0.11 python3 -m automation.bench replay SESSION_DIR --output /tmp/r2

python3 -m automation.bench synth --background FRAME.npy --output /tmp/s1 --seeds 5
python3 -m automation.bench ab --background FRAME.npy --output /tmp/ab1 --seeds 5 \
    --candidate-env SKJUTBANA_MIN_POST_AGE_S=0.11
python3 -m automation.bench compare /tmp/s1 /tmp/s2      # BETTER / WORSE / NO_SIGNIFICANT_DIFFERENCE
```

`compare` refuses runs with different physics, seeds, background or settings
hash (different calibrations are different experiments).

## Opt-in detector hypotheses (default OFF)

| Flag | Hypothesis |
|---|---|
| `SKJUTBANA_MIN_POST_AGE_S=0.11` | H1: no FAST job on frames that cannot yet show the hole |
| `SKJUTBANA_CAUSAL_OWNERSHIP=1` | H3′: older shot keeps evidence until the newer hole could be visible; later confirmation only for proposals with originating change |
| `SKJUTBANA_REQUIRE_NOVELTY=6` | H2: tracks without PRE→POST change (grey levels) cannot be selected (not recommended: hurt moving backgrounds) |
| `SKJUTBANA_PRE_AVERAGE_FRAMES=4` | H4: confirmation PRE is the mean of up to N frames ending at the normal PRE frame, never before the previous shot's hole (no measurable effect; keep off) |
| `SKJUTBANA_PREFER_LOCAL_CONTRAST=3` | H5: prefer this shot's tracks with compact local darkening (REJECTED 2026-11-18: synthetic WORSE p=0.03, doubles 9→5; helped textured/frozen-moving only) |
| `SKJUTBANA_EVIDENCE_OVER_TIMING=1` | H8a: onset only gates the window; winner = highest ranking value (ties: hits, earlier onset, lower id) NO EFFECT 2026-11-18 (synthetic 41/48 = base; replay 28/89 = base) |
| `SKJUTBANA_BEST_READY_TOPK=3` | H8b: emit the first READY track among the top-k eligible instead of waiting on the top one NO EFFECT 2026-11-18 (synthetic and replay identical to base) |
| `SKJUTBANA_UNCAPPED_RANK=1` | H6: rank by the V2 score without the 35 clip (`v26_rank_excess`); `score` unchanged NO EFFECT 2026-11-18 (synthetic and replay identical to base) |
| `SKJUTBANA_HOLE_SHAPE_PRIOR=1` | H7a: rank × hole-likeness [0.3,1] from equivalent radius (`SKJUTBANA_HOLE_RADIUS_PX=0.7,2.5`) and circularity MIXED 2026-11-18 (synthetic = base; replay 28/89, +4/-4) |
| `SKJUTBANA_SIGNED_DARKENING_PRIOR=1` | H7b: rank × [0.3,1] from `v2_darkening/v2_absdiff` (brightening/weak darkening loses) SLIGHTLY WORSE 2026-11-18 (replay 27/89, -1) |
| `SKJUTBANA_NO_OLD_HOLE_SNAP=1` | H9a: no snap to an old hole when the track is ≥4 px away and its own change ≥5 grey NO EFFECT 2026-11-18 (synthetic and replay identical to base) |
| `SKJUTBANA_SOFT_KNOWN_HOLE_PENALTY=1` | H9b: known-hole penalty ×0.6/0.8/0.9 instead of ×0.15/0.4/0.7 REJECTED 2026-11-18 (synthetic 39/48 vs 41, +1 false; replay 27/89, -1) |
| `SKJUTBANA_LATE_REPROPOSE_S=0.25` | H10: one extra global FULL (rescue) extraction per shot on the first frame ≥ peak+value, same worker/ownership/next-peak cutoff/timeout; candidates join tracks + local confirmation (`audio_trigger['v26_late_repropose']`) NO EFFECT 2026-11-18 (replay 47/149 vs 46) |
| `SKJUTBANA_AUDIO_COOLDOWN_S=0.12` | audio peak cooldown override (CO2 second peak 66–94 ms after the muzzle peak) — audio events = real shots, but emissions unchanged/worse in replay; KEPT OFF 2026-11-18 |
| `SKJUTBANA_HOLE_LIKENESS_RANK=1` | H11: candidates get `h11_hole_snr` (matched filter of the signed PRE→POST darkening with the mean real-stamp radial profile ÷ local RMS); tracks rank by it and the winner is the highest rank among eligible tracks (implies H8a ordering). Offline re-rank evidence positive; live A/B: see AUDIT_FINDINGS §14 |

## Nightly (cron 02:30, `crontab -l`)

```bash
python3 -m automation.bench nightly --dry-run      # show plan and preconditions
python3 -m automation.bench nightly                # run now
cat /data/skjutbana/nightly/LATEST.md              # last report
```

Each night: code regression check (defaults + pilot flags, fixed seeds, compared with the
previous night), new synthetic training data (new seeds), export of every recorded session
with `ground_truth.jsonl` (once), training of a logistic candidate scorer on all training
data and evaluation on a frozen holdout (synthetic seeds + held-out session). Models are
`OFFLINE_REPLAY_ONLY_UNVALIDATED`, `live_authority=false`, never installed. Skips when the
game runs, memory < 12 GB or disk < 10 GB; at most two worker processes.

Promote a flag only after synthetic A/B, recorded-source replay **and**
physical validation agree.

## Real hole stamps per weapon (2026-11-18)
- Airgun (default stamp set): `--physics '{"hole_model":"stamps","hole_stamp_scale":[1.0,1.0]}'`
- CO2 AR15 as they are (real AR15 holes, true size):
  `--physics '{"hole_model":"stamps","hole_stamp_path":"/data/skjutbana/hole_library/ar15_co2_20261118/clean_stamps.npz","hole_stamp_scale":[0.95,1.0],"hole_stamp_extra_blur_px":0.0}'`
  (preset `AR15_REAL_STAMPS` in `src/engine/offline/ar15_scenarios.py`). The nightly `stamps` variant
  is unchanged (shrunk airgun proxy) so its baseline stays comparable.
- AI test (projected, F8): `co2` = real AR15 stamps (gain 4.5-6), `airgun` = real airgun stamps,
  `co2_shrunk` = old proxy, `legacy` = drawn holes.

## Hole-likeness measurement tools (2026-11-18, measurement only)
- `python3 -m automation.bench hole-likeness SESSION REPLAY_DIR` — re-rank the candidates exported
  by `replay --export-candidates` with matched-filter/purity/persistence features; GT rank per background.
- `python3 -m automation.bench hole-likeness-dense SESSION [--rank dense_s]` — the matched filter itself
  as a proposal source over the viewport (recall + ranking, independent of the V2 generator).
- `python3 -m automation.bench optical-holes --hole-style stamps_ar15|stamps_airgun` — projects REAL hole
  stamps exactly as the AI test does (profiles in `ai_training_content.PROJECTED_PROFILE`).
