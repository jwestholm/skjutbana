# Audit findings — 2026-11-18 (autonomous review session)

Status legend: **FIXED** (changed + tested), **OPEN** (needs decision/work),
**INFO** (documented, no action). All detector changes are default-off; nothing is committed.

## Contents

The opening note describes the original review. Section 15 records a test that
intermittently reported a saved frame as missing. Section 23 documents the fix.

| Sections 1–12 | Sections 13–23 |
|---|---|
| [1. H6–H9 selection experiments](#audit-1) · INFO | [13. Projected F2 recording](#audit-13) · INFO |
| [2. Stale selftest fixtures](#audit-2) · FIXED | [14. Textured-background experiments](#audit-14) · OPEN |
| [3. Recall and test timing](#audit-3) · OPEN | [15. Selftest baseline and missing-frame report](#audit-15) · INFO |
| [4. Calibration persistence](#audit-4) · FIXED | [16. F2 and replay clocks](#audit-16) · INFO |
| [5. Other maintenance fixes](#audit-5) · FIXED | [17. Replay thread on claude/latency](#audit-17) · FIXED |
| [6. Decisions pending](#audit-6) · OPEN | [18. Worker benchmarks and shutdown report](#audit-18) · OPEN |
| [7. Code inventory](#audit-7) · INFO | [19. Offline bench maintenance](#audit-19) · INFO |
| [8. Outstanding research](#audit-8) · OPEN | [20. Isolated projected replay](#audit-20) · INFO |
| [9. First physical recording](#audit-9) · INFO | [21. Offline scorer exclusions](#audit-21) · FIXED |
| [10. Board wear](#audit-10) · INFO | [22. Selftest runner](#audit-22) · FIXED |
| [11. Fresh-paper recording](#audit-11) · INFO | [23. Opt-in shutdown and hardware validation](#audit-23) · OPEN |
| [12. Hole and stamp comparison](#audit-12) · INFO |  |

<a id="audit-1"></a>

## 1. Winner selection hypotheses H6–H9 (measured, default off)

Offline evidence: synthetic real-hole stamps at CO2 scale, pilot flags as base, 3 backgrounds,
4 seeds, scenarios single/double/string3/echo/false_audio (48 cases), paired compare
(`/data/skjutbana/evaluation_runs/winner_selection_ab_20261118`), plus replay of three recorded
optical sessions (89 shots, `/data/skjutbana/evaluation_runs/winner_selection_replay_ab_20261118`).

| Flag | Synthetic | Replay | Verdict |
|---|---|---|---|
| base | 41/48, impacts 28/28, false 7 | 28/89 | — |
| H8a evidence over timing | identical | identical | no effect |
| H8b best ready top-k | identical | identical | no effect |
| H6 uncapped rank | identical | identical | no effect |
| H7a hole shape prior | identical | 28/89 (+4/−4) | mixed |
| H7b signed darkening | identical | 27/89 (−1) | slightly worse |
| H9a no old-hole snap | identical | identical | no effect |
| H9b soft known-hole penalty | 39/48 (+1 false) | 27/89 (−1) | rejected |

**Why (funnel on exported candidates):** the wrong outputs are not decided at the final track
selection. Realistic CO2-contrast projected holes: the true hole is **not among the candidates in
38/41** shots (recall). Photo background: missing 9/24, rank 1 7/24, rank >5 8/24 (ranking).
Moving: rank 1 13/24, rank 2–5 3, >5 4, missing 4. In synthetic stamps every impact is already
correct; the remaining synthetic failures are **false emissions on echo/false-audio** (7/48).

Next: (a) recall for small faint holes in candidate generation, (b) hole-likeness ranking on
textured backgrounds, (c) IMPACT/no-hole rejection for echoes/claps.

<a id="audit-2"></a>

## 2. Long-failing selftests — FIXED (tests were stale, code correct)

- `automation/object_hit_selftest.py`: since b59caa4 OBJECT_HIT waits for later causal frames when
  zero or several objects are supported (`_await_object_disambiguation`,
  `object_disambiguation_selftest` asserts this). The fixture fed frames only to peak+0.065 s.
  Fix: feed one more frame at the 0.20 s deadline in `damage()`.
- `automation/v2231_selftest.py`: V2.23.2 (2a0f6d0) added fresh-domain requirements to
  `research_promotion_gate`; the old `good` fixture lacked them (`v2232_selftest` asserts that).
  Fix: fixture mirrors v2232 (baseline eligible_ranked_shots + domain metrics).
- `automation/object_physical_query_selftest.py`: made independent of the local calibration's
  lens model (`h.lens=None`).


<a id="audit-3"></a>

## 3. Recall: the true hole is usually never proposed — mostly a TIMING problem (OPEN, experiment H10 running)

Measured from camera.bin of the AI-test recording (in-game, realistic CO2-contrast projected holes):
- the first global FAST proposal frame is 0.11–0.21 s after the shot (median ~0.136 s);
- the hole first becomes visible 0.17–0.33 s after the shot in that rig (render + projector latency);
- later frames only LOCALLY CONFIRM the first frame's proposals (`shot_fast_v2225.py` ~1083–1151);
- the FULL rescue extraction runs only when there are zero proposals or none confirm, which never
  happens (130–200 noise proposals per shot).
So the hole (8–42 grey darkening on bright backgrounds, the strongest change in the ROI) is
never proposed. ~15/41 shots had no recoverable signal at all (dark backgrounds: a projector
cannot darken dark content — rig artefact, not a CO2 property). Real holes can also appear late
(impact ~30 ms after audio + exposure + ~66 ms receive latency), so this is a real design risk.
Git history: no detector/matched filter was ever removed; `rescue_blob` (built for tiny plateaus)
runs only in FULL rescue since 0898b5e (v2.22.5).
Experiment: H10 `SKJUTBANA_LATE_REPROPOSE_S` = one extra global extraction on a frame ≥ peak+X,
evaluated alone and with H8a (the selector ranks onset time before score).

Projected contrast sweep (gray / photo, `--hole-factor-range 0.5,0.95`, 30 shots each,
`/data/skjutbana/evaluation_runs/recall_contrast_20261118`): on gray, holes with ≥30 % projected
darkening are emitted ~70 %, 20–30 % about half, <20 % almost never; on the photo the hit rate
stays low even for strong holes (ranking/texture). Small samples.

### 3b. Results (2026-11-18 afternoon)
- **H10 late global re-proposal** (`SKJUTBANA_LATE_REPROPOSE_S` 0.22 / 0.32, alone and with H8a):
  replay 46→47/149 (5 sessions), synthetic identical (41/48, no new false emissions). Essentially no
  effect: a wrong track from the early proposals is usually already READY and emitted (~0.33–0.37 s)
  before the late candidates can win. Kept default off ("no effect").
- **Waiting longer before the first global proposal works** (`SKJUTBANA_MIN_POST_AGE_S`):

  | MIN_POST_AGE_S | AI-test session (41) | gray contrast session (30) |
  |---|---|---|
  | 0.11 (pilot) | 6 correct, 34 wrong | 15 correct, 14 wrong |
  | 0.20 | 10 correct, 26 wrong | 15 correct, 13 wrong |
  | 0.30 | 14 correct, 21 wrong | 18 correct, 6 wrong |

  The 0.11 s pilot value was tuned on the 2026-11-17 optical session (onset 0.12–0.156 s). Physical
  onset = impact (~15–25 ms after the audio peak for CO2: the muzzle blast reaches the board mic
  before the slower pellet) + exposure + ~66 ms receive latency, i.e. roughly 0.10–0.15 s, so
  0.11 s is borderline. **Tonight's recordings decide the value**: record once, replay with 0.11 /
  0.20 / 0.30 offline. Latency cost of 0.2–0.3 s stays inside the ~1 s budget.
- **Test-tool artefact fixed:** the AI-test F2 progress text ("Headless: i/N") was drawn on the board
  and changed between rounds; 5/30 wrong emissions in the AI-test replay sat exactly on it. The
  counter now goes to stdout.

### 3c. AI-test rig timing fixed (test tool only)
The in-game AI test stamped the synthetic audio event BEFORE the hole was drawn; with reveal delay,
frame and projector latency the hole appeared 0.17–0.33 s after the peak (physical: ~0.10–0.15 s).
Now synthetic events are stamped `now + synthetic_audio_align_s` (runtime setting, default 0.06 s),
used consistently for ground truth and `synthetic_audio.jsonl`. Limitation (Codex): only for the test
tool; a real shot inside the same 60 ms window during a test run could be reordered; results from
the rig are optical/projected evidence, not physical accuracy.

New F2 recording with the fix (36 rounds, 55 holes, CO2-calibrated projected stamps,
`/data/skjutbana/source_sessions/20261118T123405Z_76d489ed`), offline replay with funnel
(`/data/skjutbana/evaluation_runs/aitest_timing_fix_20261118`): MIN_POST_AGE 0.11 → 11/55,
0.20 → 14/55. Per background (0.20): white + white_grid **7/8 correct**, coord_grid 3/7 (text in
every cell), gray 1/4, black/dark 0/9 (projector cannot show holes on dark content), checker /
bubbles / scrolling / moving targets 1–2 each (candidates missing AND hole ranked >1).
Conclusion: with realistic CO2 contrast and physical timing the chain works on clean bright
surfaces; textured and moving game content is the real weakness (recall and ranking), dark content
cannot be judged with a projector rig.

Synthetic check of a longer fixed wait: MIN_POST_AGE 0.30 drops doubles 12→8/12; 0.20 impacts
28→26/28 (false 7→5). Decision (with Codex): keep 0.11 as pilot; replay tonight's physical
recordings at 0.11 / 0.15 / 0.20 and decide per measured onset; an adaptive onset-triggered first
proposal is a later hypothesis.

<a id="audit-4"></a>

## 4. Calibration saves silently dropped the lens model — FIXED

Menu "Kalibrera viewport via kamera", AI-test auto-calibration and `bench calibration-check --save`
replaced the whole calibration (menu path without backup) and dropped `camera_distortion`.
Now (reviewed with Codex): `save_camera_calibration` always writes a unique backup first
(`/data/skjutbana/backups`, override `SKJUTBANA_BACKUP_DIR`) and raises instead of overwriting
if the backup fails; `ArucoCalibrator.save_and_apply` fits the lens model in a background thread
(fit only, ~8 s) and the live loop applies it on the main thread (`lens_model.apply_pending`)
only if the saved calibration still has the same fingerprint; `calibration-check --save` always
evaluates the lens model synchronously; the AI-test auto-calibration never replaces a lens-model
calibration. Test: `automation/calibration_save_selftest.py`.

<a id="audit-5"></a>

## 5. Other fixes
- `src/engine/scenes/loading.py`: bootstrap failures are logged with traceback instead of `pass`
  (a silent failure made later patch layers wrap unpatched methods).
- Docs: stale flash numbers/claims in CURRENT_STATE, bench README calibration semantics, README
  (calibration backup, parec, F2 matrix), new "CURRENT capture method" section at the top of
  PHYSICAL_CAPTURE_RUNBOOK.md, status note in ROADMAP "NOW".

<a id="audit-6"></a>

## 6. Needs the user's decision (OPEN)
- **Security:** `src/engine/output/led_test.py` (tracked in git, never imported) hard-codes a Tuya
  device id, IP and LOCAL_KEY (lines ~11–13). Treat the key as exposed if the repo was ever shared;
  recommend rotating it in the Tuya app and removing the literals (history still contains them).
  Not changed (credentials).
- Stray git-ignored backups: `src/engine/settings.py.v253r2-broken.bak`,
  `src/engine/camera/__init__.py.bak_v27_1`, `src/engine/ai/bootstrap.py.bak_v27_1` — delete?
- Never-imported modules: `ai/ranker_v5_extension.py`, `ai/training_data.py`,
  `offline/replay_context.py`, `output/led_test.py`.

<a id="audit-7"></a>

## 7. Information (no action taken)
- Import graph: 204 modules under src/: 143 live, 57 automation/offline-only, 4 never imported.
- Superseded patch layers: v244 wraps v243's *previous* ROI so `frame_roi_v243` never runs;
  v241's extract wrapper is installed then removed by v243; v25 detector features are all off in
  `detector_v25.json` (telemetry only); v2221 fully replaces the base `_resolve_audio_events`.
- `camera/object_physical_query.py:21` imports a private helper from the diagnostic
  `demo_object_zone_extension` module.
- Duplicated helpers (`_safe_float` ×30, `_runtime_settings` ×11); ~31 never-referenced functions.
- Unread settings keys: top-level `visual_hits_enabled`/`visual_hits_mode` in content/settings.json
  (code reads nested `visual_hits.*`); seven V1 keys in content/ai/settings.json (never staged —
  documented only).
- 40 `SKJUTBANA_*` env vars, 16 undocumented (capture plumbing etc.); full table in the audit notes.
- Docs still missing: ARCHITECTURE.md stops at 2026-10-18 (no shot_path_step/virtual runtime/
  source recording/lens model/precision levels/parec); nightly holdout session 98e68fe9 is not
  untouched any more and should be replaced by a fresh physical session.

<a id="audit-8"></a>

## 8. Recommended next steps (priority)
1. Tonight: record physical shots (per weapon), label GT, replay at MIN_POST_AGE 0.11/0.15/0.20 →
   choose the value from measured onset; measure real CO2 hole contrast vs the stamp model.
2. Textured/moving backgrounds: hole-likeness ranking (real-stamp template / learned scorer) and
   recall on textured content — evaluate on the AI-test recordings with the funnel.
3. IMPACT/no-hole rejection for echo/false audio (synthetic: 7/48 false emissions in every arm).
4. Replace the nightly holdout session with a fresh physical one.
5. User decisions in section 6 (Tuya key, stray .bak files, never-imported modules).

<a id="audit-9"></a>

## 9. First physical CO2 AR15 round on the capture target (2026-11-18 ~15:00)

Target: new static `content/games/physical_capture_target.py` (menu "Fysisk inspelning – tavla").
Recording `/data/skjutbana/source_sessions/20261118T135339Z_da06f87a` (window copy `_ar15cut`,
GT in `ground_truth.jsonl`), analysis `/data/skjutbana/evaluation_runs/physical_ar15_20261118`.
17 pellets (5 singles, 3 doubles, 2 triples) + 6 empty-magazine shots; operator photos in
`/data/skjutbana/skott18nov1500/`.
- **Visibility:** only 7/17 holes are visible to the camera (clean white paper 15–19 grey, others
  4–9); holes in the tape patches are invisible (also in photos). Board needs fresh paper.
- **Onset (physical, receive timestamps):** 0.139 / 0.147 / 0.153 / 0.162 s after the audio peak.
- **Live:** 9 emissions, 4 correct (2,3,4,7), 0 on empty magazine.
- **Two audio peaks per shot**, 66–94 ms apart (muzzle + impact/echo): 45 audio events for 23 shots.
- **Replay** (visible holes, 30 px): age 0.11 → 6/7, 18 false; **0.15 → 6/7, 14 false**;
  0.20 → 5/7, 12 false. New default-off flag `SKJUTBANA_AUDIO_COOLDOWN_S` (0.12/0.15) brings
  audio events to 23–24 (= real shots) but emissions rise to 23 (5/7 correct): the engine emits for
  every shot even when no hole is visible — false emissions come from missing no-hole/IMPACT
  rejection, not from the double peaks alone. Flag kept off.
- Suggested: MIN_POST_AGE 0.15 (confirm on airgun/pistol); next hypothesis = no-hole rejection.

<a id="audit-10"></a>

## 10. Board wear map (2026-11-18 ~16:00, after removing ~150 g of tape)
New `python3 -m automation.bench board-snapshot` (white projection, camera, hole blobs → screen
coords, heatmap; `--compare` lists new blobs). Result in
`/data/skjutbana/evaluation_runs/board_wear_20261118/`: 55 % of holes in the central 25 % of the
area. Uses agreed with Codex → ROADMAP "Game design rule: spread the aim points". The capture
target now moves its cross grid randomly per sequence.

<a id="audit-11"></a>

## 11. Physical CO2 AR15 round 2 on fresh paper (2026-11-18 ~17:00)
Session `/data/skjutbana/source_sessions/20261118T155321Z_04f1be81` (GT from camera diff + operator
photos `/data/skjutbana/skott18nov1700/`): 16 pellets (row 1 singles, row 2 doubles, row 3 triple on
9, one shot on 10) + 6 empty-magazine shots; 15 GT positions (aim 6 double = one shared hole).
On fresh paper 12/15 holes are clearly visible (12–31 grey); aims 1, 3, 9 weak (4–8).
| | correct /15 | false (other) | emissions on empty magazine |
|---|---|---|---|
| live (age 0.11) | 5 (+aim 3 at 1.6 s, outside window) | 4 (3 on the board label text) | 0 |
| replay age 0.11 | 10 | 6 | 7 |
| **replay age 0.15** | **12** | **3** | 7 |
Second physical session agreeing with section 9: **MIN_POST_AGE 0.15 is better** (runbook updated).
Live is far below replay: worker 400–600 ms per shot, latency up to 1.6 s, each shot spawns two
audio shots (66–94 ms second peak) — live CPU budget is the next big item. Empty-magazine shots
still emit in replay (no-hole rejection missing). Capture target: board label text now shown only
3 s after a key press; numpad keys accepted. First real AR15 stamps:
`/data/skjutbana/hole_library/ar15_co2_20261118/`.

<a id="audit-12"></a>

## 12. Real AR15 vs airgun holes; real stamps used as synthetic holes
Same camera, same method: airgun core area 40 px / depth 0.28 (38 grey), AR15 22 px / 0.05-0.19
(10-39 grey) → AR15 diameter ~0.74x and about half as dark, ~1/4 the integrated darkening (the old
"50-80 % smaller" proxy was in the right range). Both real sets are now synthetic holes "as they
are": AI test F8 `co2` = AR15 library (true size), offline preset `AR15_REAL_STAMPS`.
Projector check (`evaluation_runs/stamp_projection_20261118`, white, gain 3): camera sees airgun
39.5 px/0.20 (real 36.5/0.21, match) and AR15 24.5 px/0.056 (real 22/0.095) → size right, AR15 gain
raised to 4.5-6. Operator: airgun holes look slightly too big by eye → footprint scale random 0.95-1.0 for both, random 0-360 deg rotation + mirror; weak AR15 stamps (low gas, barely through) kept on purpose. Selftest ai_training_content OK.

<a id="audit-13"></a>

## 13. AI-test F2 with real AR15 stamps (projected, 48 rounds, MIN_POST_AGE 0.15)
Run `/data/skjutbana/ai_test_runs/ai_test_20261118T172714.json`, recording
`/data/skjutbana/source_sessions/20261118T162349Z_66d659ce`. Live path ±5 mm: **14/78 (18 %)**,
40 false; white 3/6, white_grid 5/7, coord_grid 3/9, gray 1/5, checker_anim 1/7, images 1/5, all
dark/textured/moving 0. Median error of correct hits 1.8 mm. Bright plain surfaces ≈ physical
fresh-paper live result (5–6/15); textured/moving/dark content stays the weakness (recall+ranking),
false emissions dominate there. (A first attempt scored 0/76 because the stamp-projection script
still held the camera — invalid, deleted. AI-test auto-calibration failing silently when there is
no camera frame now logs the reason.)

<a id="audit-14"></a>

## 14. Hole recognition on textured backgrounds — H11 hole-likeness ranking (default OFF)
Measurement first (`automation/bench/hole_likeness.py`: `hole-likeness`, `-dense`, `-emissions`):
re-ranking exported candidates by a matched filter of the change image with the mean radial
profile of the REAL stamps (59 airgun + 8 AR15) put the GT first more often on every session
(AI-test 19→25 of 36 with GT present, morning AI-test 9→12/21, physical 11→12/13).
New independent validation data: `optical-holes --hole-style stamps_ar15` (real AR15 stamps exactly as
the AI test projects them) on pattern, moving, three photos — 5×24 shots, recorded 2026-11-18 18:01–18:05
(sessions 20261118T1701..1704*), never used for design.
**H11 `SKJUTBANA_HOLE_LIKENESS_RANK=1`** (winner_selection_flags.py; FAST + FULL extractors attach
`h11_hole_snr`; selector ranks by it among eligible tracks within earliest onset + 0.06 s):

| session (paired replay, age 0.15 + causal) | base | H11 v3 |
|---|---|---|
| AI-test real AR15 stamps (78) | 20 (53 false) | **29** (44) |
| morning AI-test, shrunk stamps (55) | 11 (42) | **19** (34) |
| validation moving (24) | 3 | **9** |
| validation photo testtavla (24) | 2 | **9** |
| validation photo woods (24) | 1 | 3 |
| validation photo building (24) | 0 | 0 |
| validation sinusoid pattern (24) | 11 | 9 |
| physical AR15 fresh paper (22 incl. 7 empty-mag) | 15 (4 false) | 15 (4) |

Total projected 48 → 78 correct of 253, fewer false emissions, physical unchanged. Version history:
v1 without onset window (pattern 11→7); v2 SNR normalised by a local RMS that included the blob itself
(every isolated blob saturated at ~13–14, faint true holes lost); v3 = annulus noise (61×61 minus 21×21).
Remaining: recall (GT not among candidates in ~half the textured shots), pattern −2, abstention
(emission study: SNR<6 would drop 13/43 false emissions, 0/18 correct). Evidence level: optical
projected + one physical session — not physical validation.

### 14b. H11 review fixes and the decisive control (v4–v6)
Codex review: tracks whose candidates lack `h11_hole_snr` were ranked 0 (missing evidence treated as a
bad hole). Debug showed those are the v24 tile-probe/accumulator candidates (detector_v2=1 but no
extractor fields) — not legacy. v4 (fallback to default order when any unmeasured track is in the pool)
and v5 (also measuring merged V1+V2 candidates) were identical to base because such a track is almost
always present. **v6 (final)**: within the earliest-onset pool, measured tracks are ordered by
hole-likeness, unmeasured ones follow in the default timing order. v6 reproduces v3 exactly.
**Control arm** `SKJUTBANA_H11_DIAG_RANK_BY_SCORE=1` (same measured/unmeasured split, measured ranked
by the old best_score): AI-test 21/78 (H11 29), moving 3 (9), testtavla 5 (9), pattern 11 (9),
physical 14/22 (15) → the gain comes from the matched filter, not from demoting the v24 source.
TODO (proper fix): measure every candidate uniformly at the tracking seam (PRE-by-shot + current frame
patch), then drop the measured/unmeasured split. Pattern background (−2) remains open.

### 14c. H11 v6 final numbers + LIVE projector test
Paired replay (base → H11 v6): AI-test 20→29/78, morning AI-test 11→18/55, moving 3→9, testtavla 2→9,
woods 1→3, pattern 11→9, building 0→0 (/24), physical AR15 15→15/22. Projected total 48→77.
**Live F2 AI-test with H11 on** (real AR15 stamps, 48 rounds, `/data/skjutbana/ai_test_runs/ai_test_20261118T191433.json`,
recording 20261118T181112Z_6cb18ae7): **30/79 (38 %)**, 32 false, median error 1.3 mm — previous live run
without H11: 14/78 (18 %), 40 false (different random holes, not paired). white 8/9, white_grid 7/9,
coord_grid 4/5, gray 5/7; dark/photo/scroll/moving still ≤ 1.
Latency: hit median 1056 ms (p90 1857) vs 976 (1759) without H11; FAST extract +38 ms. NOTE: the live
hit latency is ~1 s even without H11 — above the game budget; CPU/latency is the next structural item.
Status: H11 stays DEFAULT OFF (optical/projected evidence + one physical session; needs physical
validation on game-like content before default-on). H12 (matched-filter proposals): no effect in the
two sessions measured (identical to H11), kept off.

### 14d. H11 as an evidence family (v7–v9) + code study
- v7/v8 (per-candidate PATCH source on raw PRE/CUR, ±local gain/offset, ±3×3 shift search): AI-test
  22–23/78, moving 5, testtavla 5–7 — far below the V2-map source (29/9/9); shift search costs ~9×
  (186 ms/200 candidates). Cause: a single noisy PRE frame vs V2's registered, photometric, multi-frame
  reference. Patch source kept as SHADOW evidence (`SKJUTBANA_HOLE_EVIDENCE=1`, fields ev_hole_morph_patch_*).
- v9 (final structure): H11 ranks by `ev_hole_morph_snr` from the V2-map source (attach_h11), with the
  evidence contract (valid/version; cost in `last_window_debug['hole_morph_v2map_ms']`). Verified
  identical to v6: AI-test 29/78, moving 9, testtavla 9, physical 15/22.
- Code study of the whole evidence chain → new `EVIDENCE_CHAIN.md` (26 items, status, context maps,
  plan agreed with Codex).

<a id="audit-15"></a>

## 15. Offline selftest baseline at 765580a — 2026-11-18

All 140 modules matching `automation/*selftest*.py` ran in separate Python processes
on `codex/astra-latency`. Each process used dummy SDL video/audio drivers, a private
temporary `/data` for fixtures, and no camera device. Three modules ran concurrently.

- First pass: 139/140 modules exited successfully. In `physical_gt_capture_selftest`,
  `test_interrupt_preserves_completed_frames_and_explicit_missing` saw an extra
  `PRE` in the missing-frame list. The full module passed 37/37 on a separate rerun.
  The cause of the first failure is open.
- Four test cases skipped on the first pass. With the stamp fixture library mounted
  read-only, `ai_training_content_selftest` passed 8/8. With a private loopback
  network, `physical_optical_lab_selftest` passed 25/25, including its TCP round trip.
- After reruns, all 140 modules had a successful run. The passing
  `object_freshness_selftest` also logged trace finalization errors for temporary
  fixture paths. The effect on fixture cleanup is unverified.

Work packages 1–6 remain open. There was no detector change, paired replay,
latency measurement or physical validation in this run.

<a id="audit-16"></a>

## 16. What the F2 event timestamps and the replay clock measure

The F2 report stores a timestamp and an associated synthetic peak timestamp for
each delivered camera event (`AITrainingScene._on_live_hit`). For each of the
two reports in §14c, we calculated `1000 * (ts - peak_ts)` for every entry in
`hits` and used NumPy's linear percentiles. All entries had both timestamps.
False emissions are included. A projected hole without an emission has no
latency observation.

| F2 report on 2026-11-18 | Delivered events | Median (ms) | p90 (ms) | Correct / projected holes | False emissions |
|---|---:|---:|---:|---:|---:|
| 17:27:14 | 54 | 990 | 1643 | 14/78 | 40 |
| 19:14:33 | 62 | 1067 | 1950 | 30/79 | 32 |

These were separate live optical runs with different projected holes. The exact
latency figures in §14c were not reproduced from these two event lists using
this definition. The input hashes, calculation and clock check are in
`/data/skjutbana/evaluation_runs/f2_clock_audit_20261118_vPwRuW35/`.

For the recorded-session replay command, the default `--worker-cost-s` is
0.15. `VirtualExecutor` models a single FIFO worker and freezes the virtual
clock during each job. `DetectorStageProfilerV2224` reads the patched
`time.perf_counter()`. As a control, a standalone job that waited 25.06 ms
on the host registered 0 ms inside the job and 150 ms from submission to
completion in virtual time. Accordingly the stock replay timings cannot
attribute the observed F2 percentiles to worker stages or duplicate jobs.

All 140 selftest modules exited successfully in a fresh sequential run. The
single skipped socket case passed with its full module on isolated loopback
(25/25). The earlier missing-frame test also passed. Work package 1 remains
open. This audit adds no paired replay or physical measurements.

The following two sections reproduce the findings from branch `claude/latency` through `19eb44e`.

<a id="audit-17"></a>

## 17. Replay fidelity bug: replays ran the FULL/legacy path, not the live path (fixed on claude/latency)
`VirtualExecutor.run_due` (src/engine/offline/virtual_runtime.py) executed detector jobs on MainThread,
but five live stages are gated on the CV worker thread name "shot-cv-v2224": V2.22.5 FAST extract
(shot_fast_v2225.extract_v2225), V2.24.1 object-local, V2.25.1 region proposal, V2.25.2 freshness,
V2.25.3 cross-shot novelty. Every `bench replay` so far therefore replayed the FULL extractor
(cProfile: `_extract_candidates` 475 ms/job, 0 "FAST-EXTRACT" log lines; live logs show FAST).
Fix: jobs run under the thread name "shot-cv-v2224-virtual"; `SKJUTBANA_REPLAY_LEGACY_THREAD=1`
restores the old behaviour for comparisons with earlier baselines. bench_selftest/bench_fidelity pass.
**All earlier replay numbers in §9–§14 used the legacy path.** Re-measured with the live path
(`/data/skjutbana/evaluation_runs/replay_fidelity_20261118/`, age 0.15 + causal):

| session | base | H11 |
|---|---|---|
| physical AR15 fresh paper (22) | 14 (5 false) | **16** (4) |
| AI-test real AR15 stamps (78) | 21 (52) | **26** (47) |
| validation moving (24) | 3 | 5 |
| validation testtavla (24) | 9 | 9 |
| validation pattern (24) | 11 | 9 |

H11 on the live path: +7 overall (58→65) incl. +2 physical — smaller than the legacy-path +29 but
consistent with the live F2 result (30/79 vs 14/78). Pattern −2 persists.

<a id="audit-18"></a>

## 18. Detector worker CPU: exact-equivalent speedups (claude/latency)
cProfile of every async detector job while replaying the physical AR15 session on the live (FAST)
path (82 jobs; profiler overhead inflates pure-Python code): 474 ms/job before. The two largest pure-
Python O(n·m) loops were rewritten with a uniform-grid spatial index (new
`src/engine/camera/spatial_grid.py`), keeping the exact distance tests and tie order:
- `detector_v24_extension._update_shot_accumulator`: 37 → 7 ms per 220-candidate frame (bench);
  reference kept as `_update_shot_accumulator_reference` (`SKJUTBANA_V24_ACC_REFERENCE=1`).
- `CandidateGeneratorV2._update_candidate_bank`: 23 → 5 ms per frame; reference
  `_update_candidate_bank_reference` (`SKJUTBANA_V2_BANK_REFERENCE=1`).
Proof: new `automation/v24_accumulator_equivalence_selftest.py` and
`automation/v2_candidate_bank_equivalence_selftest.py` (random frames, duplicates and ties, JSON-identical
outputs and engine state); physical replay emissions identical (22 emissions, 14/22 correct).
Profiled job: **474 → 304 ms (−36 %)**. Next candidates: `_refine_peak` (~63 ms/job, many tiny numpy
calls — batching may change float32 rounding, needs an equivalence argument), v26 `build_output`.
Live hit latency still has to be measured live (replay clock is virtual — Codex package A).
**Live check of §18 (2026-11-18 ~21:40, F2 AI-test 48 rounds each, H11 off):** optimized worker median
723 ms / hit median 1248 ms (p90 1887) vs reference implementations (`SKJUTBANA_V24_ACC_REFERENCE=1
SKJUTBANA_V2_BANK_REFERENCE=1`) 742 ms / 1300 ms (p90 1789). Real gain only ~20–50 ms: cProfile
overstated it (it inflates pure-Python loops). Both runs were CONTAMINATED by host load (load average
~5 on 4 cores; a parallel Codex selftest run): worker 720–740 ms vs ~560 ms earlier, accuracy 4–6/80–89
vs 14/78 earlier. Repeat on an idle host before drawing conclusions.
**Shutdown bug:** in both runs `main.py` did not exit within 30 s of SIGINT (one earlier instance kept
running at ~185 % CPU / 11.7 GB RSS and held the camera); TERM/KILL needed. Reproducible tonight.

<a id="17-offline-bench-maintenance"></a>

<a id="audit-19"></a>

## 19. Offline bench maintenance

**A.** The optional recorded-replay sidecar records job boundaries on the
host monotonic clock. A synthetic 25 ms sleep must measure at least 25 ms,
while the virtual job cost and event schedule stay fixed. Internal stage
timers retain their original clock. This measures the offline executor.
Each row records the observed thread name. At base 765580a, replay calls
jobs on MainThread. Five runtime modules guard code with the worker name
prefix, so method-installation parity alone overstates runtime coverage.

**B.** Ran `bench replay --use-current-settings --wall-clock-profile` and
`bench latency-report` on recorded projected session `20261118T162349Z_66d659ce`.
Of 121 submitted jobs, 116 had both endpoints, all on `MainThread`. The other
five had no observed start. The 116 intervals summed to 20.807 s from submit
to start and 143.313 s from start to finish. These are sums over offline jobs
on the legacy path. Replay used shared host resources and a read-only source
tree, so optional writes to the local diagnostic log failed. Results are in
`/data/skjutbana/evaluation_runs/offline_bench_20261118_gkK5l9M8/projected/`.
Per-stage and per-physical-shot attribution remain open. The physical session
was not run for this report.

**C.** A shared immutable observation represents either a scalar value
or a missing reason. It carries the family and version, with optional
time window and cost. Three compatibility fixtures passed on the baseline
and on the refactor. They compare the patch and map candidate fields,
including existing keys, missing values and a disabled map source. A fourth
test checks the new observation type. No accuracy experiment was performed.

**D.** Measured the standalone image-context function on synthetic 640 × 480
grayscale arrays with a 4 × 6 grid and one OpenCV thread. Over 30 samples after
three warmups, one frame cost 4.34 ms median / 4.83 ms p90. Three frames cost
5.86 / 6.38 ms. Seed 20261118. Timings exclude file decoding and JSON output.
The offline command emits a region grid for inspection. With one image,
temporal variation is unknown. This run made no candidate-level measurements.
Reproduce with `PYTHONPATH=. python3 /data/skjutbana/evaluation_runs/offline_bench_20261118_gkK5l9M8/measure.py`
from the repository root. Individual samples are in `image_context_cost.json` alongside it.

### E. Static reachability and removals

At `6f74b9b`, the AST import graph for 667 tracked Python files had zero incoming
edges for each of the four modules in §6. The scan resolved relative imports and
checked both module imports and `from package import name`. A second pass found
zero string literals containing their full module names or file paths outside the
modules themselves. No module in the table appeared in `sys.modules` after either
headless installer probe in `automation/bench_fidelity_selftest.py`. Those probes
stop before the application loop. The game loader also accepts a script path and
manual CLI use and imports from outside the repository remain possible.
To reproduce the static scan from the repository root, run
`python3 /data/skjutbana/evaluation_runs/offline_bench_20261118_gkK5l9M8/import_graph.py 6f74b9b`.

| Module | Source and history | Decision |
|---|---|---|
| `ai/ranker_v5_extension.py` | The installer is exported through `__all__`. Its bootstrap call was removed in `41bb312`, which explicitly retained the implementation for rollback. | Keep the public installer. |
| `ai/training_data.py` | `9ccef61` removed the in-tree call to `save_training_example`. The public helper writes session JSON. | Keep the callable for scripts. |
| `offline/replay_context.py` | `ReplayScannerContext` constructs `ReplayFrame` and `ReplayAudioEvent`. All three names occur only in this module in tracked Python source. The file is listed with a hash in `PACKAGE_V2_12_MANIFEST.txt`. | Keep the packaged module. |
| `output/led_test.py` | `__main__` calls `main()` and the script has its own argument parser. | Keep the standalone tool. |

To check the call sites, run `git grep -nwE 'save_training_example|install_ranker_v5_extension|ReplayScannerContext|ReplayFrame|ReplayAudioEvent' 6f74b9b -- '*.py'`. The installer probe uses `main.py`'s v241 → v243 → v244 install order. V243 restores v241's saved extractor, and v244 reads v243's saved ROI function. The v241/v243 installers and the v2221 replacement for `HitScanner._resolve_audio_events` participate in that sequence. V25 has three disabled feature switches in the checked-in config, but its top-level `enabled` is true and its wrapper writes telemetry. `object_physical_query.py` directly imports `_regions_in_work_space` from the demo extension.

Within `src`, the 30 definitions of `_safe_float` have 10 distinct argument/body AST combinations and the 11 definitions of `_runtime_settings` have three. For example, the result repository permits a `None` default for invalid floats while ranker V4 returns a numeric default. Name counts do not establish an unused definition. All live wrappers and helper definitions remain in place.

The deletion is confined to `automation/temporal_impact_research.py`. Commit `fb44e37` retained the old single-coordinate branch after an unconditional `continue` in the same loop. Removing the unreachable tail and the now-redundant final `continue` leaves all 553 decoded instructions of `run` equal on CPython 3.10. The comparison also covers instructions of nested code objects loaded by `run`. Reproduce from the repository root after applying this change:

```sh
python3 - <<'PY'
import dis, subprocess, types
from pathlib import Path

path = Path("automation/temporal_impact_research.py")
old = subprocess.check_output(["git", "show", f"6f74b9b:{path}"], text=True)

def instructions(code):
    return [(i.opname, instructions(i.argval) if isinstance(i.argval, types.CodeType)
             else i.argval) for i in dis.get_instructions(code)]

def run_code(text):
    module = compile(text, str(path), "exec")
    return next(c for c in module.co_consts if isinstance(c, types.CodeType)
                and c.co_name == "run")

before, after = (instructions(run_code(s)) for s in (old, path.read_text()))
assert before == after
print(len(after))
PY
```

### F. Live code navigation

The opening section of `ARCHITECTURE.md` follows event creation, frame
ownership, proposal and confirmation work, tracking, selection and delivery.
Source inspection covered the installed resolver wrappers as well as the base
scanner. A syntax-tree lookup verified 20 named functions across 10 modules.
The documentation distinguishes a track seeded before confirmation, evidence
from a subsequent frame, and the state assigned on delivery.

All discovered `automation/*selftest*.py` modules passed at each checkpoint
(A 140, B 141, C 142, D 143, E 143, F 143). The optical-lab module's TCP
round trip also passed on isolated loopback at every checkpoint.

<a id="audit-20"></a>

## 20. Projected replay with the worker-name correction in isolation

Replayed `20261118T162349Z_66d659ce` from the recorded projector and camera
inputs using `--use-current-settings --wall-clock-profile`. The checkout used
the code at `0457b07` with only `src/engine/offline/virtual_runtime.py` from
`43f70bd` mounted read-only for the process. This run used the detector
implementation from `0457b07`. The manifest records the mounted Git blob
and settings hashes. The legacy thread override was unset.

| Host-clock interval | Observed jobs / submitted | Sum (s) | Median (ms) | p90 (ms) |
|---|---:|---:|---:|---:|
| Submit to start | 115/120 | 17.822 | 155.98 | 196.50 |
| Start to finish | 115/120 | 39.568 | 347.78 | 544.46 |

All 115 observed starts recorded `shot-cv-v2224-virtual`. Five submissions
had no observed start. The run logged 97 FAST extraction messages and read
5,025 camera frames and 13,448 PCM blocks. These are elapsed host intervals
of offline executor jobs. Submit to start includes other replay-loop work.
The timers inside each stage still read the virtual clock. No physical
session was profiled and no job was classified as duplicate work.

The report, individual timestamps, log and input manifest are in
`/data/skjutbana/evaluation_runs/projected_thread_check_20261118_DvDJU49m/`.
All 143 selftest modules from the working branch passed with the same
read-only runtime overlay and an isolated loopback network.

<a id="audit-21"></a>

## 21. Image-context fields in offline training

A synthetic candidate row containing `sample`, `ctx_brightness`,
`ctx_texture` and `ctx_motion` contributed all four numeric fields to
the trainer's feature list at `b9406e6`. With the three context fields
explicitly excluded, only `sample` is selected. A regression test covers
a measured motion value of zero alongside numeric brightness and texture.
The image-context command remains standalone. Worker cost for generating
context on candidates remains unmeasured.

<a id="audit-22"></a>

## 22. Selftest runner

The runner reports each module's exit status and elapsed time, returns nonzero
on failures and timeouts, and gives each module a private data and device
namespace. Its first three-way run reported 142/143 modules passing. The
failure reproduced the extra `PRE` missing-frame entry described in §15.
Separate runner fixtures exercised success, failure, timeout and interrupted
cleanup, including a test that left a detached descendant running.
A subsequent two-way suite run passed all 143 modules at this checkpoint.

<a id="audit-23"></a>

## 23. Application shutdown and capture finalization

The legacy installed loop joins both thread pools before stopping camera and
audio. A device-free fixture submitted a task blocked on an unset event to
`AsyncDetectorV2224`. A direct call to its original `shutdown()` still
blocked after 2.000 s, at which point the test terminated its child. This is a
possible cause of the §18 hang. The operator's process had no stack trace
identifying which resource was actually blocked.

`SKJUTBANA_BOUNDED_SHUTDOWN=1` enables supervised exit. Communication stops
accepting commands, the audio backend is terminated and reaped, the camera
reader is stopped, and the source recorder can drain before the executor join.
Trace writers can finish and pygame can close on the orderly path. After
15 seconds the supervisor can kill its child process group, including a
stuck executor and its subprocesses. An interrupted write may remain partial.

With the production deadline and synthetic processes, clean exit took 0.083 s
after SIGINT and 0.075 s after SIGTERM (both exit status 0). A blocked executor
was force-stopped 15.043 s after SIGTERM (status 124). These are single
signal-to-exit measurements. Release of a busy physical camera remains open.
The capture wrapper forwards SIGTERM and allows the supervisor to finish
before finalizing its own exit record.

The review checkpoint now samples the writer queue before checking frame
metadata on disk. A deterministic fixture publishes the sidecar at the queue
transition and checks that the saved frame is retained. This reproduces the
failure mechanism behind the extra missing-frame entry in §§15 and 22.
All 145 selftest modules passed in the isolated three-process run, including
the process-signal, resource-cleanup, capture-wrapper and checkpoint tests.

## 24. 2026-11-19: merge, flaky selftest, live per-shot timing log
- **Merged** into `codex/physical-ai-lab-foundation` (de0b1f1): claude/latency (replay live-path fix,
  grid speedups) + codex/astra-2 (test runner, bounded shutdown flag, audit index, evidence contract,
  latency_report, image_context). Audit conflict resolved by taking astra-2's file (verified: contains
  every line of the claude/latency audit). `python3 -m automation.run_selftests`: 147/147 twice.
- **OPEN — flaky `automation.physical_collection_selftest`**: occasionally exits 143 (SIGTERM) when run
  as a plain process (2 of ~10 runs; once inside run_selftests, once directly), never in its own session
  (`setsid`, 4/4) nor via `python3 -m unittest`. Individual tests pass. Suspects: the new SIGTERM
  forwarding tests (`signal.raise_signal(SIGTERM)` relies on `_run_capture` having installed its handler)
  or a concurrent selftest run sending SIGTERM to a process group. Not reproduced deterministically.
- **New diagnostic `SKJUTBANA_SHOT_TIMING_LOG=1`** (`src/engine/shot_timing_log.py`, hooks in
  shot_async_v2224.apply_result and shot_critical_v2223.update_finished): one JSONL line per finished
  audio shot with every applied detector job (frame age, submit age, queue, worker, applied age, stages,
  candidates) and the event timing fields. Analysis: `python3 -m automation.bench shot-timing LOG`.
  Selftest `automation/shot_timing_log_selftest.py`. Replay smoke test works (virtual times); the
  physical replay shows 42 audio shots for 22 real shots (second CO2 peak) — 20 end "missed".
- **TODO (user, verify later): live check of `SKJUTBANA_BOUNDED_SHUTDOWN=1`** with the camera open
  under load (does it release the camera within 15 s; no orphan process). Not yet run.

## 25. 2026-11-19 live timing (idle host) and silent calibration drift
**Per-shot live timing** (`SKJUTBANA_SHOT_TIMING_LOG=1`, F2 AI-test 48 rounds, load avg 0.07, H11 off;
`/data/skjutbana/shot_timing/live_f2_base_20261119T075019Z.jsonl`), matched shots, median:
analysed frame +168 ms after the peak → job submitted +195 ms → **worker 562 ms** (`generate` 390 ms, of
which `_register_current` 73 ms) → result applied on the main thread +812 ms → confirmation/selection
+180 ms → **hit 1032 ms** (p90 1665). AI-test = 1 job per shot (synthetic audio has one peak). The worker
is the dominant cost and is slower live (≈560–630 ms) than offline on the same path (≈348 ms median,
Codex §20) — GIL/CPU contention with camera decode + rendering is the main suspect.
**Calibration drift:** live accuracy had dropped to 4–6/77–89 with median error 3.3–4.6 mm. calibration-check:
saved calibration marker error median 6.35 px (≈4 mm; 1.65 px on 2026-11-18 morning), mean offset −5.9 px x
— after re-papering and projector restarts. Recalibrated with backup
(`/data/skjutbana/backups/settings.backup_20261119T091343_571393.json`; lens model LOO median 1.13 screen px;
validation median 1.31 px). Live F2 afterwards: **11/83, median error 1.2 mm** (white_grid 5/6) — accuracy back.
The AI-test auto-calibration had seen the markers but silently KEPT the drifted lens calibration. It now
validates the kept calibration against the on-screen markers and prints/shows a warning when the median
error exceeds 3 px (`ai_training._try_auto_calibrate`). Recommendation: run `bench calibration-check` before
every physical session (runbook).

## 26. 2026-11-19 why the live worker is slow: contention measurements (idle host, F2 48 rounds each)
New diagnostics in `_run_job`: `stages_ms['job_thread_cpu_ms']` (worker thread CPU, excludes OpenCV helper
threads) and `job_elapsed_ms`; the shot-timing log also records `cv2_threads` (4 by default on this host).

| run | worker thread CPU (median) | worker elapsed (median / p90) | hit (matched, median / p90) | correct |
|---|---|---|---|---|
| offline replay, same code path | 253 | 251 / 314 | — | — |
| live default (cv2 4 threads) | 351 | 555 / 724 | 1024 / 1735 | 19/84 |
| live `SKJUTBANA_CV2_THREADS=1` | 358 | 509 / 645 | 1003 / 1724 | 16/86 |
| live `SKJUTBANA_CV2_THREADS=2` | 355 | 511 / 702 | 922 / 1316 | 14/79 |
| live `SKJUTBANA_AI_SHADOW_OFF=1` | 354 | 524 / 679 | 911 / 1574 | 19/80 |

Live costs ~+100 ms more worker CPU (cache/memory-bandwidth contention — the camera thread decodes,
rotates and converts 4K MJPG at 25 fps) and ~+200 ms of waiting (GIL/scheduling). Fewer OpenCV threads
and skipping the shadow-only AI observation each save only ~30–45 ms of worker elapsed; run-to-run noise
on the hit time is ±50–100 ms. Both flags are default OFF (new: `SKJUTBANA_CV2_THREADS`, `SKJUTBANA_AI_SHADOW_OFF`;
the latter stops AI-memory training observations, no hit decision uses them).
Larger options (need design + validation): camera decode cost (grayscale/reduced decode changes inputs and
hole scale → detector validation), moving the CV worker to its own process (Codex: persistent child owning
generator state, explicit snapshots, start with copied messages before shared memory).

## 27. 2026-11-19 colour channels, H13 detector luma, and the offline MAX evaluator
**Colour study** (matched-filter SNR at GT, local gain fit; physical AR15 + AI-test sessions;
`/data/skjutbana/evaluation_runs/color_channels_20261119/`): CIE L* beats plain grey for real holes —
physical AR15 median SNR gray 169, Lab L 221, **L*-LUT on grey 226 (+34 %)**; black 8→14 (L), grey 26→37 (L),
white_grid 303→342 (L); plain white/checker slightly lower. The LUT version costs ~1 ms per frame.
**H13 `SKJUTBANA_DETECTOR_LUMA=L`** (default off): new `src/engine/camera/detector_luma.py::detector_gray`;
all nine live BGR→grey conversions (hit_scanner ×5, hit_scanner_v2221, shot_critical_v2223 ×2,
object_physical_query) now go through it so PRE and POST always match. Caveat: the AI-test board
reference (`ai_training` white/black exposures) still uses plain grey. Paired replay (live path, new
calibration): physical fresh paper 14/22 → 14, taped 9/13 → 9, AI-test 21/78 → 22, moving 3/24 → 5;
no regressions, small gains — the SNR gain at GT does not yet translate into more hits in the chain.
**Offline MAX evaluator** `python3 -m automation.bench hole-max SESSIONS [--channel L] [--region-radius-px R]`
(dense matched filter, multi-frame PRE, all POST frames to next shot or 1 s; OBJECT hint = circle around
an aim point randomly offset ≤ R/2 from GT, seeded before scoring): 7 sessions, 213 shots —
full viewport gray 68 (32 %), L 70 (33 %); **with a 60 px (~4 cm) object region gray 115 (54 %), L 118 (55 %)**;
real AR15 holes 11/15 → 12/15 with the region. The object hint is by far the largest single gain. In the
projected rig grey/black/dark backgrounds and several photos stay at ~0 (projected AR15 holes invisible to
the camera) — the rig, not the detector, caps those. (Label "physical" in that output = the five
optical-holes validation sessions, which have no background field.)

## 28. 2026-11-19 game object hints (OBJECT path), evidence nightly started
**AI-test object hint** (`SKJUTBANA_AI_TEST_OBJECT_HINT_PX`, `SKJUTBANA_AI_TEST_OBJECT_MODE=regions|object_hit`,
default off): every scheduled hole gets a square target around an aim point drawn within r/2 of the hole
before the shot. Live (r = 35 screen px ≈ 4 cm): `regions` → ordinary hits 3/79 (vs 11–19 without),
only 29 shots got a decision, worker median 840 ms; `object_hit` → of 37 shot queries OBJECT_HIT 14,
UNKNOWN 22 (13 insufficient change, 5 deadline exceeded), multiple-objects 3, NO_OBJECT_HIT 1; the main loop
slowed so much that F2 stopped at 26/48 rounds (no object-id correctness scored).
**Replay region hint** (new `SKJUTBANA_REPLAY_REGION_HINT_PX`, `_REPLAY_REGION_MODE`, `_REPLAY_REGION_SEED`;
virtual_runtime.REGION_TARGETS via session_replay): with r = 35 the same chain IMPROVES — physical 14→15/22,
AI-test 21→28/78, moving 3→5/24 (FAST extract is replaced by the v251 region proposals in those shots).
So the live collapse is a live-only effect (load/latency: region path heavier; decisions not reached in time),
not a property of the evidence. Next: live regions run on an idle host with the shot-timing log.
**Evidence nightly** started in a separate root `/data/skjutbana/nightly_evidence` (live-path replay,
MIN_POST_AGE 0.15, `SKJUTBANA_H11_FULL_ROI_MAP=1` + `SKJUTBANA_HOLE_EVIDENCE=1` as candidate features,
holdout = physical AR15 fresh paper + the 5 validation sessions + 98e68fe9; 4 AI-test sessions with GT
from the drifted calibration excluded via new config `exclude_sessions`; new config `pilot_env_override`).

### §28b Evidence nightly result + CO2 labelling bug (2026-11-19)

- `label_session_candidates` gave the physical CO2 sessions **0 labelled candidates**: truth matches the
  first of the two CO2 audio peaks (66-94 ms apart) but causal ownership hands the candidates to the second.
  Fix: an event within `CO2_PAIR_S=0.12` s of a truth-matched event inherits its truth; the shot group uses the
  producing event. Old exports kept in `/data/skjutbana/nightly_evidence/dataset_mislabelled_co2_20261119/`.
  The regular `/data/skjutbana/nightly` root has the same bug for its CO2 sessions (not re-exported yet).
- Evidence nightly (`/data/skjutbana/nightly_evidence/2026-11-19_relabel/`; H11 V2-map + patch evidence fields,
  live-path replay, MIN_POST_AGE 0.15): holdout top-1 **model 161/285 vs live score 154/285**, AUC 0.973 vs 0.921.
  Physical AR15 11/15 vs 10/15 (13 have a correct candidate), 98e68fe9 13/24 vs 7/24, synthetic equal
  (double 27/36 vs 30/36 is the model's weak spot). OFFLINE_REPLAY_ONLY_UNVALIDATED; not live authority.
- The validation sessions 1701..1704 are limited by **candidate recall** (5/8/4/11 of 24 shots have any correct
  candidate), not by ranking — the selector cannot fix them.

### §29 Live object regions, idle host (2026-11-19 13:2x-13:4x)

AI test 48 rounds, all backgrounds, calibration median 1.31 px, shot-timing log on
(`/data/skjutbana/shot_timing/{noregions,regions,regions_failopen}_20261119.jsonl`).

| arm | correct | false/extra | timing-log missed | worker median |
|---|---|---|---|---|
| no regions | 20/85 | 44 | 19 | 505 ms |
| regions r=35 | 18/86 | 26 | 34 | 576 ms |
| regions + `SKJUTBANA_REGION_AUTHORITY_FAIL_OPEN=1` | 22/88 | 21 | 39 | 650 ms |

- The earlier live collapse (3/79) did not reproduce: it was host load, not the region path.
- With regions the V2.25.3 registered-region authority replaces the base/H11 selector; 42 shots waited for a
  full rescue (requested at 1.8 s) and missed. The new default-off fail-open flag (delegate to the base selector
  after 0.9 s, `_S` overrides) removes the rescue waits; replay unchanged (AI-test 29=29, physical 16 vs 15,
  moving 5=5). Live the misses did not drop: the base selector has nothing better either.
- Net: regions halve false emissions (44 -> 21-26) at equal correct hits. The correct-hit ceiling (~20/86) is
  candidate recall (no correct candidate exists), consistent with §28b. Black background: 0 correct in all arms.
- Region path costs ~70 ms extra worker CPU per job (REGION-POOL median 83 ms).

### §30 Recall/first-loss study on live AI-test sessions (2026-11-19, offline replay)

Sessions 20261119T1223/1227/1336/1343 (live AI test, all backgrounds). Scripts: session scratchpad
(visibility_probe.py, firstloss.py, h14_offline*.py); outputs `/data/skjutbana/evaluation_runs/{recall_bg,h12_recall,firstloss}_20261119*`.
- Visibility probe (MF SNR at GT vs 200 random points): ~20/85 shots visible but without a candidate
  (moving, bubbles, coord_grid, scroll, checker); black/dark/checker_anim holes essentially invisible -> flash only.
- H12 n=8: candidate recall 68 -> 81/174, emissions 61 -> 63; n=30 / min SNR 5 add nothing.
- First loss (174 shots, H11+H12): ok 60, no candidate 82, selection rank 19 (chosen track same onset frame,
  94-985 px away, slightly higher H11), no audio event 11 (REPLAY artefact: live injects synthetic peaks directly,
  replay re-detects mixed pulses), other 2.
- H14 (re-rank earliest-onset tracks by multi-frame MF SNR, post median up to 0.35/0.6/1.0 s, top-K 3..all,
  ±4 px local search): **falsified** - 92-115 vs base 121/330 events. Not wired.
- MAX evaluator (whole-viewport dense MF peak) 172/348 vs chain ~37 %: its wins are largely shots where the
  chain has NO track at GT (any GT track: 166/330), i.e. proposal location, not ranking.

### §31 H15 dense second opinion (2026-11-19, replay; default OFF)

`src/engine/shot_dense_h15.py`, `SKJUTBANA_H15_DENSE=1` (hook in `shot_fast_v2225.detect_v2225`). At peak+0.6 s
one job on the existing detector worker: H11-profile matched filter over the viewport on
D = 1 - median(post)/median(pre) (frames from `scanner.frame_history`, POST bounded by next peak+0.10).
Accepted only if score >= 16, >= 1.15x the next peak, core <= 0.4, bipolar <= 1 (rule fitted on AI-test dev
sessions 1223/1227, holdout 1336/1343: 84 vs 80 always-dense vs 65 chain, wrong H15 picks 88 -> 12); else the
ordinary selector decides. The shot waits for H15 until the result or 1.3 s (explicit, bounded).
Replay, live-path, H11 rank + H13 L, `/data/skjutbana/evaluation_runs/h15_replay_20261119/`:

| session | base correct/false | H15 correct/false |
|---|---|---|
| AI 1343 | 30/52 | 43/40 |
| AI 1336 | 36/48 | 46/40 |
| AI 1223 | 35/50 | 46/39 |
| AI 1227 | 31/48 | 38/41 |
| AI 1623 (18 nov) | 29/46 | 35/40 |
| moving 1702 | 6/18 | 8/16 |
| physical AR15 fresh | 16/4 | 15/6 |
| physical AR15 taped | 9/9 | 10/9 |
| **total 485** | **192 (39.6 %) / 275** | **241 (49.7 %) / 232** |

Decisions arrive ~0.8-0.9 s after the peak in replay (worker cost modelled). Without the acceptance rule H15 raised
physical false emissions 4 -> 14 / 9 -> 22 (CO2 echo and no-impact events always produced a pick). NOT live-verified.

§31 addendum (after Codex review): spatial 24 px exclusion removed; state reset per scanner/frame shape with
generation-checked jobs; deadline enforced on first sight of the result; authority track carries the real POST
window (first/last POST frame, n POST frames); no unsafe PRE fallback - doubles use PRE after prev+0.20 s, else the
ordinary PRE window with ONLY the previous shot's own decided XY excluded (burst ownership); second peak of a CO2
pair gets no H15 job and the pair is one shot (no second emission of a hole its partner already reported).
Physical after the fixes: fresh 16/22 false 5 (base 16/4), taped 9/13 false 7 (base 9/9). AI-session numbers of
the reviewed version: pending (run h15v5 in the evaluation dir).
