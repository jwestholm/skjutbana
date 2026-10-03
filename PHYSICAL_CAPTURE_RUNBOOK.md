# Physical capture runbook

## Kompetensprov diagnostic recovery lesson — 2026-09-27

`launch-kompetensprov` is a **silent, static diagnostic**, without the normal
spoken commands/competition sequence. ENTER starts it; ENTER ends the observation
interval after the final POST wait. Normal game mode is `python3 main.py` without
capture environment overrides, followed by the Kompetensprov menu entry.
These are different modes; diagnostic silence does not mean audio assets failed.

The completed `kompetensprov_capture_20260926_215855` must not be restarted or
overwritten. Its six labels and raw arrays are preserved; selected XY can be
measured from the runtime log. All missing trace metadata followed a recorded
SIGSEGV. The full recovery report is
`/data/skjutbana/evaluation_runs/kompetensprov_recovery_20260927_01/REPORT.md`.
No replacement physical session is requested.

Completed bounded capture windows now save trace metadata through the existing
writer before process exit. Early normal shutdown still writes incomplete
coverage explicitly. Check the bound session after closing: `trace_error.json`,
`runtime_exit.json` and `[PHYSICAL-TRACE]` messages expose finalization/process
failures. A successful process exit alone does not establish complete capture.
Missing trace JSON means unavailable metadata; `physical_test check` inventories
existing PRE/POST/maps/labels and stays unhealthy. Do not replace missing timing
with NPY indices, file mtimes or assumed frame rate. Do not relabel from AUTO.

## Completed D02: preserve actual event mapping — 2026-09-26

`tonight_D02/capture_20260926_162528` is captured and analyzed; do not relaunch it.
Its explicit root is `session_20260926_162528_D02_4e1d6abd`. Actual impacts are
runtime 1 (grid) and 3 (white); 2/4/5/6 are operator-confirmed NO_PHYSICAL_SHOT.
Events 4/5 belong to one right-edge cough/knock action, with attribution unknown;
6 is a left-edge control. Preserve this deviation rather than assigning protocol
truth by order. The accepted plan below remains a plan, not a record of execution.

The existing event/geometry note files are still blank templates. Later operator
corrections are preserved in the forensic package with their source. For future
captures, record actual runtime IDs/action grouping and setup/surface notes;
generic completion checks cannot prove each planned control was performed.
Explicit NO_PHYSICAL_SHOT assignments take precedence over stale unresolved
labeler placeholders. They do not require fake positive coordinates.

All requested native frames were saved: 50 PRE-history and 62–63 POST per event,
no recorded sequence/cap/writer losses. Only 5–13 POST frames precede selection;
later frames are observational, and overlapping event windows are not independent
evidence. Verify unchanged projection separately: `marker_frame` records a display
flip after event completion, not whether pixels actually changed. This capture
has no authoritative scene-invariance record. No new capture is requested.
Report: `/data/skjutbana/evaluation_runs/D02_forensics_20260926_01/REPORT.md`.

## Next calibration action: one flat-surface discrimination run (2026-09-24)

The newer `geometry_retry2_20260924_212534` run physically verifies client
placement: setup/stability MATCH, measured X11 top decoration 37 px, final client
origin (2135,95). Its status is **CALIBRATION_FIT_POOR**. Do not repeat the old
placement-isolation procedure below or relax the 3-camera-pixel tolerance.

Offline analysis establishes bottom-row noncollinearity beyond what any single
homography can fit at 3 px. It does not distinguish surface shape from optical
distortion or systematic detection bias. **No shooting or immediate test is
requested.** When convenient, perform this one controlled substitution:

1. Close the normal application. Keep camera orientation, focus/zoom, projector
   pose/settings and desktop/client placement fixed. Temporarily cover/replace
   the target with a demonstrably rigid, flat, matte light surface spanning all
   24 markers, approximately at the original board plane. Do not move the camera
   or projector to fit it. Record the surface used, any gap/tilt and any unavoidable
   setup change in an operator note alongside the NEW report. An unverified
   flexible sheet would not distinguish the hypotheses.
2. Retain the current calibration; from the repository root run:

   ```bash
   python3 -m automation.physical_foundation_check geometry --output "/data/skjutbana/evaluation_runs/geometry_flat_panel_$(date +%Y%m%d_%H%M%S)"
   ```

3. Leave the restored window in place, let the panel settle, then press **SPACE**.
   The existing tool saves five frames, setup and hashes. Expected exit code may
   still be 2 / CALIBRATION_FIT_POOR because the retained fit is already poor.
   This is a discrimination measurement, not a request to obtain PASS.

Compare the new bottom row's minimum enclosing line half-width with the prior
4.724 px (fresh refined centers 4.501–4.895 px). A much straighter row on the flat
surface supports original-surface nonplanarity; persistent curvature points to
the optical/projector/detection path. A new plane can change global homography,
so raw residual against the old mapping is not the deciding statistic. Record
the displacement, never install a compensating offset. Reflectance/edge quality
also changes with surface material, limiting causal attribution; one run cannot
separate camera optics from projector optics. No new calibration model or
physical PASS follows automatically.

The offline reader is reproducible and uses only an explicit matched-setup
report and its hash-verified frames:

```bash
python3 -m automation.calibration_residual_research --report /path/to/new/report.json --output /data/skjutbana/evaluation_runs/new_unique_calibration_analysis
```

Its output includes residual tables, vector fields, row/column bounds and spatial
holdouts. It never saves calibration/settings or modifies a captured session.
Return the normal surface afterwards; do not run a Can shooting test until the
existing geometry and temporal prerequisites have actually passed.

## Next action: placement-isolation geometry run (2026-09-24 evening)

Latest physical report `no_shoot_20260924_184227/geometry` FAILED: saved SDL
position (2135,95), observed diagnostic position (2135,132). The fresh calibration
also has poor all-marker fit. This run is now classified SETUP_MISMATCH with a
separate CALIBRATION_FIT_POOR finding, not simply a generic geometry failure.

**No more shooting is requested. No test is required immediately.** When ready,
close the normal application so it does not own the camera, keep camera/projector/
board and the CURRENT calibration unchanged, and run this one command from the
repository root:

```bash
python3 -m automation.physical_foundation_check geometry --output "/data/skjutbana/evaluation_runs/geometry_retry_$(date +%Y%m%d_%H%M%S)"
```

The tool creates the window using the saved setup, waits for mapped readback,
and permits one measured recreation if decoration/WM placement displaced the
client. It never uses a guessed -37, changes calibration, or assigns Window.position.
It records each placement attempt and optional native X11 client origin/frame
extents. Do not manually move the window after placement; a mismatch aborts before
camera startup. Press SPACE once it is positioned to collect the marker frames.
A later move invalidates the run. ESC aborts. Every failure returns exit code 2.

Statuses:
- SETUP_MISMATCH: calibration/runtime client, display, viewport, orientation or
  frame geometry differ (or required setup provenance is missing).
- CALIBRATION_FIT_POOR: setup matches, but all-marker calibration error exceeds
  the unchanged 3-camera-pixel bound, coverage is insufficient or inverse is bad.
  Fresh observations are still saved to characterize the remaining fit error.
- PHYSICAL_GEOMETRY_FAIL: setup and stored fit pass, but fresh projected-marker
  observations fail coverage/residual checks.
- PASS: all prerequisite geometry checks and fresh observations pass. This says
  nothing about PRE timing or physical hit accuracy and does not unlock shooting.

Do not recalibrate first for this particular test: retaining the same fit isolates
whether restoring the client origin removes the dominant displacement. The
remaining right/bottom errors need review before another calibration approach.
No threshold relaxation, collision margin or camera-coordinate offset is applied.

## NO-SHOOT foundation checks first — 2026-09-24

**Do not run another Can shooting trial yet.** The captured September 15 run has
invalid physical mapping and post-damage PRE. The `launch-can` workflow now fails
before creating a binding unless a READY foundation report is supplied. Today's
checker reports BLOCKED because live PRE selection is not yet validated; do not
hand-write a READY report or use the older launcher instructions to bypass it.
D02 and CAN_OBJECT remain separate; no new physical session has been captured.

1. Connect camera, microphone, projector and speakers. Start `python3 main.py`.
   Move the existing 800x600 client window to its final projector position, set
   the intended viewport and camera orientation, and leave the physical setup
   fixed. Open **Kamera -> Kalibrera viewport via kamera**, press **ENTER**, and
   wait for the saved-calibration message. This deliberate user calibration uses
   the corrected inverse and records window setup; the development pass itself
   did not modify either settings file. Do not press R to clear calibration.
   Close the application before running the diagnostics (one camera owner).
2. From the repository root, run:

   ```bash
   FOUNDATION_RUN=/data/skjutbana/evaluation_runs/no_shoot_$(date +%Y%m%d_%H%M%S)
   python3 -m automation.physical_foundation_check geometry --output "$FOUNDATION_RUN/geometry"
   ```

   The diagnostic restores window position from a newly recorded calibration.
   Verify it is on the correct projector, then press **SPACE**. It projects the
   existing 24 markers, waits for presentation and saves five spaced camera
   samples. ESC aborts. Each frame must have at least eight distributed markers,
   maximum error <=3 camera pixels and round-trip error <=0.05 client pixels.
   This is a measured diagnostic tolerance, not a hit margin. FAIL returns exit
   code 2. Review report.json expected/observed coordinates and saved images.
   A successful software round trip alone is not physical calibration proof.
3. Only after geometry PASS, with unchanged setup:

   ```bash
   python3 -m automation.physical_foundation_check sync --output "$FOUNDATION_RUN/sync"
   ```

   Press SPACE; four paired flashes and short original tones run about five
   seconds apart. No weapon or shot is involved. The microphone must hear the
   speaker. Captured PCM, block/sample provenance, camera brightness/sequence,
   command timestamps and report.json are saved. Missing signal, gaps or moved
   setup produce an incomplete report. The measured audio-minus-visual receipt
   offset/jitter includes speaker/projector output latency, acoustic propagation
   and both input paths; it cannot by itself calibrate device time or choose PRE.
   Keep the same window/client geometry and do not open competing capture apps.
4. Preserve both output directories for review. The next code step must validate
   causal PRE selection using this provenance and the three immutable recorded
   cases, including projection transitions and rapid events. The current offline
   visual-step proposal finds preceding frames in all three cases but is not a
   live policy. No further shooting is required to perform these two diagnostics.

Optional machine-readable prerequisite report (expected exit 2 / BLOCKED today):

```bash
python3 -m automation.physical_foundation_check preflight \
  --geometry "$FOUNDATION_RUN/geometry/report.json" \
  --timing "$FOUNDATION_RUN/sync/report.json" \
  --pre-selection /data/skjutbana/evaluation_runs/physical_foundations_20260924_01/research_v2/event_specific_pre.json
```

Neither diagnostic creates a physical trace binding or claims a successful hit.
Once geometry AND causal timing/PRE are verified, a fresh collision-protected
CAN_OBJECT binding and OBJECT crop capture remain mandatory. The original CAN
capture, prior forensics, settings and physical_traces symlink stay untouched.

## Captured retry reviewed — prerequisite checks before another shot (2026-09-23)

Do not immediately repeat the shooting instructions below. The actual captured
run `can_retry_20260915_150509` was gold hit / between-object physical miss / duck
hit, despite its planned order. Awards were gold / beans / mini (1/3 correct).
Preserve this run and use the user's observations; planned rows are not labels.

Offline forensics reproduced all three decisions exactly. New visible marks
already exist in the chosen PRE frames, and frozen camera AABBs are above the
photographed targets. Extra POSTs alone do not repair either problem. The
following prerequisites are recommended, not captured or claimed complete:

1. **No shooting:** with the same board, projector/window and camera setup,
   verify the existing calibration against visible projected target boundaries.
   Save calibration identity/matrices, viewport/display/window geometry and
   corresponding camera observations. Compare frozen outlines in a nonprojected
   diagnostic view; do not add detection overlays to the physical scan area.
   Do not infer a translation correction from a desired hit or the three marks.
2. **No projectile required:** record several independently observable events
   with both a visible change and audible contact, for example a visible clapper
   away from the target. Record audio sample/stream timing and camera acquisition
   timing using existing capture owners. A valid common timing reference is
   required; PCM receipt time alone cannot prove absolute latency. The current
   capture lacks those timestamps, so this is a prerequisite for the next code
   pass, not an assertion that today's launcher records them already.
3. Once timing and alignment are defensible, design a fresh, SFX-OFF three-shot
   verification: gold/can hit, deliberate between-object miss, duck hit. Preserve
   projection through the planned late-evidence observation window if testing
   settled persistence. The current immediate-reaction launcher does not do that;
   a fixed visual observation window needs an explicit diagnostic implementation.
   Do not silently change normal gameplay or use late animation pixels as proof.

The existing source is replayable without camera/game startup:

```bash
REPLAY_OUT=/data/skjutbana/evaluation_runs/can_replay_$(date +%Y%m%d_%H%M%S)
python3 -m automation.object_physical_replay \
  --capture /data/skjutbana/evaluation_runs/can_retry_20260915_150509 \
  --output "$REPLAY_OUT"
python3 -m automation.object_physical_replay_selftest \
  --capture /data/skjutbana/evaluation_runs/can_retry_20260915_150509
```

The replay verifies saved OBJECT hashes and equality to corresponding full-frame
pixels before using later frames. It preserves presentation flags, rejects an
existing output destination, and records source/output hashes. Continuation and
older-history PRE are explicitly offline counterfactuals, not production settings.
The three recorded-output tests deliberately reproduce gold/beans/mini. They do
not claim corrected physical performance. S03 and original sessions stay untouched.

## Can Shootout: fresh captured three-shot retry (2026-09-15)

The first real 20 m, SFX-OFF run produced two independently observed physical
misses outside objects (correct game MISS) and one observed duck hit (false game
MISS). Its capture was refused because settings still pointed to the populated
D01 directory. Those shots are not replayable. The next run must use a new binding.

Close the current application. From the repository root, run:

```bash
CAN_RUN=/data/skjutbana/evaluation_runs/can_retry_$(date +%Y%m%d_%H%M%S)
python3 -m automation.physical_collection launch-can --output-dir "$CAN_RUN"
```

This single launcher prepares an explicit three-shot CAN_OBJECT plan/binding,
creates a unique trace root, verifies it, and starts the application with
**SFX OFF and immediate OBJECT feedback**. It saves stdout/stderr to
`$CAN_RUN/runtime.log`. main verifies and atomically claims the binding **before
hardware/game imports**; reused/wrong/populated roots are refused. An application
restart requires a new output directory and binding, even if no shots were taken.
No saved settings file is changed and no restore command is needed. Do not use
plain `python3 main.py` for this captured retry; it does not prepare a fresh binding.
The old settings-based `start` workflow remains available for other capture plans.

Choose **Skjut burkar – 20 m**. Wait for LIVE and a settled projection before each:

1. Intentional can hit.
2. Intentional duck hit.
3. Intentional physical impact just outside an active object.

Independently record what the projectile actually did in `event_notes.json`, with
actual shot/event IDs. Object intent and collision are not physical GT. A physical
near miss is still a physical shot; it is not NO_PHYSICAL_SHOT. No audio capture
means NOT_CAPTURED; extra events need independent notes. Record board/surface,
calibration and camera/projection observations in `geometry_notes.json`, leaving
unavailable details explicit. Do not use or bind D02 for these shots.

After the last shot, wait five seconds and close normally. Then:

```bash
python3 -m automation.physical_test check --binding "$CAN_RUN/binding.json"
python3 -m automation.physical_test label --binding "$CAN_RUN/binding.json"
python3 -m automation.physical_test evaluate --binding "$CAN_RUN/binding.json" --output "$CAN_RUN/current_evaluation_01"
```

`evaluate` measures ordinary global selectors using independent labels, not OBJECT
collisions. Normal immediate game reactions may change CURRENT's later frames;
that contamination is logged. OBJECT gameplay does **not** wait for CURRENT in
this retry. Do not set `SKJUTBANA_OBJECT_COMPARE=1`; the launcher forces it off.
The old optional comparison visual hold remains for a separately designed trial.

One `[OBJECT-SHOT]` JSON record joins OBJECT and later CURRENT. It includes frozen
camera/game shapes, player/distance, top-three XY/evidence/provenance, exact
collision, timing, previous-hole context, and **every object's** bounded raw
component and observed-candidate ledger. Raw entries report reserve/cap decisions;
observations report exact-XY confirmation reasons, local/global ranks, AABB
membership/distance, exact-shape membership and final retention/drop reason.
Frames/ledgers are bounded, not printed every frame. AABB distance is camera
pixels; exact-shape distance is not approximated with a fake hit tolerance.

The existing asynchronous physical recorder now saves the actual OBJECT PRE/POST
crop inputs, with camera crop bounds, timestamps and content hashes. They live in
`object_inputs/` and `trace.json.object_evidence_crops`, **separate from global
replay frames**. Full per-object diagnostics remain in `object_query_diagnostic`.
Global and OBJECT outputs are independently inspectable; neither is physical GT.

The query uses the latest safe causal PRE surface and freezes gameplay until its
own decision. Old unchanged holes cannot supply new change; adjacent damage or
visible enlargement may. Missing safe PRE or unresolved change returns UNCERTAIN.
Duck collision now shares its unchanged body/head/beak drawing, without a hit
margin. Software regression coverage does not prove this physical false negative
is resolved. Do not change thresholds before reviewing this captured retry.

Later, separately, finish the existing-hole/distance/SFX checks in
`research/physical_capture_plans/CAN_OBJECT.json`. Speaker playback remains
instrumented, not automatically suppressed. The three-shot retry is deliberately
silent; it is not a sound-feedback test or a physical reliability estimate.

## D02 six-event diagnostic package — process-only launch (2026-09-26)

Use `research/physical_capture_plans/D02.json`. It has **two physical-shot rows**
and six separate protocol events; planned control categories are never labels.
Prepared local package:
`/data/skjutbana/evaluation_runs/ranking_families_20260914_112457/tonight_D02/`.
The preparation itself is a reusable template; the September 26 completed run
and its protocol deviations are recorded above. Each future launch copies the accepted
plan and operator notes into a new run directory and allocates a unique explicit
D02 DEVELOPMENT trace root. It never rewrites saved settings or the trace symlink.
`saved_configuration.json` is a read-only setup aid with saved calibration identity,
viewport/content geometry and analysis-crop settings. It does not prove the actual
camera frame, projector placement or physical board bounds. Working crop geometry
is derived per frame and remains pending until recorded metadata is exported.

**Do not reuse the old D01 directory.** It now mixes September 11 runtime data
with September 9 labels. Original D01 comparisons use only the hash-verified
September 10 counterfactual cache. The recorder now refuses a populated capture
root; restarting the application requires a fresh binding. Existing files are
never restored, removed or overwritten by that guard.

1. Configure the camera/projector with the ordinary calibration and a static
   Testtavla/image scene. Keep the can game separate. Record actual camera-space
   physical board bounds, board/surface revision, observable seam endpoints (or
   an explicit unavailable reason), framebuffer/viewport/content rectangle and
   display description in `geometry_notes.json`. Use an actual camera view to
   measure physical bounds; game target positions cannot supply those bounds.
2. Fill the prepared geometry notes, then close the application. From the
   repository root, use the shared process-only launcher below. It copies the
   existing plan/notes, records saved calibration/geometry and settings identity,
   checks the exact binding before hardware startup and saves runtime output to
   the new run's `runtime.log`. Independent geometry notes remain operator truth;
   the launcher does not claim to validate physical placement. Blank notes copied
   from preparation must be completed in the new run before physical events.

```bash
D02_PACKAGE=/data/skjutbana/evaluation_runs/ranking_families_20260914_112457/tonight_D02
D02_RUN="$D02_PACKAGE/capture_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_collection launch-d02 \
  --plan "$D02_PACKAGE/plan.json" --output-dir "$D02_RUN" \
  --trace-root /data/skjutbana/physical_traces
```

Launch in the same display/camera configuration and verify the recorded geometry
still matches. The preflight checks saved configuration; it cannot independently
prove the current desktop/projector placement. If calibration, projection or
surface changes, stop and record a fresh revision in a new package. Do not start
another collection helper while this binding is active.
An existing run directory or claimed/populated binding is refused. A restart
requires another fresh launch/run directory. Do not use the old settings-writing
`physical_collection start` or `physical_test start` for this D02. No settings
restore is needed after the process-only launch. Existing CAN_OBJECT launch and
its foundation gate are unchanged.
The plan identity/rows and its exact snapshot hash must match the binding. To
refresh the setup aid after configuration changes, write a new file with
`python3 -m automation.physical_capture_provenance inspect --output "$D02_RUN/saved_configuration_02.json"`.
For D02, CURRENT remains the physical authority; no experimental global ranker
opt-in is offered. The separate can-game OBJECT capability is not part of D02.

**Capture window:** warm up the enabled scanner for at least five seconds. D02
requests all available native-cadence grayscale history in [peak−4 s, peak], up
to 240 PRE frames, plus the original separately named PRE snapshot; and
(peak, peak+5 s], up to 300 POST frames. POST collection continues even after the
detector completes/removes the event. It does not extend the detector deadline,
modify its frozen decision, or authorize using later/next-event pixels for that
decision. Keep projection unchanged for the entire session, including hit-marker
or scene reactions. Allow five seconds of POST after each event except the
planned short-delay sound control; record its actual delay and preserve the
overlap as diagnostic evidence.

The capture-only configuration is frozen in the binding and trace provenance;
defaults outside an explicit extended capture remain 24 PRE / 64 POST frames.
There is no extra image ring. Existing scanner history (360 frames), writer
queue (128 jobs / 1 GiB), configured frame caps and 16 captured-event cap bound
D02 storage in memory. Disk volume is larger and I/O drops remain possible.
Each trace reports requested windows, available/attempted/saved counts, available
frame timestamps and camera identities (up to 1,024 per window), actual saved
range/gaps, frame-cap losses, missing acquisition metadata, sequence gaps and
writer losses. Missing history or early shutdown is never claimed continuous.
Camera timestamps remain receipt observations; sensor exposure and comprehensive
worker delivery/occupancy telemetry remain unavailable.

3. Execute protocol IDs **3 → 1 → 2 → 4 → 5 → 6**. This places the isolated
   sound control before physical shots and the after-shot control directly after
   the second physical shot. Keep exact runtime event IDs in `event_notes.json`.

| Protocol ID | Action | Independent observation / hypothesis |
|---|---|---|
| 3 | Sound-only trigger with settled board | No discharge/damage; can sound alone produce a false candidate? |
| 1 | One low-contrast physical impact | Label the actual hole; does weak evidence survive retention/ranking? |
| 2 | One printed-line/near-hole physical impact | Label actual change, including enlargement; do not reject old-hole proximity. |
| 4 | Sound-only trigger shortly after ID 2 | Record actual delay and absence of discharge/new damage; test residual-change contamination. No fixed timing gate is implied. |
| 5 | Observed no-damage movement of one panel side | Record which side and visible motion. If needed, use a separate sound cue to trigger camera capture; record it. |
| 6 | Observed no-damage seam/alternate-panel movement | Record location and lack of damage; compare structured false change. |

Control sounds must involve no projectile. A control that produces no audio event
is **NOT_CAPTURED**, not a successful NO_IMPACT rejection. An unintended tear or
uncertain physical event stays UNKNOWN/AMBIGUOUS pending review. Do not fill six
successful labels merely because six actions were planned. Keep the observer's
notes independent of detector/game outcomes.

4. After the last event wait at least five seconds, then close normally and check
   the exact binding. Label the two observed physical impacts, then classify each
   human-confirmed no-impact event explicitly; S in the labeler means unresolved.

```bash
python3 -m automation.physical_test check --binding "$D02_RUN/binding.json"
python3 -m automation.physical_capture_provenance geometry-postflight --binding "$D02_RUN/binding.json" --output "$D02_RUN/recorded_geometry.json"
python3 -m automation.physical_test label --binding "$D02_RUN/binding.json"
# Repeat with each actual control event ID and its specific human observation:
python3 -m automation.physical_test classify --binding "$D02_RUN/binding.json" --shot-id ACTUAL_EVENT_ID --no-physical-shot --reason "HUMAN_OBSERVED_REASON"
python3 -m automation.physical_test evaluate --binding "$D02_RUN/binding.json" --output "$D02_RUN/evaluation_01"
python3 -m automation.physical_collection finalize --plan "$D02_RUN/plan.json" --session D02 --binding "$D02_RUN/binding.json" --in-capture-order --preview
python3 -m automation.physical_collection finalize --plan "$D02_RUN/plan.json" --session D02 --binding "$D02_RUN/binding.json" --in-capture-order --output "$D02_RUN/finalized_01.json"
```

Use `--in-capture-order` only after confirming the two physical ordinals map to
the physical event IDs in order. Otherwise supply `--shot-map 1=EVENT` and
`--shot-map 2=EVENT`. Generic finalization checks the two physical rows and all
event labels; separately review that all four planned controls have real captured
events and independent notes. Never restart into D02 for the optional can-game
test. Bind a fresh separate session. Can-game console `OBJECT-SHOT` results and
emitted/gameplay XY are separate diagnostics, never D02 truth or global detector gains.
Review `recorded_geometry.json` for actual frame shapes and per-stage crop/debug
metadata. Missing fields stay UNAVAILABLE; neither saved settings nor a game AABB
may silently replace missing physical geometry. The exporter reads only the bound
D02 and leaves source traces/labels unchanged.

The existing S01/S02 plan requests **10 physical discharges per session**.
Count discharges, audio-triggered events and visible physical changes separately;
they are not necessarily one-to-one. Keep runtime event IDs; never shift labels
to hide extra triggers or silently invent a single impact for an ambiguous event.
Store generated reports and large data under `/data/skjutbana`.
`content/ai/physical_traces` is intentionally a symlink to
`/data/skjutbana/physical_traces`; preserve it.

The accuracy target is **100% correct selected/emitted physical results**, with
95% the minimum acceptable level. Preserve @5/@10/@20/@42 comparisons. A high
candidate oracle is not success. CURRENT remains live authority; the common
verifier experiments in [ACCURACY_95_100_RESEARCH.md](ACCURACY_95_100_RESEARCH.md)
are frozen offline references only. S03 must remain entirely untouched.

## Ambiguous physical events and current research state

S02 event 1 has an unconfirmed possible second visible change. Preserve its
native label and report the caveat plus exclusion sensitivity. An audio event
can have no discharge; a discharge can have no new visible hole; tape/paper can
change or a previous hole can reopen. Do not infer truth from timing, appearance
or one-to-one shot numbering alone. Rapid genuine shots remain valid.

The minimum research truth contract uses `impacts: [...]` and explicit
SINGLE_IMPACT, NO_PHYSICAL_SHOT, UNKNOWN/AMBIGUOUS states. Multiple impacts can be
represented later without changing that array. The current single-output
evaluator excludes unresolved/multiple-impact truth explicitly. The existing
label GUI's S key still means unresolved; it does not classify physical truth.
Legacy finalization deliberately refuses unresolved assignments. Keep uncertain
cases pending with a reason rather than forcing a label to make finalization pass.

The original six-shot D01 development capture was finalized and contains no
intentionally collected no-impact events. Its current raw directory is invalid
for the original labels; use the frozen research cache described above. The D01
findings are reviewed and D02 preparation is authorized; capture remains pending
the user's physical session. Do not reuse or inspect S03.

Physical trace capture now retains upstream contour/filter evidence and cleanup
boundaries, with camera versus crop coordinate provenance. When hybrid merge
runs with physical trace capture enabled, the new ledger records each input's
identity, spatial geometry changes, quota/cap outcome and retained/output rank.
It is observational and adds capture payload/cost; it does not change detector
authority. Earlier traces lack that ledger. The D01 reconstruction matches one
recorded stage exactly for events 1–5, but only 199/200 points for event 6; this
is not complete asynchronous replay. Missing RAW evidence remains UNAVAILABLE.
A trace/frame PASS establishes artifact
integrity; it does not establish perfect physical truth, all missing source
proposals, or the live timing of an offline verifier.

## Before shooting

Choose a fresh report directory. Close the application before binding a session:

```bash
mkdir -p /data/skjutbana/evaluation_runs
CAPTURE_RUN=$(mktemp -d /data/skjutbana/evaluation_runs/capture_XXXXXXXX)
python3 -m automation.physical_capture_plan --sessions 3 --shots-per-session 10 --output "$CAPTURE_RUN/plan.json"
python3 -m automation.physical_collection start --plan "$CAPTURE_RUN/plan.json" --session S01 --output "$CAPTURE_RUN/S01_binding.json"
python3 -m automation.physical_collection preflight --plan "$CAPTURE_RUN/plan.json" --session S01 --binding "$CAPTURE_RUN/S01_binding.json"
python3 main.py
```

Keep the same plan for S02/S03 and use a separate binding for each session.
`physical_collection start` reserves the trace root, enables capture and saves
a byte-for-byte settings backup. **Do not then run `physical_test start`: that
separate lifecycle creates another root and overrides the collection binding.**
The intended category sequence is collection guidance, never detector truth.

After 3–5 shots, check the exact binding from another terminal:

```bash
python3 -m automation.physical_test check --binding "$CAPTURE_RUN/S01_binding.json"
```

Wait at least five seconds after the final event before closing the application
normally. Check again after shutdown to confirm that all evidence flushed.

## Labeling and extra audio events

```bash
python3 -m automation.physical_test label --binding "$CAPTURE_RUN/S01_binding.json"
```

Click the physical hole, then Enter/P for precise or A for approximate. **S saves
an unresolved skip; it does not mean `NO_PHYSICAL_SHOT`.** Use S when the event
cannot yet be assigned, then explicitly classify any human-confirmed false event:

```bash
python3 -m automation.physical_test classify --binding "$CAPTURE_RUN/S01_binding.json" \
  --shot-id 4 --no-physical-shot --reason "Human confirmed extra audio event, no physical impact"
```

Classification writes only `physical_assignments.json`. For an immutable review,
put the same assignments in a new external JSON instead and pass `--mapping`
to quality/evaluate. Example:

```json
{"4": {"state": "NO_PHYSICAL_SHOT", "label_shot_id": null, "reason": "Human-confirmed extra audio event"}}
```

Missing labels or S markers alone must never imply a nonphysical event.

## Reset labels safely

Close the application and labeler. Preview the bound session:

```bash
python3 -m automation.physical_collection reset-labels --binding "$CAPTURE_RUN/S01_binding.json" --session S01
```

The preview lists the exact files and changes nothing. To perform the reset,
repeat with `--apply`, then relabel through the same binding:

```bash
python3 -m automation.physical_collection reset-labels --binding "$CAPTURE_RUN/S01_binding.json" --session S01 --apply
python3 -m automation.physical_test label --binding "$CAPTURE_RUN/S01_binding.json"
```

The allowlist is **only** `shots/shot_XXXXXXXX/ground_truth.json`,
`shots/shot_XXXXXXXX/ground_truth_status.json`, and session-level
`physical_assignments.json`. Each existing file is copied and hash-verified in
a fresh `/data/skjutbana/label_resets/` directory before removal; the terminal
prints that backup path. `--backup-root` can select another directory outside
the session. Repeating the reset on an unlabeled session is a no-op.

PRE/POST frames, candidate evidence, selectors, trace metadata, runtime logs,
settings, bindings and external reports remain unchanged. Existing evaluation
and finalization reports still describe the old labels; generate fresh reports
after relabeling. Reset refuses incomplete/pending captures, an active automation
port, wrong binding/session identity, symlinks inside the label/shot paths and
untouched validation bindings. The storage-root symlink is supported.
Legacy captured sessions can use an explicit `--root` instead of `--binding`.

## Quality, evaluation and finalization

Resolve the bound root once; do not use latest-session discovery or scan the
whole trace storage during development:

```bash
SESSION_ROOT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace_root"])' "$CAPTURE_RUN/S01_binding.json")
python3 -m automation.physical_trace_quality --session-root "$SESSION_ROOT" --output "$CAPTURE_RUN/S01_quality.json"
python3 -m automation.physical_test evaluate --binding "$CAPTURE_RUN/S01_binding.json" --output "$CAPTURE_RUN/S01_evaluation"
```

For external assignments, add `--mapping /path/to/assignments.json` to both
commands. Quality opens frame/map artifacts and checks label identity, bounds,
timing and assignment consistency. Replay readiness is structural; evaluation
must additionally report `CURRENT_EXACT_REPLAY` MATCH before claiming exact
recorded-input replay. This is not regenerated detector replay.

**No manually written labels manifest is needed.** After labeling, use the
binding to read the exact captured root, verify its plan/session/class, and build
the aggregate manifest from saved ground truth and physical assignments. The
command always checks current trace/frame/label quality and hashes the session;
`--quality` optionally cross-checks an existing report.

If physical event IDs are in planned shot order, explicitly confirm that order
with `--in-capture-order`. This excludes human-classified NO_PHYSICAL_SHOT events.
It never infers false events from missing labels or silently assumes ordinal
mapping. Preview first:

```bash
python3 -m automation.physical_collection finalize --plan "$CAPTURE_RUN/plan.json" \
  --session S01 --binding "$CAPTURE_RUN/S01_binding.json" \
  --in-capture-order --preview
```

When the displayed mapping is correct, use a fresh output path:

```bash
python3 -m automation.physical_collection finalize --plan "$CAPTURE_RUN/plan.json" \
  --session S01 --binding "$CAPTURE_RUN/S01_binding.json" \
  --in-capture-order --output "$CAPTURE_RUN/S01_finalized.json"
```

For a different order, replace `--in-capture-order` with repeated
`--shot-map PLANNED=EVENT` arguments, for example `--shot-map 1=1 --shot-map 2=3`.
Supply every physical ordinal unless already explicitly mapped in binding/plan
rows or assignments. Existing mappings must agree with the CLI. A coordinate
`label_shot_id` is a label-file reference, **not** a planned physical ordinal.
For S02-style external assignments add `--mapping /path/to/assignments.json`.
Recovered sessions may use their explicit recovery binding; missing/incomplete
capture evidence still blocks finalization.

The new report includes `label_manifest`, fresh `quality`, the mapping basis and
input SHA-256 hashes. No labels, settings or session files are changed. Existing
outputs are refused. UNKNOWN/AMBIGUOUS/unresolved labels, reused coordinate labels,
duplicate or missing physical mappings, wrong bindings and stale quality claims
are refused with an actionable error. `--preview` writes nothing.

The historical aggregate-manifest CLI remains supported:

```bash
python3 -m automation.physical_collection finalize --plan "$CAPTURE_RUN/plan.json" \
  --session S01 --labels "$CAPTURE_RUN/S01_labels.json" \
  --quality "$CAPTURE_RUN/S01_quality.json" --output "$CAPTURE_RUN/S01_finalized.json"
```

Finalization requires explicit trace/frame/label PASS and complete event/shot
mapping. A string `FAIL`, an unresolved label, or another session's quality
report cannot pass. Candidate/oracle recall and actual selected/emitted accuracy
must remain separate. Report unavailable terminal evidence separately from a
measured miss or timeout.

For the existing D01 capture, use the preserved plan
`research/physical_capture_plans/D01.json` (byte-identical to the original local
`evaluation_runs/physical_capture_plan_D01.json`), session `D01`, and
`evaluation_runs/D01_binding.json`. Its human-confirmed mapping is event 1–6 to
planned shot 1–6. An existing `D01_finalized.json` is a preserved baseline;
choose a new report path for rechecks. Bindings and runtime reports stay local.

After capture, optionally restore the binding's original settings:

```bash
python3 -m automation.physical_collection restore --binding "$CAPTURE_RUN/S01_binding.json"
```

## Protected validation and recovery

S03 remains `VALIDATION_UNTOUCHED`. Do not inspect it for development, reset its
labels, or include it in inventory/dataset rebuilds while fitting or choosing a
configuration. Use explicit development roots only. The current generic patch
builder uses post-decision frames and does not enforce the validation split;
its output is unsuitable for live-equivalent verifier claims without a causal
cutoff and an explicit development-session allowlist.

Keep incomplete/crashed sessions and every original result. Repair capture
before collecting a fresh uniquely bound session; never delete a trace to fix
labels. Missing frame artifacts block use. S01's recovered data is preserved at
`session_20260909_S01_recovered_20260909`: labels are complete, event 4 is
nonphysical, and event 11 has usable images/GT but incomplete terminal evidence.

S02's completed 2026-09-09 capture uses **`evaluation_runs/S02_retry_binding.json`**,
not the earlier S02 binding. Physical events are 1,3,4,5,6,7,9,11,12,13;
events 2,8,10 are human-confirmed nonphysical events. Their original S markers
remain untouched; the finalized external mapping and checks are documented in
[S02_PHYSICAL_FINDINGS.md](S02_PHYSICAL_FINDINGS.md).
