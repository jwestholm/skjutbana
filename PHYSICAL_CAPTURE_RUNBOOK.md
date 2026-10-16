# Physical capture runbook

## Optical/HIL lab v0.1 — no shooting

**OPTICAL/HIL TEST != PROJECTILE VALIDATION. SYNTHETIC != PHYSICAL.**
PASSIVE_VARIABILITY is environmental/camera variability; PROJECTOR_RESPONSE is a
deliberate optical probe, not impact evidence. No physical optical run has been
validated yet. This does not replace the pending three-shot GT acceptance below.

When camera/projector access is available, start a dedicated research process:

```bash
SKJUTBANA_OPTICAL_LAB=1 python3 main.py
```

Use the existing calibrated projector window placement. `setWindowPos` is allowed
only between runs; no menu entry or game is needed. This process never starts audio
acquisition or executes the detector/game loop. Do not combine it with shot/GT
capture environment variables. Quit/restart normally to return to gameplay.

In a second terminal, from the repository:

```bash
PASSIVE="/data/skjutbana/evaluation_runs/optical_hil_passive_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_optical_lab passive --duration 30 --output-dir "$PASSIVE"

PROBES="/data/skjutbana/evaluation_runs/optical_hil_probes_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_optical_lab probes --settle 0.6 --output-dir "$PROBES"

GEOMETRY="/data/skjutbana/evaluation_runs/optical_hil_geometry_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_optical_lab geometry --input "$PROBES" --output-dir "$GEOMETRY"
```

The CLI subscribes before sending the start command and waits on explicit events.
Ctrl+C requests cancellation; Escape/window close preserves completed records.
New directories only; source captures and existing runs are never overwritten.
Geometry auditing is offline, hashes the saved images and uses only saved mapping
and probe plans. Missing triplets/calibration are UNAVAILABLE. It does not calibrate
or save settings. Optical centroid displacement includes photometric/perspective
effects; do not interpret it as pure calibration error or a physical-hole XY.

Passive: constant MID_GRAY, streaming tile moments/range and1DN histogram MAD of
tile means, per-frame brightness/difference/timing, relative low/high-variability
tiles, brightness lag correlation and diagnostic translation. Translation is not
applied to images. Native-pixel temporal noise is not the tile-mean statistic.
Probes: bright/dark squares plus vertical/horizontal edge patches at center/four
quadrants, each with baseline/probe/recovery. Full-camera difference minus measured
recovery variation gives a weighted response centroid/spread; no expected-position
ROI or snapping. Broad/multiple optical changes can make it ambiguous.

Artifacts: `config.json`, `manifest.json`, `summary.json`, `REPORT.md`,
`transitions.jsonl` (physical runs), `frames.jsonl`, `timing.csv`,
`passive/tile_statistics.npz`, sampled native NPY+JSON under `passive/` or
`projector_probes/`. Records include original acquisition metadata, raw backend
timestamps, pixel/file hashes, source sequence, projector command/apply/flip times,
source code hashes, processing/copy/write cost and gaps. Flip is not photon time;
receipt is not exposure. No image is tagged as projectile GT.

Default30s passive saves roughly3 native frames (~75MB at3840x2160 BGR), plus compact
statistics/logs; the image window is30s after0.6s settle, then writer drain. Probes
save60 frames (~1.49GB at4K), with36s intentional settle plus acquisition/analysis
time (budget roughly45–70s, physically unverified). `--save-all-frames` is explicit:
30s×30fps×4K BGR is about22.4GB before metadata and may exceed write throughput.
The4K synthetic processing profile is~93ms/frame, so native30fps is not guaranteed.
Backpressure/write losses are counted and mark INCOMPLETE; do not treat irregular
sampling as a full-rate flicker measurement. No deadline or evidence rule is relaxed.

Offline without devices/server (deterministic seed1729; never physical evidence):

```bash
python3 -m automation.physical_optical_lab passive --synthetic --duration 30 \
  --output-dir "/data/skjutbana/evaluation_runs/optical_hil_synthetic_passive_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_optical_lab probes --synthetic \
  --output-dir "/data/skjutbana/evaluation_runs/optical_hil_synthetic_probes_$(date +%Y%m%d_%H%M%S)"
```

`--dry-run` is an alias for synthetic execution, not a hardware preflight. Share
manifest/summary/config, frame and transition logs, timing CSV and relevant complete
NPY+JSON pairs when troubleshooting. Socket execution is not verified in Codex's
restricted environment; queue/event parsing and instrument lifecycle are tested.

## Independent full-frame GT review capture v1 — 2026-10-13

**PRE causality correction:** in the first three-shot run
`physical_gt_capture_v1_20261013_142407`, the operator saw the new hole in PRE.
Those references are not certified before-damage truth. The repaired adapter
uses the existing ring selector with a hard500ms receipt/publication margin before
audio; missing clocks or older frames produce PRE UNAVAILABLE. Do not use a
near-audio or static fallback. No original capture or labels were rewritten.

**2026-10-14 offline hardening:** the qualified native reference is pinned at the
first dispatched event observation, before camera/object/scanner work. The existing
300-frame ring need not retain PRE until live processing ends. No live image copy
or disk I/O is added; saving still waits for closure. Missing history at event start
remains UNAVAILABLE even if later frames arrive. The operator is currently unavailable;
physical acceptance is pending, and no additional capture is needed for this audit.

**LIVE PHYSICAL EVIDENCE != GT REVIEW CAPTURE.** The operator could not identify
holes independently in the old paired object crops, even with blink/zoom. That
series remains UNKNOWN; candidates and phone photographs cannot supply its XY.
The following opt-in test captures new human-inspection material, without changing
same-shot detection, thresholds, selection or game scoring. It has software tests;
physical visibility is not validated yet.

From the repository, use a new output directory:

```bash
GT_RUN="/data/skjutbana/evaluation_runs/physical_gt_capture_v1_$(date +%Y%m%d_%H%M%S)"
python3 -m automation.physical_collection play-can \
  --object-crops --mature-candidates --gt-review --gt-shots 3 \
  --output-dir "$GT_RUN"
```

Select **Spel → Skjutprylar**. Perform only three deliberate shots, individually:
1. Clean, visibly inspectable surface.
2. Near old damage/tape.
3. Dark/projected detail.

Do not move the camera or board between live and review. Wait after each shot:
both OBJECT_HIT and ordinary CURRENT must finish, and the mature worker must be
idle. After the game result has been displayed, the same scene is paused; audio
acquisition and scanner are stopped. Wait0.5s, then display WHITE, BLACK, MID_GRAY
in sequence, waiting0.6s before each original full-camera frame acquisition.
No calibration/reference scene is invoked. Only pixels are filled on the display;
no board reference, evidence, gameplay hit or label is derived from the neutral
background. After normal scene restoration and1s plus a new camera frame, the
review ring is cleared and the existing scanner/audio lifecycle rearms.
**Do not shoot during this phase or rearming. Wait for scanner READY.** Keyboard
game actions are paused; Escape/window close terminates capture cleanly.
Each audio event consumes a review slot, including false triggers. Enter actual
event IDs and independently observed discharge in `event_notes.json`; never assume
three audio events mean three physical impacts. Overlapping events are explicitly
unavailable for independent review association.

Saved under `$GT_RUN/gt_review/`:
- `config.json`, `manifest.json`, `REPORT.md`, `gt_capture_timing.csv`.
- `shots/shot_XXXXXXXX/pre_0.npy`: latest clock-qualified native ring frame with
  BOTH receipt and publication at least500ms before the audio estimate. If none
  exists at event dispatch, no PRE file is saved; the manifest records UNAVAILABLE.
- `live_post_0.npy`, `live_post_1.npy`: first two available normal POSTs before
  live closure. A long live job can exhaust POST history; PRE is now pinned earlier.
  No missing frame is invented.
- `white.npy`, `black.npy`, `mid_gray.npy`: separate **GT_SETTLED** native BGR
  camera frames, never ordinary detector `trace.frames` or same-shot evidence.
- Each NPY has a JSON sidecar: shot/event, role, canonical dimensions/orientation,
  acquisition sequence/stream, frame timestamp, file/pixel hashes, display RGB,
  flip-return time, delay and write/copy timing. Read/receive times do not certify
  sensor exposure or actual projector photometry. Missing/dropped/interrupted
  images stay explicit. A failed camera restore or writer drain stops before rearm.

PRE sidecars also record `audio_peak_ts`, original `audio_acquisition`,
`delta_ms_to_audio` (negative means earlier), receipt/read/publication clocks,
`selection_rule`, allowed cutoffs and `pre_cutoff_verified`. **The500ms guard is
not a measured upper bound on audio latency.** `causal_pre_certified=false` and
`physical_impact_timestamp=null` remain honest limits until causality can be
independently established. Backend timestamps are not exposure truth. Legacy
references show PRE UNQUALIFIED; guarded references still show pre-damage
UNVERIFIED. If the purported new hole is already visible in PRE, do not certify
newness from that comparison; preserve UNKNOWN and report the provenance.

`physical_gt_timing_v1` records source stream/application sequence, backend,
read start/end, receipt and publication, audio estimate, signed deltas, configured
guard and explicit unavailable fields. Application sequence is not device sequence.
Publication is after transform, before ring insertion. Raw POS_MSEC is retained
without assuming V4L2 clock/exposure flags. **CAMERA RECEIVE TIME != PROVEN
EXPOSURE TIME; pre_cutoff_verified != causal_pre_certified.**

Read-only audit (no hardware, no candidate data, no capture mutation):

```bash
python3 -m automation.physical_gt_audit \
  --capture /data/skjutbana/evaluation_runs/physical_gt_capture_v1_20261013_142407
```

It reports all three old PREs UNQUALIFIED, and availability/SHA256 for the nine
neutral images. Optional `--output NEW_JSON_PATH` writes outside the source capture
under `/data/skjutbana/`; existing output files are refused.

The wait schedule is at least3.3s per event plus camera, audio stop/start and write
cost. In the software benchmark, six3840×2160 BGR images (~149MB) drained in0.81s;
copy6–23ms/image and writer126–160ms/image occurred after live closure. Actual
capture times and drops are recorded, not estimated as detector latency.

Review later with the existing UI (keep labels outside original capture):

```bash
python3 -m automation.physical_trace_label --gt-review \
  --capture "$GT_RUN" \
  --output "${GT_RUN}_labels/exact_camera_ground_truth.json"
```

`4` PRE, `3` review/POST, up/down cycles LIVE_POST and neutral frames; the title
shows the actual role and WHITE/BLACK/MID_GRAY. `B` blink, `+/-` zoom, right-drag
pan, `0` fit; original pixels only. Inspect PRE, return to a GT_SETTLED image with
blink off, click the independently visible NEW hole, Enter to save. Final clicks
on LIVE_POST are refused in this mode. `U/S` UNKNOWN; `N` no visible hole also means
UNKNOWN, never NO_PHYSICAL_SHOT. Missing PRE blocks the normal comparison workflow.
No candidates, support, selected XY, boxes, target suggestions or heatmaps are
loaded/shown; `--show-candidates` is refused. Full-frame origin is(0,0): inverse
display scale/pan produces canonical camera XY directly, without recalibration.

If the human has independent physical knowledge identifying the hole in a settled
frame, `--independent-hole-observation "actual human observation and basis"` permits
that original GT_SETTLED click without PRE. Supply a real observation, not aim or
detector output. This explicit note applies to labels saved in that invocation.
`review_truth_basis` distinguishes this from PRE/POST inspection and records
`causal_pre_certified=false` in both cases. Neither workflow proves hole birth from
an unqualified PRE. If identification is uncertain, save UNKNOWN instead.

Labels use `physical_gt_review_exact_camera_v1`, extending the existing manual
camera_x/y convention with immutable image/metadata hashes, inspected PRE,
GT_SETTLED role, projector state, click transform and revision history. Optional
read-only contact sheets use `--prepare-only --review-package NEW_DIRECTORY`.
After independently defensible labels, existing `object_candidate_reconciliation
--paired --capture "$GT_RUN" --exact-camera-gt "${GT_RUN}_labels/exact_camera_ground_truth.json"
--output NEW_ANALYSIS_DIRECTORY` accepts them; missing pools/GT remain UNAVAILABLE.

**Future observability acceptance, when the operator is available:** only three
shots; verify the new hole is absent from the guarded PRE and visible in at least
one neutral frame for each. Label camera XY only if unambiguous. Otherwise retain UNKNOWN and report which
information is missing. This is not detector accuracy acceptance. Share runtime.log,
play.json, event_notes.json, gt_review config/manifest/timings and relevant complete
NPY+JSON pairs when troubleshooting. No physical test has been performed by Codex.

## Independent exact camera-XY labels from paired crops — 2026-10-12

Historical adapter status: later operator review found this specific series
unlabelable. All its physical labels remain UNKNOWN; do not force labels from
candidate suggestions. The adapter remains available for other suitable crops. Reuse the
existing `physical_trace_label` UI in its strict crop mode:

```bash
python3 -m automation.physical_trace_label \
  --paired-object-crops \
  --capture /data/skjutbana/evaluation_runs/can_play_candidate_compare_20261011_203905 \
  --output /data/skjutbana/evaluation_runs/paired_manual_label_prep_20261012_152909/review/exact_camera_ground_truth.json
```

Event1–4 have two PRE and four POST timestamps, event5 two PRE/three POST;
event6 is IMAGE DATA UNAVAILABLE. Six spatial tiles exist for each recorded event.
Last POST on2/4 contains only4/3 of those tiles. The UI lists only saved frames for
the selected tile. Saved grayscale pixels are already canonical camera oriented
(recorded rotation180 applied); no current calibration, rotation or guessed scale
is used. PRE means before audio, **not certification of a clean pre-impact board**.
If a new physical hole cannot be distinguished confidently, leave UNKNOWN.

Controls:
- Left/right: event; brackets `[ ]`: spatial tile; up/down: saved POST frame.
- `4`: original PRE; comma/period: PRE frame; `3`: original POST; `B`: blink.
- `+/-`: zoom; right mouse drag: pan; `0`: fit. Nearest-neighbor display preserves
  grayscale values; no per-image contrast stretch, registration or smoothing.
- Inspect original PRE at the same tile, switch to `3` POST with blink off, click
  the independently visible NEW hole. The green cross is only your pending click.
  `R` cancels/retries. `Enter` explicitly saves; navigate back to revise later.
- `U` or `S`: UNKNOWN / NOT_LABELABLE. `N`: NO_VISIBLE_HOLE, still UNKNOWN—not
  NO_PHYSICAL_SHOT. `Q`: exit; already saved revisions persist. No save means no GT.

Candidate/selected/support/heatmap/target overlays are disabled, including keyboard
shortcuts; `--show-candidates` is refused. No candidate sidecars are opened by the
label adapter. Only original POST can supply a final click. Object-level results,
intended aim and discharge truth remain separate; XY never invents them.

The separate file uses `paired_exact_camera_ground_truth_v1`, retaining existing
`space=camera`, `camera_x/y`, `quality=precise` conventions plus `camera_xy` and
crop-local/display coordinates. Each exact row binds capture play hash, shot/event
ID, trace hash, source NPY file and pixel hashes, bbox/origin/size, full camera
size, timestamp/acquisition identity, inspected PRE entries, manual method and
`candidate_overlay=false`. A file contains explicit UNKNOWN for other events.
Atomic validated replacement preserves every previous row in revision history.
Existing labels from another capture/schema are refused rather than overwritten.
Original capture and original operator truth are never written.

Read-only review aids (not labels): `/data/skjutbana/evaluation_runs/paired_manual_label_prep_20261012_152909/review/` contains
manifest.json, truth_status.json (preparation snapshot), per-event frames.json and
candidate-free tile contact sheets. Tiles at the same timestamp/acquisition are
one camera observation, not independent frames. No missing full-board pixels are
reconstructed. A new package can be generated headlessly with the same command
using `--prepare-only --review-package /data/skjutbana/evaluation_runs/NEW_REVIEW`
instead of `--output`.

After human labeling, use a NEW output directory:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m automation.object_candidate_reconciliation \
  --paired \
  --capture /data/skjutbana/evaluation_runs/can_play_candidate_compare_20261011_203905 \
  --exact-camera-gt /data/skjutbana/evaluation_runs/paired_manual_label_prep_20261012_152909/review/exact_camera_ground_truth.json \
  --output /data/skjutbana/evaluation_runs/object_paired_manual_oracle_01
```

Invalid/incomplete exact labels fail closed. With only UNKNOWN, oracle remains
UNAVAILABLE and no oracle files are emitted. Valid independent labels enable
candidate_nearest_gt.csv and oracle_summary.json: saved local retained/supported
pools and each complete mature boundary packet. Missing raw/intermediate pools
remain UNAVAILABLE. Nearest candidate order is native order, not a shared score
rank. First/last saved mature boundaries are summarized separately, never merged
into a retrospective best pool. CURRENT emitted-coordinate metrics are separate
from candidate oracle and from OBJECT_HIT truth. This is offline measurement only.


## Paired saved-pool analysis — 2026-10-12

Analyze the verified paired capture without launching hardware:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m automation.object_candidate_reconciliation \
  --paired \
  --capture /data/skjutbana/evaluation_runs/can_play_candidate_compare_20261011_203905 \
  --output /data/skjutbana/evaluation_runs/object_paired_review_02
```

Output must be new. It contains shot/stage/timing CSVs, candidate JSONL with original
XY and source references, frozen-shape membership, input/source hashes, truth status
and `operator_ground_truth.worksheet.json` using the existing `shots` schema.
Optional `--operator-truth /path/to/separate/operator_ground_truth.json` reads
independent observations without writing into the original capture. A filled
worksheet remains separate and must retain its capture hash and explicit shot IDs.
Do not bind planned shot ordinals to audio events automatically.

For this run the operator explicitly reports no independent physical observations:
all six events stay UNKNOWN. Do not fill labels from aim, game reactions or candidate
locations. TARGET_MISS means a real board impact outside the intended object;
NO_PHYSICAL_SHOT means no discharge/impact and is never inferred from UNKNOWN.
All packet counts are version/stage counts, not hit accuracy. A missing stage stays
UNAVAILABLE, including event6 whose export calls did not create candidate files.

The original full-frame mode cannot display `object_evidence_crops` and can write
unresolved statuses into its source. Use the strict paired mode described above
with separate output. It reuses the same UI with native crop origins and source
hashes; it never zero-fills a fake full frame or uses current calibration.
F1/F2 scene framepacks cannot retroactively restore unsaved images or provide
independent physical GT. No exact-camera oracle is emitted without valid labels.

## Operator console and complete diagnostic logs — 2026-10-12

`physical_collection play-can` now defaults to startup instructions, one compact
line per physical/audio event, explicit warnings/errors, and a final summary:

```text
[SHOT 3] OBJECT_HIT=UNKNOWN supports=4 mature=?/? exact=pending decision=337ms
```

This line is printed when the object decision arrives. `?` means unavailable at
that time, not zero; mature pool delivery and ordinary exact-coordinate results
can arrive later. Their updates are saved without printing another shot line.
Events interrupted before a decision are reported as UNAVAILABLE at shutdown.
Mouse/debug activity is not assigned a physical shot ID.

Files under the chosen capture directory:

- `runtime.log`: complete original child stdout/stderr bytes, including all
  OBJECT-HIT/CURRENT, mature-candidate, detector, resolver and latency details.
- `logs/event_summary.jsonl`: append-only shot/status/timing updates and warnings,
  with source, explicit shot IDs, original timing where available, parent receipt
  time (not an evidence clock), and byte ranges pointing into `runtime.log`.
- `runtime_exit.json`: exit/interruption status, log-drain completeness, counts,
  logging cost and logging failures. Ctrl+C keeps draining the child's final output.
- `play.json` and `event_notes.json`: capture configuration and independent operator
  observations. Existing `object_crops/shots/shot_XXXXXXXX/trace.json`, crop files
  and optional `mature_candidates/pool_NNNN.json` remain the canonical artifacts;
  the summary does not duplicate their payloads.

Missing/incomplete requested traces and truncated mature exports produce visible
warnings. Raw legacy lines without a shot ID stay unassigned; interleaved output
is retained and flagged where the summary cannot parse it reliably.

For detailed terminal debugging only, opt in:

```bash
python3 -m automation.physical_collection play-can \
  --object-crops --mature-candidates --verbose-console \
  --output-dir /data/skjutbana/evaluation_runs/can_play_verbose_01
```

Use a new output directory for every run. Omit `--verbose-console` for normal
operator use. Other application launches retain their existing console logging.
For troubleshooting share `runtime.log`, `logs/event_summary.jsonl`, `play.json`,
`runtime_exit.json`, independent event notes/operator truth, and the relevant shot
directories including trace/crops/mature sidecars when enabled. Existing replay
and candidate reconciliation commands continue consuming their original files.

## Compare object-local and mature native candidates — 2026-10-11

Analyze preserved captures without camera access, model fitting or regeneration:

```bash
python3 -m automation.object_candidate_reconciliation \
  --capture /data/skjutbana/evaluation_runs/can_play_03 \
  --capture /data/skjutbana/evaluation_runs/can_play_04 \
  --historical-cache /data/skjutbana/evaluation_runs/D01_counterfactual_20260910_152433 \
  --output /data/skjutbana/evaluation_runs/object_candidate_review_01
```

Output must be new. This comparison takes seconds, not an overnight training run.
It reads named inputs only and records hashes. Missing mature pools are UNAVAILABLE,
not zero candidates. Historical stage-oracle numbers are separate from original
CURRENT selected results. Object GT cannot establish exact hole-coordinate error.

The next short capture is justified only to fill the missing native-pool/delivery
ledger. It does not install a new object selector:

```bash
python3 -m automation.physical_collection play-can \
  --object-crops --mature-candidates \
  --output-dir /data/skjutbana/evaluation_runs/can_play_candidate_compare_01
```

Use the existing Skjutprylar game. Independently record the actual impacted object
for each physical shot and any verified no-discharge audio event; planned aim is
not truth. Do not infer hole XY from object labels or game reactions. After the
run, preserve operator observations in `operator_ground_truth.json` using the
existing can_play_03/04 schema (`shots`, `shot_id`, `physical_truth`, `object_id`),
only when independently verified. Without that file the comparator reports UNKNOWN
truth and still measures support/timing. Keep the original event notes.

The flag adds `mature_candidates/pool_NNNN.json` under each saved object-crop shot,
including late worker/confirmation deliveries after OBJECT_HIT closes. Limits are
32 pools/event, 1024 candidates/pool, 2 MiB/pool and 8 MiB/event; truncation/queue
loss invalidates unique shadow qualification. No full frames or settings changes.
The runtime log includes capture copy/enqueue cost. Default playback has no export.
Mouse/debug tests remain separate from physical camera shots.

can_play_03/04 supersede the proposed retest below: eight real shots had the correct
object among several supported objects, so all remained UNKNOWN. can_play_04:4 is a
verified no-discharge control with no crop inputs; do not fabricate its replay.

## Analyze saved OBJECT_HIT crops locally — 2026-10-08

`can_play_02` is preserved. Replay its saved query inputs without camera/projector
access or current calibration:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m automation.object_hit_replay \
  --capture /data/skjutbana/evaluation_runs/can_play_02 \
  --output /data/skjutbana/evaluation_runs/can_play_02_analysis_01 \
  --repeats 20 --profile --compare-scalar
```

Output must be new. After interruption, repeat the identical command with
`--resume`; completed shots are reused and changed inputs/code/options are refused.
`REPORT.md`, `summary.json`, `objects.csv`/JSON, `candidate_recurrence.json` and
per-shot checkpoints contain decisions, reasons, diagnostics and CPU measurements.
For `can_play_01`, use its path and a fresh output: compact logs are analyzed without
inventing image replay. No physical labels are inferred or edited.

Evidence replay neutralizes CPU deadlines, so its terminal result can differ from a
live timeout. Saved frame timestamps/cutoffs remain enforced. Known-hole context and
original scheduling are unavailable. All recorded frame ledgers are checked; missing
crops or mapping fail closed. Timing measures CPU with observer copies, excluding
loading/camera waits. Recurrence is diagnostic, never a re-hit veto.

The next `can_play_03` test checks post-batching latency and independently observed
object decisions. It does not validate a new support rule. Record actual fresh damage,
object/outside/uncertain and shot_id after every shot; aim and game response are not GT.
Two PRE observations in `can_play_02` cannot certify a clean baseline. A later rule
experiment needs longer timestamped PRE crop history across nuisance variation plus
independent placement observations. The current flag saves actual query inputs only;
it does not yet capture that longer diagnostic history.

## Play today: existing Skjutprylar OBJECT_HIT test — 2026-10-08

This is normal gameplay with compact diagnostics, not the old full research
capture/preflight workflow. Use the existing calibrated camera, projection and
audio setup. No new game, model, labels or saved settings are installed.

From the repository root:

```bash
python3 -m automation.physical_collection play-can \
  --object-crops \
  --output-dir /data/skjutbana/evaluation_runs/can_play_03
```

Choose **Spel > Skjutprylar** (the existing `shoot_stuff_range` entry), at **20 m**.
The launcher starts the normal menu, mutes can SFX for the initial controls, turns
off comparison visual hold and disables full-frame physical traces **only in this
process**. The output directory must be new; choose a new suffix if03 exists.
Never reuse an old capture directory. Existing `launch-can` still means a full
research capture with its own foundation preflight; it is not this command.

Before shooting, check projected alignment with the usual setup. If visibly wrong,
use the existing calibration menu. Wait until the HUD says **SKJUT!** and the image
has settled for at least one second before each shot. The old late-PRE limitation
is not fixed by OBJECT_HIT; this test measures practical behavior rather than
asserting that the physical reference is already validated.

1. **Player 1, shots 1–5:** one clear physical hit inside each of MINIBURK,
   LÄSKBURK, GULDBURK, BÖNBURK XXL and SARDINER, in that order. The sardine tin is
   the fifth can; leave the duck and toilet roll alone.
2. Wait for **Player 2** and the restored rack. **Shots 6–10:** repeat those five
   objects, one hit each. Use a visibly different spot within each object rather
   than assuming another projectile through the same hole creates new damage.
3. After the match, press **R**, wait for SKJUT! and settled projection.
   **Shots 11–15:** deliberately hit paper between/outside all active objects:
   left of the rack, between mini/soda, between roll/gold, below the shelf, right
   of the rack. Stay inside the calibrated shooting surface. Avoid all seven
   active shapes, including the duck and roll. If a false hit removes an object,
   reset with R and wait again before the next control.

For every shot, independently observe the physical placement; intended aim is
not truth. Fill `event_notes.json` with the corresponding console `shot_id`, actual
object (or outside), and result. Unknown placement is marked uncertain, not guessed.
Unexpected audio-triggered events get separate notes rather than silently shifting
shot numbers. **After each shot, independently identify the new physical damage
and record the observed object or outside.** If placement cannot be established,
write `uncertain` in notes; never copy intended_object into observed truth. Record
wrong/no reaction too. Preserve the original10-UNKNOWN run as the baseline.

| Human-observed physical placement | Record |
| --- | --- |
| Inside intended active object | correct_object / wrong_object / no_object_detected |
| Outside all active objects | correctly_no_object / false_object_hit |

Grade object identity, **not camera @5 accuracy**. Diagnostic XY is retained.
The console/runtime.log has immediate `[OBJECT-HIT]` records and later
`[OBJECT-HIT-CURRENT]` comparisons with ordinary exact-coordinate output. Fields
include frozen regions/shapes, per-object best candidate/evidence, UNKNOWN reasons,
timing, and `global_rescue_used_for_object=false`. Ordinary global rescue can still
run independently and retains its existing shot-id runtime messages. The optional
`--object-crops` now records only actual query crops (up to3 PRE/8 POST,20 events;
11M uint8 pixels/event maximum) in `object_crops/shots/shot_*/object_inputs/`.
Each `trace.json` contains frame timestamps, crop origins/hashes, exact query ledger
and persistence completeness/errors. No full camera frames are saved. Omit the flag
for ordinary compact-only gameplay. `runtime_exit.json` records process exit;
`play.json` records source hashes and the settings hash.

For this retest inspect per-object `event_freshness`: new change beyond observed
PRE, overlap with PRE variation, insufficient history, missing previous POST
newness, or polarity reversal. Non-support remains physical UNKNOWN, not validated
NO_IMPACT. Logs preserve old persistence separately, all retained alternatives,
geometry membership, final ambiguity and timing. If all objects still pass, stop:
use these crops to distinguish projection/registration effects before another change.

Initial practical target is at least95% correct object identity and zero false
object hits on outside shots. With only10 target trials that means10/10 for95%;
**9/10 targets and5/5 clean controls is PLAYABLE for this initial check**, not a
general95% validation claim. Preserve a playable result and play. If poor, inspect
the observed failures and compact evidence before changing mechanics. Do not start
another selector/locality sweep. UNKNOWN is an abstention, not proof of NO_IMPACT.

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
