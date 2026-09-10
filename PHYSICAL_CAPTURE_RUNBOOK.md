# Physical capture runbook

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

The six-shot D01 development capture is complete. It contains no intentionally
collected no-impact events. See [D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md)
for the results and the smallest proposed D02 follow-up. Do not create D02 before
the D01 findings are reviewed; do not reuse or inspect S03.

Physical trace capture now retains upstream contour/filter evidence and cleanup
boundaries, with camera versus crop coordinate provenance. Missing full hybrid
RAW evidence remains UNAVAILABLE. A trace/frame PASS establishes artifact
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
