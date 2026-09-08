# Next physical test

Close the application first. From the repository root:

```bash
python3 -m automation.physical_test start
```

This creates a unique session, validates the frozen SHADOW manifest, records the
old trace settings, enables capture without changing AI mode, and launches
`python3 -u main.py` with stdout/stderr in the session's `runtime.log`.
Select the target and shoot. Write down actual shot count and order.

```bash
python3 -m automation.physical_test check
python3 -m automation.physical_test label
```

`check` may report incomplete traces while a shot is still processing. It verifies
PRE/POST/map artifacts and producer completeness, and reports labels, effective
mode and audio/challenger diagnostics. Close the application after shooting so all
traces flush before final evaluation. Label from the physical hole evidence, not
from the detector overlay (candidate overlays default off).

If a runtime event has no physical counterpart, explicitly classify it. Never infer
that from a missing click. Example syntax, **not a prescribed next-session label**:

```bash
python3 -m automation.physical_test classify --shot-id 2 --no-physical-shot --reason "verified duplicate of first physical shot"
python3 -m automation.physical_test classify --shot-id 3 --label-shot-id 2 --reason "second labeled hole belongs to event 3"
```

The assignments are a sidecar; original click files remain unchanged. An explicit
label mapping must reference an existing label. Inspect events close in time and
image progression before changing ordinal correspondence.

```bash
python3 -m automation.physical_test evaluate
python3 -m automation.physical_test stop
```

Evaluation creates a new ignored run directory, exports/evaluates real and
unresolved events separately from explicitly nonphysical events, and saves
`physical_comparison.json` with deterministic/AI coordinates, ranks and confidence.
False events are in `false_events.json`, outside bullet-hit denominators. No
existing evaluation run is overwritten. `stop` restores only helper-owned trace
settings and refuses conflicts; close the launched app first.

Use `--root PATH` with check/label/evaluate for an older session. For the audited
biathlon session also pass
`--mapping evaluation_runs/overnight_20260907/biathlon_mapping.json` to evaluate.
`--challenger PATH` selects another explicitly frozen SHADOW manifest;
`--prepare-only` prepares settings without launching the app. Models/results are
local ignored artifacts; regenerate on another checkout before running start:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m automation.conditional_ranking --output evaluation_runs/overnight_20260907/ranking
```

Choose a new output directory if that path already exists, then pass its
`challenger.json` explicitly. These commands do not promote any model.

The present physical series supports diagnosis only. The next test should record
at least several deliberately separated shots with exact physical counts, include
a few rapid pairs, and retain `runtime.log`. Inspect whether the corrected async
track timestamp stays tied to its candidate frame, whether PRE already contains
the new hole, and whether the same old hole wins again. Do not tune audio thresholds
from a single extra trigger.
