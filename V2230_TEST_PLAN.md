# V2.23.0 test plan

## Phase A — install / software only

From repository root:

```bash
python3 -m automation.v2230_selftest
python3 -m automation.v2230_verify_install
python3 -m automation.v2230_audit
```

Expected:

- all selftests pass,
- V2.22.6 is still present in `main.py`,
- V2.23 reports `live authority: NO`,
- audit lists whichever old models/modules/data actually exist on the shooting PC,
- `content/ai/training_v223/reports/inventory_latest.json` is written.

The audit is deliberately dynamic because the shooting PC contains historical model/data artifacts that are not all versioned in Git.

## Phase B — current data, no projector required

Run:

```bash
python3 -m automation.v2230_status
python3 -m automation.v2230_train --quick
```

If legacy candidate packs can be imported, training should produce:

```text
content/ai/training_v223/models/challengers/<trial>/
    model.npz
    model.json
content/ai/training_v223/models/registry.json
content/ai/training_v223/models/champion.json   # when a first research champion is accepted
content/ai/training_v223/reports/latest.json
```

Important checks:

- audit must state how many legacy shot JSONs were loaded vs skipped,
- `protected_holdout_evaluated_for_selection` must be false,
- `eligible_for_live_authority` must be false,
- if split is provisional, treat metrics as engineering feedback only.

If legacy importer reports 0 usable shots despite V2.16 packs existing, send `inventory_latest.json` plus one representative `shot_*.json` (not NPZ yet). The parser is intentionally fail-open rather than guessing an undocumented old schema.

## Phase C — F2 integration

Start the game normally:

```bash
python3 main.py
```

Startup must include the existing V2.22 chain and then:

```text
[V2.23.0] unified training/model pipeline installed (capture + shadow champion/challenger; live authority unchanged)
```

Open AI Training and run a **small F2 test first**. The old F2 behavior should remain, plus:

```text
[V2.23 CAPTURE] session=... ready
...
[V2.23 AUTOTRAIN] F2 completed; captured=... scheduled=True
[V2.23 AUTOTRAIN] background challenger training started (shadow only)
```

After it completes:

```bash
python3 -m automation.v2230_status
```

Verify the new session contains shot JSON files and that the registry/champion is shadow-only.

## Phase D — manual physical GT

In AI Training, make one physical shot using the current audio workaround if necessary, then click the true hole as usual. Expected log:

```text
[V2.23 CAPTURE] physical GT saved candidates=N
```

The saved record must have `source_kind=physical_manual` and the clicked camera GT. This is the beginning of the genuinely valuable multi-session physical dataset.

## Phase E — unattended offline work

Only after A–D work:

```bash
python3 -m automation.v2230_autotrain --hours 1 --quick
```

Then:

```bash
python3 -m automation.v2230_status
```

For an overnight run later:

```bash
python3 -m automation.v2230_autotrain --hours 8
```

Do not interpret repeated improvement on a provisional one-session split as production accuracy. Collect independent physical sessions and freeze a protected holdout before any live-authority gate.
