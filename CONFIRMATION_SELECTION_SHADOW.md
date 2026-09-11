# Confirmation selection shadow

`CONFIRMATION_SELECTION_SHADOW` is the frozen development replay selector
accepted from the 10-shot session. Its exact formula is:

```
v2225_confirm_center_abs + v2225_confirm_darkening - v2225_confirm_compact
```

The module records a configuration SHA-256 in every trace and evaluates only
local-confirmation candidates. It never changes the scanner's deterministic
selection or emitted coordinates. The status is `PHYSICAL_REPLAY_CHALLENGER`;
the existing 10-shot session is development evidence, not validation.

Each new trace contains independent `selectors` entries for:

- `CURRENT_DETERMINISTIC`
- `CONFIRMATION_SELECTION_SHADOW`
- `CANONICAL_AI_SHADOW`

Run the next independent validation session with:

```bash
python3 -m automation.physical_test start
# shoot exactly the planned number of shots
python3 -m automation.physical_test check
python3 -m automation.physical_test label
python3 -m automation.physical_test evaluate
```

The evaluation writes selector metrics, oracle candidate availability, and
per-shot distances to a new `evaluation_runs/physical_*` directory. Do not
change the selector formula or weights after labels from that session are
available; report the frozen results first.
