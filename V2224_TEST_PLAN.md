# V2.22.4 physical test plan

## 0. Install verification

From repository root:

```bash
python3 -m automation.v2224_selftest
python3 -m automation.v2224_verify_install
python3 -m automation.v2224_apply_docs
```

Then run the older regression selftests that are present in the checkout:

```bash
python3 -m automation.v2223_selftest
python3 -m automation.v2222_selftest
python3 -m automation.v2221_selftest
python3 -m automation.runtime_v222_selftest
```

## 1. Startup only

Start `python3 main.py` and verify:

```text
[V2.22.1] ... installed
[V2.22.2] ... installed
[V2.22.3] ... installed
[V2.22.4] async CV + off-critical AI shadow pipeline installed
```

Stop if any relevant patch says `unavailable`.

## 2. Three deliberate physical shots

Use AI Training in `advisory`, no F2.

Shoot one ordinary hole and wait for the visible result.  Repeat twice.  The important new logs are:

```text
[V2.22.3 AUDIO-PRIORITY] ... main_ack=...
[V2.22.4 CV] shot=... queue=... worker=... v2=... other=... cand=... pre=... ref=... reg=... extract=... bank=... merge=...
[SHOT #...] HIT ...
[V2.22.4 AI-SHADOW] shot=... queue=... compute=...
[V2.22.3 VISIBLE] ...
```

Expected behavioural change: while `[V2.22.4 CV]` is still pending, the mouse/window/rendering should not freeze for the whole CV compute time.  In advisory mode the visible HIT must not wait for `[V2.22.4 AI-SHADOW]` to finish.

## 3. Two moderately close shots — concurrency test

Only after the first three are correct, fire two shots about 0.7–1.0 s apart at clearly different positions.  This is intentionally not yet a rapid-fire gameplay test.

Expected:

- two audio peaks and two distinct shot ids;
- shot 2 `main_ack` should stay low even if shot 1 CV is running;
- CV worker may show queue delay because it is intentionally single-owner;
- no candidate/state leakage between shot ids;
- both final coordinates must correspond to their physical holes.

If shot 2 audio is acknowledged quickly but its CV queue grows, the next optimisation is detector throughput / fast-rescue architecture rather than main-loop priority.

## 4. What to send back

Send the terminal block containing:

- all `[V2.22.3 AUDIO-PRIORITY]` rows;
- all `[V2.22.4 CV]` rows;
- `[SHOT #] HIT/MISS`;
- `[V2.22.4 AI-SHADOW]`;
- `[V2.22.3 LATENCY]` and `VISIBLE`.

Also note whether the spinner/window stayed responsive and whether each marker landed on the correct physical hole.

Do not run F2/night training yet.
