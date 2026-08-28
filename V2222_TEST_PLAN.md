# V2.22.2 test plan

Do not start an F2/night run yet. First verify the live fast path with a very
small number of physical shots.

## A. Software checks

From repository root:

```bash
python3 -m automation.v2222_selftest
python3 -m automation.v2221_selftest
python3 -m automation.offline_v222_selftest
python3 -m automation.runtime_v222_selftest
python3 -m automation.v2222_verify_install
```

All must pass.

## B. Start game and regression-check menus

```bash
python3 main.py
```

Expected startup includes:

```text
[V2.22.1] Perspective ROI/edge-guard HitScanner patch installed
[V2.22.2] cursor/novelty/ridge/backlog fast-path patch installed
```

Open AI Results once. It must still open without `Unknown item type: ai_results`.

Use AI mode `advisory`; do not enable AI authority.

## C. Cursor test before firing a group

Go to manual AI Training (do **not** press F2).

1. Move the mouse to a clearly visible position inside the projected playfield.
2. Stop moving it.
3. Fire one physical shot.
4. At the audio peak the cursor should disappear during detection.
5. After HIT/MISS resolves the cursor should reappear so the true impact can be
   clicked/labeled.

Report if the cursor stays visible during detection or fails to return.

## D. Five ordinary physical shots

Fire five shots at fresh locations, spread over the playfield. Confirm for each:

- hit marker appears at the correct physical location;
- perspective/viewport coordinates are still correct;
- previous outer-edge candidate rows remain gone;
- the long horizontal candidate bands are substantially reduced;
- old physical holes no longer dominate the top candidate display.

Copy terminal output as text if possible. The important lines are:

```text
[V2.22.1 ROI] ...
[V2.22.2 CLEAN] ...
[V2.22.2 FAST] ...
[SHOT #N] HIT: ... e2e=...ms
[V2.22 RESOLVER] ...
```

## E. Re-hit safety test

After the five ordinary shots, deliberately fire **two** shots very close to a
hole registered during this same run (only if safe/practical).

Purpose: prove that the old-hole cleanup does not make a genuine fresh re-hit
impossible. A fresh re-hit should survive because `pre_shot_change` is new even
though the location is near a known hole.

## F. What to send back

For each shot or as one pasted block, send:

- `candidates=A->B`
- `stale=` / `demoted=`
- `ridges=` / `ridge_removed=` / `fresh_saved=`
- `frames=N->K drop=`
- `detector=`, `cleanup=`, `overhead=`, `update=`
- HIT `e2e=`
- resolver `e2e=`
- whether the drawn marker matched the physical hole.

Also state whether:

1. cursor disappeared and returned correctly;
2. old holes moved down in the displayed ranking;
3. horizontal bands disappeared/reduced;
4. the two intentional re-hits were still detectable.

## G. Decision rule for V2.22.3

We choose the next optimization from measured timing rather than guessing:

- detector >500–700 ms => build coarse-to-fine image detection;
- overhead/backlog large => optimize ingestion/history;
- detector fast but event e2e slow => reduce/parallelize track confirmation;
- accuracy noise remains dominant => tune novelty/ridge thresholds using the
  captured physical evidence before any night training.
