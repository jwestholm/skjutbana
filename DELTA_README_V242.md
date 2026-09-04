# V2.24.2 — Game Context Testscene delta

Base: tested V2.24.1 / V2.24.0 dev checkpoint.

## What changes

- Adds `content/games/hit_context_test_v242.py`.
- Adds an idempotent menu installer for `content/menu.json`.
- Adds shot-time frozen-region visual diagnostics and console logging.
- Covers target, no-shoot, moving, overlap, edge, outside-region and EMPTY
  global-fallback cases.
- Updates GAME_DEVELOPMENT, ROADMAP and the main project MD files through the
  append-only documentation patch.
- Changes **no detector authority** and introduces **no V2.25 object engine**.

## Install / prepare

Unzip over the current dev checkout, then run:

```bash
python3 -m automation.v242_prepare
python3 -m automation.v242_selftest
python3 -m automation.v242_verify_install
python3 -m automation.v242_status
python3 main.py
```

The prepare step patches `content/menu.json` in place. This is intentional: the
full menu file belongs to the user's current checkout and is not reconstructed
inside the delta.

Open **Spel -> Hit Context Test (V2.24.2)** and follow `V242_TEST_PLAN.md`.
