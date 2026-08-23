# V2.7.2 direct integration verification

V2.7.2 moves installation to the actual game entry point (`main.py`).

## Required startup proof

Start the game:

```bash
python3 main.py
```

The GAME terminal must print:

```text
[RANKER-V6] V2.7.2 DIRECT integration installed (... source=main.py)
```

Before any F2 test, run in another terminal:

```bash
python3 -m automation.detector_v27_verify
```

The verifier is read-only. It must report:

```text
PASS: V2.7.2 is installed in a live game process.
Install source: main.py
```

## First test only

```bash
python3 -m automation.ai_training_loop 1 1 --seed 12345
python3 -m automation.detector_v27_verify
python3 -m automation.detector_v27_analyze
```

After the 100-shot test, `rank_with_funnel calls`, `GT calls`, and
`Diagnostic rows` must all be greater than zero.

Do not run 10x100 until this path is proven.
