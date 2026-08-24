# Detector / AI V2.8 — recall-preserving hypothesis pool

V2.8 follows the first successful V2.7 hypothesis benchmark. That test showed:

- filtered GT <=42 px: 62/99;
- micro-clusters retained 60/99;
- the 120-item final pool retained only 44/99;
- clustering itself lost only 2 GT cases, while spatial pooling lost 16.

Therefore V2.8 deliberately freezes the detector/Shot Vault and micro-cluster
logic and changes the post-cluster pool.

## Architecture

```text
V2.6 detector / Shot Vault
        |
filtered observations
        |
V2.7 micro-clustering (unchanged)
        |
        +--> V2.7-style core pool (120)
        |
        +--> V2.8 recall pool (up to 220)
                  |
                  +--> core-first baseline (actual until gate opens)
                  +--> full-baseline shadow
                  +--> fresh V6 shadow / training
```

If a shot has <=220 micro-clusters, V2.8 keeps all of them. For larger pools,
selection uses reserved baseline/support/signal/diversity/Vault heads plus
spatial local winners.

The actual baseline is intentionally conservative: the old 120 core is ordered
first. This means the V2.8 recall experiment cannot lower the old baseline Top-1
merely by adding rescued candidates. V6 can nevertheless see and learn from the
full recall pool, and it may take authority only through the existing validation
gate.

## Diagnostics

Authoritative V2.8 diagnostics are per-shot atomic JSON files:

```text
content/ai/detector_v28/sessions/<runtime_session>/shot_000001.json
...
```

A JSONL mirror is also kept at:

```text
content/ai/detector_v28/shot_diagnostics.jsonl
```

Use:

```bash
python3 -m automation.detector_v28_verify
python3 -m automation.detector_v28_analyze
```

The V2.8 analyzer compares filtered input, all clusters, old 120 core and the
new recall pool at 10/20/42 px, plus core-first baseline, full-baseline shadow,
V6 shadow and actual selection.
