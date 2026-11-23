"""Model tournament + cooperation for dense candidate peaks (offline research).

Each model type (logreg, mlp, stumps) is scored on DEV out-of-fold, VAL and REAL by can_peak_model.py
(preds_<kind>.npz).  Here they compete (each alone) and cooperate:
  mean  - average probability
  min   - consensus: a candidate is only as strong as the most sceptical model
  vote2 - at least two of three models above their own tau
Tau for every combination is chosen on DEV OOF only (0 wrong / 0 false, plus margin), as for single models.

    python3 -m automation.can_peak_ensemble DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from automation.can_peak_model import load, shot_eval


def pick_tau(meta, p, manifest):
    taus = np.linspace(0.5, 0.999, 200)
    safe = [t for t in taus if (lambda r: r["wrong"] == 0 and r["false"] == 0)(shot_eval(meta, p, t, manifest))]
    t0 = min(safe) if safe else 0.999
    return min(0.999, t0 + 0.5 * (1 - t0))


def main():
    d = Path(sys.argv[1])
    _, md, _ = load(d, "dev"); _, mv, _ = load(d, "val"); _, mr, _ = load(d, "real")
    man = {n: [json.loads(l) for l in open(d / f"{n}_shots.jsonl")] for n in ("dev", "val", "real")}
    kinds = [k for k in ("logreg", "mlp", "stumps") if (d / f"preds_{k}.npz").exists()]
    P = {k: np.load(d / f"preds_{k}.npz") for k in kinds}
    combos = {k: {s: P[k][s] for s in ("oof", "val", "real")} for k in kinds}
    if len(kinds) >= 2:
        combos["mean"] = {s: np.mean([P[k][s] for k in kinds], 0) for s in ("oof", "val", "real")}
        combos["min"] = {s: np.min([P[k][s] for k in kinds], 0) for s in ("oof", "val", "real")}
    rows = {}
    for name, c in combos.items():
        tau = pick_tau(md, c["oof"], man["dev"])
        rows[name] = dict(tau=round(float(tau), 3), dev_oof=shot_eval(md, c["oof"], tau, man["dev"]),
                          val=shot_eval(mv, c["val"], tau, man["val"]), real=shot_eval(mr, c["real"], tau, man["real"]))
        r = rows[name]
        print(f"{name:7s} tau {r['tau']:.3f} | DEV {r['dev_oof']['correct']}/{r['dev_oof']['on']} w{r['dev_oof']['wrong']} f{r['dev_oof']['false']}"
              f" amb{r['dev_oof']['ambiguous']} | VAL {r['val']['correct']}/{r['val']['on']} w{r['val']['wrong']} f{r['val']['false']}"
              f" amb{r['val']['ambiguous']} | REAL {r['real']['correct']}/{r['real']['on']} w{r['real']['wrong']} f{r['real']['false']}")
    (d / "ensemble.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
