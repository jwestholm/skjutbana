"""Train/evaluate a small numpy logistic model that scores dense candidate peaks (offline research only).

Positives: projected-hole peaks within 3 game px of the drawn hole.  Negatives: peaks > 6 px away.
Shot-level evaluation: take the best-scoring candidate of the shot; accept if p >= tau; outcome by exact
shape: correct (inside truth object), wrong (inside another object), false (truth = miss, inside any object).
Tau is chosen on DEV only (largest recall with 0 wrong/false on DEV, then a safety margin); VAL and REAL are
reported, never used for fitting or tau.

    python3 -m automation.can_peak_model DIR      (expects DIR/dev.npz, val.npz, real.npz + *_meta.jsonl)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AMBIG_P = 0.85   # chosen on DEV out-of-fold (0.5:300, 0.7:323, 0.85:335, 0.95:338 of 359; 0 wrong)


def load(d, name):
    z = np.load(d / f"{name}.npz")
    meta = [json.loads(l) for l in open(d / f"{name}_meta.jsonl")]
    return z["X"].astype(np.float64), meta, list(z["features"])


def fit_logreg(X, y, l2=1.0, iters=3000, lr=0.1):
    w = np.zeros(X.shape[1]); b = 0.0
    pos = y.mean()
    sw = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))   # class balance
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def outcome(m):
    return m["inside"][0] if len(m["inside"]) == 1 else (None if not m["inside"] else "AMBIGUOUS_SHAPE")


def shot_eval(meta, p, tau, manifest):
    shots = defaultdict(list)
    for m, pi in zip(meta, p):
        shots[(m["session"], m["sid"])].append((pi, m))
    res = dict(on=0, correct=0, wrong=0, misses=0, false=0, ambiguous=0)
    for sh in manifest:                       # every attempt counts, also shots without candidates
        truth = sh["truth"]
        res["on" if truth else "misses"] += 1
        rows = sorted(shots.get((sh["session"], sh["sid"]), []), key=lambda r: -r[0])
        if not rows or rows[0][0] < tau:
            continue
        best = rows[0]
        # object-level ambiguity guard: strong candidates supporting different outcomes -> abstain
        outs = {outcome(r[1]) for r in rows if r[0] >= AMBIG_P}
        if len(outs) > 1 or "AMBIGUOUS_SHAPE" in outs:
            res["ambiguous"] += 1
            continue
        if not best[1]["inside"]:
            continue
        obj = best[1]["inside"][0] if len(best[1]["inside"]) == 1 else None
        if obj is None:
            continue
        if truth is None:
            res["false"] += 1
        elif obj == truth:
            res["correct"] += 1
        else:
            res["wrong"] += 1
    return res


def labels(meta):
    return np.array([1 if m["dist"] is not None and m["dist"] <= 3 else (0 if m["dist"] is None or m["dist"] > 6 else -1)
                     for m in meta])


def train(X, meta):
    lab = labels(meta)
    keep = lab >= 0
    mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-6
    w, b = fit_logreg((X[keep] - mu) / sd, lab[keep].astype(float))
    return lambda Y: 1 / (1 + np.exp(-np.clip(((Y - mu) / sd) @ w + b, -30, 30))), (w, b, mu, sd)


def main():
    d = Path(sys.argv[1])
    Xd, md, feats = load(d, "dev")
    Xv, mv, _ = load(d, "val")
    Xr, mr, _ = load(d, "real")
    man = {n: [json.loads(l) for l in open(d / f"{n}_shots.jsonl")] for n in ("dev", "val", "real")}
    # out-of-fold (leave-one-session-out) predictions on DEV for tau selection
    sessions = sorted({m["session"] for m in md})
    p_oof = np.zeros(len(md))
    for s in sessions:
        tr = np.array([m["session"] != s for m in md])
        P, _ = train(Xd[tr], [m for m, t in zip(md, tr) if t])
        p_oof[~tr] = P(Xd[~tr])
    taus = np.linspace(0.5, 0.999, 200)
    safe = [t for t in taus if (lambda r: r["wrong"] == 0 and r["false"] == 0)(shot_eval(md, p_oof, t, man["dev"]))]
    tau0 = min(safe) if safe else 0.999
    tau = min(0.999, tau0 + 0.5 * (1 - tau0))          # margin towards 1 (no statistical guarantee)
    P, (w, b, mu, sd) = train(Xd, md)
    out = dict(features=feats, w=w.tolist(), b=b, mu=mu.tolist(), sd=sd.tolist(), tau_dev_min=tau0, tau=tau, ambig_p=AMBIG_P,
               dev_oof=shot_eval(md, p_oof, tau, man["dev"]), val=shot_eval(mv, P(Xv), tau, man["val"]),
               real_dev_only=shot_eval(mr, P(Xr), tau, man["real"]))
    order = np.argsort(-np.abs(w))
    out["top_weights"] = [(feats[i], round(float(w[i]), 3)) for i in order[:10]]
    (d / "model.json").write_text(json.dumps(out, indent=1))
    for k in ("tau_dev_min", "tau", "dev_oof", "val", "real_dev_only", "top_weights"):
        print(k, out[k])


if __name__ == "__main__":
    main()
