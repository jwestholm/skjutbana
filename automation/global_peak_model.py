"""Universal (whole-viewport) candidate scorer vs the H15-REF-1 rule (offline research, SHADOW only).

Shot level: the best candidate of a shot is accepted when p >= tau and no other candidate further than 8 px
has p >= AMBIG; correct = within 8 camera px of truth, otherwise WRONG XY (the dangerous error).
H15-REF-1 on the same candidates: strongest s; accept if s>=16, s/s2>=1.15, core<=0.4, bipolar<=1.0.
Tau and AMBIG come from DEV leave-one-session-out only (0 wrong), VAL/TEST/REAL are reported.

    python3 -m automation.global_peak_model DIR [logreg|mlp]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from automation.can_peak_model import fit_logreg, fit_mlp

TOL = 8.0


def load(d, n):
    z = np.load(d / f"{n}.npz")
    meta = [json.loads(l) for l in open(d / f"{n}_meta.jsonl")]
    shots = [json.loads(l) for l in open(d / f"{n}_shots.jsonl")]
    return z["X"].astype(np.float64), meta, shots


def labels(meta):
    return np.array([1 if m["dist"] <= 5 else (0 if m["dist"] > 12 else -1) for m in meta])


def train(X, meta, kind):
    lab = labels(meta); keep = lab >= 0
    mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-6
    Z = (X[keep] - mu) / sd
    if kind == "mlp":
        W1, b1, W2, b2 = fit_mlp(Z, lab[keep].astype(float))
        f = lambda Y: 1 / (1 + np.exp(-np.clip(np.tanh(((Y - mu) / sd) @ W1 + b1) @ W2 + b2[0], -30, 30)))
        return f, dict(kind="mlp", W1=W1.tolist(), b1=b1.tolist(), W2=W2.tolist(), b2=float(b2[0]), mu=mu.tolist(), sd=sd.tolist())
    w, b = fit_logreg(Z, lab[keep].astype(float))
    return (lambda Y: 1 / (1 + np.exp(-np.clip(((Y - mu) / sd) @ w + b, -30, 30))),
            dict(kind="logreg", w=w.tolist(), b=b, mu=mu.tolist(), sd=sd.tolist()))


def group(meta, p=None):
    S = defaultdict(list)
    for i, m in enumerate(meta):
        S[(m["session"], m["sid"])].append((None if p is None else p[i], m, i))
    return S


def eval_model(meta, shots, p, tau, ambig, by=None):
    S = group(meta, p)
    res = defaultdict(lambda: dict(n=0, correct=0, wrong=0, ambiguous=0))
    for sh in shots:
        key = sh.get(by) if by else "all"
        r = res[key]; r["n"] += 1
        rows = sorted(S.get((sh["session"], sh["sid"]), []), key=lambda x: -x[0])
        if not rows or rows[0][0] < tau:
            continue
        b = rows[0][1]
        if any(x[0] >= ambig and math.hypot(x[1]["cx"] - b["cx"], x[1]["cy"] - b["cy"]) > TOL for x in rows[1:]):
            r["ambiguous"] += 1
            continue
        r["correct" if b["dist"] <= TOL else "wrong"] += 1
    return dict(res)


def eval_rule(X, meta, shots, feats, by=None):
    S = group(meta)
    fi = {f: feats.index(f) for f in ("s", "core", "bipolar")}
    res = defaultdict(lambda: dict(n=0, correct=0, wrong=0))
    for sh in shots:
        key = sh.get(by) if by else "all"
        r = res[key]; r["n"] += 1
        rows = sorted(S.get((sh["session"], sh["sid"]), []), key=lambda x: -x[1]["s"])
        if not rows:
            continue
        top = X[rows[0][2]]
        s2 = rows[1][1]["s"] if len(rows) > 1 else 0.0
        ok = top[fi["s"]] >= 16 and (s2 <= 0 or top[fi["s"]] / s2 >= 1.15) and top[fi["core"]] <= 0.4 and top[fi["bipolar"]] <= 1.0
        if ok:
            r["correct" if rows[0][1]["dist"] <= TOL else "wrong"] += 1
    return dict(res)


def main():
    d = Path(sys.argv[1]); kind = sys.argv[2] if len(sys.argv) > 2 else "logreg"
    feats = list(np.load(d / "dev.npz")["features"])
    Xd, md, sd_ = load(d, "dev")
    sessions = sorted({m["session"] for m in md})
    p_oof = np.zeros(len(md))
    for s in sessions:
        tr = np.array([m["session"] != s for m in md])
        P, _ = train(Xd[tr], [m for m, t in zip(md, tr) if t], kind)
        p_oof[~tr] = P(Xd[~tr])
    best = None
    for ambig in (0.5, 0.7, 0.85, 0.95):
        for tau in np.linspace(0.5, 0.999, 120):
            r = eval_model(md, sd_, p_oof, tau, ambig)["all"]
            if r["wrong"] == 0 and (best is None or r["correct"] > best[0]):
                best = (r["correct"], float(tau), ambig)
    _, tau0, ambig = best
    tau = min(0.999, tau0 + 0.5 * (1 - tau0))
    P, params = train(Xd, md, kind)
    out = dict(features=feats, **params, tau=tau, tau_dev_min=tau0, ambig_p=ambig, tol_px=TOL)
    out["dev_oof"] = eval_model(md, sd_, p_oof, tau, ambig)
    out["dev_rule"] = eval_rule(Xd, md, sd_, feats)
    for n in ("val", "test", "real"):
        if (d / f"{n}.npz").exists():
            X, m, sh = load(d, n)
            p = P(X)
            out[n] = eval_model(m, sh, p, tau, ambig)
            out[n + "_rule"] = eval_rule(X, m, sh, feats)
            out[n + "_by_bg"] = dict(model=eval_model(m, sh, p, tau, ambig, by="background" if n != "test" or sh[0].get("background") else "session"),
                                     rule=eval_rule(X, m, sh, feats, by="background" if n != "test" or sh[0].get("background") else "session"))
    (d / f"model_{kind}.json").write_text(json.dumps(out, indent=1))
    for k in ("tau", "ambig_p", "dev_oof", "dev_rule", "val", "val_rule", "test", "test_rule", "real", "real_rule"):
        print(k, out.get(k))


if __name__ == "__main__":
    main()
