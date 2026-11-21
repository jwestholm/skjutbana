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


def fit_mlp(X, y, hidden=16, iters=4000, lr=0.01, l2=1e-3, seed=0):
    """1-hidden-layer tanh MLP, Adam, class-balanced log loss (numpy only)."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), hidden); b2 = 0.0
    pos = y.mean(); sw = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    params = [W1, b1, W2, np.array([b2])]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    for t in range(1, iters + 1):
        H = np.tanh(X @ params[0] + params[1])
        z = H @ params[2] + params[3][0]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw / len(y)
        gW2 = H.T @ g + l2 * params[2]; gb2 = np.array([g.sum()])
        gH = np.outer(g, params[2]) * (1 - H ** 2)
        gW1 = X.T @ gH + l2 * params[0]; gb1 = gH.sum(0)
        for i, gr in enumerate((gW1, gb1, gW2, gb2)):
            m[i] = 0.9 * m[i] + 0.1 * gr; v[i] = 0.999 * v[i] + 0.001 * gr ** 2
            params[i] = params[i] - lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return params


def fit_stumps(X, y, rounds=150, lr=0.3, bins=16):
    """Gradient-boosted decision stumps (numpy, histogram split search): axis-aligned thresholds, a
    different inductive bias from the linear model and the MLP."""
    qs = [np.unique(np.quantile(X[:, j], np.linspace(0.03, 0.97, bins))) for j in range(X.shape[1])]
    B = [np.searchsorted(q, X[:, j], side="left") for j, q in enumerate(qs)]   # bin k: q[k-1] < x <= q[k]
    pos = y.mean(); sw = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    F = np.zeros(len(y)); trees = []
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-F)); g = (y - p) * sw; h = p * (1 - p) * sw + 1e-9
        G, H = g.sum(), h.sum(); best = None
        for j, q in enumerate(qs):
            gl = np.cumsum(np.bincount(B[j], g, len(q) + 1))[:len(q)]
            hl = np.cumsum(np.bincount(B[j], h, len(q) + 1))[:len(q)]
            gain = gl * gl / (hl + 1) + (G - gl) ** 2 / (H - hl + 1)
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (gain[k], j, float(q[k]), gl[k] / (hl[k] + 1), (G - gl[k]) / (H - hl[k] + 1))
        _, j, t, vl, vr = best
        trees.append((j, t, lr * vl, lr * vr))
        F += np.where(X[:, j] <= t, lr * vl, lr * vr)
    return trees


def stumps_predict(trees, X):
    F = np.zeros(len(X))
    for j, t, vl, vr in trees:
        F += np.where(X[:, j] <= t, vl, vr)
    return 1 / (1 + np.exp(-np.clip(F, -30, 30)))


def labels(meta):
    return np.array([1 if m["dist"] is not None and m["dist"] <= 3 else (0 if m["dist"] is None or m["dist"] > 6 else -1)
                     for m in meta])


KIND = "logreg"


def train(X, meta):
    lab = labels(meta)
    keep = lab >= 0
    mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-6
    if KIND == "stumps":
        trees = fit_stumps((X[keep] - mu) / sd, lab[keep].astype(float))
        return (lambda Y: stumps_predict(trees, (Y - mu) / sd),
                dict(kind="stumps", trees=[list(t) for t in trees], mu=mu.tolist(), sd=sd.tolist()))
    if KIND == "mlp":
        W1, b1, W2, b2 = fit_mlp((X[keep] - mu) / sd, lab[keep].astype(float))
        f = lambda Y: 1 / (1 + np.exp(-np.clip(np.tanh(((Y - mu) / sd) @ W1 + b1) @ W2 + b2[0], -30, 30)))
        return f, dict(kind="mlp", W1=W1.tolist(), b1=b1.tolist(), W2=W2.tolist(), b2=float(b2[0]), mu=mu.tolist(), sd=sd.tolist())
    w, b = fit_logreg((X[keep] - mu) / sd, lab[keep].astype(float))
    return (lambda Y: 1 / (1 + np.exp(-np.clip(((Y - mu) / sd) @ w + b, -30, 30))),
            dict(kind="logreg", w=w.tolist(), b=b, mu=mu.tolist(), sd=sd.tolist()))


def main():
    global KIND
    d = Path(sys.argv[1])
    KIND = sys.argv[2] if len(sys.argv) > 2 else "logreg"
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
    P, params = train(Xd, md)
    extra = {}
    if (d / "test.npz").exists():                       # untouched final test (reported, never used)
        Xt, mt, _ = load(d, "test")
        man["test"] = [json.loads(l) for l in open(d / "test_shots.jsonl")]
        extra["test"] = P(Xt)
    np.savez(d / f"preds_{params['kind']}.npz", oof=p_oof, val=P(Xv), real=P(Xr), **extra)
    out = dict(features=feats, **params, tau_dev_min=tau0, tau=tau, ambig_p=AMBIG_P,
               dev_oof=shot_eval(md, p_oof, tau, man["dev"]), val=shot_eval(mv, P(Xv), tau, man["val"]),
               real_dev_only=shot_eval(mr, P(Xr), tau, man["real"]),
               test=shot_eval(mt, extra["test"], tau, man["test"]) if "test" in extra else None)
    if params["kind"] == "logreg":
        w = np.array(params["w"]); order = np.argsort(-np.abs(w))
        out["top_weights"] = [(feats[i], round(float(w[i]), 3)) for i in order[:10]]
    (d / f"model_{params['kind']}.json").write_text(json.dumps(out, indent=1))
    if params["kind"] == "logreg":
        (d / "model.json").write_text(json.dumps(out, indent=1))
    for k in ("tau_dev_min", "tau", "dev_oof", "val", "test", "real_dev_only", "top_weights"):
        print(k, out.get(k))


if __name__ == "__main__":
    main()
