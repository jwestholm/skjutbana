"""Compare the dense rule, saved peak models and rule-OR-model cooperation on a candidate dataset.

Rule = the live dense rule applied to the dark candidates of the shot (s, core, bipolar; MIN_S 11, ratio 1.15
against the next hole-like peak, isolated strong peak s >= 18 / ratio 1.5).  Model = PeakModel with its own
tau + object-level ambiguity guard (as live shadow).  Union = rule OR model; when both decide on different
outcomes -> abstain.  Outcomes by exact shape: correct / wrong (other object) / false (truth = miss).

    python3 -m automation.can_peak_compare DATA_DIR SET MODEL.json [MODEL.json ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.engine.camera.dense_peak_model import FEATURES, PeakModel


def rule_pick(rows):
    dark = [m for f, m in rows if not m.get("bright")]
    dark.sort(key=lambda m: -m["s"])
    holey = [m for m in dark if m["core"] >= 0.15 and (m["core"] >= 0.4 or m["bipolar"] <= 1.8)]
    if holey:
        s2 = holey[1]["s"] if len(holey) > 1 else 0.0
        if holey[0]["s"] >= 11 and (s2 <= 0 or holey[0]["s"] / s2 >= 1.15):
            return holey[0]
    if dark:
        s2 = dark[1]["s"] if len(dark) > 1 else 0.0
        if dark[0]["s"] >= 18 and (s2 <= 0 or dark[0]["s"] / s2 >= 1.5):
            return dark[0]
    return None


def outcome_obj(m):
    if m is None or not m["inside"]:
        return None
    return m["inside"][0] if len(m["inside"]) == 1 else "AMBIG"


def model_pick(rows, model):
    rows = [(f, m) for f, m in rows if model.uses_bright or not m.get("bright")]
    if not rows:
        return None, False
    F = np.array([f if model.uses_bright else f[:len(FEATURES)] for f, _ in rows])
    p = model.p(F)
    order = np.argsort(-p)
    if p[order[0]] < model.tau:
        return None, False
    outs = {outcome_obj(rows[i][1]) for i in order if p[i] >= model.ambig_p}
    if len(outs) > 1 or "AMBIG" in outs:
        return None, True
    return rows[order[0]][1], False


def score(res, truth, obj):
    if obj is None or obj == "AMBIG":
        return
    if truth is None:
        res["false"] += 1
    elif obj == truth:
        res["correct"] += 1
    else:
        res["wrong"] += 1


def main():
    d, name, models = Path(sys.argv[1]), sys.argv[2], [PeakModel(p) for p in sys.argv[3:]]
    z = np.load(d / f"{name}.npz")
    X = z["X"]; feats = list(z["features"])
    meta = [json.loads(l) for l in open(d / f"{name}_meta.jsonl")]
    shots = [json.loads(l) for l in open(d / f"{name}_shots.jsonl")]
    ci, bi = feats.index("core"), feats.index("bipolar")
    by = defaultdict(list)
    for f, m in zip(X, meta):
        m = dict(m, core=float(f[ci]), bipolar=float(f[bi]))
        by[(m["session"], m["sid"])].append((f, m))
    keys = ["rule"] + [f"m{i}" for i in range(len(models))] + [f"rule|m{i}" for i in range(len(models))]
    res = {k: dict(on=0, misses=0, correct=0, wrong=0, false=0) for k in keys}
    for sh in shots:
        truth = sh["truth"]
        rows = by.get((sh["session"], sh["sid"]), [])
        r_obj = outcome_obj(rule_pick(rows))
        for k in keys:
            res[k]["on" if truth else "misses"] += 1
        score(res["rule"], truth, r_obj)
        for i, mdl in enumerate(models):
            pick, _ = model_pick(rows, mdl)
            m_obj = outcome_obj(pick)
            score(res[f"m{i}"], truth, m_obj)
            u = r_obj if r_obj not in (None, "AMBIG") else None
            v = m_obj if m_obj not in (None, "AMBIG") else None
            score(res[f"rule|m{i}"], truth, None if (u and v and u != v) else (u or v))
    print(json.dumps(dict(set=name, models=[m.path for m in models], **res)))


if __name__ == "__main__":
    main()
