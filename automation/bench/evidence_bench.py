"""Per-evidence benchmark on bench candidate exports (offline measurement only; 2026-11-20).

  python3 -m automation.bench evidence-bench --root /data/skjutbana/nightly_evidence/dataset --out DIR

Uses the frozen nightly train/holdout split (whole sessions / seeds, never mixed) and answers, per evidence
field and per evidence FAMILY (field-name prefix):
  * presence: fraction of holdout candidates where the field is measured
  * AUC (holdout, impact groups only): how well the field alone separates the true hole from the rest
  * single-field top-1: ranking candidates of each shot by this field alone (sign chosen on TRAIN)
  * redundancy: |Spearman| >= 0.9 pairs (risk of over-weighting one observation)
  * family ablation: retrain the nightly logistic scorer without one family -> holdout top-1 change
  * AI vs functional: per shot, model-only wins vs live-score-only wins, and which fields differ
Candidate recall (shots with any correct candidate) is reported as the ceiling every ranking shares.
Nothing here changes detector behaviour.  Selection uses TRAIN only; holdout is reported, never tuned on.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from automation.bench.train import _auc, _feature_names, _fit_logistic, _load, _matrix, _top1


def _family(name: str) -> str:
    parts = name.split("_")
    if parts[0] in ("ev", "v2", "v24", "v25", "v26", "v2224", "v2225", "v251", "v252", "v253", "h11", "h12",
                    "ctx", "shot", "candidate", "local", "ai"):
        return "_".join(parts[:2]) if len(parts) > 1 else parts[0]
    return parts[0]


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def _ceiling(shots, cands) -> int:
    pos = {c["group"] for c in cands if c["label"]}
    return sum(1 for s in shots if s["group"] in pos)


def run(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    train_dirs = sorted(str(d) for d in (root / "train").glob("*") if (d / "candidates.jsonl").exists())
    hold_dirs = sorted(str(d) for d in (root / "holdout").glob("*") if (d / "candidates.jsonl").exists())
    tr_shots, tr, _ = _load(train_dirs)
    ho_shots, ho, _ = _load(hold_dirs)
    tr = [r for r in tr if r["producer_has_impact"]]
    names = _feature_names(tr)
    Xtr, med = _matrix(tr, names)
    Xho, _ = _matrix(ho, names, med)
    ytr = np.array([1.0 if r["label"] else 0.0 for r in tr])
    yho = np.array([1.0 if r["label"] else 0.0 for r in ho])
    imp = np.array([bool(r["producer_has_impact"]) for r in ho])
    present = np.array([[isinstance(r["candidate"].get(k), (int, float)) and not isinstance(r["candidate"].get(k), bool)
                         for k in names] for r in ho])
    res: dict = dict(train_dirs=len(train_dirs), holdout_dirs=len(hold_dirs), holdout_shots=len(ho_shots),
                     candidate_ceiling=_ceiling(ho_shots, ho), features=len(names))
    # single fields
    rows = []
    for j, k in enumerate(names):
        a_tr = _auc(Xtr[:, j], ytr)
        sign = 1.0 if (a_tr is None or a_tr >= 0.5) else -1.0
        a_ho = _auc(sign * Xho[imp, j], yho[imp]) if imp.any() else None
        t1 = _top1(ho_shots, ho, sign * Xho[:, j])["hits"]
        rows.append(dict(field=k, family=_family(k), presence=float(present[:, j].mean()), sign=sign,
                         auc_train=a_tr if a_tr is None else (a_tr if sign > 0 else 1 - a_tr),
                         auc_holdout=a_ho, top1_holdout=t1))
    rows.sort(key=lambda r: -(r["auc_holdout"] or 0))
    res["fields"] = rows
    # redundancy (Spearman on a train sample)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=min(20000, len(Xtr)), replace=False)
    R = np.stack([_rankdata(Xtr[idx, j]) for j in range(len(names))], axis=1)
    R = (R - R.mean(0)) / np.where(R.std(0) > 0, R.std(0), 1)
    C = (R.T @ R) / len(idx)
    pairs = [(names[i], names[j], float(C[i, j])) for i in range(len(names)) for j in range(i + 1, len(names))
             if abs(C[i, j]) >= 0.9]
    res["redundant_pairs"] = sorted(pairs, key=lambda p: -abs(p[2]))
    # full model + family ablation
    def fit_eval(cols):
        mu, sd = Xtr[:, cols].mean(0), Xtr[:, cols].std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        w, b = _fit_logistic((Xtr[:, cols] - mu) / sd, ytr)
        s = (Xho[:, cols] - mu) / sd @ w + b
        return _top1(ho_shots, ho, s)["hits"], _auc(s[imp], yho[imp]), s
    all_cols = list(range(len(names)))
    full_t1, full_auc, full_s = fit_eval(all_cols)
    live = np.array([float(r["candidate"].get("score", 0.0) or 0.0) for r in ho])
    res["model_full"] = dict(top1=full_t1, auc=full_auc)
    res["live_score"] = dict(top1=_top1(ho_shots, ho, live)["hits"], auc=_auc(live[imp], yho[imp]))
    fams = sorted({_family(k) for k in names})
    abl = []
    for f in fams:
        cols = [i for i, k in enumerate(names) if _family(k) != f]
        if not cols:
            continue
        t1, auc, _ = fit_eval(cols)
        only = [i for i, k in enumerate(names) if _family(k) == f]
        t1o, auco, _ = fit_eval(only)
        abl.append(dict(family=f, n_fields=len(only), top1_without=t1, delta_top1=t1 - full_t1,
                        auc_without=auc, top1_only=t1o, auc_only=auco))
    res["family_ablation"] = sorted(abl, key=lambda a: a["delta_top1"])
    # AI vs functional per shot
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(ho):
        by_group.setdefault(r["group"], []).append(i)
    model_only, live_only, both, neither = [], [], 0, 0
    for s in ho_shots:
        ix = by_group.get(s["group"], [])
        if not ix:
            neither += 1
            continue
        m = max(ix, key=lambda i: full_s[i]); l = max(ix, key=lambda i: live[i])
        mo, lo = bool(ho[m]["label"]), bool(ho[l]["label"])
        if mo and lo:
            both += 1
        elif mo:
            model_only.append((m, l, s["_source"]))
        elif lo:
            live_only.append((m, l, s["_source"]))
        else:
            neither += 1
    diffs = []
    for j, k in enumerate(names):   # which fields separate the model's correct pick from live's wrong pick
        d = [Xho[m, j] - Xho[l, j] for m, l, _ in model_only]
        if d:
            sd = float(np.std(Xho[:, j])) or 1.0
            diffs.append((k, float(np.median(d)) / sd))
    res["ai_vs_functional"] = dict(both=both, model_only=len(model_only), live_only=len(live_only), neither=neither,
                                   model_only_by_source=_count(model_only), live_only_by_source=_count(live_only),
                                   fields_driving_model_wins=sorted(diffs, key=lambda kv: -abs(kv[1]))[:15])
    (out / "evidence_bench.json").write_text(json.dumps(res, indent=1))
    _report(res, out / "EVIDENCE_BENCH.md")
    return res


def _count(items):
    c: dict[str, int] = {}
    for _m, _l, src in items:
        c[Path(src).name] = c.get(Path(src).name, 0) + 1
    return c


def _report(res: dict, path: Path) -> None:
    L = ["# Evidence benchmark (offline, frozen nightly split)", "",
         f"Holdout shots {res['holdout_shots']}, candidate ceiling (shots with any correct candidate) "
         f"{res['candidate_ceiling']}, fields {res['features']}.",
         f"Model (all fields) top-1 {res['model_full']['top1']} AUC {res['model_full']['auc']:.3f}; "
         f"live score top-1 {res['live_score']['top1']} AUC {res['live_score']['auc']:.3f}.", "",
         "## Single fields (sign fixed on train)", "", "| field | family | presence | AUC holdout | top-1 alone |",
         "|---|---|---|---|---|"]
    for r in res["fields"][:40]:
        L.append(f"| {r['field']} | {r['family']} | {r['presence']:.2f} | "
                 f"{(r['auc_holdout'] or 0):.3f} | {r['top1_holdout']} |")
    L += ["", "## Family ablation (retrain without / only this family)", "",
          "| family | fields | top-1 without | delta | top-1 only | AUC only |", "|---|---|---|---|---|---|"]
    for a in res["family_ablation"]:
        L.append(f"| {a['family']} | {a['n_fields']} | {a['top1_without']} | {a['delta_top1']:+d} | "
                 f"{a['top1_only']} | {(a['auc_only'] or 0):.3f} |")
    L += ["", f"## Redundant pairs (|Spearman| >= 0.9): {len(res['redundant_pairs'])}", ""]
    for a, b, c in res["redundant_pairs"][:40]:
        L.append(f"- {a} ~ {b}: {c:+.2f}")
    av = res["ai_vs_functional"]
    L += ["", "## AI (learned scorer) vs functional (live score)", "",
          f"both correct {av['both']}, model only {av['model_only']}, live only {av['live_only']}, neither {av['neither']}",
          f"model-only by source: {av['model_only_by_source']}", f"live-only by source: {av['live_only_by_source']}",
          "", "Fields where the model's correct pick differs most from live's wrong pick (median diff / sd):"]
    for k, v in av["fields_driving_model_wins"]:
        L.append(f"- {k}: {v:+.2f}")
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m automation.bench evidence-bench")
    ap.add_argument("--root", default="/data/skjutbana/nightly_evidence/dataset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Path(a.root), Path(a.out))
    print(open(Path(a.out) / "EVIDENCE_BENCH.md").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
