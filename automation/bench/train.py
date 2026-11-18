"""Train/evaluate a simple candidate scorer from bench exports (offline only).

  python3 -m automation.bench train --train DIR [DIR ...] --holdout DIR [DIR ...] --output DIR

Inputs are `candidates.jsonl` files written by `bench synth --export-candidates`
(rows of kind "shot" and "candidate"). Splits are by whole run directory
(scenario/seed or recorded session) and fixed BEFORE feature selection,
imputation and scaling. The model is never installed into the runtime:
status OFFLINE_REPLAY_ONLY_UNVALIDATED, live_authority=false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

STATUS = "OFFLINE_REPLAY_ONLY_UNVALIDATED"
EXCLUDED_EXACT = {"camera_x", "camera_y", "timestamp", "analysis_crop_x0", "analysis_crop_y0",
                  "v2224_producer_shot_id", "v2225_origin_frame_ts", "track_id", "hole_id",
                  "ctx_brightness", "ctx_texture", "ctx_motion"}  # diagnostic image context
EXCLUDED_SUBSTRINGS = ("shot_id", "_ts", "origin_xy", "ground_truth", "training_")


def _load(dirs: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    shots, cands, hashes = [], [], []
    for d in dirs:
        path = Path(d) / "candidates.jsonl"
        data = path.read_bytes()
        hashes.append(f"{path}:{hashlib.sha256(data).hexdigest()}")
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source"] = str(d)
            (shots if row.get("kind") == "shot" else cands).append(row)
    return shots, cands, hashes


def _feature_names(rows: list[dict], min_presence: float = 0.5) -> list[str]:
    counts: dict[str, int] = {}
    for r in rows:
        for k, v in r["candidate"].items():
            if k in EXCLUDED_EXACT or any(s in k for s in EXCLUDED_SUBSTRINGS):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                counts[k] = counts.get(k, 0) + 1
    n = max(1, len(rows))
    return sorted(k for k, c in counts.items() if c / n >= min_presence)


def _matrix(rows, names, medians=None):
    X = np.full((len(rows), len(names)), np.nan)
    for i, r in enumerate(rows):
        c = r["candidate"]
        for j, k in enumerate(names):
            v = c.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                X[i, j] = float(v)
    if medians is None:
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    X = np.where(np.isfinite(X), X, medians)
    return X, medians


def _fit_logistic(X, y, l2=1.0, iters=400, lr=0.5):
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    w_sample = np.where(y > 0, len(y) / (2 * pos), len(y) / (2 * neg))   # balance classes
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = w_sample * (p - y)
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def _auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _top1(shots, cands, scores):
    """Per impact shot: does the best-scored candidate of its group hit? No candidates = miss."""
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(cands):
        by_group.setdefault(r["group"], []).append(i)
    hits = 0
    per_scenario: dict[str, list[int]] = {}
    for s in shots:
        idx = by_group.get(s["group"], [])
        ok = bool(idx) and bool(cands[max(idx, key=lambda i: scores[i])]["label"])
        hits += ok
        per_scenario.setdefault(s["scenario"], [0, 0])
        per_scenario[s["scenario"]][0] += ok
        per_scenario[s["scenario"]][1] += 1
    return dict(shots=len(shots), hits=hits, rate=hits / len(shots) if shots else None,
                per_scenario={k: f"{v[0]}/{v[1]}" for k, v in sorted(per_scenario.items())})


def train(train_dirs: list[str], holdout_dirs: list[str], output: Path) -> dict[str, Any]:
    if set(map(str, train_dirs)) & set(map(str, holdout_dirs)):
        raise SystemExit("A directory is in both train and holdout")
    output.mkdir(parents=True, exist_ok=False)
    tr_shots, tr, tr_hashes = _load(train_dirs)
    ho_shots, ho, ho_hashes = _load(holdout_dirs)
    tr = [r for r in tr if r["producer_has_impact"]]       # learn ranking within real shots
    names = _feature_names(tr)                               # fit on TRAIN only
    Xtr, medians = _matrix(tr, names)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    ytr = np.array([1.0 if r["label"] else 0.0 for r in tr])
    w, b = _fit_logistic((Xtr - mu) / sd, ytr)
    Xho, _ = _matrix(ho, names, medians)
    model_scores = (Xho - mu) / sd @ w + b
    live_scores = np.array([float(r["candidate"].get("score", 0.0) or 0.0) for r in ho])
    yho = np.array([1.0 if r["label"] else 0.0 for r in ho])
    impact_mask = np.array([bool(r["producer_has_impact"]) for r in ho])
    metrics = dict(
        train=dict(dirs=len(train_dirs), candidates=len(tr), positives=int(ytr.sum()), shots=len(tr_shots)),
        holdout=dict(dirs=len(holdout_dirs), candidates=len(ho), positives=int(yho.sum()), shots=len(ho_shots)),
        holdout_auc=dict(model=_auc(model_scores[impact_mask], yho[impact_mask]) if impact_mask.any() else None,
                         live_score=_auc(live_scores[impact_mask], yho[impact_mask]) if impact_mask.any() else None),
        holdout_top1=dict(model=_top1(ho_shots, ho, model_scores), live_score=_top1(ho_shots, ho, live_scores)),
        holdout_top1_by_source={},
    )
    # per holdout set (synthetic disk / stress / real-hole stamps / recorded session)
    for src in holdout_dirs:
        idx = [i for i, r in enumerate(ho) if r["_source"] == str(src)]
        shots_s = [s for s in ho_shots if s["_source"] == str(src)]
        sub = [ho[i] for i in idx]
        m = _top1(shots_s, sub, model_scores[idx]) if idx else _top1(shots_s, [], [])
        l = _top1(shots_s, sub, live_scores[idx]) if idx else _top1(shots_s, [], [])
        metrics["holdout_top1_by_source"][Path(src).name] = dict(model=f"{m['hits']}/{m['shots']}",
                                                                 live_score=f"{l['hits']}/{l['shots']}")
    weights = sorted(zip(names, w.tolist()), key=lambda kv: -abs(kv[1]))
    model = dict(status=STATUS, live_authority=False, kind="logistic_regression_candidate_scorer",
                 created=time.time(), features=names, impute_medians=medians.tolist(), mean=mu.tolist(),
                 std=sd.tolist(), weights=w.tolist(), bias=float(b), top_weights=weights[:15],
                 training_data=tr_hashes, holdout_data=ho_hashes,
                 note="Never installed into the runtime. Synthetic/optical replay evidence only.")
    blob = json.dumps(model, sort_keys=True).encode()
    model["sha256"] = hashlib.sha256(blob).hexdigest()
    (output / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    t1m, t1l = metrics["holdout_top1"]["model"], metrics["holdout_top1"]["live_score"]
    report = [f"# Nightly candidate-scorer ({STATUS})", "",
              f"live_authority=false. Model sha256 `{model['sha256']}`.", "",
              f"Holdout shots: {t1m['shots']} — model top-1 {t1m['hits']} vs live score top-1 {t1l['hits']}",
              f"Holdout AUC (impact groups): model {metrics['holdout_auc']['model']} vs live score {metrics['holdout_auc']['live_score']}",
              "", "Per holdout set (model / live):"]
    for k, v in metrics["holdout_top1_by_source"].items():
        report.append(f"- {k}: {v['model']} / {v['live_score']}")
    report += ["", "Per scenario (model / live):"]
    for k in t1m["per_scenario"]:
        report.append(f"- {k}: {t1m['per_scenario'][k]} / {t1l['per_scenario'].get(k)}")
    report += ["", "Top weights:"] + [f"- {k}: {v:+.3f}" for k, v in weights[:15]]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    return dict(metrics=metrics, model_sha256=model["sha256"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m automation.bench train")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    result = train(a.train, a.holdout, Path(a.output))
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
