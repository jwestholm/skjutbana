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


def _cached_top1(exports, scores, *, source_scoped=True):
    """Source identity is part of a shot group. Legacy mode is diagnostic only."""
    hits = total = 0; scenarios = {}; offset = 0; best = {}
    for e in exports:
        inverse = {v: k for k, v in e.manifest['groups'].items()}
        for i, row in enumerate(e.meta):
            raw = inverse[int(row['group'])]
            g = (e.source, raw) if source_scoped else raw
            score = scores[offset + i]
            if g not in best or score > best[g][0]: best[g] = (score, bool(row['label']))
        offset += len(e.meta)
    for e in exports:
        for shot in e.shots:
            raw = str(shot['group']); g = (e.source, raw) if source_scoped else raw
            ok = best.get(g, (0, False))[1]
            hits += ok; total += 1
            v = scenarios.setdefault(shot['scenario'], [0, 0]); v[0] += ok; v[1] += 1
    return dict(shots=total, hits=hits, rate=hits / total if total else None,
                per_scenario={k: f'{v[0]}/{v[1]}' for k, v in sorted(scenarios.items())})


def train(train_dirs: list[str], holdout_dirs: list[str], output: Path, *, cache_root=None,
          negative_cap=None, sample_seed=20261122, max_matrix_gb=2.0) -> dict[str, Any]:
    from .candidate_cache import Export, selection
    import resource
    started = time.monotonic()
    train_dirs = [str(Path(d).resolve()) for d in train_dirs]
    holdout_dirs = [str(Path(d).resolve()) for d in holdout_dirs]
    if set(train_dirs) & set(holdout_dirs):
        raise ValueError('A directory is in both train and holdout')
    if not train_dirs or not holdout_dirs: raise ValueError('Both splits are required')
    output.mkdir(parents=True, exist_ok=False)
    cache_root = Path(cache_root or output.parent / 'candidate_cache')
    te = [Export(d, cache_root) for d in train_dirs]
    he = [Export(d, cache_root) for d in holdout_dirs]
    tr_hashes = [f'{e.source}/candidates.jsonl:{e.manifest["sha256"]}' for e in te]
    ho_hashes = [f'{e.source}/candidates.jsonl:{e.manifest["sha256"]}' for e in he]
    if {e.manifest['sha256'] for e in te} & {e.manifest['sha256'] for e in he}:
        raise ValueError('Identical export content in train and holdout')
    counts = {}; n_impact = sum(e.manifest['impact_candidates'] for e in te)
    for e in te:
        for k, n in zip(e.manifest['features'], e.manifest['impact_presence']): counts[k] = counts.get(k, 0) + n
    names = sorted(k for k, n in counts.items() if n / max(n_impact, 1) >= 0.5)
    chosen = [selection(e, seed=sample_seed, negative_cap=negative_cap) for e in te]
    n = sum(map(len, chosen)); nh = sum(len(e.meta) for e in he)
    if not n or not names: raise ValueError('No training candidates/features')
    matrix_bytes = (n + nh) * len(names) * 8
    if not math.isfinite(max_matrix_gb) or max_matrix_gb <= 0 or matrix_bytes > max_matrix_gb * 1e9:
        raise ValueError(f'Matrix budget exceeded: {matrix_bytes} bytes; limit {max_matrix_gb} GB')
    # Float32 storage; float64 working arithmetic retains the previous optimizer.
    def matrix(exports, selections, path):
        N = sum(map(len, selections)); out = np.lib.format.open_memmap(path, mode='w+', dtype='float64', shape=(N, len(names)))
        out[:] = np.nan; off = 0
        for e, idx in zip(exports, selections):
            cols = {k: j for j, k in enumerate(e.manifest['features'])}
            for j, k in enumerate(names):
                if k in cols: out[off:off+len(idx), j] = e.X[idx, cols[k]]
            off += len(idx)
        return out
    import tempfile
    with tempfile.TemporaryDirectory(prefix="bench_train_work_") as work:
        Xtr = matrix(te, chosen, Path(work) / 'train_work.npy')
        medians = np.array([np.nanmedian(Xtr[:, j]) for j in range(len(names))])
        medians = np.where(np.isfinite(medians), medians, 0)
        for j in range(len(names)):
            col = Xtr[:, j]; col[~np.isfinite(col)] = medians[j]
        mu = np.array([Xtr[:, j].mean() for j in range(len(names))])
        sd = np.array([Xtr[:, j].std() for j in range(len(names))]); sd = np.where(sd > 1e-9, sd, 1)
        for lo in range(0, n, 8192): Xtr[lo:lo+8192] = (Xtr[lo:lo+8192] - mu) / sd
        Xtr.flush()
        ytr = np.concatenate([e.meta['label'][idx].astype(float) for e, idx in zip(te, chosen)])
        sample = dict(seed=sample_seed, negative_cap=negative_cap, policy='all positive groups; capped other impact groups', exports=[])
        for j, (e, idx) in enumerate(zip(te, chosen)):
            np.save(output / f'selection_{j:03d}.npy', idx, allow_pickle=False)
            sample['exports'].append(dict(source=e.source, sha256=e.manifest['sha256'], rows=len(idx), selection=f'selection_{j:03d}.npy'))
        (output / 'sampling.json').write_text(json.dumps(sample, indent=2) + '\n')
        w, b = _fit_logistic(Xtr, ytr)
        all_ho = [np.arange(len(e.meta)) for e in he]
        Xho = matrix(he, all_ho, Path(work) / 'holdout_work.npy')
        model_scores = np.empty(nh); off = 0
        for lo in range(0, nh, 8192):
            block = np.asarray(Xho[lo:lo+8192]); block = np.where(np.isfinite(block), block, medians)
            model_scores[lo:lo+len(block)] = (block - mu) / sd @ w + b
        live_scores = np.concatenate([e.meta['live_score'] for e in he])
        yho = np.concatenate([e.meta['label'] for e in he]); impact_mask = np.concatenate([e.meta['impact'] for e in he])
        metrics = dict(train=dict(dirs=len(te), candidates=n, available_candidates=n_impact, positives=int(ytr.sum()), shots=sum(len(e.shots) for e in te)),
            holdout=dict(dirs=len(he), candidates=nh, positives=int(yho.sum()), shots=sum(len(e.shots) for e in he)),
            holdout_auc=dict(model=_auc(model_scores[impact_mask], yho[impact_mask]) if impact_mask.any() else None,
                             live_score=_auc(live_scores[impact_mask], yho[impact_mask]) if impact_mask.any() else None),
            holdout_top1=dict(model=_cached_top1(he, model_scores), live_score=_cached_top1(he, live_scores)), holdout_top1_by_source={})
        metrics['legacy_global_group_top1'] = dict(note='DIAGNOSTIC ONLY: historical cross-export group collision', model=_cached_top1(he, model_scores, source_scoped=False), live_score=_cached_top1(he, live_scores, source_scoped=False))
        for e in he:
            end = off + len(e.meta)
            m = _cached_top1([e], model_scores[off:end]); l = _cached_top1([e], live_scores[off:end])
            metrics['holdout_top1_by_source'][Path(e.source).name] = dict(model=f"{m['hits']}/{m['shots']}", live_score=f"{l['hits']}/{l['shots']}")
            off = end
        metrics['resources'] = dict(peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            elapsed_s=time.monotonic()-started, working_matrix_bytes=matrix_bytes,
            selected_cache_bytes=sum(p.stat().st_size for e in te+he for p in e.path.iterdir() if p.is_file()),
            sampling=sample, evaluation_sampling=False)
        weights = sorted(zip(names, w.tolist()), key=lambda kv: -abs(kv[1]))
        model = dict(status=STATUS, live_authority=False, kind="logistic_regression_candidate_scorer",
                     created=time.time(), features=names, impute_medians=medians.tolist(), mean=mu.tolist(),
                     std=sd.tolist(), weights=w.tolist(), bias=float(b), top_weights=weights[:15],
                     training_data=tr_hashes, holdout_data=ho_hashes, sampling=sample, cache_version=1,
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
    p.add_argument("--cache-root")
    p.add_argument("--negative-cap", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=20261122)
    p.add_argument("--max-matrix-gb", type=float, default=2.0)
    p.add_argument('--address-limit-gb', type=float, default=None)
    a = p.parse_args(argv)
    if a.address_limit_gb is not None:
        import resource
        if not math.isfinite(a.address_limit_gb) or a.address_limit_gb <= 0: p.error('Invalid address limit')
        limit = int(a.address_limit_gb * 1e9)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    result = train(a.train, a.holdout, Path(a.output), cache_root=a.cache_root, negative_cap=a.negative_cap, sample_seed=a.sample_seed, max_matrix_gb=a.max_matrix_gb)
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
