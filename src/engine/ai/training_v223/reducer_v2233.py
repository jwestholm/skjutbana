from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .dense_v2233 import DenseShotV2233, REDUCER_FEATURE_NAMES

EPS = 1e-7


def _sigmoid_neg_diff(diff: np.ndarray) -> np.ndarray:
    # sigmoid(-diff), numerically stable enough after clipping.
    return 1.0 / (1.0 + np.exp(np.clip(diff, -40.0, 40.0)))


@dataclass
class ReducerModelV2233:
    kind: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float32, copy=False) - self.mean) / self.scale

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros((0,), dtype=np.float32)
        z = self._norm(x)
        if self.kind == "linear":
            return z @ self.arrays["w"] + float(self.arrays["b"][0])
        if self.kind == "mlp":
            h = np.tanh(z @ self.arrays["w1"] + self.arrays["b1"])
            return (h @ self.arrays["w2"] + self.arrays["b2"]).reshape(-1)
        raise ValueError(f"Unknown reducer kind: {self.kind}")

    def rank_indices(self, shot: DenseShotV2233) -> np.ndarray:
        return np.argsort(-self.score_matrix(shot.features), kind="stable")

    def save(self, directory: Path | str) -> tuple[Path, Path]:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        npz = directory / "model.npz"; meta = directory / "model.json"
        arrays = {"mean": self.mean.astype(np.float32), "scale": self.scale.astype(np.float32)}
        arrays.update({k: np.asarray(v, dtype=np.float32) for k, v in self.arrays.items()})
        np.savez_compressed(npz, **arrays)
        meta.write_text(json.dumps({
            "schema_version": "2.23.3-reducer-1",
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
        }, indent=2, sort_keys=True), encoding="utf-8")
        return npz, meta

    @classmethod
    def load(cls, directory: Path | str) -> "ReducerModelV2233":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz", allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            scale = np.asarray(data["scale"], dtype=np.float32)
            arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files if k not in {"mean", "scale"}}
        return cls(str(meta["kind"]), tuple(str(x) for x in meta["feature_names"]), mean, scale, arrays, dict(meta.get("metadata", {})))


def _sample_for_scaler(shots: Sequence[DenseShotV2233], rng: np.random.Generator, per_shot: int = 1536) -> np.ndarray:
    chunks = []
    for shot in shots:
        n = len(shot.features)
        if n == 0: continue
        if n <= per_shot:
            idx = np.arange(n)
        else:
            # Include positive/near-positive rows plus random physical background.
            important = np.flatnonzero(shot.distances <= 42.0)
            remaining = np.setdiff1d(np.arange(n), important, assume_unique=False)
            take = max(0, per_shot - len(important))
            if take and len(remaining) > take:
                remaining = rng.choice(remaining, size=take, replace=False)
            idx = np.unique(np.concatenate([important, remaining]))[:per_shot]
        chunks.append(shot.features[idx].astype(np.float32, copy=False))
    if not chunks:
        return np.zeros((0, len(REDUCER_FEATURE_NAMES)), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def fit_reducer_scaler(shots: Sequence[DenseShotV2233], *, seed: int = 2233) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample = _sample_for_scaler(shots, rng)
    if sample.size == 0:
        return np.zeros(len(REDUCER_FEATURE_NAMES), dtype=np.float32), np.ones(len(REDUCER_FEATURE_NAMES), dtype=np.float32)
    mean = np.mean(sample.astype(np.float64), axis=0)
    scale = np.std(sample.astype(np.float64), axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _mining_indices(shot: DenseShotV2233, *, rng: np.random.Generator, max_pos: int = 6, hard_each: int = 128, random_neg: int = 160) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(shot.distances <= 20.0)
    neg = np.flatnonzero(shot.distances > 42.0)  # 20..42 is deliberately neutral.
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    # Prefer the closest positive hypotheses but retain some diversity when many
    # dense points fall within the tolerance disk.
    pos = pos[np.argsort(shot.distances[pos], kind="stable")]
    if len(pos) > max_pos:
        pos = pos[:max_pos]

    dense_idx = REDUCER_FEATURE_NAMES.index("dense_score")
    rich_idx = REDUCER_FEATURE_NAMES.index("v2233_newhole_heuristic")
    hard_dense = neg[np.argsort(-shot.features[neg, dense_idx], kind="stable")[:hard_each]]
    hard_rich = neg[np.argsort(-shot.features[neg, rich_idx], kind="stable")[:hard_each]]
    chosen = np.unique(np.concatenate([hard_dense, hard_rich]))
    remaining = np.setdiff1d(neg, chosen, assume_unique=False)
    if random_neg > 0 and len(remaining):
        take = min(int(random_neg), len(remaining))
        rnd = rng.choice(remaining, size=take, replace=False)
        chosen = np.unique(np.concatenate([chosen, rnd]))
    return pos.astype(np.int64), chosen.astype(np.int64)


def train_reducer(
    shots: Sequence[DenseShotV2233],
    *,
    kind: str = "mlp",
    hidden: int = 48,
    epochs: int = 36,
    learning_rate: float = 0.006,
    l2: float = 0.0015,
    seed: int = 2233,
    progress: Callable[[int, int, float], None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[ReducerModelV2233, dict[str, Any]]:
    usable = [s for s in shots if s.oracle20 and np.any(s.distances > 42.0)]
    if len(usable) < 8:
        raise ValueError("Need at least 8 dense shots with <=20px positives")
    mean, scale = fit_reducer_scaler(usable, seed=seed)
    rng = np.random.default_rng(seed)
    d = len(REDUCER_FEATURE_NAMES)
    if kind == "linear":
        params = {"w": rng.normal(0.0, 0.025, d).astype(np.float32), "b": np.zeros(1, np.float32)}
    elif kind == "mlp":
        params = {
            "w1": rng.normal(0.0, 0.14 / max(1.0, math.sqrt(d)), (d, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "w2": rng.normal(0.0, 0.14 / max(1.0, math.sqrt(hidden)), (hidden, 1)).astype(np.float32),
            "b2": np.zeros(1, np.float32),
        }
    else:
        raise ValueError("kind must be linear or mlp")
    m = {k: np.zeros_like(v) for k, v in params.items()}
    vv = {k: np.zeros_like(v) for k, v in params.items()}
    step = 0; history: list[float] = []

    for epoch in range(int(epochs)):
        losses = []
        for sidx in rng.permutation(len(usable)):
            shot = usable[int(sidx)]
            pos_idx, neg_idx = _mining_indices(shot, rng=rng)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            idx = np.concatenate([pos_idx, neg_idx])
            x = (shot.features[idx].astype(np.float32) - mean) / scale
            pcount = len(pos_idx)
            if kind == "linear":
                scores = x @ params["w"] + params["b"][0]
                cache = None
            else:
                h = np.tanh(x @ params["w1"] + params["b1"])
                scores = (h @ params["w2"] + params["b2"]).reshape(-1)
                cache = h
            sp = scores[:pcount][:, None]
            sn = scores[pcount:][None, :]
            diff = sp - sn
            pair_weight = np.exp(-(shot.distances[pos_idx][:, None] ** 2) / (2.0 * 9.0 * 9.0)).astype(np.float32)
            g = -_sigmoid_neg_diff(diff) * pair_weight
            denom = max(float(np.sum(pair_weight)) * max(1, len(neg_idx)), 1.0)
            g /= denom
            ds = np.zeros(len(idx), dtype=np.float32)
            ds[:pcount] = np.sum(g, axis=1)
            ds[pcount:] = -np.sum(g, axis=0)
            # Weighted pairwise logistic loss for diagnostics.
            losses.append(float(np.mean(np.logaddexp(0.0, -diff) * pair_weight)))
            if kind == "linear":
                grads = {"w": x.T @ ds + l2 * params["w"], "b": np.asarray([np.sum(ds)], np.float32)}
            else:
                assert cache is not None
                grads = {"w2": cache.T @ ds[:, None] + l2 * params["w2"], "b2": np.asarray([np.sum(ds)], np.float32)}
                dh = ds[:, None] @ params["w2"].T
                dz = dh * (1.0 - cache * cache)
                grads["w1"] = x.T @ dz + l2 * params["w1"]
                grads["b1"] = np.sum(dz, axis=0)
            step += 1
            for key, grad in grads.items():
                grad = np.clip(np.asarray(grad, np.float32), -5.0, 5.0)
                m[key] = 0.9 * m[key] + 0.1 * grad
                vv[key] = 0.999 * vv[key] + 0.001 * (grad * grad)
                mh = m[key] / (1.0 - 0.9 ** step)
                vh = vv[key] / (1.0 - 0.999 ** step)
                params[key] -= learning_rate * mh / (np.sqrt(vh) + 1e-7)
        loss = float(np.mean(losses)) if losses else float("nan")
        history.append(loss)
        if progress and (epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(1, epochs // 6) == 0):
            progress(epoch + 1, epochs, loss)

    meta = dict(metadata or {})
    meta.update({
        "schema_version": "2.23.3-reducer-1", "seed": seed, "epochs": epochs,
        "hidden": hidden if kind == "mlp" else 0, "learning_rate": learning_rate,
        "l2": l2, "training_shots": len(usable), "neutral_band_px": [20.0, 42.0],
        "feature_contract": list(REDUCER_FEATURE_NAMES), "gt_in_model_features": False,
    })
    model = ReducerModelV2233(kind, REDUCER_FEATURE_NAMES, mean, scale, params, meta)
    return model, {"loss_history": history, "usable_training_shots": len(usable)}


def evaluate_reducer(model: ReducerModelV2233, shots: Sequence[DenseShotV2233], *, ks: Sequence[int] = (32, 64, 128, 256, 512, 1024)) -> dict[str, Any]:
    shots_n = len(shots); oracle20 = oracle42 = 0; top1_20 = top1_42 = 0
    retained20 = {int(k): 0 for k in ks}; retained42 = {int(k): 0 for k in ks}
    ranks20: list[int] = []; ranks42: list[int] = []
    for shot in shots:
        if len(shot.features) == 0: continue
        has20 = shot.oracle20; has42 = shot.oracle42
        oracle20 += int(has20); oracle42 += int(has42)
        order = model.rank_indices(shot)
        if len(order):
            top1_20 += int(shot.distances[int(order[0])] <= 20.0)
            top1_42 += int(shot.distances[int(order[0])] <= 42.0)
        if has20:
            rank = next((r for r, idx in enumerate(order, 1) if shot.distances[int(idx)] <= 20.0), len(order) + 1)
            ranks20.append(rank)
            for k in ks: retained20[int(k)] += int(rank <= int(k))
        if has42:
            rank = next((r for r, idx in enumerate(order, 1) if shot.distances[int(idx)] <= 42.0), len(order) + 1)
            ranks42.append(rank)
            for k in ks: retained42[int(k)] += int(rank <= int(k))
    return {
        "shots": shots_n,
        "oracle20": oracle20,
        "oracle20_rate": oracle20 / shots_n if shots_n else 0.0,
        "oracle42": oracle42,
        "oracle42_rate": oracle42 / shots_n if shots_n else 0.0,
        "top1_20": top1_20,
        "top1_20_rate": top1_20 / shots_n if shots_n else 0.0,
        "conditional_top1_20_rate": top1_20 / oracle20 if oracle20 else 0.0,
        "top1_42": top1_42,
        "conditional_top1_42_rate": top1_42 / oracle42 if oracle42 else 0.0,
        "median_positive_rank20": float(np.median(ranks20)) if ranks20 else None,
        "p90_positive_rank20": float(np.percentile(ranks20, 90)) if ranks20 else None,
        "median_positive_rank42": float(np.median(ranks42)) if ranks42 else None,
        "retention20_at_k": {str(k): retained20[int(k)] / oracle20 if oracle20 else 0.0 for k in ks},
        "retention42_at_k": {str(k): retained42[int(k)] / oracle42 if oracle42 else 0.0 for k in ks},
    }
