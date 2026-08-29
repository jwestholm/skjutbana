from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .schema import FEATURE_NAMES, ShotTrainingRecord

EPS = 1e-8


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    z = x - float(np.max(x))
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / max(float(e.sum()), EPS)


def _feature_matrix(record: ShotTrainingRecord, feature_names: Sequence[str]) -> np.ndarray:
    if not record.candidates:
        return np.zeros((0, len(feature_names)), dtype=np.float32)
    return np.asarray([[float(row.features.get(k, 0.0)) for k in feature_names] for row in record.candidates], dtype=np.float32)


def _positive_mask(record: ShotTrainingRecord, radius: float = 20.0) -> np.ndarray:
    return np.asarray([
        (row.gt_distance_px is not None and float(row.gt_distance_px) <= radius)
        for row in record.candidates
    ], dtype=bool)


def fit_scaler(records: Iterable[ShotTrainingRecord], feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    chunks = [_feature_matrix(r, feature_names) for r in records if r.candidates]
    if not chunks:
        return np.zeros(len(feature_names), dtype=np.float32), np.ones(len(feature_names), dtype=np.float32)
    x = np.concatenate(chunks, axis=0).astype(np.float64)
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean.astype(np.float32), scale.astype(np.float32)


@dataclass
class RankModelV223:
    kind: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros((0,), dtype=np.float32)
        z = (x.astype(np.float32) - self.mean) / self.scale
        if self.kind == "linear":
            return z @ self.arrays["w"] + float(self.arrays["b"][0])
        if self.kind == "mlp":
            h = np.tanh(z @ self.arrays["w1"] + self.arrays["b1"])
            return (h @ self.arrays["w2"] + self.arrays["b2"]).reshape(-1)
        raise ValueError(f"Unknown model kind: {self.kind}")

    def score_record(self, record: ShotTrainingRecord) -> np.ndarray:
        return self.score_matrix(_feature_matrix(record, self.feature_names))

    def rank_indices(self, record: ShotTrainingRecord) -> np.ndarray:
        scores = self.score_record(record)
        return np.argsort(-scores, kind="stable")

    def save(self, directory: Path | str) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz_path = directory / "model.npz"
        json_path = directory / "model.json"
        arrays = {"mean": self.mean, "scale": self.scale}
        arrays.update(self.arrays)
        np.savez_compressed(npz_path, **arrays)
        payload = {
            "schema_version": "2.23.0",
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return npz_path, json_path

    @classmethod
    def load(cls, directory: Path | str) -> "RankModelV223":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz", allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            scale = np.asarray(data["scale"], dtype=np.float32)
            arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files if k not in {"mean", "scale"}}
        return cls(
            kind=str(meta["kind"]),
            feature_names=tuple(str(x) for x in meta["feature_names"]),
            mean=mean, scale=scale, arrays=arrays, metadata=dict(meta.get("metadata", {})),
        )


def _targets(record: ShotTrainingRecord, *, positive_radius: float = 20.0, sigma: float = 8.0) -> np.ndarray | None:
    distances = np.asarray([
        float(row.gt_distance_px) if row.gt_distance_px is not None else 1e9
        for row in record.candidates
    ], dtype=np.float32)
    if not np.any(distances <= positive_radius):
        return None
    q = np.exp(-(distances * distances) / (2.0 * sigma * sigma)).astype(np.float32)
    q[distances > 42.0] = 0.0
    total = float(q.sum())
    if total <= EPS:
        q = (distances <= positive_radius).astype(np.float32)
        total = float(q.sum())
    return q / max(total, EPS)


def _subsample_indices(record: ShotTrainingRecord, *, max_candidates: int, rng: np.random.Generator) -> np.ndarray:
    n = len(record.candidates)
    if n <= max_candidates:
        return np.arange(n, dtype=np.int64)
    pos = [i for i, row in enumerate(record.candidates) if row.gt_distance_px is not None and row.gt_distance_px <= 42.0]
    hard = sorted(
        [i for i in range(n) if i not in set(pos)],
        key=lambda i: (
            record.candidates[i].baseline_rank if record.candidates[i].baseline_rank is not None else 10**9,
            -(record.candidates[i].baseline_score if record.candidates[i].baseline_score is not None else -1e9),
        ),
    )
    keep: list[int] = []
    for i in pos + hard[: max_candidates // 2]:
        if i not in keep:
            keep.append(i)
        if len(keep) >= max_candidates:
            break
    remaining = [i for i in range(n) if i not in set(keep)]
    if len(keep) < max_candidates and remaining:
        take = min(max_candidates - len(keep), len(remaining))
        sampled = rng.choice(np.asarray(remaining), size=take, replace=False)
        keep.extend(int(x) for x in sampled)
    return np.asarray(sorted(keep), dtype=np.int64)


def train_rank_model(
    records: Sequence[ShotTrainingRecord],
    *,
    kind: str = "linear",
    feature_names: Sequence[str] = FEATURE_NAMES,
    hidden: int = 24,
    epochs: int = 80,
    learning_rate: float = 0.02,
    l2: float = 0.001,
    seed: int = 2230,
    max_candidates_per_shot: int = 256,
    metadata: dict[str, Any] | None = None,
) -> tuple[RankModelV223, dict[str, Any]]:
    usable = [r for r in records if r.candidates and _targets(r) is not None]
    if len(usable) < 2:
        raise ValueError("Need at least two training shots with a <=20px actual candidate")
    names = tuple(feature_names)
    mean, scale = fit_scaler(usable, names)
    rng = np.random.default_rng(seed)
    d = len(names)
    if kind == "linear":
        params = {
            "w": rng.normal(0.0, 0.02, size=(d,)).astype(np.float32),
            "b": np.zeros((1,), dtype=np.float32),
        }
    elif kind == "mlp":
        params = {
            "w1": rng.normal(0.0, 0.08 / max(1.0, math.sqrt(d)), size=(d, hidden)).astype(np.float32),
            "b1": np.zeros((hidden,), dtype=np.float32),
            "w2": rng.normal(0.0, 0.08 / max(1.0, math.sqrt(hidden)), size=(hidden, 1)).astype(np.float32),
            "b2": np.zeros((1,), dtype=np.float32),
        }
    else:
        raise ValueError("kind must be linear or mlp")

    # Adam state.
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(vv) for k, vv in params.items()}
    step = 0
    history: list[float] = []

    for epoch in range(int(epochs)):
        order = rng.permutation(len(usable))
        losses: list[float] = []
        for ridx in order:
            record = usable[int(ridx)]
            idx = _subsample_indices(record, max_candidates=max_candidates_per_shot, rng=rng)
            x_raw = _feature_matrix(record, names)[idx]
            x = (x_raw - mean) / scale
            full_q = _targets(record)
            assert full_q is not None
            q = full_q[idx]
            q_sum = float(q.sum())
            if q_sum <= EPS:
                continue
            q = q / q_sum
            if kind == "linear":
                scores = x @ params["w"] + params["b"][0]
                p = _softmax(scores)
                ds = p - q
                grads = {
                    "w": x.T @ ds + l2 * params["w"],
                    "b": np.asarray([ds.sum()], dtype=np.float32),
                }
            else:
                hpre = x @ params["w1"] + params["b1"]
                h = np.tanh(hpre)
                scores = (h @ params["w2"] + params["b2"]).reshape(-1)
                p = _softmax(scores)
                ds = (p - q).reshape(-1, 1)
                grads = {
                    "w2": h.T @ ds + l2 * params["w2"],
                    "b2": ds.sum(axis=0),
                }
                dh = ds @ params["w2"].T
                dhpre = dh * (1.0 - h * h)
                grads["w1"] = x.T @ dhpre + l2 * params["w1"]
                grads["b1"] = dhpre.sum(axis=0)
            losses.append(float(-np.sum(q * np.log(np.clip(p, EPS, 1.0)))))
            step += 1
            # Conservative gradient clipping makes unattended trials robust.
            for key, grad in grads.items():
                grad = np.clip(grad.astype(np.float32), -5.0, 5.0)
                m[key] = 0.9 * m[key] + 0.1 * grad
                v[key] = 0.999 * v[key] + 0.001 * (grad * grad)
                mh = m[key] / (1.0 - 0.9 ** step)
                vh = v[key] / (1.0 - 0.999 ** step)
                params[key] -= learning_rate * mh / (np.sqrt(vh) + 1e-7)
        history.append(float(np.mean(losses)) if losses else float("nan"))

    model_meta = dict(metadata or {})
    model_meta.update({
        "training_shots": len(usable), "epochs": int(epochs), "seed": int(seed),
        "learning_rate": float(learning_rate), "l2": float(l2),
        "hidden": int(hidden) if kind == "mlp" else 0,
        "max_candidates_per_shot": int(max_candidates_per_shot),
        "final_loss": history[-1] if history else None,
    })
    model = RankModelV223(kind=kind, feature_names=names, mean=mean, scale=scale, arrays=params, metadata=model_meta)
    return model, {"loss_history": history, "usable_training_shots": len(usable)}


def evaluate_model(model: RankModelV223, records: Sequence[ShotTrainingRecord]) -> dict[str, Any]:
    shots = 0
    oracle20 = oracle42 = 0
    top1 = top3 = 0
    top1_42 = top3_42 = 0
    ranks: list[int] = []
    ranks42: list[int] = []
    errors: list[float] = []
    for record in records:
        shots += 1
        if not record.candidates:
            continue
        positive20 = _positive_mask(record, 20.0)
        positive42 = _positive_mask(record, 42.0)
        has20 = bool(np.any(positive20))
        has42 = bool(np.any(positive42))
        oracle20 += int(has20)
        oracle42 += int(has42)
        order = model.rank_indices(record)
        if order.size:
            d0 = record.candidates[int(order[0])].gt_distance_px
            if d0 is not None:
                errors.append(float(d0))
            top1 += int(bool(positive20[int(order[0])]))
            top3 += int(any(bool(positive20[int(i)]) for i in order[:3]))
            top1_42 += int(bool(positive42[int(order[0])]))
            top3_42 += int(any(bool(positive42[int(i)]) for i in order[:3]))
        if has20:
            for rank, idx in enumerate(order, start=1):
                if positive20[int(idx)]:
                    ranks.append(rank)
                    break
        if has42:
            for rank, idx in enumerate(order, start=1):
                if positive42[int(idx)]:
                    ranks42.append(rank)
                    break
    return {
        "shots": shots,
        "oracle20": oracle20,
        "oracle20_rate": oracle20 / shots if shots else 0.0,
        "oracle42": oracle42,
        "oracle42_rate": oracle42 / shots if shots else 0.0,
        "top1_20": top1,
        "top1_20_rate": top1 / shots if shots else 0.0,
        "conditional_top1_20_rate": top1 / oracle20 if oracle20 else 0.0,
        "top3_20_rate": top3 / shots if shots else 0.0,
        "conditional_top3_20_rate": top3 / oracle20 if oracle20 else 0.0,
        "top1_42": top1_42,
        "conditional_top1_42_rate": top1_42 / oracle42 if oracle42 else 0.0,
        "conditional_top3_42_rate": top3_42 / oracle42 if oracle42 else 0.0,
        "median_positive_rank": float(np.median(ranks)) if ranks else None,
        "median_positive_rank42": float(np.median(ranks42)) if ranks42 else None,
        "mrr20": float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0,
        "mrr42": float(np.mean([1.0 / r for r in ranks42])) if ranks42 else 0.0,
        "median_top1_error_px": float(np.median(errors)) if errors else None,
        "p95_top1_error_px": float(np.percentile(errors, 95)) if errors else None,
    }


def evaluate_baseline(records: Sequence[ShotTrainingRecord]) -> dict[str, Any]:
    """Evaluate a reproducible non-learned reference ranking.

    V2.23.1 required explicit baseline_rank and therefore reported zero eligible
    shots for many imported/expanded pools. V2.23.2 falls back to baseline_score
    (combined/detector/dense score captured with the candidate) when explicit
    ranks are absent. This makes challenger-vs-reference comparisons meaningful.
    """
    shots = oracle20 = oracle42 = top1 = top3 = top1_42 = top3_42 = 0
    ranks: list[int] = []
    ranks42: list[int] = []
    eligible = 0
    explicit_ranked = 0
    score_ranked = 0
    for record in records:
        shots += 1
        pos20 = [r for r in record.candidates if r.gt_distance_px is not None and r.gt_distance_px <= 20.0]
        pos42 = [r for r in record.candidates if r.gt_distance_px is not None and r.gt_distance_px <= 42.0]
        oracle20 += int(bool(pos20)); oracle42 += int(bool(pos42))
        ranked_rows = [r for r in record.candidates if r.baseline_rank is not None and r.baseline_rank > 0]
        if ranked_rows:
            ranked_rows.sort(key=lambda r: int(r.baseline_rank or 10**9))
            explicit_ranked += 1
        else:
            ranked_rows = [r for r in record.candidates if r.baseline_score is not None]
            ranked_rows.sort(key=lambda r: float(r.baseline_score or 0.0), reverse=True)
            if ranked_rows:
                score_ranked += 1
        if not ranked_rows:
            continue
        eligible += 1
        top1 += int(ranked_rows[0].gt_distance_px is not None and ranked_rows[0].gt_distance_px <= 20.0)
        top3 += int(any(r.gt_distance_px is not None and r.gt_distance_px <= 20.0 for r in ranked_rows[:3]))
        top1_42 += int(ranked_rows[0].gt_distance_px is not None and ranked_rows[0].gt_distance_px <= 42.0)
        top3_42 += int(any(r.gt_distance_px is not None and r.gt_distance_px <= 42.0 for r in ranked_rows[:3]))
        pos_ranks = [rank for rank, r in enumerate(ranked_rows, start=1) if r.gt_distance_px is not None and r.gt_distance_px <= 20.0]
        pos_ranks42 = [rank for rank, r in enumerate(ranked_rows, start=1) if r.gt_distance_px is not None and r.gt_distance_px <= 42.0]
        if pos_ranks: ranks.append(min(pos_ranks))
        if pos_ranks42: ranks42.append(min(pos_ranks42))
    return {
        "shots": shots,
        "eligible_ranked_shots": eligible,
        "explicit_ranked_shots": explicit_ranked,
        "score_fallback_ranked_shots": score_ranked,
        "oracle20": oracle20,
        "oracle20_rate": oracle20 / shots if shots else 0.0,
        "oracle42": oracle42,
        "oracle42_rate": oracle42 / shots if shots else 0.0,
        "top1_20": top1,
        "top1_20_rate_on_all": top1 / shots if shots else 0.0,
        "top1_20_rate_on_ranked": top1 / eligible if eligible else 0.0,
        "conditional_top1_20_rate": top1 / oracle20 if oracle20 else 0.0,
        "top3_20_rate_on_ranked": top3 / eligible if eligible else 0.0,
        "conditional_top3_20_rate": top3 / oracle20 if oracle20 else 0.0,
        "conditional_top1_42_rate": top1_42 / oracle42 if oracle42 else 0.0,
        "conditional_top3_42_rate": top3_42 / oracle42 if oracle42 else 0.0,
        "median_positive_rank": float(np.median(ranks)) if ranks else None,
        "median_positive_rank42": float(np.median(ranks42)) if ranks42 else None,
        "mrr20": float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0,
        "mrr42": float(np.mean([1.0 / r for r in ranks42])) if ranks42 else 0.0,
        "reference_policy": "explicit_baseline_rank_else_baseline_score_desc",
    }

