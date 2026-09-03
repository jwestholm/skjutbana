from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .dense_v2233 import REDUCER_FEATURE_NAMES
from .evidence_patch_v2235 import (
    EVIDENCE_CHANNELS,
    EVIDENCE_PATCH_SIZE,
    PATCH_NEGATIVE_RADIUS_PX,
    PATCH_POSITIVE_RADIUS_PX,
    EvidenceShotRefV2235,
    load_evidence_shot,
)

EPS = 1e-8


def evidence_vector(patches: np.ndarray) -> np.ndarray:
    """Compact but spatial representation of registered evidence patches.

    Input is still a real multi-channel local image.  We retain a coarse 3x3
    view, the unpooled central 3x3, and per-channel shape/contrast statistics.
    This avoids billions of unnecessary raw-pixel MLP multiplies while keeping
    candidate-centred spatial information.
    """
    p = np.asarray(patches, dtype=np.float32)
    if p.ndim != 4 or p.shape[1:] != (EVIDENCE_CHANNELS, EVIDENCE_PATCH_SIZE, EVIDENCE_PATCH_SIZE):
        raise ValueError(f"Expected [B,{EVIDENCE_CHANNELS},{EVIDENCE_PATCH_SIZE},{EVIDENCE_PATCH_SIZE}], got {p.shape}")
    p = p / 255.0
    b = len(p)
    coarse = p.reshape(b, EVIDENCE_CHANNELS, 3, 3, 3, 3).mean(axis=(3, 5)).reshape(b, -1)
    center = p[:, :, 3:6, 3:6].reshape(b, -1)
    mean = p.mean(axis=(2, 3))
    std = p.std(axis=(2, 3))
    vmax = p.max(axis=(2, 3))
    cmean = p[:, :, 3:6, 3:6].mean(axis=(2, 3))
    ring_sum = p.sum(axis=(2, 3)) - p[:, :, 3:6, 3:6].sum(axis=(2, 3))
    ring_mean = ring_sum / float(EVIDENCE_PATCH_SIZE * EVIDENCE_PATCH_SIZE - 9)
    contrast = cmean - ring_mean
    x = np.concatenate([coarse, center, mean, std, vmax, cmean, contrast], axis=1).astype(np.float32)
    # Map normalisation is already robust per shot; centring improves MLP fit.
    x[:, : coarse.shape[1] + center.shape[1]] = (x[:, : coarse.shape[1] + center.shape[1]] - 0.25) * 2.0
    x[~np.isfinite(x)] = 0.0
    return x


EVIDENCE_VECTOR_SIZE = EVIDENCE_CHANNELS * (9 + 9 + 5)


@dataclass
class EvidenceModelV2235:
    kind: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _forward_features(self, x: np.ndarray, *, need_cache: bool = False):
        if self.kind == "linear":
            score = x @ self.arrays["w"] + self.arrays["b"][0]
            return (score, (x,)) if need_cache else score
        if self.kind == "mlp":
            h = np.tanh(x @ self.arrays["w1"] + self.arrays["b1"])
            score = (h @ self.arrays["w2"] + self.arrays["b2"]).reshape(-1)
            return (score, (x, h)) if need_cache else score
        raise ValueError(f"Unknown evidence model kind: {self.kind}")

    def score_patches(self, patches: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
        n = len(patches)
        out = np.empty(n, dtype=np.float32)
        step = max(1, int(batch_size))
        for start in range(0, n, step):
            stop = min(n, start + step)
            x = evidence_vector(patches[start:stop])
            out[start:stop] = np.asarray(self._forward_features(x), dtype=np.float32)
        return out

    def rank_indices(self, patches: np.ndarray) -> np.ndarray:
        return np.argsort(-self.score_patches(patches), kind="stable")

    def save(self, directory: Path | str) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz = directory / "model.npz"
        js = directory / "model.json"
        np.savez_compressed(npz, **self.arrays)
        js.write_text(json.dumps({
            "schema_version": "2.23.5-evidence-model-1",
            "kind": self.kind,
            "metadata": self.metadata,
        }, indent=2, sort_keys=True), encoding="utf-8")
        return npz, js

    @classmethod
    def load(cls, directory: Path | str) -> "EvidenceModelV2235":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz", allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(str(meta["kind"]), arrays, dict(meta.get("metadata", {})))


@dataclass
class SampledEvidenceShotV2235:
    session_id: str
    shot_id: str
    patches: np.ndarray
    distances: np.ndarray


def _init_model(kind: str, *, hidden: int, seed: int) -> EvidenceModelV2235:
    rng = np.random.default_rng(seed)
    d = EVIDENCE_VECTOR_SIZE
    if kind == "linear":
        arrays = {"w": np.zeros(d, np.float32), "b": np.zeros(1, np.float32)}
    elif kind == "mlp":
        arrays = {
            "w1": rng.normal(0.0, math.sqrt(2.0 / d) * 0.45, (d, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "w2": rng.normal(0.0, math.sqrt(2.0 / max(1, hidden)) * 0.30, (hidden, 1)).astype(np.float32),
            "b2": np.zeros(1, np.float32),
        }
    else:
        raise ValueError("kind must be linear or mlp")
    return EvidenceModelV2235(kind, arrays, {"live_authority": False})


def _feature_index(name: str) -> int | None:
    try:
        return tuple(REDUCER_FEATURE_NAMES).index(name)
    except ValueError:
        return None


def _fixed_negative_indices(shot, *, rng: np.random.Generator, each: int = 48, random_neg: int = 64) -> np.ndarray:
    neg = np.flatnonzero(shot.dense.distances > PATCH_NEGATIVE_RADIUS_PX)
    if len(neg) == 0:
        return np.zeros(0, dtype=np.int64)
    selected: list[np.ndarray] = []
    for name in ("dense_score", "dense_map_percentile_max", "v2233_newhole_heuristic"):
        fi = _feature_index(name)
        if fi is not None:
            order = neg[np.argsort(-shot.dense.features[neg, fi], kind="stable")[:each]]
            selected.append(order)
    # GT-free image-hard negatives: strongest registered-map centre response.
    centre_strength = shot.patches[neg, :, 4, 4].astype(np.float32).mean(axis=1)
    selected.append(neg[np.argsort(-centre_strength, kind="stable")[:each]])
    chosen = np.unique(np.concatenate(selected)) if selected else np.zeros(0, dtype=np.int64)
    remaining = np.setdiff1d(neg, chosen, assume_unique=False)
    if len(remaining) and random_neg > 0:
        take = min(int(random_neg), len(remaining))
        chosen = np.unique(np.concatenate([chosen, rng.choice(remaining, size=take, replace=False)]))
    return chosen.astype(np.int64)


def build_training_samples(
    refs: Sequence[EvidenceShotRefV2235],
    *,
    seed: int,
    mined: Mapping[tuple[str, int], np.ndarray] | None = None,
    mined_limit: int = 320,
) -> list[SampledEvidenceShotV2235]:
    rng = np.random.default_rng(seed)
    out: list[SampledEvidenceShotV2235] = []
    for idx, ref in enumerate(refs, start=1):
        shot = load_evidence_shot(ref)
        # Candidate positives are tight: the evidence event must be centred in
        # the local patch.  6..42px is deliberately neutral, not negative.
        pos = np.flatnonzero(shot.dense.distances <= PATCH_POSITIVE_RADIUS_PX)
        if len(pos):
            pos = pos[np.argsort(shot.dense.distances[pos], kind="stable")[:6]]
        fixed = _fixed_negative_indices(shot, rng=rng)
        mined_idx = np.asarray((mined or {}).get((ref.session_id, ref.sequence), np.zeros(0, np.int64)), dtype=np.int64)
        if len(mined_idx):
            mined_idx = mined_idx[: max(0, int(mined_limit))]
        neg = np.unique(np.concatenate([fixed, mined_idx])) if len(mined_idx) else fixed
        # Strict safety: model-mined rows still must be definitely NOT current-new-hole.
        neg = neg[(neg >= 0) & (neg < len(shot.dense.distances))]
        neg = neg[shot.dense.distances[neg] > PATCH_NEGATIVE_RADIUS_PX]

        parts = [shot.anchors]
        dparts = [shot.anchor_distances]
        if len(pos):
            parts.append(shot.patches[pos])
            dparts.append(shot.dense.distances[pos].astype(np.float32))
        if len(neg):
            parts.append(shot.patches[neg])
            dparts.append(shot.dense.distances[neg].astype(np.float32))
        out.append(SampledEvidenceShotV2235(
            ref.session_id,
            ref.shot_id,
            np.concatenate(parts, axis=0),
            np.concatenate(dparts, axis=0).astype(np.float32),
        ))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(
                f"[V2.23.5 SAMPLE] {idx}/{len(refs)} shot={ref.shot_id} "
                f"anchors={len(shot.anchors)} candidate_pos={len(pos)} negatives={len(neg)} mined={len(mined_idx)}"
            )
    return out


def _pair_ds(scores: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, float] | None:
    pos = np.flatnonzero(distances <= PATCH_POSITIVE_RADIUS_PX)
    neg = np.flatnonzero(distances > PATCH_NEGATIVE_RADIUS_PX)
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = scores[pos][:, None] - scores[neg][None, :]
    weight = np.exp(-(distances[pos][:, None] ** 2) / (2.0 * 3.0 * 3.0)).astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(np.clip(diff, -40.0, 40.0)))
    g = -sig * weight
    denom = max(float(weight.sum()) * len(neg), 1.0)
    g /= denom
    ds = np.zeros(len(scores), np.float32)
    ds[pos] = np.sum(g, axis=1)
    ds[neg] = -np.sum(g, axis=0)
    loss = float(np.mean(np.logaddexp(0.0, -diff) * weight))
    return ds, loss


def _grads(model: EvidenceModelV2235, x: np.ndarray, ds: np.ndarray, *, l2: float) -> dict[str, np.ndarray]:
    if model.kind == "linear":
        return {
            "w": x.T @ ds + l2 * model.arrays["w"],
            "b": np.asarray([ds.sum()], np.float32),
        }
    scores, cache = model._forward_features(x, need_cache=True)
    xx, h = cache
    dz = (ds[:, None] @ model.arrays["w2"].T) * (1.0 - h * h)
    return {
        "w2": h.T @ ds[:, None] + l2 * model.arrays["w2"],
        "b2": np.asarray([ds.sum()], np.float32),
        "w1": xx.T @ dz + l2 * model.arrays["w1"],
        "b1": dz.sum(axis=0),
    }


def train_stage(
    model: EvidenceModelV2235,
    samples: Sequence[SampledEvidenceShotV2235],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    stage_name: str,
    progress: Callable[[str, int, int, float], None] | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.arrays.items()}
    vv = {k: np.zeros_like(v) for k, v in model.arrays.items()}
    step = 0
    history: list[float] = []
    for epoch in range(int(epochs)):
        losses: list[float] = []
        for sidx in rng.permutation(len(samples)):
            sample = samples[int(sidx)]
            patches = sample.patches
            aug = patches
            # Registered evidence is orientation-independent; flips cheaply
            # prevent the learner from keying on scene direction.
            if rng.random() < 0.5:
                aug = aug[:, :, :, ::-1]
            if rng.random() < 0.5:
                aug = aug[:, :, ::-1, :]
            x = evidence_vector(aug)
            scores = np.asarray(model._forward_features(x), dtype=np.float32)
            pair = _pair_ds(scores, sample.distances)
            if pair is None:
                continue
            ds, loss = pair
            losses.append(loss)
            grads = _grads(model, x, ds, l2=l2)
            step += 1
            for key, grad in grads.items():
                grad = np.clip(np.asarray(grad, dtype=np.float32), -5.0, 5.0)
                m[key] = 0.9 * m[key] + 0.1 * grad
                vv[key] = 0.999 * vv[key] + 0.001 * (grad * grad)
                mh = m[key] / (1.0 - 0.9 ** step)
                vh = vv[key] / (1.0 - 0.999 ** step)
                model.arrays[key] -= learning_rate * mh / (np.sqrt(vh) + 1e-7)
        loss = float(np.mean(losses)) if losses else float("nan")
        history.append(loss)
        if progress and (epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(1, epochs // 5) == 0):
            progress(stage_name, epoch + 1, epochs, loss)
    return history


def mine_hard_negatives(
    model: EvidenceModelV2235,
    refs: Sequence[EvidenceShotRefV2235],
    *,
    per_shot: int,
    label: str,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    mined: dict[tuple[str, int], np.ndarray] = {}
    ranks20: list[int] = []
    retained512 = 0
    oracle20 = 0
    for idx, ref in enumerate(refs, start=1):
        shot = load_evidence_shot(ref)
        scores = model.score_patches(shot.patches)
        neg = np.flatnonzero(shot.dense.distances > PATCH_NEGATIVE_RADIUS_PX)
        hard = neg[np.argsort(-scores[neg], kind="stable")[: min(int(per_shot), len(neg))]] if len(neg) else np.zeros(0, np.int64)
        mined[(ref.session_id, ref.sequence)] = hard.astype(np.int64)
        if shot.dense.oracle20:
            oracle20 += 1
            order = np.argsort(-scores, kind="stable")
            rank = next((r for r, j in enumerate(order, 1) if shot.dense.distances[int(j)] <= 20.0), len(order) + 1)
            ranks20.append(rank)
            retained512 += int(rank <= 512)
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.5 MINE] {label} {idx}/{len(refs)} shot={ref.shot_id} hard={len(hard)}")
    stats = {
        "oracle20": oracle20,
        "median_positive_rank20": float(np.median(ranks20)) if ranks20 else None,
        "retention20_at_512": retained512 / oracle20 if oracle20 else 0.0,
        "mined_per_shot": int(per_shot),
    }
    return mined, stats


def train_evidence_model(
    refs: Sequence[EvidenceShotRefV2235],
    *,
    kind: str,
    hidden: int,
    quick: bool,
    learning_rate: float,
    seed: int,
    progress: Callable[[str, int, int, float], None] | None = None,
) -> tuple[EvidenceModelV2235, dict[str, Any]]:
    if len(refs) < 8:
        raise ValueError("Need >=8 evidence training shots")
    model = _init_model(kind, hidden=hidden, seed=seed)
    stage_epochs = (7, 5, 5) if quick else (18, 12, 12)
    histories: dict[str, list[float]] = {}

    s1 = build_training_samples(refs, seed=seed)
    histories["stage1"] = train_stage(model, s1, epochs=stage_epochs[0], learning_rate=learning_rate, l2=0.0008, seed=seed+10, stage_name="stage1", progress=progress)

    mined1, mine1_stats = mine_hard_negatives(model, refs, per_shot=256, label="round1")
    s2 = build_training_samples(refs, seed=seed+1, mined=mined1, mined_limit=256)
    histories["stage2"] = train_stage(model, s2, epochs=stage_epochs[1], learning_rate=learning_rate*0.65, l2=0.0008, seed=seed+20, stage_name="stage2", progress=progress)

    mined2, mine2_stats = mine_hard_negatives(model, refs, per_shot=384, label="round2")
    s3 = build_training_samples(refs, seed=seed+2, mined=mined2, mined_limit=384)
    histories["stage3"] = train_stage(model, s3, epochs=stage_epochs[2], learning_rate=learning_rate*0.42, l2=0.0010, seed=seed+30, stage_name="stage3", progress=progress)

    model.metadata = {
        "schema_version": "2.23.5-evidence-model-1",
        "kind": kind,
        "hidden": hidden if kind == "mlp" else 0,
        "training_shots": len(refs),
        "positive_radius_px": PATCH_POSITIVE_RADIUS_PX,
        "neutral_band_px": [PATCH_POSITIVE_RADIUS_PX, PATCH_NEGATIVE_RADIUS_PX],
        "negative_radius_gt_px": PATCH_NEGATIVE_RADIUS_PX,
        "hard_negative_rounds": [256, 384],
        "gt_anchors_training_only": True,
        "candidate_pool_modified_by_gt": False,
        "live_authority": False,
        "seed": seed,
    }
    return model, {"histories": histories, "mine_round1": mine1_stats, "mine_round2": mine2_stats}


def evaluate_evidence_model(
    model: EvidenceModelV2235,
    refs: Sequence[EvidenceShotRefV2235],
    *,
    ks: Sequence[int] = (32, 64, 128, 256, 512, 1024),
) -> dict[str, Any]:
    oracle20 = oracle42 = top1_20 = top1_42 = 0
    ranks20: list[int] = []
    ranks42: list[int] = []
    kept20 = {int(k): 0 for k in ks}
    kept42 = {int(k): 0 for k in ks}
    for idx, ref in enumerate(refs, start=1):
        shot = load_evidence_shot(ref)
        scores = model.score_patches(shot.patches)
        order = np.argsort(-scores, kind="stable")
        has20 = shot.dense.oracle20
        has42 = shot.dense.oracle42
        oracle20 += int(has20)
        oracle42 += int(has42)
        if len(order):
            top1_20 += int(shot.dense.distances[int(order[0])] <= 20.0)
            top1_42 += int(shot.dense.distances[int(order[0])] <= 42.0)
        if has20:
            rank = next((r for r, j in enumerate(order, 1) if shot.dense.distances[int(j)] <= 20.0), len(order) + 1)
            ranks20.append(rank)
            for k in ks:
                kept20[int(k)] += int(rank <= int(k))
        if has42:
            rank = next((r for r, j in enumerate(order, 1) if shot.dense.distances[int(j)] <= 42.0), len(order) + 1)
            ranks42.append(rank)
            for k in ks:
                kept42[int(k)] += int(rank <= int(k))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.5 EVAL] {idx}/{len(refs)} shot={ref.shot_id}")
    n = len(refs)
    return {
        "shots": n,
        "oracle20": oracle20,
        "oracle20_rate": oracle20 / n if n else 0.0,
        "oracle42": oracle42,
        "oracle42_rate": oracle42 / n if n else 0.0,
        "top1_20": top1_20,
        "top1_20_rate": top1_20 / n if n else 0.0,
        "conditional_top1_20_rate": top1_20 / oracle20 if oracle20 else 0.0,
        "top1_42": top1_42,
        "conditional_top1_42_rate": top1_42 / oracle42 if oracle42 else 0.0,
        "median_positive_rank20": float(np.median(ranks20)) if ranks20 else None,
        "p90_positive_rank20": float(np.percentile(ranks20, 90)) if ranks20 else None,
        "median_positive_rank42": float(np.median(ranks42)) if ranks42 else None,
        "retention20_at_k": {str(k): kept20[int(k)] / oracle20 if oracle20 else 0.0 for k in ks},
        "retention42_at_k": {str(k): kept42[int(k)] / oracle42 if oracle42 else 0.0 for k in ks},
    }
