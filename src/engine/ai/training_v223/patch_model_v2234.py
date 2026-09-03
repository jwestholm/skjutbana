from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .dense_v2233 import REDUCER_FEATURE_NAMES
from .patch_v2234 import PATCH_CHANNELS, PATCH_SIZE, PatchShotRefV2234, extract_gt_anchor_patches, load_patch_shot

EPS = 1e-8


def _norm_patch_u8(patches: np.ndarray) -> np.ndarray:
    p = np.asarray(patches, dtype=np.float32)
    if p.ndim != 4 or p.shape[1:] != (PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Expected [B,{PATCH_CHANNELS},{PATCH_SIZE},{PATCH_SIZE}], got {p.shape}")
    out = np.empty_like(p, dtype=np.float32)
    out[:, 0] = (p[:, 0] - 128.0) / 64.0
    out[:, 1] = (p[:, 1] - 128.0) / 64.0
    out[:, 2] = p[:, 2] / 64.0
    out[:, 3] = (p[:, 3] - 128.0) / 64.0
    out[:, 4] = p[:, 4] / 255.0
    return np.clip(out, -4.0, 4.0)


def _windows3(x: np.ndarray) -> np.ndarray:
    xp = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(xp, (3,3), axis=(2,3))


@dataclass
class PatchModelV2234:
    kind: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _forward(self, patches: np.ndarray, *, need_cache: bool = False):
        x = _norm_patch_u8(patches)
        if self.kind == "patch_mlp":
            flat = x.reshape(len(x), -1)
            h = np.tanh(flat @ self.arrays["w1"] + self.arrays["b1"])
            score = (h @ self.arrays["w2"] + self.arrays["b2"]).reshape(-1)
            return (score, (flat, h)) if need_cache else score
        if self.kind == "tiny_cnn":
            win = _windows3(x)  # [B,C,H,W,3,3]
            cols = win.transpose(0,2,3,1,4,5).reshape(-1, PATCH_CHANNELS*9)
            wf = self.arrays["conv_w"].reshape(self.arrays["conv_w"].shape[0], -1).T
            zflat = cols @ wf + self.arrays["conv_b"]
            z = zflat.reshape(len(x), PATCH_SIZE, PATCH_SIZE, -1).transpose(0,3,1,2)
            a = np.maximum(z, 0.0)
            pooled = a.reshape(len(x), a.shape[1], PATCH_SIZE//2, 2, PATCH_SIZE//2, 2).mean(axis=(3,5))
            flat = pooled.reshape(len(x), -1)
            h = np.tanh(flat @ self.arrays["fc1_w"] + self.arrays["fc1_b"])
            score = (h @ self.arrays["out_w"] + self.arrays["out_b"]).reshape(-1)
            return (score, (cols, z, flat, h)) if need_cache else score
        raise ValueError(f"Unknown patch model kind: {self.kind}")

    def score_patches(self, patches: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
        n = len(patches)
        out = np.empty(n, dtype=np.float32)
        for start in range(0, n, max(1, int(batch_size))):
            stop = min(n, start + max(1, int(batch_size)))
            out[start:stop] = np.asarray(self._forward(patches[start:stop]), dtype=np.float32)
        return out

    def rank_indices(self, patches: np.ndarray) -> np.ndarray:
        return np.argsort(-self.score_patches(patches), kind="stable")

    def save(self, directory: Path | str) -> tuple[Path, Path]:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        npz = directory / "model.npz"; js = directory / "model.json"
        np.savez_compressed(npz, **self.arrays)
        js.write_text(json.dumps({
            "schema_version": "2.23.4-patchmodel-1",
            "kind": self.kind,
            "metadata": self.metadata,
        }, indent=2, sort_keys=True), encoding="utf-8")
        return npz, js

    @classmethod
    def load(cls, directory: Path | str) -> "PatchModelV2234":
        directory = Path(directory)
        meta = json.loads((directory/"model.json").read_text(encoding="utf-8"))
        with np.load(directory/"model.npz", allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(str(meta["kind"]), arrays, dict(meta.get("metadata", {})))


@dataclass
class SampledPatchShotV2234:
    session_id: str
    shot_id: str
    patches: np.ndarray
    distances: np.ndarray


def _hard_negative_indices(dense, *, rng: np.random.Generator, hard_each: int = 48, random_neg: int = 64) -> np.ndarray:
    neg = np.flatnonzero(dense.distances > 42.0)
    if len(neg) == 0:
        return np.zeros(0, dtype=np.int64)
    names = tuple(REDUCER_FEATURE_NAMES)
    dense_idx = names.index("dense_score") if "dense_score" in names else 0
    rich_idx = names.index("v2233_newhole_heuristic") if "v2233_newhole_heuristic" in names else dense_idx
    a = neg[np.argsort(-dense.features[neg, dense_idx], kind="stable")[:hard_each]]
    b = neg[np.argsort(-dense.features[neg, rich_idx], kind="stable")[:hard_each]]
    chosen = np.unique(np.concatenate([a,b]))
    rem = np.setdiff1d(neg, chosen, assume_unique=False)
    if len(rem) and random_neg > 0:
        take = min(int(random_neg), len(rem))
        chosen = np.unique(np.concatenate([chosen, rng.choice(rem, size=take, replace=False)]))
    return chosen.astype(np.int64)


def build_training_samples(refs: Sequence[PatchShotRefV2234], *, seed: int = 2340) -> list[SampledPatchShotV2234]:
    rng = np.random.default_rng(seed)
    out: list[SampledPatchShotV2234] = []
    for idx, ref in enumerate(refs, start=1):
        shot = load_patch_shot(ref)
        pos = np.flatnonzero(shot.dense.distances <= 20.0)
        if len(pos):
            pos = pos[np.argsort(shot.dense.distances[pos], kind="stable")[:4]]
        neg = _hard_negative_indices(shot.dense, rng=rng)
        parts = []; dparts = []
        # GT anchors are training-only positive examples. They do not repair or
        # alter proposal pools and therefore cannot inflate proposal metrics.
        anchors, adist = extract_gt_anchor_patches(ref)
        parts.append(anchors); dparts.append(adist)
        if len(pos):
            parts.append(shot.patches[pos]); dparts.append(shot.dense.distances[pos].astype(np.float32))
        if len(neg):
            parts.append(shot.patches[neg]); dparts.append(shot.dense.distances[neg].astype(np.float32))
        if parts:
            out.append(SampledPatchShotV2234(ref.session_id, ref.shot_id, np.concatenate(parts, axis=0), np.concatenate(dparts)))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.4 SAMPLE] {idx}/{len(refs)} shot={ref.shot_id} positives={len(anchors)+len(pos)} negatives={len(neg)}")
    return out


def _init_model(kind: str, *, seed: int, hidden: int, filters: int) -> PatchModelV2234:
    rng = np.random.default_rng(seed)
    if kind == "patch_mlp":
        d = PATCH_CHANNELS * PATCH_SIZE * PATCH_SIZE
        arrays = {
            "w1": rng.normal(0.0, math.sqrt(2.0/d)*0.35, (d, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "w2": rng.normal(0.0, math.sqrt(2.0/max(1,hidden))*0.25, (hidden,1)).astype(np.float32),
            "b2": np.zeros(1, np.float32),
        }
    elif kind == "tiny_cnn":
        arrays = {
            "conv_w": rng.normal(0.0, math.sqrt(2.0/(PATCH_CHANNELS*9))*0.45, (filters,PATCH_CHANNELS,3,3)).astype(np.float32),
            "conv_b": np.zeros(filters, np.float32),
            "fc1_w": rng.normal(0.0, math.sqrt(2.0/(filters*(PATCH_SIZE//2)**2))*0.35, (filters*(PATCH_SIZE//2)**2, hidden)).astype(np.float32),
            "fc1_b": np.zeros(hidden, np.float32),
            "out_w": rng.normal(0.0, math.sqrt(2.0/max(1,hidden))*0.25, (hidden,1)).astype(np.float32),
            "out_b": np.zeros(1, np.float32),
        }
    else:
        raise ValueError("kind must be patch_mlp or tiny_cnn")
    return PatchModelV2234(kind, arrays, {"live_authority": False})


def _pair_ds(scores: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, float] | None:
    pos = np.flatnonzero(distances <= 20.0)
    neg = np.flatnonzero(distances > 42.0)
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = scores[pos][:,None] - scores[neg][None,:]
    pair_weight = np.exp(-(distances[pos][:,None]**2)/(2.0*8.0*8.0)).astype(np.float32)
    sig = 1.0/(1.0 + np.exp(np.clip(diff, -40, 40)))
    g = -sig * pair_weight
    denom = max(float(pair_weight.sum()) * len(neg), 1.0)
    g /= denom
    ds = np.zeros(len(scores), np.float32)
    ds[pos] = np.sum(g, axis=1)
    ds[neg] = -np.sum(g, axis=0)
    loss = float(np.mean(np.logaddexp(0.0, -diff) * pair_weight))
    return ds, loss


def _grads(model: PatchModelV2234, patches: np.ndarray, ds: np.ndarray, *, l2: float) -> dict[str, np.ndarray]:
    scores, cache = model._forward(patches, need_cache=True)
    if model.kind == "patch_mlp":
        flat, h = cache
        dz = (ds[:,None] @ model.arrays["w2"].T) * (1.0 - h*h)
        return {
            "w2": h.T @ ds[:,None] + l2*model.arrays["w2"],
            "b2": np.asarray([ds.sum()], np.float32),
            "w1": flat.T @ dz + l2*model.arrays["w1"],
            "b1": dz.sum(axis=0),
        }
    cols, z, flat, h = cache
    dhpre = (ds[:,None] @ model.arrays["out_w"].T) * (1.0 - h*h)
    g = {
        "out_w": h.T @ ds[:,None] + l2*model.arrays["out_w"],
        "out_b": np.asarray([ds.sum()], np.float32),
        "fc1_w": flat.T @ dhpre + l2*model.arrays["fc1_w"],
        "fc1_b": dhpre.sum(axis=0),
    }
    dflat = dhpre @ model.arrays["fc1_w"].T
    f = model.arrays["conv_w"].shape[0]
    dpool = dflat.reshape(len(patches), f, PATCH_SIZE//2, PATCH_SIZE//2)
    da = np.repeat(np.repeat(dpool/4.0, 2, axis=2), 2, axis=3)
    dzc = da * (z > 0.0)
    dzflat = dzc.transpose(0,2,3,1).reshape(-1, f)
    gw = cols.T @ dzflat
    g["conv_w"] = gw.T.reshape(model.arrays["conv_w"].shape) + l2*model.arrays["conv_w"]
    g["conv_b"] = dzflat.sum(axis=0)
    return g


def train_patch_model(
    refs: Sequence[PatchShotRefV2234],
    *,
    kind: str = "tiny_cnn",
    hidden: int = 32,
    filters: int = 6,
    epochs: int = 12,
    learning_rate: float = 0.002,
    l2: float = 0.0008,
    seed: int = 2340,
    progress: Callable[[int,int,float],None] | None = None,
) -> tuple[PatchModelV2234, dict[str, Any]]:
    samples = build_training_samples(refs, seed=seed)
    if len(samples) < 8:
        raise ValueError("Need >=8 patch training shots")
    model = _init_model(kind, seed=seed, hidden=hidden, filters=filters)
    m = {k: np.zeros_like(v) for k,v in model.arrays.items()}
    vv = {k: np.zeros_like(v) for k,v in model.arrays.items()}
    rng = np.random.default_rng(seed+999)
    step = 0; history = []
    for epoch in range(int(epochs)):
        losses = []
        for sidx in rng.permutation(len(samples)):
            sample = samples[int(sidx)]
            patches = sample.patches
            # Lightweight orientation augmentation; labels remain new-hole/not-new.
            aug = patches.copy()
            if rng.random() < 0.5: aug = aug[:,:,:,::-1]
            if rng.random() < 0.5: aug = aug[:,:,::-1,:]
            scores = np.asarray(model._forward(aug), dtype=np.float32)
            pair = _pair_ds(scores, sample.distances)
            if pair is None: continue
            ds, loss = pair; losses.append(loss)
            grads = _grads(model, aug, ds, l2=l2)
            step += 1
            for key, grad in grads.items():
                grad = np.clip(np.asarray(grad, np.float32), -5.0, 5.0)
                m[key] = 0.9*m[key] + 0.1*grad
                vv[key] = 0.999*vv[key] + 0.001*(grad*grad)
                mh = m[key]/(1.0-0.9**step); vh = vv[key]/(1.0-0.999**step)
                model.arrays[key] -= learning_rate*mh/(np.sqrt(vh)+1e-7)
        loss = float(np.mean(losses)) if losses else float("nan")
        history.append(loss)
        if progress and (epoch == 0 or epoch+1 == epochs or (epoch+1) % max(1,epochs//6) == 0):
            progress(epoch+1, epochs, loss)
    model.metadata = {
        "schema_version": "2.23.4-patchmodel-1",
        "kind": kind, "hidden": hidden, "filters": filters if kind=="tiny_cnn" else 0,
        "epochs": epochs, "learning_rate": learning_rate, "l2": l2, "seed": seed,
        "training_shots": len(samples), "gt_anchor_training_only": True,
        "gt_anchor_inserted_into_candidate_pool": False,
        "neutral_band_px": [20.0,42.0], "live_authority": False,
    }
    return model, {"loss_history": history, "training_shots": len(samples)}


def evaluate_patch_model(model: PatchModelV2234, refs: Sequence[PatchShotRefV2234], *, ks: Sequence[int]=(32,64,128,256,512,1024)) -> dict[str, Any]:
    oracle20=oracle42=top1_20=top1_42=0; ranks20=[]; ranks42=[]
    kept20={int(k):0 for k in ks}; kept42={int(k):0 for k in ks}
    for idx, ref in enumerate(refs, start=1):
        shot = load_patch_shot(ref)
        order = model.rank_indices(shot.patches)
        has20=shot.dense.oracle20; has42=shot.dense.oracle42
        oracle20 += int(has20); oracle42 += int(has42)
        if len(order):
            top1_20 += int(shot.dense.distances[int(order[0])] <= 20.0)
            top1_42 += int(shot.dense.distances[int(order[0])] <= 42.0)
        if has20:
            r = next((i for i,j in enumerate(order,1) if shot.dense.distances[int(j)]<=20.0), len(order)+1)
            ranks20.append(r)
            for k in ks: kept20[int(k)] += int(r<=int(k))
        if has42:
            r = next((i for i,j in enumerate(order,1) if shot.dense.distances[int(j)]<=42.0), len(order)+1)
            ranks42.append(r)
            for k in ks: kept42[int(k)] += int(r<=int(k))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.4 EVAL] {idx}/{len(refs)} shot={ref.shot_id}")
    n=len(refs)
    return {
        "shots":n,"oracle20":oracle20,"oracle20_rate":oracle20/n if n else 0.0,
        "oracle42":oracle42,"oracle42_rate":oracle42/n if n else 0.0,
        "top1_20":top1_20,"top1_20_rate":top1_20/n if n else 0.0,
        "conditional_top1_20_rate":top1_20/oracle20 if oracle20 else 0.0,
        "top1_42":top1_42,"conditional_top1_42_rate":top1_42/oracle42 if oracle42 else 0.0,
        "median_positive_rank20":float(np.median(ranks20)) if ranks20 else None,
        "p90_positive_rank20":float(np.percentile(ranks20,90)) if ranks20 else None,
        "median_positive_rank42":float(np.median(ranks42)) if ranks42 else None,
        "retention20_at_k":{str(k):kept20[int(k)]/oracle20 if oracle20 else 0.0 for k in ks},
        "retention42_at_k":{str(k):kept42[int(k)]/oracle42 if oracle42 else 0.0 for k in ks},
    }
