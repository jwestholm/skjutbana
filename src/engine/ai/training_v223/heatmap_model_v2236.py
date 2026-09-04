from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .heatmap_v2236 import (
    HEATMAP_CHANNELS,
    HEATMAP_STRIDE,
    HeatmapShotRefV2236,
    HeatmapShotV2236,
    camera_to_coarse,
    coarse_to_camera,
    load_heatmap_shot,
)

KERNEL_SIZE = 5
POSITIVE_RADIUS_PX = 6.5
NEGATIVE_RADIUS_PX = 42.0
EPS = 1e-8


@dataclass
class HeatmapModelV2236:
    kind: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "HeatmapModelV2236":
        return HeatmapModelV2236(self.kind, {k: np.asarray(v).copy() for k, v in self.arrays.items()}, copy.deepcopy(self.metadata))

    def score_patches(self, patches: np.ndarray) -> np.ndarray:
        p = np.asarray(patches, dtype=np.float32) / 255.0
        if p.ndim != 4 or p.shape[1:] != (HEATMAP_CHANNELS, KERNEL_SIZE, KERNEL_SIZE):
            raise ValueError(f"Expected [N,{HEATMAP_CHANNELS},{KERNEL_SIZE},{KERNEL_SIZE}], got {p.shape}")
        if self.kind == "linear_conv":
            return np.einsum("nchw,chw->n", p, self.arrays["kernel"], optimize=True) + float(self.arrays["bias"][0])
        if self.kind == "spatial_conv":
            hidden = np.einsum("nchw,fchw->nf", p, self.arrays["kernel"], optimize=True) + self.arrays["bias"][None, :]
            hidden = np.tanh(hidden)
            return hidden @ self.arrays["out_w"] + float(self.arrays["out_b"][0])
        raise ValueError(f"Unknown heatmap model kind: {self.kind}")

    def score_map(self, maps: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for full-map V2.23.6 inference") from exc
        src = np.asarray(maps, dtype=np.float32) / 255.0
        if src.ndim != 3 or src.shape[0] != HEATMAP_CHANNELS:
            raise ValueError(f"Expected [{HEATMAP_CHANNELS},H,W], got {src.shape}")
        border = cv2.BORDER_REFLECT_101
        if self.kind == "linear_conv":
            out = np.full(src.shape[1:], float(self.arrays["bias"][0]), dtype=np.float32)
            for c in range(HEATMAP_CHANNELS):
                out += cv2.filter2D(src[c], cv2.CV_32F, self.arrays["kernel"][c], borderType=border)
            return out
        if self.kind == "spatial_conv":
            filters = int(self.arrays["kernel"].shape[0])
            out = np.full(src.shape[1:], float(self.arrays["out_b"][0]), dtype=np.float32)
            for f in range(filters):
                h = np.full(src.shape[1:], float(self.arrays["bias"][f]), dtype=np.float32)
                for c in range(HEATMAP_CHANNELS):
                    h += cv2.filter2D(src[c], cv2.CV_32F, self.arrays["kernel"][f, c], borderType=border)
                np.tanh(h, out=h)
                out += h * float(self.arrays["out_w"][f])
            return out
        raise ValueError(f"Unknown heatmap model kind: {self.kind}")

    def save(self, directory: Path | str) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz = directory / "model.npz"
        js = directory / "model.json"
        np.savez_compressed(npz, **self.arrays)
        js.write_text(json.dumps({
            "schema_version": "2.23.6-heatmap-model-1",
            "kind": self.kind,
            "metadata": self.metadata,
        }, indent=2, sort_keys=True), encoding="utf-8")
        return npz, js

    @classmethod
    def load(cls, directory: Path | str) -> "HeatmapModelV2236":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "model.npz", allow_pickle=False) as data:
            arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(str(meta["kind"]), arrays, dict(meta.get("metadata", {})))


@dataclass
class HeatmapTrainingSampleV2236:
    session_id: str
    shot_id: str
    patches: np.ndarray
    distances: np.ndarray


def init_heatmap_model(kind: str, *, hidden: int = 8, seed: int = 2360) -> HeatmapModelV2236:
    rng = np.random.default_rng(seed)
    scale = 0.08
    if kind == "linear_conv":
        arrays = {
            "kernel": rng.normal(0.0, scale, size=(HEATMAP_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float32),
            "bias": np.zeros(1, np.float32),
        }
    elif kind == "spatial_conv":
        h = max(2, int(hidden))
        arrays = {
            "kernel": rng.normal(0.0, 0.06, size=(h, HEATMAP_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float32),
            "bias": np.zeros(h, np.float32),
            "out_w": rng.normal(0.0, 0.08, size=h).astype(np.float32),
            "out_b": np.zeros(1, np.float32),
        }
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return HeatmapModelV2236(kind, arrays, {"live_authority": False})


def extract_grid_patches(maps: np.ndarray, coarse_xy: np.ndarray) -> np.ndarray:
    src = np.asarray(maps, dtype=np.uint8)
    pts = np.asarray(coarse_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"coarse_xy must be [N,2], got {pts.shape}")
    half = KERNEL_SIZE // 2
    c, h, w = src.shape
    pad = np.pad(src, ((0, 0), (half, half), (half, half)), mode="reflect")
    out = np.empty((len(pts), c, KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    offsets = np.arange(-half, half + 1, dtype=np.int32)
    xs = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, w - 1) + half
    ys = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, h - 1) + half
    yy = ys[:, None, None] + offsets[None, :, None]
    xx = xs[:, None, None] + offsets[None, None, :]
    for ci in range(c):
        out[:, ci] = pad[ci][yy, xx]
    return out


def _camera_distance_for_coarse(points: np.ndarray, gt_xy: tuple[float, float]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    xs = pts[:, 0] * HEATMAP_STRIDE + (HEATMAP_STRIDE - 1) * 0.5
    ys = pts[:, 1] * HEATMAP_STRIDE + (HEATMAP_STRIDE - 1) * 0.5
    return np.sqrt((xs - gt_xy[0]) ** 2 + (ys - gt_xy[1]) ** 2).astype(np.float32)


def _local_maxima(score: np.ndarray, *, top_k: int, nms_radius: int = 4, allowed: np.ndarray | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV required") from exc
    s = np.asarray(score, dtype=np.float32)
    valid = np.isfinite(s)
    if allowed is not None:
        valid &= np.asarray(allowed, dtype=bool)
    work = np.where(valid, s, -np.inf)
    dil = cv2.dilate(np.where(np.isfinite(work), work, -1e30).astype(np.float32), np.ones((3, 3), np.uint8))
    mask = valid & (work >= dil - 1e-7)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.float32)
        flat = int(np.nanargmax(np.where(valid, s, -np.inf)))
        y, x = np.unravel_index(flat, s.shape)
        return np.asarray([[x, y]], dtype=np.float32)
    vals = work[ys, xs]
    order = np.argsort(-vals, kind="stable")
    chosen: list[tuple[int, int]] = []
    r2 = float(max(1, nms_radius) ** 2)
    for oi in order:
        x = int(xs[int(oi)]); y = int(ys[int(oi)])
        if any((x - xx) ** 2 + (y - yy) ** 2 < r2 for xx, yy in chosen):
            continue
        chosen.append((x, y))
        if len(chosen) >= int(top_k):
            break
    return np.asarray(chosen, dtype=np.float32)


def peak_camera_xy(score: np.ndarray, *, top_k: int = 5, nms_radius_px: float = 24.0) -> np.ndarray:
    coarse = _local_maxima(score, top_k=top_k, nms_radius=max(1, int(round(nms_radius_px / HEATMAP_STRIDE))))
    if len(coarse) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    offset = (HEATMAP_STRIDE - 1) * 0.5
    out = coarse.copy()
    out[:, 0] = coarse[:, 0] * HEATMAP_STRIDE + offset
    out[:, 1] = coarse[:, 1] * HEATMAP_STRIDE + offset
    return out.astype(np.float32)


def _initial_negative_points(shot: HeatmapShotV2236, *, random_count: int, hard_count: int, seed: int) -> np.ndarray:
    maps = shot.maps.astype(np.float32)
    h, w = maps.shape[1:]
    gx, gy = camera_to_coarse(*shot.gt_xy)
    yy, xx = np.ogrid[:h, :w]
    dist_px = np.sqrt(((xx - gx) * HEATMAP_STRIDE) ** 2 + ((yy - gy) * HEATMAP_STRIDE) ** 2)
    allowed = dist_px > NEGATIVE_RADIUS_PX

    # Strong deterministic false evidence: maxima of max-channel, fused and
    # compact/persistence mixtures. GT only masks the neutral/positive area.
    fused = maps[6]
    maxch = maps.max(axis=0)
    mix = 0.45 * maps[2] + 0.25 * maps[4] + 0.20 * maps[7] + 0.10 * maps[6]
    coords: list[np.ndarray] = []
    each = max(8, int(hard_count) // 3)
    for score in (fused, maxch, mix):
        coords.append(_local_maxima(score, top_k=each, nms_radius=3, allowed=allowed))

    rng = np.random.default_rng(seed)
    flat_allowed = np.flatnonzero(allowed.ravel())
    if len(flat_allowed):
        take = min(int(random_count), len(flat_allowed))
        picked = rng.choice(flat_allowed, size=take, replace=False)
        ry, rx = np.unravel_index(picked, (h, w))
        coords.append(np.stack([rx, ry], axis=1).astype(np.float32))
    if not coords:
        return np.zeros((0, 2), np.float32)
    allp = np.concatenate(coords, axis=0)
    # stable unique integer grid coordinates
    seen: set[tuple[int, int]] = set(); out: list[tuple[int, int]] = []
    for p in allp:
        key = (int(round(float(p[0]))), int(round(float(p[1]))))
        if key in seen:
            continue
        seen.add(key); out.append(key)
    return np.asarray(out, dtype=np.float32)


def build_training_samples(
    refs: Sequence[HeatmapShotRefV2236], *, seed: int, mined: Mapping[tuple[str, int], np.ndarray] | None = None,
) -> list[HeatmapTrainingSampleV2236]:
    samples: list[HeatmapTrainingSampleV2236] = []
    for idx, ref in enumerate(refs, start=1):
        shot = load_heatmap_shot(ref)
        gx, gy = camera_to_coarse(*shot.gt_xy)
        cx = int(round(gx)); cy = int(round(gy))
        pos_points = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x = cx + dx; y = cy + dy
                if x < 0 or y < 0 or x >= shot.maps.shape[2] or y >= shot.maps.shape[1]:
                    continue
                cam = coarse_to_camera(x, y)
                if math.hypot(cam[0] - shot.gt_xy[0], cam[1] - shot.gt_xy[1]) <= POSITIVE_RADIUS_PX:
                    pos_points.append((x, y))
        if not pos_points:
            pos_points = [(max(0, min(shot.maps.shape[2]-1, cx)), max(0, min(shot.maps.shape[1]-1, cy)))]
        neg = _initial_negative_points(shot, random_count=160, hard_count=144, seed=seed + ref.sequence * 17)
        mined_pts = np.asarray((mined or {}).get((ref.session_id, ref.sequence), np.zeros((0, 2), np.float32)), dtype=np.float32)
        if len(mined_pts):
            neg = np.concatenate([neg, mined_pts], axis=0)
        pts = np.concatenate([np.asarray(pos_points, np.float32), neg], axis=0)
        distances = _camera_distance_for_coarse(pts, shot.gt_xy)
        keep = (distances <= POSITIVE_RADIUS_PX) | (distances > NEGATIVE_RADIUS_PX)
        pts = pts[keep]; distances = distances[keep]
        patches = extract_grid_patches(shot.maps, pts)
        samples.append(HeatmapTrainingSampleV2236(ref.session_id, ref.shot_id, patches, distances))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(
                f"[V2.23.6 SAMPLE] {idx}/{len(refs)} shot={ref.shot_id} "
                f"positive={int(np.sum(distances <= POSITIVE_RADIUS_PX))} negative={int(np.sum(distances > NEGATIVE_RADIUS_PX))} mined={len(mined_pts)}"
            )
    return samples


def _pair_gradient(scores: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, float] | None:
    pos = np.flatnonzero(distances <= POSITIVE_RADIUS_PX)
    neg = np.flatnonzero(distances > NEGATIVE_RADIUS_PX)
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = scores[pos][:, None] - scores[neg][None, :]
    sig = 1.0 / (1.0 + np.exp(np.clip(diff, -40.0, 40.0)))
    # Prefer anchors nearest GT but retain all <=6.5px positives.
    pw = np.exp(-(distances[pos] ** 2) / (2.0 * 3.0 * 3.0)).astype(np.float32)
    g = -sig * pw[:, None]
    denom = max(float(pw.sum()) * len(neg), 1.0)
    g /= denom
    ds = np.zeros(len(scores), np.float32)
    ds[pos] = np.sum(g, axis=1)
    ds[neg] = -np.sum(g, axis=0)
    loss = float(np.mean(np.logaddexp(0.0, -diff) * pw[:, None]))
    return ds, loss


def _model_grads(model: HeatmapModelV2236, patches: np.ndarray, ds: np.ndarray, *, l2: float) -> dict[str, np.ndarray]:
    x = np.asarray(patches, dtype=np.float32) / 255.0
    if model.kind == "linear_conv":
        return {
            "kernel": np.einsum("n,nchw->chw", ds, x, optimize=True) + l2 * model.arrays["kernel"],
            "bias": np.asarray([ds.sum()], np.float32),
        }
    hidden_pre = np.einsum("nchw,fchw->nf", x, model.arrays["kernel"], optimize=True) + model.arrays["bias"][None, :]
    hidden = np.tanh(hidden_pre)
    dh = ds[:, None] * model.arrays["out_w"][None, :] * (1.0 - hidden * hidden)
    return {
        "out_w": hidden.T @ ds + l2 * model.arrays["out_w"],
        "out_b": np.asarray([ds.sum()], np.float32),
        "kernel": np.einsum("nf,nchw->fchw", dh, x, optimize=True) + l2 * model.arrays["kernel"],
        "bias": dh.sum(axis=0),
    }


def train_stage(
    model: HeatmapModelV2236,
    samples: Sequence[HeatmapTrainingSampleV2236],
    *, epochs: int, learning_rate: float, l2: float, seed: int, stage_name: str,
    progress: Callable[[str, int, int, float], None] | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.arrays.items()}
    vv = {k: np.zeros_like(v) for k, v in model.arrays.items()}
    step = 0; history: list[float] = []
    for epoch in range(int(epochs)):
        losses = []
        for sidx in rng.permutation(len(samples)):
            sample = samples[int(sidx)]
            patches = sample.patches
            # Evidence geometry is orientation independent; flips are safe.
            aug = patches
            if rng.random() < 0.5: aug = aug[:, :, :, ::-1]
            if rng.random() < 0.5: aug = aug[:, :, ::-1, :]
            scores = model.score_patches(aug)
            pair = _pair_gradient(scores, sample.distances)
            if pair is None: continue
            ds, loss = pair; losses.append(loss)
            grads = _model_grads(model, aug, ds, l2=l2)
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


def mine_heatmap_negatives(
    model: HeatmapModelV2236, refs: Sequence[HeatmapShotRefV2236], *, per_shot: int = 96,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    mined: dict[tuple[str, int], np.ndarray] = {}
    top1_20 = 0; errors = []
    for idx, ref in enumerate(refs, start=1):
        shot = load_heatmap_shot(ref)
        score = model.score_map(shot.maps)
        gx, gy = camera_to_coarse(*shot.gt_xy)
        yy, xx = np.ogrid[:score.shape[0], :score.shape[1]]
        dist = np.sqrt(((xx - gx) * HEATMAP_STRIDE) ** 2 + ((yy - gy) * HEATMAP_STRIDE) ** 2)
        allowed = dist > NEGATIVE_RADIUS_PX
        hard = _local_maxima(score, top_k=per_shot, nms_radius=3, allowed=allowed)
        mined[(ref.session_id, ref.sequence)] = hard
        peaks = peak_camera_xy(score, top_k=1)
        if len(peaks):
            d = math.hypot(float(peaks[0,0]) - shot.gt_xy[0], float(peaks[0,1]) - shot.gt_xy[1])
            errors.append(d); top1_20 += int(d <= 20.0)
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.6 MINE] {idx}/{len(refs)} shot={ref.shot_id} hard={len(hard)}")
    return mined, {
        "shots": len(refs),
        "top1_20_rate_before_mining": top1_20 / len(refs) if refs else 0.0,
        "median_error_before_mining": float(np.median(errors)) if errors else None,
        "mined_per_shot": int(per_shot),
    }


def _snap_peak_to_dense(peak: np.ndarray, shot: HeatmapShotV2236, radius_px: float = 18.0) -> np.ndarray:
    if shot.dense_xy.size == 0:
        return peak
    delta = shot.dense_xy - peak[None, :]
    d = np.sqrt(np.sum(delta * delta, axis=1))
    idx = int(np.argmin(d))
    if float(d[idx]) <= radius_px:
        return shot.dense_xy[idx].astype(np.float32)
    return peak


def _metric_from_score(score: np.ndarray, shot: HeatmapShotV2236) -> dict[str, Any]:
    peaks = peak_camera_xy(score, top_k=5)
    ds = [math.hypot(float(p[0]) - shot.gt_xy[0], float(p[1]) - shot.gt_xy[1]) for p in peaks]
    snapped = [_snap_peak_to_dense(p, shot) for p in peaks]
    sds = [math.hypot(float(p[0]) - shot.gt_xy[0], float(p[1]) - shot.gt_xy[1]) for p in snapped]
    return {
        "top1_error": ds[0] if ds else float("inf"),
        "top3_error": min(ds[:3]) if ds else float("inf"),
        "top5_error": min(ds[:5]) if ds else float("inf"),
        "snap_top1_error": sds[0] if sds else float("inf"),
        "snap_top3_error": min(sds[:3]) if sds else float("inf"),
    }


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(metrics)
    top1 = np.asarray([float(m["top1_error"]) for m in metrics], np.float32)
    top3 = np.asarray([float(m["top3_error"]) for m in metrics], np.float32)
    top5 = np.asarray([float(m["top5_error"]) for m in metrics], np.float32)
    snap1 = np.asarray([float(m["snap_top1_error"]) for m in metrics], np.float32)
    snap3 = np.asarray([float(m["snap_top3_error"]) for m in metrics], np.float32)
    def rate(a, r): return float(np.mean(a <= r)) if len(a) else 0.0
    return {
        "shots": n,
        "top1_at5": rate(top1, 5.0),
        "top1_at10": rate(top1, 10.0),
        "top1_at20": rate(top1, 20.0),
        "top1_at42": rate(top1, 42.0),
        "top3_at20": rate(top3, 20.0),
        "top3_at42": rate(top3, 42.0),
        "top5_at20": rate(top5, 20.0),
        "median_error_px": float(np.median(top1)) if n else None,
        "p90_error_px": float(np.percentile(top1, 90)) if n else None,
        "p95_error_px": float(np.percentile(top1, 95)) if n else None,
        "snap_top1_at20": rate(snap1, 20.0),
        "snap_top3_at20": rate(snap3, 20.0),
        "snap_median_error_px": float(np.median(snap1)) if n else None,
    }


def evaluate_heatmap_model(model: HeatmapModelV2236, refs: Sequence[HeatmapShotRefV2236], *, label: str = "eval") -> dict[str, Any]:
    rows = []
    for idx, ref in enumerate(refs, start=1):
        shot = load_heatmap_shot(ref)
        rows.append(_metric_from_score(model.score_map(shot.maps), shot))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.6 EVAL] {label} {idx}/{len(refs)} shot={ref.shot_id}")
    return _aggregate(rows)


def evaluate_heatmap_baselines(refs: Sequence[HeatmapShotRefV2236]) -> dict[str, dict[str, Any]]:
    names = {
        "fused": lambda m: m[6].astype(np.float32),
        "max_channel": lambda m: m.max(axis=0).astype(np.float32),
        "mean_channel": lambda m: m.mean(axis=0).astype(np.float32),
        "physical_mix": lambda m: (0.45*m[2] + 0.25*m[4] + 0.20*m[7] + 0.10*m[6]).astype(np.float32),
    }
    acc: dict[str, list[dict[str, Any]]] = {k: [] for k in names}
    for idx, ref in enumerate(refs, start=1):
        shot = load_heatmap_shot(ref)
        mf = shot.maps.astype(np.float32)
        for name, fn in names.items():
            acc[name].append(_metric_from_score(fn(mf), shot))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.6 BASELINE] {idx}/{len(refs)} shot={ref.shot_id}")
    return {name: _aggregate(rows) for name, rows in acc.items()}


def objective(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    med = metrics.get("median_error_px")
    return (
        float(metrics.get("top1_at20", 0.0)),
        float(metrics.get("top3_at20", 0.0)),
        float(metrics.get("top1_at42", 0.0)),
        -(float(med) if med is not None else 1e9),
    )
