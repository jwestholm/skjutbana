from __future__ import annotations

"""Hole stamp engine for V2.20.

V2.19 pasted a signed residual from historical ``synt_*`` camera patches.
That preserved too much source-context and frequently produced black/white
streaks instead of compact hole appearances.

V2.20 still *uses* the historical patch bank, but only to estimate plausible
hole statistics (dark-core size/strength, torn-paper rim strength).  The final
rendered appearance is a compact procedural hole stamp, which is far more
stable across arbitrary media backgrounds.
"""

import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HolePatchAssetV220:
    image_path: str
    metadata_path: str = ""
    background_mode: str = "unknown"
    image_type: str = "synt"
    session_id: str = "unknown"
    width: int = 128
    height: int = 128


@dataclass(frozen=True)
class HolePatchStatsV220:
    core_radius_px: float
    core_depth: float
    rim_strength: float
    rim_width_px: float
    source_id: str


@dataclass(frozen=True)
class HoleStampV220:
    delta: np.ndarray  # float32 HxWx3 (preferred) or HxW signed additive change
    mask: np.ndarray   # float32 HxW compact support mask
    source_id: str
    core_radius_px: float
    core_depth: float
    rim_strength: float
    rim_width_px: float


class HolePatchBankV220:
    def __init__(self, assets: Sequence[HolePatchAssetV220], *, root: Path = Path("."), stats_cache_size: int = 512) -> None:
        self.assets = list(assets)
        self.root = Path(root)
        self.stats_cache_size = max(0, int(stats_cache_size))
        self._stats_cache: OrderedDict[str, HolePatchStatsV220] = OrderedDict()

    @classmethod
    def discover(
        cls,
        directory: Path = Path("content/ai/holes"),
        *,
        root: Path = Path("."),
        include_real: bool = False,
    ) -> "HolePatchBankV220":
        directory = Path(directory)
        assets: list[HolePatchAssetV220] = []
        patterns = ["synt_*.png"] + (["hole_*.png"] if include_real else [])
        for pattern in patterns:
            for image in sorted(directory.glob(pattern)):
                meta = image.with_suffix(".json")
                payload: dict[str, Any] = {}
                if meta.exists():
                    try:
                        row = json.loads(meta.read_text(encoding="utf-8"))
                        if isinstance(row, dict):
                            payload = row
                    except Exception:
                        pass
                assets.append(
                    HolePatchAssetV220(
                        image_path=str(image),
                        metadata_path=str(meta) if meta.exists() else "",
                        background_mode=str(payload.get("background_mode") or "unknown"),
                        image_type=str(payload.get("image_type") or ("hole" if image.name.startswith("hole_") else "synt")),
                        session_id=str(payload.get("session_id") or "unknown"),
                        width=int((payload.get("patch_size") or [128, 128])[0]) if isinstance(payload.get("patch_size"), list) else 128,
                        height=int((payload.get("patch_size") or [128, 128])[1]) if isinstance(payload.get("patch_size"), list) else 128,
                    )
                )
        return cls(assets, root=root)

    def __len__(self) -> int:
        return len(self.assets)

    def choose(self, rng: random.Random) -> HolePatchAssetV220 | None:
        if not self.assets:
            return None
        return self.assets[rng.randrange(len(self.assets))]

    def load_gray(self, asset: HolePatchAssetV220) -> np.ndarray:
        path = Path(asset.image_path)
        if not path.is_absolute():
            path = self.root / path
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read hole patch {path}")
        return image

    @staticmethod
    def _estimate_stats_from_patch(patch: np.ndarray, *, source_id: str) -> HolePatchStatsV220:
        gray = np.asarray(patch, dtype=np.float32)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError("Hole patch must be a 2D grayscale image")
        h, w = gray.shape
        sigma = max(4.0, min(h, w) / 16.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        residual = gray - background

        cy, cx = h // 2, w // 2
        search = 18
        y0, y1 = max(0, cy - search), min(h, cy + search + 1)
        x0, x1 = max(0, cx - search), min(w, cx + search + 1)
        local = residual[y0:y1, x0:x1]
        min_pos = np.unravel_index(int(np.argmin(local)), local.shape)
        py, px = y0 + int(min_pos[0]), x0 + int(min_pos[1])
        dark_peak = float(max(5.0, -residual[py, px]))

        thresh = -max(4.0, dark_peak * 0.42)
        binary = (residual < thresh).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        core_area = 0.0
        if 0 <= py < labels.shape[0] and 0 <= px < labels.shape[1]:
            label = int(labels[py, px])
            if label > 0:
                core_area = float(stats[label, cv2.CC_STAT_AREA])
        if core_area <= 0.0:
            core_radius = 4.5
        else:
            core_radius = math.sqrt(core_area / math.pi)
        core_radius = float(np.clip(core_radius, 2.4, 8.5))

        annulus = []
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt((xx - float(px)) ** 2 + (yy - float(py)) ** 2)
        annulus_mask = (rr >= core_radius * 0.9) & (rr <= core_radius * 2.5)
        if np.any(annulus_mask):
            annulus = residual[annulus_mask]
        rim_strength = float(np.percentile(annulus, 92)) if len(annulus) else 9.0
        rim_strength = float(np.clip(rim_strength, 2.0, 28.0))
        core_depth = float(np.clip(dark_peak * 1.7, 18.0, 78.0))
        rim_width = float(np.clip(core_radius * 0.65, 1.2, 3.6))
        return HolePatchStatsV220(
            core_radius_px=core_radius,
            core_depth=core_depth,
            rim_strength=rim_strength,
            rim_width_px=rim_width,
            source_id=source_id,
        )

    def _stats_for_asset(self, asset: HolePatchAssetV220) -> HolePatchStatsV220:
        key = asset.image_path
        cached = self._stats_cache.get(key)
        if cached is not None:
            self._stats_cache.move_to_end(key)
            return cached
        patch = self.load_gray(asset)
        stats = self._estimate_stats_from_patch(patch, source_id=Path(asset.image_path).name)
        if self.stats_cache_size > 0:
            self._stats_cache[key] = stats
            self._stats_cache.move_to_end(key)
            while len(self._stats_cache) > self.stats_cache_size:
                self._stats_cache.popitem(last=False)
        return stats

    @staticmethod
    def _render_stamp_from_stats(stats: HolePatchStatsV220, rng: random.Random) -> HoleStampV220:
        core_radius = float(np.clip(stats.core_radius_px * rng.uniform(0.92, 1.10), 2.3, 8.5))
        core_depth = float(np.clip(stats.core_depth * rng.uniform(0.90, 1.08), 18.0, 82.0))
        rim_strength = float(np.clip(stats.rim_strength * rng.uniform(0.85, 1.15), 2.0, 30.0))
        rim_width = float(np.clip(stats.rim_width_px * rng.uniform(0.85, 1.25), 1.0, 4.0))

        outer = core_radius + rim_width + 4.0
        size = int(math.ceil(outer * 2.0)) | 1
        half = size // 2
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        rr = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)

        irregular1 = 1.0 + 0.10 * np.cos(theta * 2.0 + rng.uniform(0, math.tau))
        irregular2 = 1.0 + 0.07 * np.sin(theta * 3.0 + rng.uniform(0, math.tau))
        irregular = np.clip(irregular1 * irregular2, 0.78, 1.28)
        local_core = core_radius * irregular

        core_alpha = np.clip(1.0 - (rr / np.maximum(local_core, 1e-3)) ** 1.85, 0.0, 1.0)
        center_radius = max(1.0, core_radius * rng.uniform(0.25, 0.45))
        center_drop = np.exp(-0.5 * (rr / center_radius) ** 2)
        punch = np.exp(-0.5 * (rr / max(1.0, core_radius * 0.75)) ** 2)
        dark = -core_depth * (0.60 * core_alpha + 0.26 * punch + 0.14 * center_drop)

        ring_center = core_radius + rim_width * rng.uniform(0.20, 0.55)
        ring_sigma = max(0.65, rim_width * rng.uniform(0.40, 0.75))
        ring = np.exp(-0.5 * ((rr - ring_center) / ring_sigma) ** 2)
        ring *= (rr >= max(0.0, core_radius * 0.72)).astype(np.float32)
        tear_gate = np.clip(0.88 + 0.20 * np.cos(theta + rng.uniform(0, math.tau)) + 0.10 * np.sin(theta * 2.0 + rng.uniform(0, math.tau)), 0.45, 1.25)
        ring *= tear_gate
        rim = rim_strength * ring

        # Add a subtle bright frayed-paper edge and a few lighter flecks inside the
        # hole so the result does not read like a flat black blob.
        tex_rng = np.random.default_rng(rng.randrange(2**32))
        small_h = max(4, size // 3)
        small_w = max(4, size // 3)
        texture = tex_rng.normal(0.0, 1.0, size=(small_h, small_w)).astype(np.float32)
        texture = cv2.resize(texture, (size, size), interpolation=cv2.INTER_CUBIC)
        texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=0.65, sigmaY=0.65)
        texture = (texture - float(texture.min())) / max(1e-6, float(texture.max() - texture.min()))

        fray_inner = np.clip((rr - core_radius * 0.50) / max(0.8, rim_width * 0.9), 0.0, 1.0)
        fray_outer = np.clip(1.0 - (rr - core_radius) / max(1.0, rim_width * 1.9), 0.0, 1.0)
        fray = fray_inner * fray_outer * (0.55 + 0.45 * texture)
        fleck_zone = (rr <= core_radius * 0.95).astype(np.float32) * np.clip(texture - 0.76, 0.0, 0.35)

        shadow = -0.18 * core_depth * np.exp(-0.5 * (rr / max(1.0, core_radius * 1.35)) ** 2)
        support = np.maximum(core_alpha, np.clip(ring * 0.9 + fray * 0.55, 0.0, 1.0))
        support = cv2.GaussianBlur(support.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)
        support = np.clip(support, 0.0, 1.0)

        paper_tint = np.array([0.82, 0.92, 1.00], dtype=np.float32).reshape(1, 1, 3)  # warm paper rim, BGR
        soot_tint = np.array([1.00, 0.99, 0.97], dtype=np.float32).reshape(1, 1, 3)
        neutral = np.ones((1, 1, 3), dtype=np.float32)

        core_rgb = dark[..., None] * soot_tint
        rim_rgb = (rim[..., None] * paper_tint)
        fray_rgb = ((0.34 * rim_strength) * fray)[..., None] * paper_tint
        fleck_rgb = ((1.8 + 0.07 * core_depth) * fleck_zone)[..., None] * neutral
        shadow_rgb = shadow[..., None] * neutral

        delta = (core_rgb + rim_rgb + fray_rgb + fleck_rgb + shadow_rgb).astype(np.float32)
        delta *= support[..., None]
        return HoleStampV220(
            delta=delta,
            mask=support,
            source_id=stats.source_id,
            core_radius_px=core_radius,
            core_depth=core_depth,
            rim_strength=rim_strength,
            rim_width_px=rim_width,
        )

    def sample_stamp(self, rng: random.Random) -> HoleStampV220 | None:
        asset = self.choose(rng)
        if asset is None:
            return None
        stats = self._stats_for_asset(asset)
        return self._render_stamp_from_stats(stats, rng)

    @staticmethod
    def default_stamp(rng: random.Random, *, radius_range: tuple[float, float] = (3.0, 6.0)) -> HoleStampV220:
        lo, hi = radius_range
        core_radius = float(rng.uniform(lo, hi))
        stats = HolePatchStatsV220(
            core_radius_px=core_radius,
            core_depth=float(rng.uniform(28.0, 68.0)),
            rim_strength=float(rng.uniform(4.0, 16.0)),
            rim_width_px=float(rng.uniform(1.2, 2.8)),
            source_id="procedural_default",
        )
        return HolePatchBankV220._render_stamp_from_stats(stats, rng)

    @property
    def cache_entries(self) -> int:
        return len(self._stats_cache)


def apply_hole_stamp_inplace(
    frame_bgr: np.ndarray,
    stamp: HoleStampV220,
    x: float,
    y: float,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a compact hole stamp in-place, touching only its small ROI.

    V2.20 originally converted/copied the *entire* camera frame to float for
    every old hole.  At 2K/4K with tens of holes that dominated generation
    time.  This path converts only the stamp-sized ROI.
    """
    delta = stamp.delta
    mask = stamp.mask
    if delta.ndim == 2:
        h, w = delta.shape
    else:
        h, w = delta.shape[:2]
    cx, cy = int(round(float(x))), int(round(float(y)))
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(frame_bgr.shape[1], x1), min(frame_bgr.shape[0], y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return frame_bgr
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    local_mask = mask[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    if delta.ndim == 2:
        d = delta[src_y0:src_y1, src_x0:src_x1].astype(np.float32)[..., None] * local_mask[..., None] * float(strength)
    else:
        d = delta[src_y0:src_y1, src_x0:src_x1].astype(np.float32) * local_mask[..., None] * float(strength)
    roi = frame_bgr[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    roi += d
    frame_bgr[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return frame_bgr


def apply_hole_stamp(
    frame_bgr: np.ndarray,
    stamp: HoleStampV220,
    x: float,
    y: float,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Copy-on-write convenience wrapper around :func:`apply_hole_stamp_inplace`."""
    out = frame_bgr.copy()
    return apply_hole_stamp_inplace(out, stamp, x, y, strength=strength)
