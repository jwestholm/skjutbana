from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]


@dataclass
class SyntheticHole:
    hole_id: str
    x: float
    y: float
    radius_px: float
    kind: str
    rotation_deg: float
    strength: float
    opacity: float
    shadow_offset: Tuple[float, float]
    seed: int
    bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))


class SyntheticHoleOverlay:
    """
    Transparent overlay layer with 0..n fake bullet holes.

    Tuned for projector->camera AI training:
    - mostly dark and fairly sharp holes
    - some smaller/weaker hits
    - slight variation downwards in size, but not larger than the current good range
    """

    DEFAULT_KIND_WEIGHTS = {
        "clean_hole": 0.66,
        "torn_hole": 0.16,
        "ragged_hole": 0.10,
        "dent_ring": 0.06,
        "weak_indent": 0.02,
    }

    def __init__(
        self,
        width: int,
        height: int,
        *,
        rng_seed: Optional[int] = None,
        default_radius_px_range: Tuple[float, float] = (1.6, 2.8),
        blur_sigma_range: Tuple[float, float] = (0.03, 0.12),
        shadow_opacity: float = 0.12,
        edge_brighten: float = 0.16,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.default_radius_px_range = default_radius_px_range
        self.blur_sigma_range = blur_sigma_range
        self.shadow_opacity = float(shadow_opacity)
        self.edge_brighten = float(edge_brighten)

        self._rng = random.Random(rng_seed)
        self._holes: Dict[str, SyntheticHole] = {}
        self._dirty = True
        self._cached_overlay_bgra: Optional[np.ndarray] = None

    def add_hole(
        self,
        x: float,
        y: float,
        *,
        kind: Optional[str] = None,
        radius_px: Optional[float] = None,
        rotation_deg: Optional[float] = None,
        strength: Optional[float] = None,
        opacity: Optional[float] = None,
        hole_id: Optional[str] = None,
    ) -> str:
        hole_id = hole_id or self._make_id()
        kind = kind or self._weighted_random_kind()
        radius_px = float(radius_px if radius_px is not None else self._rand_range(*self.default_radius_px_range))
        rotation_deg = float(rotation_deg if rotation_deg is not None else self._rand_range(0.0, 360.0))
        strength = float(strength if strength is not None else self._rand_range(0.82, 1.18))
        opacity = float(opacity if opacity is not None else self._rand_range(0.94, 1.0))

        seed = self._rng.randint(0, 2**31 - 1)
        local_rng = random.Random(seed)
        shadow_offset = (
            local_rng.uniform(-0.10 * radius_px, 0.18 * radius_px),
            local_rng.uniform(-0.10 * radius_px, 0.18 * radius_px),
        )

        hole = SyntheticHole(
            hole_id=hole_id,
            x=float(x),
            y=float(y),
            radius_px=radius_px,
            kind=kind,
            rotation_deg=rotation_deg,
            strength=strength,
            opacity=opacity,
            shadow_offset=shadow_offset,
            seed=seed,
        )
        hole.bbox = self._estimate_bbox(hole)
        self._holes[hole_id] = hole
        self._dirty = True
        return hole_id

    def clear(self) -> None:
        self._holes.clear()
        self._dirty = True

    def render_overlay_bgra(self) -> np.ndarray:
        if self._dirty or self._cached_overlay_bgra is None:
            overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            for hole in self._holes.values():
                self._draw_hole_to_overlay(overlay, hole)
            self._cached_overlay_bgra = overlay
            self._dirty = False
        return self._cached_overlay_bgra.copy()

    def composite_on(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError(
                f"Frame size {frame_bgr.shape[1]}x{frame_bgr.shape[0]} does not match overlay "
                f"size {self.width}x{self.height}."
            )
        overlay = self.render_overlay_bgra()
        return self._alpha_blend_bgra_on_bgr(frame_bgr, overlay)

    def _draw_hole_to_overlay(self, overlay_bgra: np.ndarray, hole: SyntheticHole) -> None:
        sprite = self._build_hole_sprite(hole)
        if sprite is None:
            return

        sh, sw = sprite.shape[:2]
        cx = int(round(hole.x))
        cy = int(round(hole.y))
        x0 = cx - sw // 2
        y0 = cy - sh // 2
        x1 = x0 + sw
        y1 = y0 + sh

        if x1 <= 0 or y1 <= 0 or x0 >= self.width or y0 >= self.height:
            return

        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        src_x1 = sw - max(0, x1 - self.width)
        src_y1 = sh - max(0, y1 - self.height)

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        src = sprite[src_y0:src_y1, src_x0:src_x1]
        dst = overlay_bgra[dst_y0:dst_y1, dst_x0:dst_x1]
        overlay_bgra[dst_y0:dst_y1, dst_x0:dst_x1] = self._alpha_over(dst, src)

    def _build_hole_sprite(self, hole: SyntheticHole) -> Optional[np.ndarray]:
        local_rng = random.Random(hole.seed)
        r = max(0.85, hole.radius_px)
        canvas_radius = int(math.ceil(r * 3.0))
        size = max(12, canvas_radius * 2 + 1)
        center = (size // 2, size // 2)

        alpha = np.zeros((size, size), dtype=np.float32)
        bgr = np.zeros((size, size, 3), dtype=np.float32)

        if hole.kind == "clean_hole":
            self._draw_clean_hole(alpha, bgr, center, r, hole, local_rng)
        elif hole.kind == "torn_hole":
            self._draw_torn_hole(alpha, bgr, center, r, hole, local_rng)
        elif hole.kind == "ragged_hole":
            self._draw_ragged_hole(alpha, bgr, center, r, hole, local_rng)
        elif hole.kind == "dent_ring":
            self._draw_dent_ring(alpha, bgr, center, r, hole, local_rng)
        elif hole.kind == "weak_indent":
            self._draw_weak_indent(alpha, bgr, center, r, hole, local_rng)
        else:
            self._draw_clean_hole(alpha, bgr, center, r, hole, local_rng)

        sigma = local_rng.uniform(*self.blur_sigma_range)
        if sigma > 0.01:
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
            for c in range(3):
                bgr[..., c] = cv2.GaussianBlur(bgr[..., c], (0, 0), sigmaX=sigma, sigmaY=sigma)

        alpha *= hole.opacity
        alpha = np.clip(alpha, 0.0, 1.0)
        bgr = np.clip(bgr, 0.0, 255.0)
        return np.dstack([bgr, alpha * 255.0]).astype(np.uint8)

    def _draw_clean_hole(self, alpha, bgr, center, r, hole, rng) -> None:
        cx, cy = center
        core_r = max(1.0, r * rng.uniform(0.78, 0.96))
        ring_r = r * rng.uniform(1.0, 1.12)

        cv2.circle(alpha, (cx, cy), int(round(core_r)), hole.opacity, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), int(round(core_r)), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        ring_mask = np.zeros_like(alpha)
        cv2.circle(
            ring_mask,
            (cx, cy),
            int(round(ring_r)),
            1.0,
            thickness=max(1, int(round(r * 0.24))),
            lineType=cv2.LINE_AA,
        )
        alpha[:] = np.maximum(alpha, ring_mask * hole.opacity * 0.76)
        for c in range(3):
            bgr[..., c] = np.maximum(bgr[..., c], ring_mask * 255.0 * (0.76 + self.edge_brighten * hole.strength))

        self._add_shadow(alpha, bgr, center, core_r, hole.shadow_offset, strength=0.08)

    def _draw_torn_hole(self, alpha, bgr, center, r, hole, rng) -> None:
        self._draw_irregular_penetration(alpha, bgr, center, r, hole, rng, spike_count=rng.randint(3, 5), spike_scale=0.32)

    def _draw_ragged_hole(self, alpha, bgr, center, r, hole, rng) -> None:
        self._draw_irregular_penetration(alpha, bgr, center, r, hole, rng, spike_count=rng.randint(5, 7), spike_scale=0.42)

    def _draw_irregular_penetration(self, alpha, bgr, center, r, hole, rng, *, spike_count: int, spike_scale: float) -> None:
        cx, cy = center
        points: List[Tuple[int, int]] = []
        base = r * rng.uniform(0.70, 0.88)
        angle0 = math.radians(hole.rotation_deg)

        for i in range(spike_count * 2):
            a = angle0 + (math.pi * 2.0 * i / (spike_count * 2))
            rr = base * rng.uniform(0.84, 1.06)
            if i % 2 == 1:
                rr *= 1.0 + rng.uniform(0.08, spike_scale)
            px = int(round(cx + math.cos(a) * rr))
            py = int(round(cy + math.sin(a) * rr))
            points.append((px, py))

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(alpha, [pts], hole.opacity)
        cv2.fillPoly(bgr, [pts], (0, 0, 0))

        ring = np.zeros_like(alpha)
        cv2.polylines(ring, [pts], isClosed=True, color=1.0, thickness=max(1, int(round(r * 0.34))), lineType=cv2.LINE_AA)
        alpha[:] = np.maximum(alpha, ring * hole.opacity * 0.76)
        for c in range(3):
            bgr[..., c] = np.maximum(bgr[..., c], ring * 255.0 * (0.78 + self.edge_brighten * hole.strength))

        if rng.random() < 0.40:
            a = angle0 + rng.uniform(-0.40, 0.40)
            tear_len = r * rng.uniform(0.20, 0.60)
            p1 = (int(round(cx + math.cos(a) * base * 0.8)), int(round(cy + math.sin(a) * base * 0.8)))
            p2 = (int(round(cx + math.cos(a) * (base + tear_len))), int(round(cy + math.sin(a) * (base + tear_len))))
            cv2.line(alpha, p1, p2, hole.opacity * 0.25, thickness=1, lineType=cv2.LINE_AA)
            cv2.line(bgr, p1, p2, (8, 8, 8), thickness=1, lineType=cv2.LINE_AA)

        self._add_shadow(alpha, bgr, center, base, hole.shadow_offset, strength=0.10)

    def _draw_dent_ring(self, alpha, bgr, center, r, hole, rng) -> None:
        cx, cy = center
        ring_r = r * rng.uniform(0.95, 1.08)
        thickness = max(1, int(round(r * rng.uniform(0.16, 0.25))))

        ring_mask = np.zeros_like(alpha)
        cv2.circle(ring_mask, (cx, cy), int(round(ring_r)), 1.0, thickness=thickness, lineType=cv2.LINE_AA)

        center_mask = np.zeros_like(alpha)
        cv2.circle(center_mask, (cx, cy), int(round(r * 0.40)), 1.0, thickness=-1, lineType=cv2.LINE_AA)

        alpha[:] = np.maximum(alpha, ring_mask * hole.opacity * 0.40)
        alpha[:] = np.maximum(alpha, center_mask * hole.opacity * 0.18)

        for c in range(3):
            bgr[..., c] = np.maximum(bgr[..., c], ring_mask * 255.0 * (0.74 + self.edge_brighten * 0.5 * hole.strength))
            bgr[..., c] = np.maximum(bgr[..., c], center_mask * 56.0)

    def _draw_weak_indent(self, alpha, bgr, center, r, hole, rng) -> None:
        cx, cy = center
        axes = (int(round(r * rng.uniform(0.70, 1.00))), int(round(r * rng.uniform(0.52, 0.82))))
        angle = hole.rotation_deg

        ring_mask = np.zeros_like(alpha)
        cv2.ellipse(ring_mask, (cx, cy), axes, angle, 0, 360, 1.0, thickness=max(1, int(round(r * 0.15))), lineType=cv2.LINE_AA)

        dent_mask = np.zeros_like(alpha)
        cv2.ellipse(
            dent_mask,
            (cx, cy),
            (max(1, int(axes[0] * 0.55)), max(1, int(axes[1] * 0.55))),
            angle,
            0,
            360,
            1.0,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        alpha[:] = np.maximum(alpha, ring_mask * hole.opacity * 0.24)
        alpha[:] = np.maximum(alpha, dent_mask * hole.opacity * 0.10)

        for c in range(3):
            bgr[..., c] = np.maximum(bgr[..., c], ring_mask * 255.0 * (0.66 + self.edge_brighten * 0.25))
            bgr[..., c] = np.maximum(bgr[..., c], dent_mask * 40.0)

    def _add_shadow(self, alpha, bgr, center, r, offset, *, strength: float) -> None:
        if strength <= 0:
            return

        shadow = np.zeros_like(alpha)
        sx = int(round(center[0] + offset[0]))
        sy = int(round(center[1] + offset[1]))
        cv2.circle(shadow, (sx, sy), max(1, int(round(r))), 1.0, thickness=-1, lineType=cv2.LINE_AA)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(0.28, r * 0.08), sigmaY=max(0.28, r * 0.08))

        alpha[:] = np.maximum(alpha, shadow * self.shadow_opacity * strength)
        darkness = shadow * 16.0 * strength
        for c in range(3):
            bgr[..., c] = np.maximum(bgr[..., c], darkness)

    def _estimate_bbox(self, hole: SyntheticHole) -> Tuple[int, int, int, int]:
        pad = int(math.ceil(hole.radius_px * 3.0))
        x0 = int(math.floor(hole.x - pad))
        y0 = int(math.floor(hole.y - pad))
        x1 = int(math.ceil(hole.x + pad))
        y1 = int(math.ceil(hole.y + pad))
        return x0, y0, x1, y1

    def _weighted_random_kind(self) -> str:
        kinds = list(self.DEFAULT_KIND_WEIGHTS.keys())
        weights = list(self.DEFAULT_KIND_WEIGHTS.values())
        return self._rng.choices(kinds, weights=weights, k=1)[0]

    def _make_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _rand_range(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    @staticmethod
    def _alpha_over(dst_bgra: np.ndarray, src_bgra: np.ndarray) -> np.ndarray:
        dst = dst_bgra.astype(np.float32) / 255.0
        src = src_bgra.astype(np.float32) / 255.0

        src_a = src[..., 3:4]
        dst_a = dst[..., 3:4]
        out_a = src_a + dst_a * (1.0 - src_a)

        denom = np.maximum(out_a, 1e-6)
        out_rgb = (src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)) / denom
        out = np.concatenate([out_rgb, out_a], axis=-1)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _alpha_blend_bgra_on_bgr(frame_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
        base = frame_bgr.astype(np.float32)
        over = overlay_bgra[..., :3].astype(np.float32)
        alpha = overlay_bgra[..., 3:4].astype(np.float32) / 255.0
        out = over * alpha + base * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
