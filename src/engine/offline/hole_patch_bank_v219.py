from __future__ import annotations

"""Camera-domain hole appearance bank for V2.19.

The historical ``content/ai/holes/synt_*`` patches are centred by design and
were photographed through the real projector/surface/camera chain.  V2.19 uses
those patches as *appearance reservoirs*, not as centred classification input.
A compact local residual around the known centre is extracted and can be
transplanted onto arbitrary media backgrounds.
"""

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HolePatchAssetV219:
    image_path: str
    metadata_path: str = ""
    background_mode: str = "unknown"
    image_type: str = "synt"
    session_id: str = "unknown"
    width: int = 128
    height: int = 128


class HolePatchBankV219:
    def __init__(self, assets: Sequence[HolePatchAssetV219], *, root: Path = Path("."), residual_cache_size: int = 512) -> None:
        self.assets = list(assets)
        self.root = Path(root)
        self.residual_cache_size = max(0, int(residual_cache_size))
        self._residual_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    @classmethod
    def discover(
        cls,
        directory: Path = Path("content/ai/holes"),
        *,
        root: Path = Path("."),
        include_real: bool = False,
    ) -> "HolePatchBankV219":
        directory = Path(directory)
        assets: list[HolePatchAssetV219] = []
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
                rel_image = str(image)
                rel_meta = str(meta) if meta.exists() else ""
                assets.append(
                    HolePatchAssetV219(
                        image_path=rel_image,
                        metadata_path=rel_meta,
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

    def choose(self, rng: random.Random) -> HolePatchAssetV219 | None:
        if not self.assets:
            return None
        return self.assets[rng.randrange(len(self.assets))]

    def load_gray(self, asset: HolePatchAssetV219) -> np.ndarray:
        path = Path(asset.image_path)
        if not path.is_absolute():
            path = self.root / path
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read hole patch {path}")
        return image

    @staticmethod
    def compact_residual(
        patch: np.ndarray,
        *,
        radius_px: float = 14.0,
        feather_px: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return additive camera-domain residual + soft compact mask.

        A large smooth background estimate removes most of the source patch's
        original white/grid illumination.  A radial mask prevents unrelated
        source-background structure from being transplanted far from the hole.
        """
        gray = np.asarray(patch, dtype=np.float32)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError("Hole patch must be a 2D grayscale image")
        h, w = gray.shape
        # Use a strong smooth estimate.  Gaussian avoids median-kernel limits on
        # small patches and preserves both dark core and bright torn-paper edge
        # in the residual.
        sigma = max(3.5, min(h, w) / 18.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        residual = np.clip(gray - background, -110.0, 110.0).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radius = max(3.0, float(radius_px))
        feather = max(1.0, float(feather_px))
        mask = np.clip((radius + feather - distance) / feather, 0.0, 1.0).astype(np.float32)
        # Give the actual central impact region full trust.
        mask[distance <= radius] = 1.0
        residual *= mask
        return residual, mask

    def sample_residual(self, rng: random.Random) -> tuple[np.ndarray, np.ndarray, HolePatchAssetV219] | None:
        asset = self.choose(rng)
        if asset is None:
            return None
        key = asset.image_path
        cached = self._residual_cache.get(key)
        if cached is not None:
            self._residual_cache.move_to_end(key)
            residual, mask = cached
            return residual, mask, asset
        patch = self.load_gray(asset)
        residual, mask = self.compact_residual(patch)
        if self.residual_cache_size > 0:
            self._residual_cache[key] = (residual, mask)
            self._residual_cache.move_to_end(key)
            while len(self._residual_cache) > self.residual_cache_size:
                self._residual_cache.popitem(last=False)
        return residual, mask, asset

    @property
    def cache_entries(self) -> int:
        return len(self._residual_cache)


def apply_hole_residual(
    frame_bgr: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Add a compact camera-domain hole residual at ``x,y``."""
    out = frame_bgr.astype(np.float32, copy=True)
    h, w = residual.shape
    cx, cy = int(round(float(x))), int(round(float(y)))
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(out.shape[1], x1), min(out.shape[0], y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return frame_bgr.copy()
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    r = residual[src_y0:src_y1, src_x0:src_x1] * mask[src_y0:src_y1, src_x0:src_x1] * float(strength)
    for channel in range(3):
        out[dst_y0:dst_y1, dst_x0:dst_x1, channel] += r
    return np.clip(out, 0, 255).astype(np.uint8)
