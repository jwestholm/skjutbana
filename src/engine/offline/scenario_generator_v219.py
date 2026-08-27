from __future__ import annotations

"""Deterministic offline world/scenario generator for hit detection V2.19.

This module does not claim to replace the physical projector/camera domain.
Its job is to create *new, labelled perception problems* cheaply so future
training can iterate over millions of seeds while physical sessions remain the
acceptance anchor.
"""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .media_bank_v219 import MediaAssetV219, read_media_manifest
from .hole_patch_bank_v219 import HolePatchBankV219, apply_hole_residual


@dataclass(frozen=True)
class ScenarioProfileV219:
    width: int = 0  # 0 = auto from latest candidate-pack metadata, fallback 3840
    height: int = 0  # 0 = auto from latest candidate-pack metadata, fallback 2160
    pre_frames: int = 3
    post_frames: int = 3
    old_hole_min: int = 0
    old_hole_max: int = 55
    known_hole_fraction: float = 0.70
    new_near_old_probability: float = 0.22
    new_overlap_old_probability: float = 0.035
    edge_bias_probability: float = 0.18
    camera_gain_jitter: float = 0.08
    camera_gamma_jitter: float = 0.10
    sensor_noise_sigma: float = 1.2
    blur_sigma_max: float = 0.55
    media_motion_step: int = 1
    hole_radius_min: float = 3.0
    hole_radius_max: float = 6.0
    use_camera_hole_patch_bank: bool = True

    @classmethod
    def from_file(cls, path: Path | None = None) -> "ScenarioProfileV219":
        path = Path(path or "content/ai/offline_v219.json")
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            section = payload.get("scenario") if isinstance(payload, dict) else None
            if not isinstance(section, dict):
                return cls()
            allowed = {field.name for field in __import__("dataclasses").fields(cls)}
            return cls(**{k: v for k, v in section.items() if k in allowed})
        except Exception:
            return cls()


@dataclass
class ScenarioSpecV219:
    seed: int
    media_id: str
    media_split: str
    media_category: str
    media_kind: str
    media_path: str
    media_frame_index: int
    gt_camera_xy: tuple[float, float]
    old_holes: list[tuple[float, float]]
    known_holes: list[tuple[float, float]]
    challenge_tags: list[str] = field(default_factory=list)
    generator_version: str = "2.19"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["gt_camera_xy"] = [float(self.gt_camera_xy[0]), float(self.gt_camera_xy[1])]
        row["old_holes"] = [[float(x), float(y)] for x, y in self.old_holes]
        row["known_holes"] = [[float(x), float(y)] for x, y in self.known_holes]
        return row


@dataclass
class GeneratedScenarioV219:
    spec: ScenarioSpecV219
    pre_frames: list[np.ndarray]
    recent_pre_frame: np.ndarray
    post_frames: list[np.ndarray]
    projector_pre_frames: list[np.ndarray]
    projector_post_frames: list[np.ndarray]


class _MediaReader:
    def __init__(self, asset: MediaAssetV219, *, root: Path):
        self.asset = asset
        path = Path(asset.path)
        self.path = path if path.is_absolute() else root / path

    @staticmethod
    def _cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Empty media frame")
        h, w = frame.shape[:2]
        scale = max(width / max(1.0, float(w)), height / max(1.0, float(h)))
        nw, nh = max(width, int(math.ceil(w * scale))), max(height, int(math.ceil(h * scale)))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        x0 = max(0, (nw - width) // 2)
        y0 = max(0, (nh - height) // 2)
        return np.ascontiguousarray(resized[y0:y0 + height, x0:x0 + width])

    def frame(self, index: int, width: int, height: int) -> np.ndarray:
        if self.asset.kind == "image":
            image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read media image {self.path}")
            return self._cover(image, width, height)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Could not open media video {self.path}")
        try:
            count = max(1, int(self.asset.frame_count or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
            index = int(index) % count
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not decode media video {self.path}")
            return self._cover(frame, width, height)
        finally:
            cap.release()


def _procedural_frame(name: str, width: int, height: int, phase: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + phase * 7919)
    yy, xx = np.mgrid[0:height, 0:width]
    name = name.lower()
    if name == "white":
        gray = np.full((height, width), 245, dtype=np.float32)
    elif name == "gray":
        gray = np.full((height, width), 132, dtype=np.float32)
    elif name == "black":
        gray = np.full((height, width), 20, dtype=np.float32)
    elif name == "checker":
        cell = max(12, min(width, height) // 18)
        gray = np.where(((xx // cell + yy // cell + phase) % 2) == 0, 42, 220).astype(np.float32)
    elif name == "stripes":
        freq = max(8, width // 40)
        gray = 125 + 95 * np.sin((xx + phase * 7) * 2 * np.pi / freq)
    elif name == "gradient":
        gray = 30 + 210 * ((0.62 * xx / max(1, width - 1)) + (0.38 * yy / max(1, height - 1)))
    elif name == "noise":
        small = rng.normal(128, 44, size=(max(2, height // 24), max(2, width // 24))).astype(np.float32)
        gray = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    elif name == "game_like":
        gray = np.full((height, width), 88, dtype=np.float32)
        # deterministic moving rectangles/circles/text-like edges
        for i in range(18):
            x = int((seed * (i + 3) * 17 + phase * (9 + i)) % max(1, width))
            y = int((seed * (i + 5) * 11 + phase * (5 + i)) % max(1, height))
            rw = 18 + (i * 13) % max(20, width // 7)
            rh = 14 + (i * 9) % max(16, height // 9)
            color = float(35 + (i * 37) % 205)
            cv2.rectangle(gray, (x, y), (min(width - 1, x + rw), min(height - 1, y + rh)), color, -1)
        cv2.putText(gray, "TARGET 42 SCORE", (max(5, width // 12), max(30, height // 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 1100), 225, 2, cv2.LINE_AA)
    else:
        gray = np.full((height, width), 200, dtype=np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class OfflineScenarioGeneratorV219:
    PROCEDURAL = ("white", "gray", "black", "checker", "stripes", "gradient", "noise", "game_like")

    def __init__(
        self,
        *,
        profile: ScenarioProfileV219 | None = None,
        media_assets: Sequence[MediaAssetV219] | None = None,
        media_manifest: Path | None = None,
        repo_root: Path = Path("."),
        hole_bank: HolePatchBankV219 | None = None,
    ) -> None:
        base_profile = profile or ScenarioProfileV219()
        self.repo_root = Path(repo_root)
        width, height = int(base_profile.width), int(base_profile.height)
        if width <= 0 or height <= 0:
            auto_w, auto_h = self._detect_camera_shape(self.repo_root)
            width = width if width > 0 else auto_w
            height = height if height > 0 else auto_h
        import dataclasses
        self.profile = dataclasses.replace(base_profile, width=width, height=height)
        self.media_assets = list(media_assets) if media_assets is not None else read_media_manifest(media_manifest or Path("content/ai/media_bank_v219/media_manifest.jsonl"))
        self.hole_bank = hole_bank if hole_bank is not None else HolePatchBankV219.discover(root=self.repo_root)

    @staticmethod
    def _detect_camera_shape(repo_root: Path) -> tuple[int, int]:
        # Candidate-pack JSON stores shapes even when full-frame arrays were not
        # saved. Prefer the newest capture because it reflects the actual camera
        # mode used by the current shooting PC.
        roots = [repo_root / "content" / "ai" / "candidate_shadow_v216" / "sessions"]
        candidates = []
        for root in roots:
            if root.exists():
                candidates.extend(root.glob("*/shot_*.json"))
        for path in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                shape = (row.get("frame_shapes") or {}).get("post") or (row.get("frame_shapes") or {}).get("pre")
                if isinstance(shape, list) and len(shape) >= 2:
                    h, w = int(shape[0]), int(shape[1])
                    if w > 0 and h > 0:
                        return w, h
            except Exception:
                pass
        return 3840, 2160

    def _select_media(self, rng: random.Random, split: str) -> MediaAssetV219 | None:
        candidates = [asset for asset in self.media_assets if asset.split == split]
        if not candidates:
            return None
        return candidates[rng.randrange(len(candidates))]

    def _choose_position(self, rng: random.Random, old_holes: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], list[str]]:
        p = self.profile
        margin = max(8, int(round(max(p.hole_radius_max * 5.0, 8.0))))
        tags: list[str] = []
        if old_holes and rng.random() < float(p.new_overlap_old_probability):
            ox, oy = rng.choice(list(old_holes))
            tags.append("hole_in_hole")
            return (float(np.clip(ox + rng.uniform(-2.0, 2.0), margin, p.width - margin - 1)), float(np.clip(oy + rng.uniform(-2.0, 2.0), margin, p.height - margin - 1))), tags
        if old_holes and rng.random() < float(p.new_near_old_probability):
            ox, oy = rng.choice(list(old_holes))
            angle = rng.uniform(0.0, math.tau)
            distance = rng.uniform(4.0, 34.0)
            tags.append("near_old_hole")
            return (float(np.clip(ox + math.cos(angle) * distance, margin, p.width - margin - 1)), float(np.clip(oy + math.sin(angle) * distance, margin, p.height - margin - 1))), tags
        if rng.random() < float(p.edge_bias_probability):
            tags.append("near_edge")
            edge = rng.choice(("left", "right", "top", "bottom"))
            if edge in {"left", "right"}:
                x = rng.randint(margin, max(margin, int(p.width * 0.10))) if edge == "left" else rng.randint(max(margin, int(p.width * 0.90)), p.width - margin - 1)
                y = rng.randint(margin, p.height - margin - 1)
            else:
                y = rng.randint(margin, max(margin, int(p.height * 0.10))) if edge == "top" else rng.randint(max(margin, int(p.height * 0.90)), p.height - margin - 1)
                x = rng.randint(margin, p.width - margin - 1)
            return (float(x), float(y)), tags
        return (float(rng.randint(margin, p.width - margin - 1)), float(rng.randint(margin, p.height - margin - 1))), tags

    def _media_frames(self, asset: MediaAssetV219 | None, rng: random.Random, seed: int) -> tuple[list[np.ndarray], list[np.ndarray], str, str, str, str, int]:
        p = self.profile
        total = p.pre_frames + p.post_frames
        if asset is None:
            name = rng.choice(self.PROCEDURAL)
            moving = name in {"checker", "stripes", "noise", "game_like"}
            frames = [_procedural_frame(name, p.width, p.height, i if moving else 0, seed) for i in range(total)]
            return frames[:p.pre_frames], frames[p.pre_frames:], f"procedural:{name}", "procedural", name, "", 0
        reader = _MediaReader(asset, root=self.repo_root)
        max_index = max(1, int(asset.frame_count))
        base = 0 if asset.kind == "image" else rng.randrange(max_index)
        step = max(1, int(p.media_motion_step))
        indices = [base] * total if asset.kind == "image" else [base + (i - p.pre_frames + 1) * step for i in range(total)]
        frames = [reader.frame(index, p.width, p.height) for index in indices]
        return frames[:p.pre_frames], frames[p.pre_frames:], asset.media_id, asset.kind, asset.category, asset.path, int(base)

    @staticmethod
    def _composite(frame: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
        base = frame.astype(np.float32)
        over = overlay_bgra[..., :3].astype(np.float32)
        alpha = overlay_bgra[..., 3:4].astype(np.float32) / 255.0
        return np.clip(over * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)

    def _camera_effect(self, frame: np.ndarray, rng: random.Random, *, shared_gain: float, shared_gamma: float) -> np.ndarray:
        p = self.profile
        gain = shared_gain * rng.uniform(0.985, 1.015)
        gamma = max(0.35, shared_gamma + rng.uniform(-0.015, 0.015))
        normalized = np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0)
        out = 255.0 * np.power(normalized, gamma) * gain
        if p.blur_sigma_max > 0:
            sigma = rng.uniform(0.0, float(p.blur_sigma_max))
            if sigma > 0.04:
                out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if p.sensor_noise_sigma > 0:
            np_rng = np.random.default_rng(rng.randrange(2**32))
            out = out + np_rng.normal(0.0, float(p.sensor_noise_sigma), size=out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _render_holes_from_camera_bank(
        self,
        rng: random.Random,
        backgrounds: Sequence[np.ndarray],
        hole_positions: Sequence[tuple[float, float]],
        *,
        include_new: bool,
        gt_xy: tuple[float, float],
    ) -> tuple[list[np.ndarray], list[str]]:
        frames = [frame.copy() for frame in backgrounds]
        source_ids: list[str] = []
        all_positions = list(hole_positions) + ([gt_xy] if include_new else [])
        for x, y in all_positions:
            sampled = self.hole_bank.sample_residual(rng) if self.hole_bank is not None else None
            if sampled is None:
                return [], []
            residual, mask, asset = sampled
            source_ids.append(Path(asset.image_path).name)
            strength = rng.uniform(0.80, 1.20)
            frames = [apply_hole_residual(frame, residual, mask, x, y, strength=strength) for frame in frames]
        return frames, source_ids

    def generate(self, seed: int, *, split: str = "train") -> GeneratedScenarioV219:
        p = self.profile
        rng = random.Random(int(seed))
        asset = self._select_media(rng, split)
        pre_bg, post_bg, media_id, media_kind, media_category, media_path, base_index = self._media_frames(asset, rng, int(seed))

        old_count = rng.randint(max(0, int(p.old_hole_min)), max(int(p.old_hole_min), int(p.old_hole_max)))
        margin = max(8, int(round(max(p.hole_radius_max * 5.0, 8.0))))
        old_holes = [
            (float(rng.randint(margin, p.width - margin - 1)), float(rng.randint(margin, p.height - margin - 1)))
            for _ in range(old_count)
        ]
        gt_xy, tags = self._choose_position(rng, old_holes)
        known_holes = [xy for xy in old_holes if rng.random() < float(p.known_hole_fraction)]
        if len(known_holes) < len(old_holes):
            tags.append("incomplete_known_holes")
        if old_count >= 35:
            tags.append("dense_old_holes")
        if media_kind == "video" or media_category in {"video", "game", "checker", "stripes", "noise", "game_like"}:
            tags.append("dynamic_background")
        if media_category in {"pattern", "text_ui", "game", "checker", "stripes", "game_like"}:
            tags.append("hard_edges")

        hole_sources: list[str] = []
        pre_holes: list[np.ndarray] = []
        post_holes: list[np.ndarray] = []
        if bool(p.use_camera_hole_patch_bank) and self.hole_bank is not None and len(self.hole_bank) > 0:
            # Use independently sampled camera-domain hole appearances, but keep
            # the *physical state* stable: old holes exist in both PRE/POST and
            # the new GT hole only in POST.  Sampling is deterministic per seed.
            # To keep old-hole appearance identical across PRE and POST we apply
            # the same residual set to the concatenated frame sequence once.
            combined = list(pre_bg) + list(post_bg)
            rendered = [frame.copy() for frame in combined]
            for x, y in old_holes:
                sampled = self.hole_bank.sample_residual(rng)
                if sampled is None:
                    rendered = []
                    break
                residual, mask, asset = sampled
                hole_sources.append(Path(asset.image_path).name)
                strength = rng.uniform(0.80, 1.20)
                rendered = [apply_hole_residual(frame, residual, mask, x, y, strength=strength) for frame in rendered]
            if rendered:
                pre_holes = rendered[:len(pre_bg)]
                post_holes = rendered[len(pre_bg):]
                sampled = self.hole_bank.sample_residual(rng)
                if sampled is not None:
                    residual, mask, asset = sampled
                    hole_sources.append(Path(asset.image_path).name)
                    strength = rng.uniform(0.80, 1.20)
                    post_holes = [apply_hole_residual(frame, residual, mask, gt_xy[0], gt_xy[1], strength=strength) for frame in post_holes]
        if not pre_holes or not post_holes:
            from src.engine.synthetic.synthetic_hole_overlay import SyntheticHoleOverlay
            overlay = SyntheticHoleOverlay(
                p.width, p.height, rng_seed=int(seed) ^ 0x51A7C0DE,
                default_radius_px_range=(float(p.hole_radius_min), float(p.hole_radius_max)),
            )
            for x, y in old_holes:
                overlay.add_hole(x, y)
            old_overlay = overlay.render_overlay_bgra()
            overlay.add_hole(float(gt_xy[0]), float(gt_xy[1]))
            new_overlay = overlay.render_overlay_bgra()
            pre_holes = [self._composite(frame, old_overlay) for frame in pre_bg]
            post_holes = [self._composite(frame, new_overlay) for frame in post_bg]
            tags.append("procedural_hole_fallback")
        else:
            tags.append("camera_captured_hole_appearance")

        shared_gain = rng.uniform(1.0 - float(p.camera_gain_jitter), 1.0 + float(p.camera_gain_jitter))
        shared_gamma = rng.uniform(1.0 - float(p.camera_gamma_jitter), 1.0 + float(p.camera_gamma_jitter))
        pre_observed = [self._camera_effect(frame, rng, shared_gain=shared_gain, shared_gamma=shared_gamma) for frame in pre_holes]
        post_observed = [self._camera_effect(frame, rng, shared_gain=shared_gain, shared_gamma=shared_gamma) for frame in post_holes]

        spec = ScenarioSpecV219(
            seed=int(seed),
            media_id=media_id,
            media_split=split,
            media_category=media_category,
            media_kind=media_kind,
            media_path=media_path,
            media_frame_index=base_index,
            gt_camera_xy=gt_xy,
            old_holes=list(old_holes),
            known_holes=list(known_holes),
            challenge_tags=sorted(set(tags)),
            metadata={
                "profile": asdict(p),
                "old_hole_count": old_count,
                "projector_frames_available": True,
                "hole_appearance_sources": hole_sources[:64],
                "hole_appearance_bank_size": 0 if self.hole_bank is None else len(self.hole_bank),
                "camera_geometry_jitter": False,
                "note": "Synthetic training world; never a substitute for physical holdout.",
            },
        )
        return GeneratedScenarioV219(
            spec=spec,
            pre_frames=[cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in pre_observed],
            recent_pre_frame=cv2.cvtColor(pre_observed[-1], cv2.COLOR_BGR2GRAY),
            post_frames=[cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in post_observed],
            projector_pre_frames=pre_bg,
            projector_post_frames=post_bg,
        )


def scenario_fingerprint(spec: ScenarioSpecV219) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
