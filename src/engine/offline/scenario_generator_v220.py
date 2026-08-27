from __future__ import annotations

"""Deterministic offline world/scenario generator for hit detection V2.20.

V2.20 keeps the strengths of the V2.19 media-world generator but fixes three
important realism problems:

1. Observed output remains RGB (grayscale is only a derived legacy view).
2. PRE/POST camera state is shared within a scenario to avoid global drift.
3. New/old holes are rendered as compact hole stamps instead of signed source
   residuals, so they look more like actual bullet holes.
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
from .hole_patch_bank_v220 import HolePatchBankV220, HoleStampV220, apply_hole_stamp


@dataclass(frozen=True)
class ScenarioProfileV220:
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
    camera_gain_jitter: float = 0.05
    camera_gamma_jitter: float = 0.06
    camera_channel_jitter: float = 0.035
    camera_black_level_jitter: float = 2.0
    frame_gain_jitter: float = 0.004
    sensor_noise_sigma: float = 1.0
    blur_sigma_max: float = 0.40
    media_motion_step: int = 1
    hole_radius_min: float = 3.0
    hole_radius_max: float = 6.0
    use_camera_hole_patch_bank: bool = True
    hole_render_retry_limit: int = 8
    qa_local_diff_min: float = 2.2
    qa_center_darkening_min: float = 1.0
    qa_static_global_mae_max: float = 2.2
    qa_diff_area_min: int = 8
    qa_diff_area_max: int = 260
    qa_aspect_ratio_max: float = 3.4

    @classmethod
    def from_file(cls, path: Path | None = None) -> "ScenarioProfileV220":
        path = Path(path or "content/ai/offline_v220.json")
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
class ScenarioSpecV220:
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
    generator_version: str = "2.20"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["gt_camera_xy"] = [float(self.gt_camera_xy[0]), float(self.gt_camera_xy[1])]
        row["old_holes"] = [[float(x), float(y)] for x, y in self.old_holes]
        row["known_holes"] = [[float(x), float(y)] for x, y in self.known_holes]
        return row


@dataclass
class GeneratedScenarioV220:
    spec: ScenarioSpecV220
    pre_frames: list[np.ndarray]              # legacy grayscale view
    recent_pre_frame: np.ndarray              # legacy grayscale view
    post_frames: list[np.ndarray]             # legacy grayscale view
    pre_frames_rgb: list[np.ndarray]
    recent_pre_frame_rgb: np.ndarray
    post_frames_rgb: list[np.ndarray]
    projector_pre_frames: list[np.ndarray]
    projector_post_frames: list[np.ndarray]


@dataclass(frozen=True)
class _CameraStateV220:
    gain: float
    gamma: float
    blur_sigma: float
    black_level: float
    channel_gains: tuple[float, float, float]


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
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    name = name.lower()

    if name == "white":
        out = np.full((height, width, 3), (245, 245, 245), dtype=np.uint8)
    elif name == "gray":
        out = np.full((height, width, 3), (132, 132, 132), dtype=np.uint8)
    elif name == "black":
        out = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    elif name == "checker":
        cell = max(12, min(width, height) // 18)
        pattern = ((xx // cell + yy // cell + phase) % 2).astype(np.uint8)
        a = np.array([42, 90, 215], dtype=np.uint8)
        b = np.array([220, 205, 54], dtype=np.uint8)
        out = np.where(pattern[..., None] == 0, a, b).astype(np.uint8)
    elif name == "stripes":
        out = np.zeros((height, width, 3), dtype=np.uint8)
        out[..., 0] = np.clip(120 + 85 * np.sin((xx + phase * 7) * 2 * np.pi / max(8, width // 40)), 0, 255)
        out[..., 1] = np.clip(100 + 75 * np.sin((xx + phase * 5) * 2 * np.pi / max(8, width // 55) + 0.8), 0, 255)
        out[..., 2] = np.clip(140 + 70 * np.sin((xx + phase * 3) * 2 * np.pi / max(8, width // 48) + 1.7), 0, 255)
    elif name == "gradient":
        out = np.zeros((height, width, 3), dtype=np.uint8)
        xnorm = xx / max(1, width - 1)
        ynorm = yy / max(1, height - 1)
        out[..., 0] = np.clip(20 + 175 * xnorm + 25 * ynorm, 0, 255)
        out[..., 1] = np.clip(30 + 135 * (1.0 - ynorm) + 55 * xnorm, 0, 255)
        out[..., 2] = np.clip(50 + 145 * ynorm + 30 * (1.0 - xnorm), 0, 255)
    elif name == "noise":
        out = np.zeros((height, width, 3), dtype=np.uint8)
        for channel, mean, std in ((0, 110, 42), (1, 135, 36), (2, 150, 40)):
            small = rng.normal(mean, std, size=(max(2, height // 24), max(2, width // 24))).astype(np.float32)
            out[..., channel] = np.clip(cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC), 0, 255)
    elif name == "game_like":
        out = np.full((height, width, 3), (88, 76, 66), dtype=np.uint8)
        for i in range(18):
            x = int((seed * (i + 3) * 17 + phase * (9 + i)) % max(1, width))
            y = int((seed * (i + 5) * 11 + phase * (5 + i)) % max(1, height))
            rw = 18 + (i * 13) % max(20, width // 7)
            rh = 14 + (i * 9) % max(16, height // 9)
            color = (
                int(35 + (i * 37) % 205),
                int(40 + (i * 61) % 180),
                int(50 + (i * 29) % 185),
            )
            cv2.rectangle(out, (x, y), (min(width - 1, x + rw), min(height - 1, y + rh)), color, -1)
        cv2.putText(out, "TARGET 42 SCORE", (max(5, width // 12), max(30, height // 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 1100), (225, 245, 225), 2, cv2.LINE_AA)
    else:
        out = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)
    return out


class OfflineScenarioGeneratorV220:
    PROCEDURAL = ("white", "gray", "black", "checker", "stripes", "gradient", "noise", "game_like")

    def __init__(
        self,
        *,
        profile: ScenarioProfileV220 | None = None,
        media_assets: Sequence[MediaAssetV219] | None = None,
        media_manifest: Path | None = None,
        repo_root: Path = Path("."),
        hole_bank: HolePatchBankV220 | None = None,
    ) -> None:
        base_profile = profile or ScenarioProfileV220()
        self.repo_root = Path(repo_root)
        width, height = int(base_profile.width), int(base_profile.height)
        if width <= 0 or height <= 0:
            auto_w, auto_h = self._detect_camera_shape(self.repo_root)
            width = width if width > 0 else auto_w
            height = height if height > 0 else auto_h
        import dataclasses
        self.profile = dataclasses.replace(base_profile, width=width, height=height)
        self.media_assets = list(media_assets) if media_assets is not None else read_media_manifest(media_manifest or Path("content/ai/media_bank_v219/media_manifest.jsonl"))
        self.hole_bank = hole_bank if hole_bank is not None else HolePatchBankV220.discover(root=self.repo_root)

    @staticmethod
    def _detect_camera_shape(repo_root: Path) -> tuple[int, int]:
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
        margin = max(10, int(round(max(p.hole_radius_max * 5.0, 10.0))))
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

    def _sample_camera_state(self, rng: random.Random) -> _CameraStateV220:
        p = self.profile
        base_gain = rng.uniform(1.0 - float(p.camera_gain_jitter), 1.0 + float(p.camera_gain_jitter))
        gamma = rng.uniform(1.0 - float(p.camera_gamma_jitter), 1.0 + float(p.camera_gamma_jitter))
        blur_sigma = rng.uniform(0.0, float(p.blur_sigma_max)) if p.blur_sigma_max > 0 else 0.0
        black = rng.uniform(-float(p.camera_black_level_jitter), float(p.camera_black_level_jitter))
        cg = tuple(rng.uniform(1.0 - float(p.camera_channel_jitter), 1.0 + float(p.camera_channel_jitter)) for _ in range(3))
        return _CameraStateV220(gain=base_gain, gamma=max(0.35, gamma), blur_sigma=blur_sigma, black_level=black, channel_gains=cg)

    @staticmethod
    def _apply_camera_state(frame: np.ndarray, state: _CameraStateV220) -> np.ndarray:
        out = np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0)
        for channel in range(3):
            out[..., channel] = np.power(out[..., channel], state.gamma) * state.gain * state.channel_gains[channel]
        out = out * 255.0 + state.black_level
        if state.blur_sigma > 0.04:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=state.blur_sigma, sigmaY=state.blur_sigma)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_temporal_noise(self, frame: np.ndarray, rng: random.Random) -> np.ndarray:
        p = self.profile
        out = frame.astype(np.float32)
        if p.frame_gain_jitter > 0:
            out *= rng.uniform(1.0 - float(p.frame_gain_jitter), 1.0 + float(p.frame_gain_jitter))
        if p.sensor_noise_sigma > 0:
            np_rng = np.random.default_rng(rng.randrange(2**32))
            out += np_rng.normal(0.0, float(p.sensor_noise_sigma), size=out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _render_sequence_with_stamps(
        self,
        base_pre: Sequence[np.ndarray],
        base_post: Sequence[np.ndarray],
        old_holes: Sequence[tuple[float, float]],
        gt_xy: tuple[float, float],
        rng: random.Random,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[HoleStampV220]]:
        p = self.profile
        pre = [frame.copy() for frame in base_pre]
        post = [frame.copy() for frame in base_post]
        sources: list[str] = []
        stamps: list[HoleStampV220] = []
        for x, y in old_holes:
            stamp = self.hole_bank.sample_stamp(rng) if bool(p.use_camera_hole_patch_bank) and self.hole_bank is not None and len(self.hole_bank) > 0 else None
            if stamp is None:
                stamp = HolePatchBankV220.default_stamp(rng, radius_range=(p.hole_radius_min, p.hole_radius_max))
            stamps.append(stamp)
            sources.append(stamp.source_id)
            strength = rng.uniform(0.92, 1.12)
            pre = [apply_hole_stamp(frame, stamp, x, y, strength=strength) for frame in pre]
            post = [apply_hole_stamp(frame, stamp, x, y, strength=strength) for frame in post]
        new_stamp = self.hole_bank.sample_stamp(rng) if bool(p.use_camera_hole_patch_bank) and self.hole_bank is not None and len(self.hole_bank) > 0 else None
        if new_stamp is None:
            new_stamp = HolePatchBankV220.default_stamp(rng, radius_range=(p.hole_radius_min, p.hole_radius_max))
        stamps.append(new_stamp)
        sources.append(new_stamp.source_id)
        strength = rng.uniform(0.96, 1.16)
        post = [apply_hole_stamp(frame, new_stamp, gt_xy[0], gt_xy[1], strength=strength) for frame in post]
        return pre, post, sources, stamps

    @staticmethod
    def _qa_metrics(pre_rgb: np.ndarray, post_rgb: np.ndarray, gt_xy: tuple[float, float], *, static_scene: bool) -> dict[str, float]:
        pre_g = cv2.cvtColor(pre_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        post_g = cv2.cvtColor(post_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(post_g - pre_g)
        darkening = pre_g - post_g
        x, y = map(int, map(round, gt_xy))
        local_r = 16
        center_r = 6
        y0, y1 = max(0, y - local_r), min(diff.shape[0], y + local_r + 1)
        x0, x1 = max(0, x - local_r), min(diff.shape[1], x + local_r + 1)
        cy0, cy1 = max(0, y - center_r), min(diff.shape[0], y + center_r + 1)
        cx0, cx1 = max(0, x - center_r), min(diff.shape[1], x + center_r + 1)
        local = diff[y0:y1, x0:x1]
        center_dark = darkening[cy0:cy1, cx0:cx1]
        thresh = (local > 4.0).astype(np.uint8)
        area = int(thresh.sum())
        aspect = 1.0
        if area > 0:
            ys, xs = np.where(thresh > 0)
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            aspect = max(bw / max(1, bh), bh / max(1, bw))
        global_mae = 0.0
        if static_scene:
            mask = np.ones(diff.shape, dtype=bool)
            grow = 40
            mask[max(0, y - grow):min(diff.shape[0], y + grow + 1), max(0, x - grow):min(diff.shape[1], x + grow + 1)] = False
            if np.any(mask):
                global_mae = float(np.mean(diff[mask]))
        return {
            "local_mean_abs_diff": float(np.mean(local)) if local.size else 0.0,
            "center_mean_darkening": float(np.mean(center_dark)) if center_dark.size else 0.0,
            "diff_area": float(area),
            "bbox_aspect_ratio": float(aspect),
            "static_global_mae": float(global_mae),
        }

    def _qa_accept(self, metrics: dict[str, float], *, static_scene: bool) -> bool:
        p = self.profile
        if metrics["local_mean_abs_diff"] < float(p.qa_local_diff_min):
            return False
        if metrics["center_mean_darkening"] < float(p.qa_center_darkening_min):
            return False
        if metrics["diff_area"] < float(p.qa_diff_area_min) or metrics["diff_area"] > float(p.qa_diff_area_max):
            return False
        if metrics["bbox_aspect_ratio"] > float(p.qa_aspect_ratio_max):
            return False
        if static_scene and metrics["static_global_mae"] > float(p.qa_static_global_mae_max):
            return False
        return True

    def generate(self, seed: int, *, split: str = "train") -> GeneratedScenarioV220:
        p = self.profile
        base_rng = random.Random(int(seed))
        asset = self._select_media(base_rng, split)
        pre_bg, post_bg, media_id, media_kind, media_category, media_path, base_index = self._media_frames(asset, base_rng, int(seed))

        old_count = base_rng.randint(max(0, int(p.old_hole_min)), max(int(p.old_hole_min), int(p.old_hole_max)))
        margin = max(10, int(round(max(p.hole_radius_max * 5.0, 10.0))))
        old_holes = [
            (float(base_rng.randint(margin, p.width - margin - 1)), float(base_rng.randint(margin, p.height - margin - 1)))
            for _ in range(old_count)
        ]
        gt_xy, tags = self._choose_position(base_rng, old_holes)
        known_holes = [xy for xy in old_holes if base_rng.random() < float(p.known_hole_fraction)]
        if len(known_holes) < len(old_holes):
            tags.append("incomplete_known_holes")
        if old_count >= 35:
            tags.append("dense_old_holes")
        if media_kind == "video" or media_category in {"video", "game", "checker", "stripes", "noise", "game_like"}:
            tags.append("dynamic_background")
        if media_category in {"pattern", "text_ui", "game", "checker", "stripes", "game_like"}:
            tags.append("hard_edges")

        static_scene = not (media_kind == "video" or "dynamic_background" in tags)
        camera_state = self._sample_camera_state(base_rng)
        attempt_metrics: dict[str, float] | None = None
        hole_sources: list[str] = []
        used_stamps: list[HoleStampV220] = []
        chosen_pre_obs: list[np.ndarray] | None = None
        chosen_post_obs: list[np.ndarray] | None = None
        attempts = max(1, int(p.hole_render_retry_limit))

        for attempt in range(attempts):
            attempt_rng = random.Random((int(seed) * 1000003 + 9719 * attempt + 0x220) & 0xFFFFFFFF)
            pre_holes, post_holes, hole_sources, used_stamps = self._render_sequence_with_stamps(pre_bg, post_bg, old_holes, gt_xy, attempt_rng)
            observed_pre = [self._apply_temporal_noise(self._apply_camera_state(frame, camera_state), attempt_rng) for frame in pre_holes]
            observed_post = [self._apply_temporal_noise(self._apply_camera_state(frame, camera_state), attempt_rng) for frame in post_holes]
            attempt_metrics = self._qa_metrics(observed_pre[-1], observed_post[-1], gt_xy, static_scene=static_scene)
            if self._qa_accept(attempt_metrics, static_scene=static_scene) or attempt == attempts - 1:
                chosen_pre_obs = observed_pre
                chosen_post_obs = observed_post
                break

        assert chosen_pre_obs is not None and chosen_post_obs is not None
        gray_pre = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in chosen_pre_obs]
        gray_post = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in chosen_post_obs]

        source_kind = "camera_captured_hole_appearance" if bool(p.use_camera_hole_patch_bank) and self.hole_bank is not None and len(self.hole_bank) > 0 else "procedural_hole_fallback"
        tags.append(source_kind)
        if static_scene:
            tags.append("shared_camera_state")
        tags.append("rgb_observed_output")

        spec = ScenarioSpecV220(
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
                "observed_output_color": "rgb",
                "qa": attempt_metrics or {},
                "qa_attempts": attempt + 1,
                "static_scene": static_scene,
                "shared_camera_state": asdict(camera_state),
                "hole_stamp_summary": [
                    {
                        "source_id": stamp.source_id,
                        "core_radius_px": float(stamp.core_radius_px),
                        "core_depth": float(stamp.core_depth),
                        "rim_strength": float(stamp.rim_strength),
                        "rim_width_px": float(stamp.rim_width_px),
                    }
                    for stamp in used_stamps[:12]
                ],
                "note": "Synthetic training world; never a substitute for physical holdout.",
            },
        )
        return GeneratedScenarioV220(
            spec=spec,
            pre_frames=gray_pre,
            recent_pre_frame=gray_pre[-1],
            post_frames=gray_post,
            pre_frames_rgb=chosen_pre_obs,
            recent_pre_frame_rgb=chosen_pre_obs[-1],
            post_frames_rgb=chosen_post_obs,
            projector_pre_frames=pre_bg,
            projector_post_frames=post_bg,
        )


def scenario_fingerprint(spec: ScenarioSpecV220) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
