from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214


DEFAULT_CONFIG_PATH = Path("content/ai/reports/v215/hole_v215_ensemble.json")


def _clip_probability(value: float | np.ndarray, eps: float = 1e-5):
    return np.clip(value, eps, 1.0 - eps)


def logit(value: float | np.ndarray) -> float | np.ndarray:
    clipped = _clip_probability(value)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def threshold_centered_margin(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a model probability into a margin around its own learned threshold.

    V2.14 mild and standard were trained under different domain profiles and are
    not guaranteed to have identical calibration.  Blending raw probabilities
    would silently favour whichever model happens to be more confident.  This
    representation makes each model's own decision boundary equal to zero.
    """

    return np.asarray(logit(probability), dtype=np.float32) - float(logit(float(threshold)))


@dataclass(frozen=True)
class HolePatchEnsembleConfigV215:
    standard_model_path: str
    mild_model_path: str
    standard_weight: float
    fused_threshold: float
    standard_threshold: float
    mild_threshold: float
    disagreement_warn: float = 0.35
    schema_version: str = "2.15"
    shadow_only: bool = True

    @property
    def mild_weight(self) -> float:
        return 1.0 - float(self.standard_weight)


@dataclass(frozen=True)
class HolePatchEvidenceV215:
    standard_probability: float
    mild_probability: float
    fused_probability: float
    disagreement: float
    standard_offset_px: tuple[float, float]
    mild_offset_px: tuple[float, float]
    fused_offset_px: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_candidate_patch(
    gray: np.ndarray,
    center_xy: tuple[float, float],
    crop_size: int,
) -> np.ndarray:
    """Extract a candidate-centred patch even close to image edges.

    Live candidates can sit close to Scanport/viewport borders.  Training crops
    were always valid, so V2.15 explicitly pads rather than silently dropping
    an edge candidate.  Reflect padding avoids introducing a black rectangle
    that the model never saw during training.
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2 or not gray.size:
        raise ValueError("gray frame must be a non-empty grayscale/BGR image")
    size = int(crop_size)
    if size <= 0:
        raise ValueError("crop_size must be > 0")
    if size % 2:
        size += 1

    x = int(round(float(center_xy[0])))
    y = int(round(float(center_xy[1])))
    half = size // 2
    h, w = gray.shape[:2]

    pad_left = max(0, half - x)
    pad_top = max(0, half - y)
    pad_right = max(0, x + half - w)
    pad_bottom = max(0, y + half - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        gray = cv2.copyMakeBorder(
            gray,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x += pad_left
        y += pad_top

    patch = gray[y - half : y + half, x - half : x + half]
    if patch.shape != (size, size):
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(patch)


class HolePatchEnsembleV215:
    """Dual V2.14 Hole-AI evidence source.

    This object is intentionally *shadow only*. ``annotate_candidates`` keeps
    candidate order untouched and only attaches evidence fields.  Future fusion
    code may consume those fields after separate gates are passed.
    """

    VERSION = "2.15"

    def __init__(
        self,
        standard_model: HolePatchAIV214,
        mild_model: HolePatchAIV214,
        config: HolePatchEnsembleConfigV215,
    ) -> None:
        self.standard_model = standard_model
        self.mild_model = mild_model
        self.config = config
        if not bool(config.shadow_only):
            raise ValueError("V2.15 config must remain shadow_only")
        if not (0.0 <= float(config.standard_weight) <= 1.0):
            raise ValueError("standard_weight must be 0..1")
        if standard_model.config.crop_size != mild_model.config.crop_size:
            raise ValueError("mild/standard crop_size mismatch")

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> "HolePatchEnsembleV215":
        config_path = Path(config_path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cfg_payload = payload.get("ensemble_config") if isinstance(payload, dict) else None
        if not isinstance(cfg_payload, dict):
            cfg_payload = payload
        cfg = HolePatchEnsembleConfigV215(**cfg_payload)
        standard, _ = HolePatchAIV214.load(Path(cfg.standard_model_path))
        mild, _ = HolePatchAIV214.load(Path(cfg.mild_model_path))
        return cls(standard, mild, cfg)

    @staticmethod
    def fuse_probabilities(
        standard_probability: np.ndarray,
        mild_probability: np.ndarray,
        *,
        standard_threshold: float,
        mild_threshold: float,
        standard_weight: float,
    ) -> np.ndarray:
        std_margin = threshold_centered_margin(standard_probability, standard_threshold)
        mild_margin = threshold_centered_margin(mild_probability, mild_threshold)
        w = float(np.clip(standard_weight, 0.0, 1.0))
        return np.asarray(sigmoid(w * std_margin + (1.0 - w) * mild_margin), dtype=np.float32)

    def score_patches(self, patches: Sequence[np.ndarray]) -> list[HolePatchEvidenceV215]:
        if not patches:
            return []
        std_p, std_off = self.standard_model.predict_patches(patches)
        mild_p, mild_off = self.mild_model.predict_patches(patches)
        fused = self.fuse_probabilities(
            std_p,
            mild_p,
            standard_threshold=self.config.standard_threshold,
            mild_threshold=self.config.mild_threshold,
            standard_weight=self.config.standard_weight,
        )

        # Offset is auxiliary evidence.  Weight it by both configured model
        # importance and each model's positive confidence.  It never alters the
        # authoritative camera coordinate in V2.15.
        std_support = float(self.config.standard_weight) * np.maximum(std_p, 1e-4)
        mild_support = float(self.config.mild_weight) * np.maximum(mild_p, 1e-4)
        denom = np.maximum(std_support + mild_support, 1e-6)[:, None]
        fused_off = (std_off * std_support[:, None] + mild_off * mild_support[:, None]) / denom

        result: list[HolePatchEvidenceV215] = []
        for index in range(len(patches)):
            result.append(
                HolePatchEvidenceV215(
                    standard_probability=float(std_p[index]),
                    mild_probability=float(mild_p[index]),
                    fused_probability=float(fused[index]),
                    disagreement=float(abs(float(std_p[index]) - float(mild_p[index]))),
                    standard_offset_px=(float(std_off[index, 0]), float(std_off[index, 1])),
                    mild_offset_px=(float(mild_off[index, 0]), float(mild_off[index, 1])),
                    fused_offset_px=(float(fused_off[index, 0]), float(fused_off[index, 1])),
                )
            )
        return result

    def annotate_candidates(
        self,
        gray: np.ndarray,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach Hole-AI evidence while preserving order and authority."""

        source = [dict(candidate) for candidate in candidates]
        if not source:
            return []
        crop_size = int(self.standard_model.config.crop_size)
        patches = [
            extract_candidate_patch(
                gray,
                (float(candidate.get("camera_x", 0.0)), float(candidate.get("camera_y", 0.0))),
                crop_size,
            )
            for candidate in source
        ]
        evidence = self.score_patches(patches)
        for candidate, ev in zip(source, evidence):
            cx = float(candidate.get("camera_x", 0.0))
            cy = float(candidate.get("camera_y", 0.0))
            candidate.update(
                {
                    "hole_v215_shadow_only": True,
                    "hole_v215_standard_probability": ev.standard_probability,
                    "hole_v215_mild_probability": ev.mild_probability,
                    "hole_v215_fused_probability": ev.fused_probability,
                    "hole_v215_disagreement": ev.disagreement,
                    "hole_v215_offset_dx": ev.fused_offset_px[0],
                    "hole_v215_offset_dy": ev.fused_offset_px[1],
                    "hole_v215_refined_camera_x": cx + ev.fused_offset_px[0],
                    "hole_v215_refined_camera_y": cy + ev.fused_offset_px[1],
                    "hole_v215_above_threshold": bool(ev.fused_probability >= float(self.config.fused_threshold)),
                    "hole_v215_uncertain": bool(ev.disagreement >= float(self.config.disagreement_warn)),
                }
            )
        return source


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HolePatchEnsembleConfigV215",
    "HolePatchEnsembleV215",
    "HolePatchEvidenceV215",
    "extract_candidate_patch",
    "logit",
    "sigmoid",
    "threshold_centered_margin",
]
