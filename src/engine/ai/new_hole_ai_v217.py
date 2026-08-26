from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class NewHoleAIConfigV217:
    input_size: int = 22
    hidden_size: int = 96
    offset_scale_px: float = 24.0
    feature_channels: int = 6
    scalar_features: int = 8
    learning_rate: float = 0.0012
    weight_decay: float = 2e-5
    offset_loss_weight: float = 0.30
    positive_class_weight: float = 1.25
    gradient_clip_norm: float = 5.0

    @property
    def input_dim(self) -> int:
        return int(self.input_size * self.input_size * self.feature_channels + self.scalar_features)


@dataclass
class BatchMetricsV217:
    loss: float
    classification_loss: float
    offset_loss: float
    accuracy: float
    positive_recall: float
    negative_specificity: float


def _signed_robust(values: np.ndarray, floor: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    scale = float(np.percentile(np.abs(x), 94)) if x.size else 0.0
    scale = max(float(floor), scale)
    return np.clip(x / (2.5 * scale), -1.0, 1.0).astype(np.float32)


def _positive_robust(values: np.ndarray, floor: float) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    scale = float(np.percentile(x, 96)) if x.size else 0.0
    scale = max(float(floor), scale)
    return np.clip(x / scale, 0.0, 1.0).astype(np.float32)


def _ensure_gray(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim != 2 or not arr.size:
        raise ValueError("NewHoleAIV217 expects non-empty grayscale/BGR patches")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class NewHoleAIV217:
    """Small temporal candidate classifier + offset refiner.

    Semantics are intentionally narrow: score whether a *new* hole-like physical
    change appeared at the candidate during the current shot. A pre-existing
    real hole is therefore a correct negative for this model, while remaining a
    positive example for the static Hole-AI.
    """

    schema_version = "2.17"

    def __init__(self, config: NewHoleAIConfigV217 | None = None, *, seed: int = 21701):
        self.config = config or NewHoleAIConfigV217()
        rng = np.random.default_rng(int(seed))
        in_dim = self.config.input_dim
        hidden = int(self.config.hidden_size)
        self.W1 = (rng.standard_normal((in_dim, hidden)).astype(np.float32) * math.sqrt(2.0 / max(1, in_dim))).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 3)).astype(np.float32) * math.sqrt(1.0 / max(1, hidden))).astype(np.float32)
        self.b2 = np.zeros((3,), dtype=np.float32)
        self._adam_m = [np.zeros_like(value) for value in self.parameters()]
        self._adam_v = [np.zeros_like(value) for value in self.parameters()]
        self._adam_t = 0

    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

    def feature_maps_from_pair(self, pre_patch: np.ndarray, post_patch: np.ndarray) -> list[np.ndarray]:
        pre_u8 = _ensure_gray(pre_patch)
        post_u8 = _ensure_gray(post_patch)
        if pre_u8.shape != post_u8.shape:
            post_u8 = cv2.resize(post_u8, (pre_u8.shape[1], pre_u8.shape[0]), interpolation=cv2.INTER_AREA)

        size = int(self.config.input_size)
        pre = cv2.resize(pre_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        post = cv2.resize(post_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        signed_delta = post - pre
        abs_delta = np.abs(signed_delta)
        darkening = np.maximum(pre - post, 0.0)

        pre_large = cv2.GaussianBlur(pre, (0, 0), 2.4)
        post_large = cv2.GaussianBlur(post, (0, 0), 2.4)
        local_delta = (post - post_large) - (pre - pre_large)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pre_close = cv2.morphologyEx(pre, cv2.MORPH_CLOSE, kernel)
        post_close = cv2.morphologyEx(post, cv2.MORPH_CLOSE, kernel)
        blackhat_gain = (post_close - post) - (pre_close - pre)

        pre_gx = cv2.Sobel(pre, cv2.CV_32F, 1, 0, ksize=3)
        pre_gy = cv2.Sobel(pre, cv2.CV_32F, 0, 1, ksize=3)
        post_gx = cv2.Sobel(post, cv2.CV_32F, 1, 0, ksize=3)
        post_gy = cv2.Sobel(post, cv2.CV_32F, 0, 1, ksize=3)
        gradient_gain = cv2.magnitude(post_gx, post_gy) - cv2.magnitude(pre_gx, pre_gy)

        return [
            _signed_robust(signed_delta, 0.012),
            _positive_robust(abs_delta, 0.010),
            _positive_robust(darkening, 0.008),
            _signed_robust(local_delta, 0.010),
            _signed_robust(blackhat_gain, 0.008),
            _signed_robust(gradient_gain, 0.012),
        ]

    def scalar_features_from_pair(
        self,
        pre_patch: np.ndarray,
        post_patch: np.ndarray,
        post_stack: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        pre = _ensure_gray(pre_patch).astype(np.float32) / 255.0
        post = _ensure_gray(post_patch).astype(np.float32) / 255.0
        if pre.shape != post.shape:
            post = cv2.resize(post, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)
        diff = post - pre
        absdiff = np.abs(diff)
        h, w = pre.shape
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inner = d2 <= max(4.0, min(h, w) * 0.08) ** 2
        ring = (d2 >= max(8.0, min(h, w) * 0.16) ** 2) & (d2 <= max(14.0, min(h, w) * 0.30) ** 2)
        center_abs = float(np.mean(absdiff[inner])) if np.any(inner) else 0.0
        center_dark = float(np.mean(np.maximum(-diff[inner], 0.0))) if np.any(inner) else 0.0
        ring_abs = float(np.mean(absdiff[ring])) if np.any(ring) else 0.0

        persistence = 0.0
        stack = list(post_stack or [])
        if stack:
            vals = []
            for item in stack:
                cur = _ensure_gray(item).astype(np.float32) / 255.0
                if cur.shape != pre.shape:
                    cur = cv2.resize(cur, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)
                vals.append(float(np.mean(np.abs(cur[inner] - pre[inner]))) if np.any(inner) else 0.0)
            if vals:
                peak = max(vals)
                persistence = float(sum(v >= max(0.01, peak * 0.55) for v in vals) / len(vals))

        scalars = np.array([
            float(np.mean(absdiff)),
            float(np.percentile(absdiff, 95)),
            center_abs,
            center_dark,
            max(0.0, center_abs - ring_abs),
            float(np.mean(np.maximum(diff, 0.0))),
            float(np.mean(np.maximum(-diff, 0.0))),
            persistence,
        ], dtype=np.float32)
        scale = np.array([0.08, 0.25, 0.20, 0.18, 0.15, 0.08, 0.08, 1.0], dtype=np.float32)
        return np.clip(scalars / scale, 0.0, 3.0).astype(np.float32)

    def features_from_pair(self, pre_patch: np.ndarray, post_patch: np.ndarray, post_stack: Sequence[np.ndarray] | None = None) -> np.ndarray:
        maps = self.feature_maps_from_pair(pre_patch, post_patch)
        requested = int(self.config.feature_channels)
        if requested < 1 or requested > len(maps):
            raise ValueError(f"feature_channels must be 1..{len(maps)}, got {requested}")
        image = np.concatenate([m.reshape(-1) for m in maps[:requested]]).astype(np.float32)
        scalars = self.scalar_features_from_pair(pre_patch, post_patch, post_stack)
        if int(self.config.scalar_features) != len(scalars):
            raise ValueError("NewHoleAI scalar feature count mismatch")
        return np.concatenate([image, scalars]).astype(np.float32)

    def feature_batch(self, pairs: Sequence[tuple[np.ndarray, np.ndarray, Sequence[np.ndarray] | None]]) -> np.ndarray:
        return np.stack([self.features_from_pair(pre, post, stack) for pre, post, stack in pairs], axis=0)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        p = self._sigmoid(raw[:, 0])
        offsets = raw[:, 1:3] * float(self.config.offset_scale_px)
        return p, offsets, hidden

    def predict_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p, off, _ = self._forward(features)
        return p.astype(np.float32), off.astype(np.float32)

    def predict_pairs(self, pairs: Sequence[tuple[np.ndarray, np.ndarray, Sequence[np.ndarray] | None]]) -> tuple[np.ndarray, np.ndarray]:
        if not pairs:
            return np.empty((0,), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        return self.predict_features(self.feature_batch(pairs))

    def train_batch(self, features: np.ndarray, labels: np.ndarray, target_offsets_px: np.ndarray) -> BatchMetricsV217:
        cfg = self.config
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        target_px = np.asarray(target_offsets_px, dtype=np.float32).reshape(-1, 2)
        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        p = self._sigmoid(raw[:, 0])
        pred_off_norm = raw[:, 1:3]
        target_off_norm = target_px / max(1e-6, float(cfg.offset_scale_px))

        eps = 1e-6
        weights = np.where(y > 0.5, float(cfg.positive_class_weight), 1.0).astype(np.float32)
        bce = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        classification_loss = float(np.sum(weights * bce) / max(eps, float(np.sum(weights))))

        pos_mask = (y > 0.5).astype(np.float32)[:, None]
        pos_count = max(1.0, float(np.sum(pos_mask)))
        offset_error = pred_off_norm - target_off_norm
        offset_loss = float(np.sum(pos_mask * (offset_error ** 2)) / (2.0 * pos_count))
        loss = classification_loss + float(cfg.offset_loss_weight) * offset_loss

        weight_norm = max(eps, float(np.sum(weights)))
        d_raw = np.zeros_like(raw, dtype=np.float32)
        d_raw[:, 0] = weights * (p - y) / weight_norm
        d_raw[:, 1:3] = float(cfg.offset_loss_weight) * pos_mask * offset_error / pos_count

        grad_W2 = hidden.T @ d_raw + float(cfg.weight_decay) * self.W2
        grad_b2 = np.sum(d_raw, axis=0)
        d_hidden = d_raw @ self.W2.T
        d_hidden[hidden_pre <= 0.0] = 0.0
        grad_W1 = x.T @ d_hidden + float(cfg.weight_decay) * self.W1
        grad_b1 = np.sum(d_hidden, axis=0)
        grads = [grad_W1, grad_b1, grad_W2, grad_b2]

        total_norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads))
        clip = float(cfg.gradient_clip_norm)
        if clip > 0.0 and total_norm > clip:
            scale = clip / max(total_norm, 1e-9)
            grads = [g * scale for g in grads]
        self._adam_step(grads)

        pred = p >= 0.5
        truth = y >= 0.5
        positives = truth
        negatives = ~truth
        return BatchMetricsV217(
            loss=float(loss),
            classification_loss=classification_loss,
            offset_loss=offset_loss,
            accuracy=float(np.mean(pred == truth)) if truth.size else 0.0,
            positive_recall=float(np.mean(pred[positives])) if np.any(positives) else 0.0,
            negative_specificity=float(np.mean(~pred[negatives])) if np.any(negatives) else 0.0,
        )

    def _adam_step(self, grads: Sequence[np.ndarray]) -> None:
        self._adam_t += 1
        beta1, beta2 = 0.9, 0.999
        lr = float(self.config.learning_rate)
        eps = 1e-8
        for index, (param, grad) in enumerate(zip(self.parameters(), grads)):
            self._adam_m[index] = beta1 * self._adam_m[index] + (1.0 - beta1) * grad
            self._adam_v[index] = beta2 * self._adam_v[index] + (1.0 - beta2) * (grad * grad)
            m_hat = self._adam_m[index] / (1.0 - beta1 ** self._adam_t)
            v_hat = self._adam_v[index] / (1.0 - beta2 ** self._adam_t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_meta = {
            "schema_version": self.schema_version,
            "model_type": "new_hole_before_after_mlp_v217",
            "config": asdict(self.config),
            "metadata": dict(metadata or {}),
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, metadata_json=np.array(json.dumps(payload_meta, ensure_ascii=False)))
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple["NewHoleAIV217", dict[str, Any]]:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            config = NewHoleAIConfigV217(**dict(metadata.get("config") or {}))
            model = cls(config=config)
            model.W1[...] = data["W1"].astype(np.float32)
            model.b1[...] = data["b1"].astype(np.float32)
            model.W2[...] = data["W2"].astype(np.float32)
            model.b2[...] = data["b2"].astype(np.float32)
        return model, metadata
