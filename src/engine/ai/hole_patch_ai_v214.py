from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HolePatchAIConfigV214:
    """Background-invariant candidate patch model for V2.14.

    V2.13 proved that raw pixels carry useful hole information, but the strict
    novel-background holdout exposed a strong domain dependency.  V2.14 keeps
    the tiny dependency-free MLP, while replacing intensity-heavy inputs with
    local physical residual channels that are much less sensitive to the
    projected background.
    """

    crop_size: int = 64
    input_size: int = 22
    hidden_size: int = 80
    offset_scale_px: float = 22.0
    feature_channels: int = 4
    learning_rate: float = 0.0012
    weight_decay: float = 2e-5
    offset_loss_weight: float = 0.30
    positive_class_weight: float = 1.0
    gradient_clip_norm: float = 5.0

    @property
    def input_dim(self) -> int:
        return int(self.input_size * self.input_size * self.feature_channels)


@dataclass
class BatchMetricsV214:
    loss: float
    classification_loss: float
    offset_loss: float
    accuracy: float
    positive_recall: float
    negative_specificity: float


def _signed_robust_normalize(values: np.ndarray, floor: float = 0.02) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - float(np.median(values))
    scale = float(np.percentile(np.abs(centered), 90)) if centered.size else 0.0
    scale = max(float(floor), scale)
    return (np.clip(centered / (3.0 * scale), -1.0, 1.0)).astype(np.float32)


def _positive_robust_normalize(values: np.ndarray, floor: float = 0.01) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    scale = float(np.percentile(values, 95)) if values.size else 0.0
    scale = max(float(floor), scale)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


class HolePatchAIV214:
    schema_version = "2.14"

    def __init__(self, config: HolePatchAIConfigV214 | None = None, *, seed: int = 21401):
        self.config = config or HolePatchAIConfigV214()
        rng = np.random.default_rng(int(seed))
        in_dim = self.config.input_dim
        hidden = int(self.config.hidden_size)
        self.W1 = (
            rng.standard_normal((in_dim, hidden)).astype(np.float32) * math.sqrt(2.0 / max(1, in_dim))
        ).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = (
            rng.standard_normal((hidden, 3)).astype(np.float32) * math.sqrt(1.0 / max(1, hidden))
        ).astype(np.float32)
        self.b2 = np.zeros((3,), dtype=np.float32)
        self._adam_m = [np.zeros_like(value) for value in self.parameters()]
        self._adam_v = [np.zeros_like(value) for value in self.parameters()]
        self._adam_t = 0

    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    def feature_maps_from_patch(self, patch: np.ndarray) -> list[np.ndarray]:
        """Return four local-physics maps used by the V2.14 MLP.

        None of these channels directly encodes absolute mean brightness.  The
        goal is to emphasise *local physical changes* (dark core, edge/rim,
        small-scale contrast) while suppressing slow projected-background
        variation.
        """

        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if patch.ndim != 2 or not patch.size:
            raise ValueError("HolePatchAIV214 expects a non-empty grayscale/BGR candidate patch")

        size = int(self.config.input_size)
        gray = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        small = cv2.GaussianBlur(gray, (0, 0), 0.75)
        medium = cv2.GaussianBlur(gray, (0, 0), 1.55)
        large = cv2.GaussianBlur(gray, (0, 0), 3.0)

        # Local residual: rejects broad illumination/projector fields.
        local_residual = _signed_robust_normalize(gray - large, floor=0.018)

        # DoG: compact structures survive; broad background gradients cancel.
        dog = _signed_robust_normalize(small - medium, floor=0.012)

        # Morphological black-hat: explicitly describes compact dark objects
        # relative to their immediate neighbourhood (a common physical hole cue).
        kernel_size = 5 if size >= 20 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        blackhat = _positive_robust_normalize(close - gray, floor=0.008)

        # Edge energy is useful for torn/rimmed holes and is intensity-shift
        # invariant.  It also gives the network evidence when the hole core is
        # weak but the physical edge remains visible.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = _positive_robust_normalize(cv2.magnitude(gx, gy), floor=0.01)

        return [local_residual, dog, blackhat, gradient]

    def features_from_patch(self, patch: np.ndarray) -> np.ndarray:
        maps = self.feature_maps_from_patch(patch)
        requested = int(self.config.feature_channels)
        if requested < 1 or requested > len(maps):
            raise ValueError(f"feature_channels must be 1..{len(maps)}, got {requested}")
        return np.concatenate([channel.reshape(-1) for channel in maps[:requested]]).astype(np.float32)

    def feature_batch(self, patches: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.features_from_patch(patch) for patch in patches], axis=0).astype(np.float32)

    def _forward_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        probability = self._sigmoid(raw[:, 0])
        offsets = raw[:, 1:3] * float(self.config.offset_scale_px)
        return probability, offsets, hidden

    def predict_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability, offsets, _ = self._forward_features(features)
        return probability.astype(np.float32), offsets.astype(np.float32)

    def predict_patches(self, patches: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not patches:
            return np.empty((0,), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        return self.predict_features(self.feature_batch(patches))

    def train_batch(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        target_offsets_px: np.ndarray,
    ) -> BatchMetricsV214:
        cfg = self.config
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        target_px = np.asarray(target_offsets_px, dtype=np.float32).reshape(-1, 2)

        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        p = self._sigmoid(raw[:, 0])
        pred_offset_norm = raw[:, 1:3]
        target_offset_norm = target_px / max(1e-6, float(cfg.offset_scale_px))

        eps = 1e-6
        sample_weight = np.where(y > 0.5, float(cfg.positive_class_weight), 1.0).astype(np.float32)
        bce = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        classification_loss = float(np.sum(sample_weight * bce) / max(eps, float(np.sum(sample_weight))))

        pos_mask = (y > 0.5).astype(np.float32)[:, None]
        pos_count = max(1.0, float(np.sum(pos_mask)))
        offset_error = pred_offset_norm - target_offset_norm
        offset_loss = float(np.sum(pos_mask * (offset_error ** 2)) / (2.0 * pos_count))
        loss = classification_loss + float(cfg.offset_loss_weight) * offset_loss

        weight_norm = max(eps, float(np.sum(sample_weight)))
        d_raw = np.zeros_like(raw, dtype=np.float32)
        d_raw[:, 0] = sample_weight * (p - y) / weight_norm
        d_raw[:, 1:3] = float(cfg.offset_loss_weight) * pos_mask * offset_error / pos_count

        grad_W2 = hidden.T @ d_raw + float(cfg.weight_decay) * self.W2
        grad_b2 = np.sum(d_raw, axis=0)
        d_hidden = d_raw @ self.W2.T
        d_hidden[hidden_pre <= 0.0] = 0.0
        grad_W1 = x.T @ d_hidden + float(cfg.weight_decay) * self.W1
        grad_b1 = np.sum(d_hidden, axis=0)
        grads = [grad_W1, grad_b1, grad_W2, grad_b2]

        total_norm = math.sqrt(sum(float(np.sum(grad.astype(np.float64) ** 2)) for grad in grads))
        clip = float(cfg.gradient_clip_norm)
        if clip > 0.0 and total_norm > clip:
            scale = clip / max(total_norm, 1e-9)
            grads = [grad * scale for grad in grads]
        self._adam_step(grads)

        pred = p >= 0.5
        truth = y >= 0.5
        positives = truth
        negatives = ~truth
        return BatchMetricsV214(
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
            "model_type": "hole_patch_mlp_v214_background_invariant",
            "config": asdict(self.config),
            "metadata": dict(metadata or {}),
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
                metadata_json=np.array(json.dumps(payload_meta, ensure_ascii=False)),
            )
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple["HolePatchAIV214", dict[str, Any]]:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            config = HolePatchAIConfigV214(**dict(metadata.get("config") or {}))
            model = cls(config=config)
            model.W1[...] = data["W1"].astype(np.float32)
            model.b1[...] = data["b1"].astype(np.float32)
            model.W2[...] = data["W2"].astype(np.float32)
            model.b2[...] = data["b2"].astype(np.float32)
        return model, metadata
