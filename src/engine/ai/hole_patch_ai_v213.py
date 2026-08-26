from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HolePatchAIConfig:
    """Small dependency-free pixel model used for the first Hole-AI proof.

    It intentionally uses only numpy + OpenCV already present in the project.
    This is a one-hidden-layer neural network, not the final architecture.  Its
    job in V2.13 is to answer one question cheaply and honestly: do raw pixels
    add held-out information beyond hand-written detector features?
    """

    crop_size: int = 64
    input_size: int = 24
    hidden_size: int = 64
    offset_scale_px: float = 20.0
    feature_channels: int = 2  # normalized gray + local high-pass
    learning_rate: float = 0.0015
    weight_decay: float = 1e-5
    offset_loss_weight: float = 0.35
    positive_class_weight: float = 1.0
    gradient_clip_norm: float = 5.0

    @property
    def input_dim(self) -> int:
        return int(self.input_size * self.input_size * self.feature_channels)


@dataclass
class BatchMetrics:
    loss: float
    classification_loss: float
    offset_loss: float
    accuracy: float
    positive_recall: float
    negative_specificity: float


class HolePatchAI:
    schema_version = "2.13"

    def __init__(self, config: HolePatchAIConfig | None = None, *, seed: int = 21301):
        self.config = config or HolePatchAIConfig()
        rng = np.random.default_rng(int(seed))
        in_dim = self.config.input_dim
        hidden = int(self.config.hidden_size)
        # He initialization for ReLU.
        self.W1 = (rng.standard_normal((in_dim, hidden)).astype(np.float32) * math.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 3)).astype(np.float32) * math.sqrt(1.0 / hidden)).astype(np.float32)
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

    def features_from_patch(self, patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if patch.ndim != 2 or not patch.size:
            raise ValueError("HolePatchAI expects a non-empty grayscale/BGR candidate patch")
        size = int(self.config.input_size)
        resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        mean = float(np.mean(resized))
        std = float(np.std(resized))
        normalized = (resized - mean) / max(0.06, std)
        normalized = np.clip(normalized, -4.0, 4.0) / 4.0

        local = cv2.GaussianBlur(resized, (5, 5), 0.9)
        highpass = resized - local
        hp_scale = max(0.035, float(np.std(highpass)) * 2.5)
        highpass = np.clip(highpass / hp_scale, -3.0, 3.0) / 3.0

        if int(self.config.feature_channels) == 1:
            channels = [normalized]
        else:
            channels = [normalized, highpass]
        return np.concatenate([channel.reshape(-1) for channel in channels]).astype(np.float32)

    def feature_batch(self, patches: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.features_from_patch(patch) for patch in patches], axis=0).astype(np.float32)

    def _forward_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        hidden_pre = features @ self.W1 + self.b1
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
    ) -> BatchMetrics:
        cfg = self.config
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        target_px = np.asarray(target_offsets_px, dtype=np.float32).reshape(-1, 2)
        n = max(1, x.shape[0])

        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        p = self._sigmoid(raw[:, 0])
        pred_offset_norm = raw[:, 1:3]
        target_offset_norm = target_px / max(1e-6, float(cfg.offset_scale_px))

        eps = 1e-6
        pos_weight = float(cfg.positive_class_weight)
        sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
        bce = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        classification_loss = float(np.sum(sample_weight * bce) / max(eps, float(np.sum(sample_weight))))

        pos_mask = (y > 0.5).astype(np.float32)[:, None]
        pos_count = max(1.0, float(np.sum(pos_mask)))
        offset_error = pred_offset_norm - target_offset_norm
        offset_loss = float(np.sum(pos_mask * (offset_error ** 2)) / (2.0 * pos_count))
        loss = classification_loss + float(cfg.offset_loss_weight) * offset_loss

        # BCE derivative with optional class weights.
        weight_norm = max(eps, float(np.sum(sample_weight)))
        d_raw = np.zeros_like(raw, dtype=np.float32)
        d_raw[:, 0] = sample_weight * (p - y) / weight_norm
        d_raw[:, 1:3] = (
            float(cfg.offset_loss_weight) * pos_mask * offset_error / pos_count
        )

        grad_W2 = hidden.T @ d_raw + float(cfg.weight_decay) * self.W2
        grad_b2 = np.sum(d_raw, axis=0)
        d_hidden = d_raw @ self.W2.T
        d_hidden[hidden_pre <= 0.0] = 0.0
        grad_W1 = x.T @ d_hidden + float(cfg.weight_decay) * self.W1
        grad_b1 = np.sum(d_hidden, axis=0)
        grads = [grad_W1, grad_b1, grad_W2, grad_b2]

        # Global gradient clipping keeps the tiny numpy trainer stable on unusual
        # camera patches without introducing another ML dependency.
        total_norm = math.sqrt(sum(float(np.sum(grad.astype(np.float64) ** 2)) for grad in grads))
        clip = float(cfg.gradient_clip_norm)
        if clip > 0.0 and total_norm > clip:
            scale = clip / max(total_norm, 1e-9)
            grads = [grad * scale for grad in grads]

        self._adam_step(grads)

        pred = p >= 0.5
        truth = y >= 0.5
        accuracy = float(np.mean(pred == truth)) if y.size else 0.0
        positives = truth
        negatives = ~truth
        positive_recall = float(np.mean(pred[positives])) if np.any(positives) else 0.0
        negative_specificity = float(np.mean(~pred[negatives])) if np.any(negatives) else 0.0
        return BatchMetrics(
            loss=float(loss),
            classification_loss=classification_loss,
            offset_loss=offset_loss,
            accuracy=accuracy,
            positive_recall=positive_recall,
            negative_specificity=negative_specificity,
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
            "model_type": "hole_patch_mlp_v213",
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
    def load(cls, path: Path) -> tuple["HolePatchAI", dict[str, Any]]:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            raw_meta = str(data["metadata_json"].item())
            metadata = json.loads(raw_meta)
            config = HolePatchAIConfig(**dict(metadata.get("config") or {}))
            model = cls(config=config)
            model.W1[...] = data["W1"].astype(np.float32)
            model.b1[...] = data["b1"].astype(np.float32)
            model.W2[...] = data["W2"].astype(np.float32)
            model.b2[...] = data["b2"].astype(np.float32)
        return model, metadata
