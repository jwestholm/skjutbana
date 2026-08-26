from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class NewHoleRankerConfigV218:
    hidden_size: int = 56
    learning_rate: float = 0.0025
    weight_decay: float = 2e-5
    listwise_temperature: float = 0.85
    pairwise_margin: float = 0.35
    pairwise_loss_weight: float = 0.30
    pairwise_hard_negatives: int = 24
    offset_loss_weight: float = 0.28
    residual_offset_scale_px: float = 36.0
    max_total_offset_px: float = 44.0
    gradient_clip_norm: float = 5.0


@dataclass
class GroupTrainMetricsV218:
    loss: float
    listwise_loss: float
    pairwise_loss: float
    offset_loss: float
    positive_score: float
    hardest_negative_score: float


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(1e-4, float(temperature))
    x = np.asarray(values, dtype=np.float32) / temperature
    x = x - float(np.max(x))
    exp = np.exp(np.clip(x, -40.0, 40.0))
    return exp / max(1e-9, float(np.sum(exp)))


class NewHoleRankerV218:
    """Candidate-aware listwise head over a frozen V2.17 temporal embedding.

    The V2.17 image/temporal backbone stays frozen.  V2.18 trains a small head
    against *whole candidate groups from the same shot*.  This fixes the main
    semantic failure observed in V2.17: good pointwise NEW/NOT-NEW AUC did not
    translate into useful ordering among ~384 real candidates.
    """

    schema_version = "2.18"

    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        config: NewHoleRankerConfigV218 | None = None,
        *,
        seed: int = 21801,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.context_dim = int(context_dim)
        self.config = config or NewHoleRankerConfigV218()
        if self.embedding_dim <= 0 or self.context_dim <= 0:
            raise ValueError("embedding_dim/context_dim must be >0")
        in_dim = self.embedding_dim + self.context_dim
        hidden = int(self.config.hidden_size)
        rng = np.random.default_rng(int(seed))
        self.W1 = (rng.standard_normal((in_dim, hidden)).astype(np.float32) * math.sqrt(2.0 / max(1, in_dim))).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        # score + residual dx/dy
        self.W2 = (rng.standard_normal((hidden, 3)).astype(np.float32) * 0.02).astype(np.float32)
        self.b2 = np.zeros((3,), dtype=np.float32)
        self.embedding_mean = np.zeros((self.embedding_dim,), dtype=np.float32)
        self.embedding_std = np.ones((self.embedding_dim,), dtype=np.float32)
        self.context_mean = np.zeros((self.context_dim,), dtype=np.float32)
        self.context_std = np.ones((self.context_dim,), dtype=np.float32)
        self._adam_m = [np.zeros_like(p) for p in self.parameters()]
        self._adam_v = [np.zeros_like(p) for p in self.parameters()]
        self._adam_t = 0

    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_normalisation(
        self,
        embedding_mean: np.ndarray,
        embedding_std: np.ndarray,
        context_mean: np.ndarray,
        context_std: np.ndarray,
    ) -> None:
        self.embedding_mean = np.asarray(embedding_mean, dtype=np.float32).reshape(self.embedding_dim)
        self.embedding_std = np.maximum(np.asarray(embedding_std, dtype=np.float32).reshape(self.embedding_dim), 1e-4)
        self.context_mean = np.asarray(context_mean, dtype=np.float32).reshape(self.context_dim)
        self.context_std = np.maximum(np.asarray(context_std, dtype=np.float32).reshape(self.context_dim), 1e-4)

    def _input(self, embedding: np.ndarray, context: np.ndarray) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float32)
        ctx = np.asarray(context, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.embedding_dim:
            raise ValueError("embedding shape mismatch")
        if ctx.ndim != 2 or ctx.shape != (emb.shape[0], self.context_dim):
            raise ValueError("context shape mismatch")
        emb = np.clip((emb - self.embedding_mean) / self.embedding_std, -6.0, 6.0)
        ctx = np.clip((ctx - self.context_mean) / self.context_std, -6.0, 6.0)
        return np.concatenate([emb, ctx], axis=1).astype(np.float32)

    def _forward(self, embedding: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self._input(embedding, context)
        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        score = raw[:, 0]
        residual = raw[:, 1:3] * float(self.config.residual_offset_scale_px)
        return score.astype(np.float32), residual.astype(np.float32), hidden, hidden_pre

    def predict(
        self,
        embedding: np.ndarray,
        context: np.ndarray,
        base_offsets_px: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        score, residual, _, _ = self._forward(embedding, context)
        base = np.asarray(base_offsets_px, dtype=np.float32).reshape(-1, 2)
        if len(base) != len(score):
            raise ValueError("base offset length mismatch")
        total = (base + residual).astype(np.float32)
        max_offset = float(self.config.max_total_offset_px)
        if max_offset > 0.0 and len(total):
            mag = np.linalg.norm(total, axis=1)
            scale = np.minimum(1.0, max_offset / np.maximum(mag, 1e-6)).astype(np.float32)
            total = total * scale[:, None]
        return score, total.astype(np.float32)

    def train_group(
        self,
        embedding: np.ndarray,
        context: np.ndarray,
        relevance: np.ndarray,
        target_offsets_px: np.ndarray,
        base_offsets_px: np.ndarray,
        distances_px: np.ndarray,
        *,
        group_weight: float = 1.0,
    ) -> GroupTrainMetricsV218:
        cfg = self.config
        rel = np.asarray(relevance, dtype=np.float32).reshape(-1)
        distances = np.asarray(distances_px, dtype=np.float32).reshape(-1)
        target_offsets = np.asarray(target_offsets_px, dtype=np.float32).reshape(-1, 2)
        base_offsets = np.asarray(base_offsets_px, dtype=np.float32).reshape(-1, 2)
        n = len(rel)
        if n < 2 or not np.any(rel > 0.0):
            raise ValueError("listwise group requires >=2 candidates and positive relevance")

        x = self._input(embedding, context)
        hidden_pre = x @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        raw = hidden @ self.W2 + self.b2
        scores = raw[:, 0]
        residual_norm = raw[:, 1:3]

        target_distribution = rel / max(1e-9, float(np.sum(rel)))
        pred_distribution = _softmax(scores, float(cfg.listwise_temperature))
        eps = 1e-8
        listwise_loss = -float(np.sum(target_distribution * np.log(pred_distribution + eps)))
        d_score = (pred_distribution - target_distribution) / max(1e-4, float(cfg.listwise_temperature))

        # Explicit hard pairwise pressure: closest/relevant candidate must beat
        # high-scoring candidates that are clearly far from the current GT.
        positive_candidates = np.flatnonzero(rel > 0.0)
        positive_index = int(positive_candidates[np.argmin(distances[positive_candidates])])
        negative_indices = np.flatnonzero(distances >= 55.0)
        pairwise_loss = 0.0
        if len(negative_indices):
            ordered = negative_indices[np.argsort(-scores[negative_indices])]
            ordered = ordered[: max(1, int(cfg.pairwise_hard_negatives))]
            violations = []
            for neg in ordered.tolist():
                value = float(cfg.pairwise_margin) - float(scores[positive_index]) + float(scores[neg])
                if value > 0.0:
                    violations.append((int(neg), value))
            if violations:
                scale = 1.0 / len(violations)
                pairwise_loss = float(sum(v for _, v in violations) * scale)
                pair_grad = float(cfg.pairwise_loss_weight) * scale
                d_score[positive_index] -= pair_grad * len(violations)
                for neg, _ in violations:
                    d_score[neg] += pair_grad

        # Offset refinement is graded rather than binary. Candidates within 42px
        # can contain useful NEW-hole signal even when they are outside the 20px
        # success radius.  V2.18 learns a residual over V2.17's existing offset.
        offset_weight = np.where(
            distances <= 12.0, 1.0,
            np.where(distances <= 20.0, 0.9,
                     np.where(distances <= 32.0, 0.55,
                              np.where(distances <= 42.0, 0.25, 0.0))),
        ).astype(np.float32)
        residual_target = (target_offsets - base_offsets) / max(1e-6, float(cfg.residual_offset_scale_px))
        error = residual_norm - residual_target
        weight2 = offset_weight[:, None]
        denom = max(1.0, float(np.sum(offset_weight)))
        offset_loss = float(np.sum(weight2 * (error ** 2)) / (2.0 * denom))

        d_raw = np.zeros_like(raw, dtype=np.float32)
        gw = float(max(0.05, group_weight))
        d_raw[:, 0] = gw * d_score
        d_raw[:, 1:3] = gw * float(cfg.offset_loss_weight) * weight2 * error / denom

        list_component = gw * listwise_loss
        pair_component = gw * float(cfg.pairwise_loss_weight) * pairwise_loss
        offset_component = gw * float(cfg.offset_loss_weight) * offset_loss
        loss = list_component + pair_component + offset_component

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
            factor = clip / max(1e-9, total_norm)
            grads = [g * factor for g in grads]
        self._adam_step(grads)

        hard = negative_indices[np.argmax(scores[negative_indices])] if len(negative_indices) else positive_index
        return GroupTrainMetricsV218(
            loss=float(loss),
            listwise_loss=float(list_component),
            pairwise_loss=float(pair_component),
            offset_loss=float(offset_component),
            positive_score=float(scores[positive_index]),
            hardest_negative_score=float(scores[int(hard)]),
        )

    def _adam_step(self, grads: Sequence[np.ndarray]) -> None:
        self._adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = float(self.config.learning_rate)
        for i, (param, grad) in enumerate(zip(self.parameters(), grads)):
            self._adam_m[i] = b1 * self._adam_m[i] + (1.0 - b1) * grad
            self._adam_v[i] = b2 * self._adam_v[i] + (1.0 - b2) * (grad * grad)
            m = self._adam_m[i] / (1.0 - b1 ** self._adam_t)
            v = self._adam_v[i] / (1.0 - b2 ** self._adam_t)
            param -= lr * m / (np.sqrt(v) + eps)

    def state(self) -> list[np.ndarray]:
        return [p.copy() for p in self.parameters()]

    def restore(self, state: Sequence[np.ndarray]) -> None:
        for param, saved in zip(self.parameters(), state):
            param[...] = np.asarray(saved, dtype=np.float32)

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": self.schema_version,
            "model_type": "candidate_aware_new_hole_listwise_v218",
            "embedding_dim": self.embedding_dim,
            "context_dim": self.context_dim,
            "config": asdict(self.config),
            "metadata": dict(metadata or {}),
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                embedding_mean=self.embedding_mean, embedding_std=self.embedding_std,
                context_mean=self.context_mean, context_std=self.context_std,
                metadata_json=np.array(json.dumps(meta, ensure_ascii=False)),
            )
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple["NewHoleRankerV218", dict[str, Any]]:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"].item()))
            model = cls(
                int(meta["embedding_dim"]), int(meta["context_dim"]),
                NewHoleRankerConfigV218(**dict(meta.get("config") or {})),
            )
            for name in ("W1", "b1", "W2", "b2", "embedding_mean", "embedding_std", "context_mean", "context_std"):
                getattr(model, name)[...] = np.asarray(data[name], dtype=np.float32)
        return model, meta
