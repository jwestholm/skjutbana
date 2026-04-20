from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .settings import AI_MODEL_PATH, load_ai_settings


@dataclass
class MemoryVector:
    features: list[float]
    label: int  # 1 or 0
    weight: float
    timestamp: float
    source: str = "manual"


class PrototypeMemoryModel:
    """A tiny bounded online model.

    It stores a capped set of positive/negative examples and scores a feature
    vector by comparing it against weighted prototypes plus recent memory.
    This keeps disk usage small and supports immediate learning after each shot.
    """

    def __init__(self) -> None:
        self.positive: list[MemoryVector] = []
        self.negative: list[MemoryVector] = []
        self.version = 1
        self.total_updates = 0
        self.last_saved_ts = 0.0
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        path = Path(AI_MODEL_PATH)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        self.version = int(raw.get("version", 1))
        self.total_updates = int(raw.get("total_updates", 0))
        self.positive = [self._memory_from_raw(item, 1) for item in raw.get("positive", []) if isinstance(item, dict)]
        self.negative = [self._memory_from_raw(item, 0) for item in raw.get("negative", []) if isinstance(item, dict)]

    def save(self) -> None:
        settings = load_ai_settings()
        max_pos = int(settings.get("max_positive_memories", 256))
        max_neg = int(settings.get("max_negative_memories", 384))
        self.positive = self.positive[-max_pos:]
        self.negative = self.negative[-max_neg:]
        data = {
            "version": self.version,
            "saved_at": time.time(),
            "total_updates": self.total_updates,
            "positive": [asdict(m) for m in self.positive],
            "negative": [asdict(m) for m in self.negative],
        }
        path = Path(AI_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.last_saved_ts = time.time()

    def reset(self) -> None:
        self.positive.clear()
        self.negative.clear()
        self.total_updates = 0
        self.save()

    def _memory_from_raw(self, raw: dict[str, Any], fallback_label: int) -> MemoryVector:
        feats = raw.get("features", [])
        if not isinstance(feats, list):
            feats = []
        return MemoryVector(
            features=[float(x) for x in feats],
            label=int(raw.get("label", fallback_label)),
            weight=float(raw.get("weight", 1.0)),
            timestamp=float(raw.get("timestamp", time.time())),
            source=str(raw.get("source", "manual")),
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def add_sample(self, features: list[float], label: int, *, weight: float = 1.0, source: str = "manual") -> None:
        sample = MemoryVector(
            features=[float(x) for x in features],
            label=1 if int(label) else 0,
            weight=float(weight),
            timestamp=time.time(),
            source=str(source),
        )
        target = self.positive if sample.label == 1 else self.negative
        target.append(sample)
        self.total_updates += 1
        if self.total_updates % 3 == 0:
            self.save()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, features: list[float]) -> float:
        x = np.array(features, dtype=np.float32)
        if x.size == 0:
            return 0.5
        if not self.positive and not self.negative:
            return 0.5

        pos_sim = self._avg_similarity(x, self.positive)
        neg_sim = self._avg_similarity(x, self.negative)
        margin = pos_sim - neg_sim
        return float(1.0 / (1.0 + math.exp(-4.0 * margin)))

    def _avg_similarity(self, x: np.ndarray, memories: list[MemoryVector]) -> float:
        if not memories:
            return 0.0
        sims: list[float] = []
        weights: list[float] = []
        for mem in memories[-64:]:
            m = np.array(mem.features, dtype=np.float32)
            n = min(x.size, m.size)
            if n <= 0:
                continue
            dist = float(np.linalg.norm(x[:n] - m[:n]) / math.sqrt(float(n)))
            sims.append(1.0 / (1.0 + dist))
            weights.append(max(0.05, float(mem.weight)))
        if not sims:
            return 0.0
        return float(np.average(np.array(sims, dtype=np.float32), weights=np.array(weights, dtype=np.float32)))

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "positive_count": len(self.positive),
            "negative_count": len(self.negative),
            "total_updates": self.total_updates,
            "last_saved_ts": self.last_saved_ts,
        }
