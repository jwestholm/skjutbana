from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evidence import EvidenceConfig, build_evidence, extract_overlay_candidates, merge_candidate_sources
from .live_detector_replay import LiveHybridReplayDetector
from .scenario_generator_v220 import GeneratedScenarioV220, OfflineScenarioGeneratorV220

RADII = (5.0, 10.0, 20.0, 42.0)


def _distance(candidate: dict[str, Any], gt: tuple[float, float]) -> float:
    return math.hypot(float(candidate.get("camera_x", 0.0)) - gt[0], float(candidate.get("camera_y", 0.0)) - gt[1])


def _hit(candidates: Sequence[dict[str, Any]], gt: tuple[float, float], radius: float) -> bool:
    return any(_distance(candidate, gt) <= radius for candidate in candidates)


@dataclass
class ScenarioBenchmarkConfigV220:
    seeds: int = 100
    first_seed: int = 1
    split: str = "validation"
    candidate_limit: int = 500
    use_overlay: bool = True
    save_failures: int = 20


class ScenarioBenchmarkV220:
    def __init__(self, *, generator: OfflineScenarioGeneratorV220, config: ScenarioBenchmarkConfigV220 | None = None) -> None:
        self.generator = generator
        self.config = config or ScenarioBenchmarkConfigV220()
        self.detector = LiveHybridReplayDetector()
        self.evidence_config = EvidenceConfig.from_file()

    def _one(self, scenario: GeneratedScenarioV220) -> dict[str, Any]:
        gt = scenario.spec.gt_camera_xy
        started = time.perf_counter()
        result = self.detector.detect(
            pre_frames=scenario.pre_frames,
            post_frames=scenario.post_frames,
            known_holes=scenario.spec.known_holes,
            ground_truth=gt,
            candidate_limit=int(self.config.candidate_limit),
        )
        detector_ms = 1000.0 * (time.perf_counter() - started)
        current = [dict(candidate) for candidate in result.candidates]
        overlay: list[dict[str, Any]] = []
        union = list(current)
        overlay_ms = 0.0
        if self.config.use_overlay:
            started = time.perf_counter()
            bundle = build_evidence(scenario.pre_frames, scenario.post_frames, config=self.evidence_config)
            overlay = extract_overlay_candidates(bundle.fused, config=self.evidence_config)
            union = merge_candidate_sources(
                [("current_detector", current), ("physical_fusion_v212", overlay)],
                merge_radius_px=5.0,
                limit=700,
            )
            overlay_ms = 1000.0 * (time.perf_counter() - started)
        return {
            "seed": scenario.spec.seed,
            "media_id": scenario.spec.media_id,
            "media_category": scenario.spec.media_category,
            "media_kind": scenario.spec.media_kind,
            "challenge_tags": scenario.spec.challenge_tags,
            "old_holes": len(scenario.spec.old_holes),
            "known_holes": len(scenario.spec.known_holes),
            "gt": [float(gt[0]), float(gt[1])],
            "counts": {"current": len(current), "overlay": len(overlay), "union": len(union)},
            "recall": {
                "current": {str(int(r)): _hit(current, gt, r) for r in RADII},
                "overlay": {str(int(r)): _hit(overlay, gt, r) for r in RADII},
                "union": {str(int(r)): _hit(union, gt, r) for r in RADII},
            },
            "nearest": {
                "current": min((_distance(candidate, gt) for candidate in current), default=9999.0),
                "overlay": min((_distance(candidate, gt) for candidate in overlay), default=9999.0),
                "union": min((_distance(candidate, gt) for candidate in union), default=9999.0),
            },
            "timing_ms": {"current_detector": detector_ms, "overlay": overlay_ms},
        }

    @staticmethod
    def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        total = max(1, len(rows))
        sources = ("current", "overlay", "union")
        recall: dict[str, dict[str, float]] = {}
        for source in sources:
            recall[source] = {str(int(radius)): sum(1 for row in rows if row["recall"][source][str(int(radius))]) / total for radius in RADII}
        return {
            "shots": len(rows),
            "recall": recall,
            "candidate_count_mean": float(np.mean([row["counts"]["current"] for row in rows])) if rows else 0.0,
            "detector_ms_mean": float(np.mean([row["timing_ms"]["current_detector"] for row in rows])) if rows else 0.0,
        }

    def run(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for offset in range(int(self.config.seeds)):
            seed = int(self.config.first_seed) + offset
            scenario = self.generator.generate(seed, split=self.config.split)
            rows.append(self._one(scenario))
            if (offset + 1) % 10 == 0 or offset + 1 == int(self.config.seeds):
                print(f"V2.20 benchmark: {offset + 1}/{self.config.seeds} scenarios")
        return {
            "schema_version": "2.20",
            "split": self.config.split,
            "seed_range": [int(self.config.first_seed), int(self.config.first_seed) + max(0, int(self.config.seeds) - 1)],
            "rows": rows,
            "summary": self._aggregate(rows),
            "warning": "Synthetic/media benchmark only; physical sessions remain the release authority.",
        }


def write_benchmark(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
