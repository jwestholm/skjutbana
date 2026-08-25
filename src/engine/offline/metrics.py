from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


DEFAULT_RADII = (5.0, 10.0, 20.0, 42.0)


def nearest_distance(candidates: Sequence[dict[str, Any]], gt_xy: tuple[float, float]) -> float | None:
    best = float("inf")
    gx, gy = float(gt_xy[0]), float(gt_xy[1])
    for candidate in candidates:
        try:
            distance = math.hypot(float(candidate["camera_x"]) - gx, float(candidate["camera_y"]) - gy)
        except Exception:
            continue
        best = min(best, distance)
    return None if not math.isfinite(best) else float(best)


def within(distance: float | None, radius: float) -> bool:
    return distance is not None and float(distance) <= float(radius)


@dataclass
class ReplayMetrics:
    radii: tuple[float, ...] = DEFAULT_RADII
    shots_total: int = 0
    shots_labelled: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    hits: dict[str, dict[float, int]] = field(default_factory=dict)
    overlay_rescues: dict[float, int] = field(default_factory=dict)
    detector_only: dict[float, int] = field(default_factory=dict)
    both: dict[float, int] = field(default_factory=dict)
    neither: dict[float, int] = field(default_factory=dict)
    candidate_totals: dict[str, int] = field(default_factory=dict)

    def add(self, result: dict[str, Any]) -> None:
        self.shots_total += 1
        sources = result.get("sources") if isinstance(result.get("sources"), dict) else {}
        for source, payload in sources.items():
            count = int(payload.get("candidate_count", 0)) if isinstance(payload, dict) else 0
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
            self.candidate_totals[source] = self.candidate_totals.get(source, 0) + count

        gt = result.get("ground_truth")
        if not isinstance(gt, dict):
            return
        self.shots_labelled += 1
        distances = {
            source: (payload.get("nearest_gt_distance") if isinstance(payload, dict) else None)
            for source, payload in sources.items()
        }
        for source, distance in distances.items():
            bucket = self.hits.setdefault(source, {radius: 0 for radius in self.radii})
            for radius in self.radii:
                if within(distance, radius):
                    bucket[radius] += 1

        detector_key = "current_detector" if "current_detector" in distances else ("v2" if "v2" in distances else None)
        if detector_key is not None and "overlay" in distances:
            detector_distance = distances.get(detector_key)
            overlay_distance = distances.get("overlay")
            for radius in self.radii:
                detector_hit = within(detector_distance, radius)
                overlay_hit = within(overlay_distance, radius)
                if overlay_hit and not detector_hit:
                    self.overlay_rescues[radius] = self.overlay_rescues.get(radius, 0) + 1
                elif detector_hit and not overlay_hit:
                    self.detector_only[radius] = self.detector_only.get(radius, 0) + 1
                elif detector_hit and overlay_hit:
                    self.both[radius] = self.both.get(radius, 0) + 1
                else:
                    self.neither[radius] = self.neither.get(radius, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        labelled = max(1, self.shots_labelled)
        recall: dict[str, dict[str, float]] = {}
        for source, radius_counts in self.hits.items():
            recall[source] = {
                str(int(radius) if float(radius).is_integer() else radius): round(100.0 * count / labelled, 4)
                for radius, count in radius_counts.items()
            }
        avg_candidates = {
            source: round(total / max(1, self.source_counts.get(source, 1)), 3)
            for source, total in self.candidate_totals.items()
        }
        return {
            "schema_version": "2.12",
            "shots_total": self.shots_total,
            "shots_labelled": self.shots_labelled,
            "recall_percent": recall,
            "average_candidates": avg_candidates,
            "complementarity": {
                "overlay_rescues_current_detector": {str(radius): self.overlay_rescues.get(radius, 0) for radius in self.radii},
                "current_detector_only": {str(radius): self.detector_only.get(radius, 0) for radius in self.radii},
                "both": {str(radius): self.both.get(radius, 0) for radius in self.radii},
                "neither": {str(radius): self.neither.get(radius, 0) for radius in self.radii},
                # Compatibility key for any early V2.12 scripts already written.
                "overlay_rescues_v2": {str(radius): self.overlay_rescues.get(radius, 0) for radius in self.radii},
            },
        }
