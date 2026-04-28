"""
Hotspot funnel diagnostics for Skjutbana hit detection.

Tracks where in the pipeline the true hit is lost:
  raw hotspots → noise filter → AI ranking → selected hit

Saves per-shot logs and generates summary reports as CSV.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

AI_DIR = Path("content/ai")
REPORTS_DIR = AI_DIR / "reports"


@dataclass
class RoundRecord:
    """Per-round metrics — single source of truth for all reporting."""

    round_id: int
    timestamp: float

    # Ground truth
    gt_screen_x: float = 0.0
    gt_screen_y: float = 0.0
    gt_camera_x: float = 0.0
    gt_camera_y: float = 0.0

    # Candidate counts
    candidate_count_raw: int = 0
    candidate_count_ranked: int = 0

    # Detection results
    found: bool = False
    top1_correct: bool = False
    top3_correct: bool = False
    nearest_dist: float = 9999.0

    # AI guess pre-facit
    ai_guess_camera_x: float = 0.0
    ai_guess_camera_y: float = 0.0
    ai_guess_dist_to_gt: float = 9999.0
    ai_guess_correct: bool = False

    # Context
    sampling_mode: str = "center_bias"
    match_radius_px: float = 42.0
    background_mode: str = "white"

    def to_csv_dict(self) -> Dict[str, Any]:
        """Flat dict suitable for CSV export."""
        return {
            "round_id": self.round_id,
            "timestamp": self.timestamp,
            "gt_screen_x": round(self.gt_screen_x, 1),
            "gt_screen_y": round(self.gt_screen_y, 1),
            "gt_camera_x": round(self.gt_camera_x, 1),
            "gt_camera_y": round(self.gt_camera_y, 1),
            "candidate_count_raw": self.candidate_count_raw,
            "candidate_count_ranked": self.candidate_count_ranked,
            "found": self.found,
            "top1_correct": self.top1_correct,
            "top3_correct": self.top3_correct,
            "nearest_dist": round(self.nearest_dist, 1),
            "ai_guess_camera_x": round(self.ai_guess_camera_x, 1),
            "ai_guess_camera_y": round(self.ai_guess_camera_y, 1),
            "ai_guess_dist_to_gt": round(self.ai_guess_dist_to_gt, 1),
            "ai_guess_correct": self.ai_guess_correct,
            "sampling_mode": self.sampling_mode,
            "match_radius_px": self.match_radius_px,
            "background_mode": self.background_mode,
        }


ROUND_RECORD_CSV_FIELDS = [
    "round_id", "timestamp",
    "gt_screen_x", "gt_screen_y", "gt_camera_x", "gt_camera_y",
    "candidate_count_raw", "candidate_count_ranked",
    "found", "top1_correct", "top3_correct", "nearest_dist",
    "ai_guess_camera_x", "ai_guess_camera_y", "ai_guess_dist_to_gt", "ai_guess_correct",
    "sampling_mode", "match_radius_px", "background_mode",
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


class ShotDiagnostics:
    """Diagnostics for a single shot through the hotspot funnel."""

    def __init__(
        self,
        gt_x: float,
        gt_y: float,
        match_radius_px: float = 10.0,
    ) -> None:
        self.gt_x = float(gt_x)
        self.gt_y = float(gt_y)
        self.match_radius_px = float(match_radius_px)
        self.timestamp = time.time()

        # Raw hotspots (from hit_scanner)
        self.raw_hotspot_count: int = 0
        self.raw_contains_gt: bool = False
        self.raw_closest_dist: float = 9999.0
        self.raw_closest_xy: Optional[Tuple[float, float]] = None

        # After noise filter
        self.filtered_count: int = 0
        self.gt_survived_filter: bool = False
        self.filter_closest_dist: float = 9999.0
        self.filter_killed_gt: bool = False

        # After AI ranking (top-K)
        self.ai_topk_count: int = 0
        self.gt_in_topk: bool = False
        self.ai_topk_closest_dist: float = 9999.0

        # Final selected
        self.selected_x: float = 0.0
        self.selected_y: float = 0.0
        self.selected_dist: float = 9999.0
        self.ai_selected_correct: bool = False

        # Rejection stats
        self.rejected_by: Dict[str, int] = {}

        # Multi-radius matching
        self.within_5px: bool = False
        self.within_10px: bool = False
        self.within_15px: bool = False
        self.within_20px: bool = False

    def _dist_to_gt(self, x: float, y: float) -> float:
        return math.hypot(x - self.gt_x, y - self.gt_y)

    def _check_radii(self, dist: float) -> None:
        if dist <= 5.0:
            self.within_5px = True
        if dist <= 10.0:
            self.within_10px = True
        if dist <= 15.0:
            self.within_15px = True
        if dist <= 20.0:
            self.within_20px = True

    def evaluate_raw_hotspots(self, hotspots: Sequence[Dict[str, Any]]) -> None:
        """Evaluate raw hotspots before any filtering."""
        self.raw_hotspot_count = len(hotspots)
        for hs in hotspots:
            x = _safe_float(hs.get("camera_x", 0.0))
            y = _safe_float(hs.get("camera_y", 0.0))
            d = self._dist_to_gt(x, y)
            if d < self.raw_closest_dist:
                self.raw_closest_dist = d
                self.raw_closest_xy = (x, y)
        self.raw_contains_gt = self.raw_closest_dist <= self.match_radius_px

    def evaluate_filtered(self, surviving: Sequence[Dict[str, Any]], rejection_counts: Optional[Dict[str, int]] = None) -> None:
        """Evaluate hotspots after noise filtering."""
        self.filtered_count = len(surviving)
        if rejection_counts:
            self.rejected_by = dict(rejection_counts)

        closest = 9999.0
        for hs in surviving:
            x = _safe_float(hs.get("camera_x", 0.0))
            y = _safe_float(hs.get("camera_y", 0.0))
            d = self._dist_to_gt(x, y)
            if d < closest:
                closest = d
        self.filter_closest_dist = closest
        self.gt_survived_filter = closest <= self.match_radius_px
        self.filter_killed_gt = self.raw_contains_gt and not self.gt_survived_filter

    def evaluate_ai_ranked(self, ranked: Sequence[Dict[str, Any]], top_k: int = 10) -> None:
        """Evaluate AI-ranked hotspots."""
        top = ranked[:top_k]
        self.ai_topk_count = len(top)
        closest = 9999.0
        for hs in top:
            x = _safe_float(hs.get("camera_x", 0.0))
            y = _safe_float(hs.get("camera_y", 0.0))
            d = self._dist_to_gt(x, y)
            if d < closest:
                closest = d
        self.ai_topk_closest_dist = closest
        self.gt_in_topk = closest <= self.match_radius_px

    def evaluate_selected(self, selected: Optional[Dict[str, Any]]) -> None:
        """Evaluate the final selected hit."""
        if selected is None:
            return
        self.selected_x = _safe_float(selected.get("camera_x", 0.0))
        self.selected_y = _safe_float(selected.get("camera_y", 0.0))
        self.selected_dist = self._dist_to_gt(self.selected_x, self.selected_y)
        self.ai_selected_correct = self.selected_dist <= self.match_radius_px
        self._check_radii(self.selected_dist)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gt_x": self.gt_x,
            "gt_y": self.gt_y,
            "match_radius_px": self.match_radius_px,
            "raw_hotspot_count": self.raw_hotspot_count,
            "raw_contains_gt": self.raw_contains_gt,
            "raw_closest_dist": round(self.raw_closest_dist, 1),
            "filtered_count": self.filtered_count,
            "gt_survived_filter": self.gt_survived_filter,
            "filter_killed_gt": self.filter_killed_gt,
            "filter_closest_dist": round(self.filter_closest_dist, 1),
            "ai_topk_count": self.ai_topk_count,
            "gt_in_topk": self.gt_in_topk,
            "ai_topk_closest_dist": round(self.ai_topk_closest_dist, 1),
            "selected_dist": round(self.selected_dist, 1),
            "ai_selected_correct": self.ai_selected_correct,
            "within_5px": self.within_5px,
            "within_10px": self.within_10px,
            "within_15px": self.within_15px,
            "within_20px": self.within_20px,
            "rejected_by": self.rejected_by,
        }


class FunnelTracker:
    """Accumulates ShotDiagnostics across a session and generates reports."""

    def __init__(self) -> None:
        self.shots: List[ShotDiagnostics] = []

    def add(self, diag: ShotDiagnostics) -> None:
        self.shots.append(diag)

    def clear(self) -> None:
        self.shots.clear()

    def summary(self) -> Dict[str, Any]:
        n = len(self.shots)
        if n == 0:
            return {"total_shots": 0}

        raw_gt = sum(1 for s in self.shots if s.raw_contains_gt)
        survived = sum(1 for s in self.shots if s.gt_survived_filter)
        killed = sum(1 for s in self.shots if s.filter_killed_gt)
        in_topk = sum(1 for s in self.shots if s.gt_in_topk)
        correct = sum(1 for s in self.shots if s.ai_selected_correct)
        w5 = sum(1 for s in self.shots if s.within_5px)
        w10 = sum(1 for s in self.shots if s.within_10px)
        w15 = sum(1 for s in self.shots if s.within_15px)
        w20 = sum(1 for s in self.shots if s.within_20px)

        raw_dists = [s.raw_closest_dist for s in self.shots if s.raw_closest_dist < 9000]
        sel_dists = [s.selected_dist for s in self.shots if s.selected_dist < 9000]

        return {
            "total_shots": n,
            "raw_contains_gt_pct": round(100.0 * raw_gt / n, 1),
            "gt_survived_filter_pct": round(100.0 * survived / n, 1),
            "filter_killed_gt_count": killed,
            "gt_in_topk_pct": round(100.0 * in_topk / n, 1),
            "ai_correct_pct": round(100.0 * correct / n, 1),
            "within_5px_pct": round(100.0 * w5 / n, 1),
            "within_10px_pct": round(100.0 * w10 / n, 1),
            "within_15px_pct": round(100.0 * w15 / n, 1),
            "within_20px_pct": round(100.0 * w20 / n, 1),
            "avg_raw_closest_dist": round(sum(raw_dists) / len(raw_dists), 1) if raw_dists else 0.0,
            "avg_selected_dist": round(sum(sel_dists) / len(sel_dists), 1) if sel_dists else 0.0,
        }

    def save_csv(self, label: str = "session", round_records: Optional[List["RoundRecord"]] = None) -> Optional[Path]:
        """Save per-shot diagnostics as CSV.

        If round_records is provided, writes one row per RoundRecord with all
        fields (single source of truth). Otherwise falls back to ShotDiagnostics.
        """
        if round_records is not None:
            return self._save_round_records_csv(label, round_records)

        if not self.shots:
            return None
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"funnel_{label}_{stamp}.csv"

        fieldnames = [
            "timestamp", "gt_x", "gt_y", "match_radius_px",
            "raw_hotspot_count", "raw_contains_gt", "raw_closest_dist",
            "filtered_count", "gt_survived_filter", "filter_killed_gt", "filter_closest_dist",
            "ai_topk_count", "gt_in_topk", "ai_topk_closest_dist",
            "selected_dist", "ai_selected_correct",
            "within_5px", "within_10px", "within_15px", "within_20px",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for shot in self.shots:
                writer.writerow(shot.to_dict())

        return path

    def _save_round_records_csv(self, label: str, records: List["RoundRecord"]) -> Optional[Path]:
        """Save round records as CSV — includes all RoundRecord fields + funnel data."""
        if not records:
            return None
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"funnel_{label}_{stamp}.csv"

        # Merge RoundRecord fields with ShotDiagnostics fields
        funnel_fields = [
            "raw_hotspot_count", "raw_contains_gt", "raw_closest_dist",
            "filtered_count", "gt_survived_filter", "filter_killed_gt",
            "ai_topk_count", "gt_in_topk", "selected_dist", "ai_selected_correct",
            "within_5px", "within_10px", "within_15px", "within_20px",
        ]
        all_fields = list(ROUND_RECORD_CSV_FIELDS) + funnel_fields

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for i, rec in enumerate(records):
                row = rec.to_csv_dict()
                # Merge funnel data if available
                if i < len(self.shots):
                    shot_dict = self.shots[i].to_dict()
                    for key in funnel_fields:
                        if key in shot_dict:
                            row[key] = shot_dict[key]
                writer.writerow(row)

        return path

    def format_summary_lines(self) -> List[str]:
        s = self.summary()
        if s["total_shots"] == 0:
            return ["Inga skott loggade."]
        return [
            f"Totalt: {s['total_shots']} skott",
            f"Rätt hål bland raw hotspots: {s['raw_contains_gt_pct']}%",
            f"Överlevde filter: {s['gt_survived_filter_pct']}%",
            f"Filter dödade GT: {s['filter_killed_gt_count']} gånger",
            f"I AI top-K: {s['gt_in_topk_pct']}%",
            f"AI valde rätt: {s['ai_correct_pct']}%",
            f"Inom 5px: {s['within_5px_pct']}% | 10px: {s['within_10px_pct']}% | 15px: {s['within_15px_pct']}% | 20px: {s['within_20px_pct']}%",
            f"Medel raw-avstånd: {s['avg_raw_closest_dist']}px | Medel valt: {s['avg_selected_dist']}px",
        ]
