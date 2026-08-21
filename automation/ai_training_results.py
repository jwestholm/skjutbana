from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "1.0"
DEFAULT_RESULTS_ROOT = Path("content/ai/automation_runs")


def utc_iso(timestamp: float | None = None) -> str:
    ts = time.time() if timestamp is None else float(timestamp)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def background_label(background: int | str) -> str:
    return str(background).strip().lower().replace(" ", "_")


class AITrainingRunStore:
    """
    Persist repeated AI-training runs in formats that are easy for both humans
    and future AI agents to consume.

    Files written per session:
      session.json  - session metadata/status
      run_001.json  - complete structured event incl. RoundRecord rows
      runs.jsonl    - compact one-line record per completed run
      runs.csv      - compact tabular overview
      summary.json  - aggregate metrics and trend information
    """

    CSV_FIELDS = [
        "run",
        "background",
        "iterations",
        "found",
        "found_pct",
        "top1",
        "top1_pct",
        "top3",
        "top3_pct",
        "ai_guess_correct",
        "ai_guess_correct_pct",
        "avg_nearest_distance_px",
        "avg_ai_guess_distance_px",
        "avg_raw_candidates",
        "zero_raw_candidate_rounds",
        "raw_contains_gt_pct",
        "gt_survived_filter_pct",
        "gt_in_topk_pct",
        "funnel_ai_correct_pct",
        "training_duration_seconds",
        "scene_duration_seconds",
        "counts_match",
        "file",
    ]

    def __init__(
        self,
        background: int | str,
        requested_runs: int,
        *,
        root: Path = DEFAULT_RESULTS_ROOT,
    ) -> None:
        self.background = background_label(background)
        self.requested_runs = int(requested_runs)
        self.started_ts = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started_ts))
        self.session_id = f"{stamp}_{self.background}_{self.requested_runs}runs"
        self.directory = root / self.session_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.git_commit = get_git_commit()
        self.compact_runs: list[dict[str, Any]] = []

        self._write_session(status="running")
        self._write_csv_header()

    def save_run(
        self,
        run_number: int,
        completed_event: dict[str, Any],
        *,
        wall_duration_seconds: float,
    ) -> dict[str, Any]:
        data = completed_event.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("aiTraining.completed event has invalid data")

        run_file = f"run_{int(run_number):03d}.json"
        full_record = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": "ai_training_run",
            "session_id": self.session_id,
            "run": int(run_number),
            "requested_runs": self.requested_runs,
            "captured_at": utc_iso(),
            "git_commit": self.git_commit,
            "wall_duration_seconds": round(float(wall_duration_seconds), 3),
            "event": completed_event,
        }
        self._write_json(self.directory / run_file, full_record)

        compact = self._compact_run(run_number, run_file, data, wall_duration_seconds)
        self.compact_runs.append(compact)
        self._append_jsonl(compact)
        self._append_csv(compact)
        self._write_session(status="running")
        return compact

    def finalize(self) -> dict[str, Any]:
        completed_ts = time.time()
        summary = self._build_summary(completed_ts)
        self._write_json(self.directory / "summary.json", summary)
        self._write_session(status="completed", completed_ts=completed_ts)
        return summary

    def mark_failed(self, error: str) -> None:
        self._write_session(status="failed", error=str(error), completed_ts=time.time())

    def _compact_run(
        self,
        run_number: int,
        run_file: str,
        data: dict[str, Any],
        wall_duration_seconds: float,
    ) -> dict[str, Any]:
        metrics = data.get("metrics", {})
        funnel = data.get("funnel", {})
        consistency = data.get("consistency", {})

        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(funnel, dict):
            funnel = {}
        if not isinstance(consistency, dict):
            consistency = {}

        nearest = metrics.get("nearest_distance_px", {})
        ai_distance = metrics.get("ai_guess_distance_px", {})
        raw_candidates = metrics.get("candidates_raw", {})
        if not isinstance(nearest, dict):
            nearest = {}
        if not isinstance(ai_distance, dict):
            ai_distance = {}
        if not isinstance(raw_candidates, dict):
            raw_candidates = {}

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run": int(run_number),
            "background": data.get("background"),
            "background_number": data.get("background_number"),
            "sampling_mode": data.get("sampling_mode"),
            "match_radius_px": data.get("match_radius_px"),
            "iterations": metrics.get("iterations", data.get("iterations", 0)),
            "found": metrics.get("found", data.get("found", 0)),
            "found_pct": metrics.get("found_pct"),
            "top1": metrics.get("top1", data.get("top1", 0)),
            "top1_pct": metrics.get("top1_pct"),
            "top3": metrics.get("top3", data.get("top3", 0)),
            "top3_pct": metrics.get("top3_pct"),
            "ai_guess_correct": metrics.get(
                "ai_guess_correct", data.get("ai_guess_correct", 0)
            ),
            "ai_guess_correct_pct": metrics.get("ai_guess_correct_pct"),
            "avg_nearest_distance_px": nearest.get("avg"),
            "avg_ai_guess_distance_px": ai_distance.get("avg"),
            "avg_raw_candidates": raw_candidates.get("avg"),
            "zero_raw_candidate_rounds": raw_candidates.get("zero_count"),
            "raw_contains_gt_pct": funnel.get("raw_contains_gt_pct"),
            "gt_survived_filter_pct": funnel.get("gt_survived_filter_pct"),
            "gt_in_topk_pct": funnel.get("gt_in_topk_pct"),
            "funnel_ai_correct_pct": funnel.get("ai_correct_pct"),
            "training_duration_seconds": data.get("training_duration_seconds"),
            "scene_duration_seconds": data.get("scene_duration_seconds"),
            "wall_duration_seconds": round(float(wall_duration_seconds), 3),
            "counts_match": consistency.get("counts_match"),
            "file": run_file,
        }

    def _build_summary(self, completed_ts: float) -> dict[str, Any]:
        runs = list(self.compact_runs)
        total_iterations = sum(int(run.get("iterations") or 0) for run in runs)

        aggregate_counts = {
            "iterations": total_iterations,
            "found": sum(int(run.get("found") or 0) for run in runs),
            "top1": sum(int(run.get("top1") or 0) for run in runs),
            "top3": sum(int(run.get("top3") or 0) for run in runs),
            "ai_guess_correct": sum(
                int(run.get("ai_guess_correct") or 0) for run in runs
            ),
        }
        aggregate_counts["found_pct"] = self._pct(
            aggregate_counts["found"], total_iterations
        )
        aggregate_counts["top1_pct"] = self._pct(
            aggregate_counts["top1"], total_iterations
        )
        aggregate_counts["top3_pct"] = self._pct(
            aggregate_counts["top3"], total_iterations
        )
        aggregate_counts["ai_guess_correct_pct"] = self._pct(
            aggregate_counts["ai_guess_correct"], total_iterations
        )

        best_found = self._best_run(runs, "found_pct")
        best_ai = self._best_run(runs, "ai_guess_correct_pct")
        first = runs[0] if runs else None
        last = runs[-1] if runs else None

        trend = {
            "first_run": self._trend_point(first),
            "last_run": self._trend_point(last),
            "best_found_run": self._trend_point(best_found),
            "best_ai_run": self._trend_point(best_ai),
            "first_to_last_found_pct_delta": self._delta(first, last, "found_pct"),
            "first_to_last_ai_correct_pct_delta": self._delta(
                first, last, "ai_guess_correct_pct"
            ),
        }

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": "ai_training_loop_summary",
            "session_id": self.session_id,
            "status": "completed",
            "started_at": utc_iso(self.started_ts),
            "completed_at": utc_iso(completed_ts),
            "duration_seconds": round(completed_ts - self.started_ts, 3),
            "git_commit": self.git_commit,
            "background_requested": self.background,
            "requested_runs": self.requested_runs,
            "completed_runs": len(runs),
            "total_synthetic_shots": total_iterations,
            "aggregate": aggregate_counts,
            "per_run_average": {
                "found_pct": self._avg_field(runs, "found_pct"),
                "top1_pct": self._avg_field(runs, "top1_pct"),
                "top3_pct": self._avg_field(runs, "top3_pct"),
                "ai_guess_correct_pct": self._avg_field(
                    runs, "ai_guess_correct_pct"
                ),
                "raw_contains_gt_pct": self._avg_field(
                    runs, "raw_contains_gt_pct"
                ),
                "gt_survived_filter_pct": self._avg_field(
                    runs, "gt_survived_filter_pct"
                ),
                "gt_in_topk_pct": self._avg_field(runs, "gt_in_topk_pct"),
                "avg_nearest_distance_px": self._avg_field(
                    runs, "avg_nearest_distance_px"
                ),
                "avg_raw_candidates": self._avg_field(
                    runs, "avg_raw_candidates"
                ),
                "wall_duration_seconds": self._avg_field(
                    runs, "wall_duration_seconds"
                ),
            },
            "trend": trend,
            "consistency": {
                "all_runs_counts_match": all(
                    run.get("counts_match") is True for run in runs
                ) if runs else False,
            },
            "files": {
                "runs_jsonl": "runs.jsonl",
                "runs_csv": "runs.csv",
                "individual_runs": "run_NNN.json",
            },
            "runs": runs,
        }

    def _write_session(
        self,
        *,
        status: str,
        error: str | None = None,
        completed_ts: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": "ai_training_loop_session",
            "session_id": self.session_id,
            "status": status,
            "started_at": utc_iso(self.started_ts),
            "git_commit": self.git_commit,
            "background_requested": self.background,
            "requested_runs": self.requested_runs,
            "completed_runs": len(self.compact_runs),
        }
        if completed_ts is not None:
            payload["completed_at"] = utc_iso(completed_ts)
        if error is not None:
            payload["error"] = error
        self._write_json(self.directory / "session.json", payload)

    def _append_jsonl(self, compact: dict[str, Any]) -> None:
        with (self.directory / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _write_csv_header(self) -> None:
        with (self.directory / "runs.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writeheader()

    def _append_csv(self, compact: dict[str, Any]) -> None:
        with (self.directory / "runs.csv").open(
            "a", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.CSV_FIELDS,
                extrasaction="ignore",
            )
            writer.writerow(compact)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temp.replace(path)

    @staticmethod
    def _pct(value: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round(100.0 * float(value) / float(total), 3)

    @staticmethod
    def _avg_field(runs: list[dict[str, Any]], field: str) -> float | None:
        values = [
            float(run[field])
            for run in runs
            if run.get(field) is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    @staticmethod
    def _best_run(
        runs: list[dict[str, Any]],
        field: str,
    ) -> dict[str, Any] | None:
        candidates = [run for run in runs if run.get(field) is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda run: float(run[field]))

    @staticmethod
    def _trend_point(run: dict[str, Any] | None) -> dict[str, Any] | None:
        if run is None:
            return None
        return {
            "run": run.get("run"),
            "found_pct": run.get("found_pct"),
            "top1_pct": run.get("top1_pct"),
            "top3_pct": run.get("top3_pct"),
            "ai_guess_correct_pct": run.get("ai_guess_correct_pct"),
            "avg_nearest_distance_px": run.get("avg_nearest_distance_px"),
            "file": run.get("file"),
        }

    @staticmethod
    def _delta(
        first: dict[str, Any] | None,
        last: dict[str, Any] | None,
        field: str,
    ) -> float | None:
        if first is None or last is None:
            return None
        if first.get(field) is None or last.get(field) is None:
            return None
        return round(float(last[field]) - float(first[field]), 3)
