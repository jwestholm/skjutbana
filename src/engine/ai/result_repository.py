from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AI_DIR = Path("content/ai")
MEMORY_FILE = AI_DIR / "memory.json"
TRAINING_EXAMPLES_DIR = AI_DIR / "sessions"
LEGACY_REPORTS_DIR = AI_DIR / "reports"
AUTOMATION_RUNS_DIR = AI_DIR / "automation_runs"


@dataclass(frozen=True)
class AIResultPoint:
    """
    One comparable completed benchmark run.

    A point can originate from:
      - legacy/current F1/F2 CSV in content/ai/reports
      - automation loop JSONL in content/ai/automation_runs

    Percent values use 0..100.
    """

    timestamp: float
    source: str
    source_id: str
    background: str
    iterations: int
    found_pct: float | None
    top1_pct: float | None
    top3_pct: float | None
    ai_correct_pct: float | None
    avg_distance_px: float | None
    git_commit: str | None = None
    session_id: str | None = None
    run_number: int | None = None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "ja"}


def _pct(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(100.0 * count / total, 3)


def _parse_iso_timestamp(value: Any) -> float | None:
    if not value:
        return None

    text = str(value).strip()

    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return time.mktime(time.strptime(text, fmt))
        except Exception:
            pass

    return _safe_float(value)


def _timestamp_from_filename(path: Path) -> float:
    match = re.search(r"(20\d{6})[_-](\d{6})", path.name)

    if match:
        try:
            return time.mktime(
                time.strptime(
                    match.group(1) + match.group(2),
                    "%Y%m%d%H%M%S",
                )
            )
        except Exception:
            pass

    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


class AIResultsRepository:
    """
    Read-only history facade plus explicit reset helpers.

    IMPORTANT DATA SEMANTICS

    content/ai/memory.json
        Active learned AI memory. This is what AISettingsScene's R command
        resets through runtime.memory.reset().

    content/ai/sessions/
        Saved/manual training examples. Archival examples, not the active model
        memory itself.

    content/ai/reports/
        Existing F1/F2 benchmark CSV files produced by FunnelTracker.save_csv().

    content/ai/automation_runs/
        New repeated automation-loop results. These complement the CSV reports;
        they do not replace them.

    The result graph intentionally reads reports + automation_runs, while memory
    and training examples are shown as separate storage/state counters.
    """

    def load_points(self) -> list[AIResultPoint]:
        points = []
        points.extend(self._load_legacy_csv_points())
        points.extend(self._load_automation_points())

        # De-duplicate conservatively. Automation runs usually also create a
        # legacy CSV. Prefer automation because it carries richer metadata.
        automation_keys: set[tuple[str, int, int]] = set()
        for point in points:
            if point.source == "automation":
                automation_keys.add(
                    (
                        point.background,
                        int(point.timestamp // 3),
                        point.iterations,
                    )
                )

        filtered: list[AIResultPoint] = []
        for point in points:
            if point.source == "legacy_csv":
                key = (
                    point.background,
                    int(point.timestamp // 3),
                    point.iterations,
                )
                if key in automation_keys:
                    continue
            filtered.append(point)

        return sorted(filtered, key=lambda p: (p.timestamp, p.source_id))

    def backgrounds(self, points: list[AIResultPoint] | None = None) -> list[str]:
        data = self.load_points() if points is None else points
        return sorted({p.background for p in data if p.background})

    def storage_summary(self) -> dict[str, Any]:
        return {
            "memory_file_exists": MEMORY_FILE.exists(),
            "training_example_files": self._count_files(TRAINING_EXAMPLES_DIR, "*.json"),
            "legacy_report_files": self._count_files(LEGACY_REPORTS_DIR, "*.csv"),
            "automation_session_dirs": self._count_dirs(AUTOMATION_RUNS_DIR),
            "automation_run_files": self._count_files(AUTOMATION_RUNS_DIR, "run_*.json"),
        }

    def clear_result_history(self) -> dict[str, int]:
        """
        Delete historical benchmark/result files only.

        Does NOT touch:
          memory.json
          sessions/
          settings.json
          exports/
        """
        legacy = self._delete_matching(LEGACY_REPORTS_DIR, "*.csv")
        automation = self._delete_tree_contents(AUTOMATION_RUNS_DIR)

        return {
            "legacy_reports_deleted": legacy,
            "automation_entries_deleted": automation,
        }

    def clear_training_examples(self) -> int:
        """
        Delete archived training-example JSON files only.

        This does not clear the active memory.json. Use runtime.memory.reset()
        separately for that.
        """
        return self._delete_matching(TRAINING_EXAMPLES_DIR, "*.json")

    def _load_automation_points(self) -> list[AIResultPoint]:
        result: list[AIResultPoint] = []

        if not AUTOMATION_RUNS_DIR.exists():
            return result

        for session_dir in sorted(AUTOMATION_RUNS_DIR.iterdir()):
            if not session_dir.is_dir():
                continue

            session_meta = self._read_json(session_dir / "session.json")
            summary_meta = self._read_json(session_dir / "summary.json")

            session_id = str(
                session_meta.get("session_id")
                or summary_meta.get("session_id")
                or session_dir.name
            )
            git_commit = (
                session_meta.get("git_commit")
                or summary_meta.get("git_commit")
            )
            started_ts = (
                _parse_iso_timestamp(session_meta.get("started_at"))
                or _parse_iso_timestamp(summary_meta.get("started_at"))
                or _timestamp_from_filename(session_dir)
            )

            jsonl = session_dir / "runs.jsonl"

            if jsonl.exists():
                compact_rows = self._read_jsonl(jsonl)
            else:
                compact_rows = []

            # Fallback to individual full run files if JSONL is absent.
            if not compact_rows:
                for run_file in sorted(session_dir.glob("run_*.json")):
                    full = self._read_json(run_file)
                    compact = self._compact_from_full_run(full)
                    if compact:
                        compact_rows.append(compact)

            for index, row in enumerate(compact_rows, start=1):
                if not isinstance(row, dict):
                    continue

                run_number = _safe_int(row.get("run"), index)
                background = str(
                    row.get("background")
                    or summary_meta.get("background_requested")
                    or "unknown"
                )
                iterations = _safe_int(row.get("iterations"), 0)

                # Compact rows are written sequentially. If no exact timestamp
                # exists, preserve chronological order with a tiny offset.
                timestamp = (
                    _parse_iso_timestamp(row.get("captured_at"))
                    or started_ts + (run_number * 0.001)
                )

                result.append(
                    AIResultPoint(
                        timestamp=timestamp,
                        source="automation",
                        source_id=f"{session_id}/run_{run_number:03d}",
                        background=background,
                        iterations=iterations,
                        found_pct=_safe_float(row.get("found_pct")),
                        top1_pct=_safe_float(row.get("top1_pct")),
                        top3_pct=_safe_float(row.get("top3_pct")),
                        ai_correct_pct=_safe_float(
                            row.get("ai_guess_correct_pct")
                            if row.get("ai_guess_correct_pct") is not None
                            else row.get("funnel_ai_correct_pct")
                        ),
                        avg_distance_px=_safe_float(
                            row.get("avg_nearest_distance_px")
                        ),
                        git_commit=str(git_commit) if git_commit else None,
                        session_id=session_id,
                        run_number=run_number,
                    )
                )

        return result

    def _load_legacy_csv_points(self) -> list[AIResultPoint]:
        result: list[AIResultPoint] = []

        if not LEGACY_REPORTS_DIR.exists():
            return result

        for path in sorted(LEGACY_REPORTS_DIR.glob("*.csv")):
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except Exception:
                continue

            if not rows:
                continue

            total = len(rows)
            background = str(rows[0].get("background_mode") or "unknown")

            found = sum(1 for row in rows if _bool_value(row.get("found")))
            top1 = sum(1 for row in rows if _bool_value(row.get("top1_correct")))
            top3 = sum(1 for row in rows if _bool_value(row.get("top3_correct")))
            ai_correct = sum(
                1 for row in rows if _bool_value(row.get("ai_guess_correct"))
            )

            distances = []
            for row in rows:
                value = _safe_float(row.get("nearest_dist"))
                if value is not None and value < 9000:
                    distances.append(value)

            timestamp = (
                _safe_float(rows[-1].get("timestamp"))
                or _timestamp_from_filename(path)
            )

            result.append(
                AIResultPoint(
                    timestamp=float(timestamp),
                    source="legacy_csv",
                    source_id=path.name,
                    background=background,
                    iterations=total,
                    found_pct=_pct(found, total),
                    top1_pct=_pct(top1, total),
                    top3_pct=_pct(top3, total),
                    ai_correct_pct=_pct(ai_correct, total),
                    avg_distance_px=(
                        round(sum(distances) / len(distances), 3)
                        if distances
                        else None
                    ),
                )
            )

        return result

    @staticmethod
    def _compact_from_full_run(full: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(full, dict):
            return None

        event = full.get("event", {})
        data = event.get("data", {}) if isinstance(event, dict) else {}
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}

        if not isinstance(data, dict):
            return None
        if not isinstance(metrics, dict):
            metrics = {}

        nearest = metrics.get("nearest_distance_px", {})
        if not isinstance(nearest, dict):
            nearest = {}

        return {
            "run": full.get("run"),
            "captured_at": full.get("captured_at"),
            "background": data.get("background"),
            "iterations": metrics.get("iterations", data.get("iterations", 0)),
            "found_pct": metrics.get("found_pct"),
            "top1_pct": metrics.get("top1_pct"),
            "top3_pct": metrics.get("top3_pct"),
            "ai_guess_correct_pct": metrics.get("ai_guess_correct_pct"),
            "avg_nearest_distance_px": nearest.get("avg"),
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows = []

        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
        except Exception:
            pass

        return rows

    @staticmethod
    def _count_files(root: Path, pattern: str) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.rglob(pattern) if path.is_file())

    @staticmethod
    def _count_dirs(root: Path) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.iterdir() if path.is_dir())

    @staticmethod
    def _delete_matching(root: Path, pattern: str) -> int:
        if not root.exists():
            return 0

        deleted = 0
        for path in list(root.rglob(pattern)):
            if not path.is_file():
                continue
            try:
                path.unlink()
                deleted += 1
            except Exception:
                pass

        return deleted

    @staticmethod
    def _delete_tree_contents(root: Path) -> int:
        if not root.exists():
            return 0

        deleted = 0
        for path in list(root.iterdir()):
            try:
                if path.is_dir():
                    # Count files for human-readable feedback.
                    deleted += sum(1 for p in path.rglob("*") if p.is_file())
                    shutil.rmtree(path)
                else:
                    path.unlink()
                    deleted += 1
            except Exception:
                pass

        return deleted
