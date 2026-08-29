from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import SCHEMA_VERSION, ShotTrainingRecord, candidate_rows_from_pool

DEFAULT_ROOT = Path("content/ai/training_v223")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_session_id(prefix: str = "session") -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}_{uuid.uuid4().hex[:8]}"


class TrainingCaptureV223:
    """Append-only writer for native V2.23 shot groups."""

    def __init__(
        self,
        *,
        source_kind: str,
        session_id: str | None = None,
        root: Path | str = DEFAULT_ROOT,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.session_id = session_id or make_session_id(source_kind)
        self.source_kind = str(source_kind)
        self.session_dir = self.root / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.session_dir / "index.jsonl"
        self.session_path = self.session_dir / "session.json"
        self.metadata = dict(metadata or {})
        self.started_at = time.time()
        self.shots_saved = 0
        self.closed = False
        self._write_session(status="open")

    def _write_session(self, *, status: str) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "source_kind": self.source_kind,
            "status": status,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "shots_saved": self.shots_saved,
            "metadata": self.metadata,
        }
        _atomic_write_text(self.session_path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
        self._write_session(status="closed" if self.closed else "open")

    def save_record(self, record: ShotTrainingRecord) -> Path:
        if self.closed:
            raise RuntimeError("capture session is closed")
        if record.session_id != self.session_id:
            record.session_id = self.session_id
        record.source_kind = record.source_kind or self.source_kind
        seq = self.shots_saved + 1
        path = self.session_dir / f"shot_{seq:06d}.json"
        payload = record.to_dict()
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        summary = {
            "shot_file": path.name,
            "shot_id": record.shot_id,
            "timestamp": record.timestamp,
            "source_kind": record.source_kind,
            "background": record.background,
            "candidate_count": len(record.candidates),
            "oracle20": record.oracle20,
            "nearest_distance_px": payload.get("nearest_distance_px"),
        }
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
        self.shots_saved = seq
        self._write_session(status="open")
        return path

    def save_from_candidates(
        self,
        *,
        shot_id: str | int,
        candidates: Sequence[Mapping[str, Any]],
        gt_camera_xy: tuple[float, float],
        gt_screen_xy: tuple[float, float] | None = None,
        timestamp: float | None = None,
        background: str = "unknown",
        sampling_mode: str | None = None,
        frame_shape: tuple[int, int] | None = None,
        source_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        rows = candidate_rows_from_pool(candidates, gt_camera_xy=gt_camera_xy, frame_shape=frame_shape)
        record = ShotTrainingRecord(
            session_id=self.session_id,
            shot_id=str(shot_id),
            source_kind=str(source_kind or self.source_kind),
            timestamp=float(timestamp if timestamp is not None else time.time()),
            gt_camera_x=float(gt_camera_xy[0]),
            gt_camera_y=float(gt_camera_xy[1]),
            gt_screen_x=(float(gt_screen_xy[0]) if gt_screen_xy else None),
            gt_screen_y=(float(gt_screen_xy[1]) if gt_screen_xy else None),
            background=str(background),
            sampling_mode=sampling_mode,
            candidates=rows,
            metadata=dict(metadata or {}),
        )
        return self.save_record(record)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._write_session(status="closed")
