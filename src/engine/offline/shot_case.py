from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GroundTruth:
    """Known new-hole location in camera pixels."""

    x: float
    y: float
    space: str = "camera"

    def as_xy(self) -> tuple[float, float]:
        return float(self.x), float(self.y)

    def to_dict(self) -> dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y), "space": str(self.space)}

    @classmethod
    def from_value(cls, value: Any) -> GroundTruth | None:
        if value is None:
            return None
        if isinstance(value, GroundTruth):
            return value
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return cls(float(value[0]), float(value[1]))
            except Exception:
                return None
        if isinstance(value, dict):
            # Prefer explicitly camera-space fields, then generic x/y.
            x_keys = ("camera_x", "gt_camera_x", "ground_truth_camera_x", "x")
            y_keys = ("camera_y", "gt_camera_y", "ground_truth_camera_y", "y")
            x = next((value.get(key) for key in x_keys if value.get(key) is not None), None)
            y = next((value.get(key) for key in y_keys if value.get(key) is not None), None)
            if x is None or y is None:
                xy = value.get("xy") or value.get("camera_xy") or value.get("gt_xy")
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    x, y = xy[0], xy[1]
            try:
                if x is not None and y is not None:
                    return cls(float(x), float(y), str(value.get("space", "camera")))
            except Exception:
                return None
        return None


def _string_paths(values: Iterable[Path | str]) -> tuple[str, ...]:
    return tuple(str(Path(value)) for value in values)


@dataclass
class ShotCase:
    """One offline replay case.

    The manifest stores paths as strings so it stays portable.  Paths may be
    absolute or relative to ``root`` supplied by the replay CLI.
    """

    shot_id: str
    pre_images: tuple[str, ...]
    post_images: tuple[str, ...]
    ground_truth: GroundTruth | None = None
    session_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shot_id = str(self.shot_id)
        self.session_id = str(self.session_id or "unknown")
        self.pre_images = _string_paths(self.pre_images)
        self.post_images = _string_paths(self.post_images)
        if self.ground_truth is not None and not isinstance(self.ground_truth, GroundTruth):
            self.ground_truth = GroundTruth.from_value(self.ground_truth)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.12",
            "shot_id": self.shot_id,
            "session_id": self.session_id,
            "pre_images": list(self.pre_images),
            "post_images": list(self.post_images),
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShotCase:
        pre = data.get("pre_images") or data.get("before_images") or []
        post = data.get("post_images") or data.get("after_images") or []
        if isinstance(pre, (str, Path)):
            pre = [pre]
        if isinstance(post, (str, Path)):
            post = [post]
        gt = GroundTruth.from_value(
            data.get("ground_truth")
            or data.get("gt")
            or data.get("gt_xy")
            or data.get("ground_truth_xy")
        )
        return cls(
            shot_id=str(data.get("shot_id") or data.get("id") or "unknown"),
            session_id=str(data.get("session_id") or data.get("session") or "unknown"),
            pre_images=tuple(str(item) for item in pre),
            post_images=tuple(str(item) for item in post),
            ground_truth=gt,
            metadata=dict(data.get("metadata") or {}),
        )

    def resolve_path(self, value: str | Path, root: Path | None = None) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (Path(root) / path) if root is not None else path

    def resolved_pre_paths(self, root: Path | None = None) -> tuple[Path, ...]:
        return tuple(self.resolve_path(value, root) for value in self.pre_images)

    def resolved_post_paths(self, root: Path | None = None) -> tuple[Path, ...]:
        return tuple(self.resolve_path(value, root) for value in self.post_images)


def write_manifest(path: Path, cases: Iterable[ShotCase]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_manifest(path: Path) -> list[ShotCase]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(text)
        return [ShotCase.from_dict(row) for row in payload if isinstance(row, dict)]
    cases: list[ShotCase] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(row, dict):
            cases.append(ShotCase.from_dict(row))
    return cases
