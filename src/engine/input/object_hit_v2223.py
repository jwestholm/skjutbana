"""V2.22.3 object-hit foundation (shadow only).

Games may register a small set of screen/game-space hit regions and ask a much
simpler question than global localisation: "did this object receive the shot?".

V2.22.3 deliberately keeps this path SHADOW-only.  The ordinary global
HitScanner/HitInput path remains authoritative.  The registry snapshots object
geometry at the audio-shot timestamp, then scores the already-produced camera
candidates against those frozen regions.  A later version may replace the
candidate-backed shadow evaluator with a direct PRE->POST per-object fast path
without changing the game API introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "2.22.3"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-cast point-in-polygon; boundary counts as inside."""
    pts = list(polygon)
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        # Boundary check first.
        dx = xj - xi
        dy = yj - yi
        cross = (x - xi) * dy - (y - yi) * dx
        if abs(cross) <= 1e-6:
            dot = (x - xi) * dx + (y - yi) * dy
            if -1e-6 <= dot <= dx * dx + dy * dy + 1e-6:
                return True
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def viewport_center_prior(screen_x: float, screen_y: float, rect_xywh: Sequence[float]) -> tuple[float, float]:
    """Return (centre_prior, edge_distance_norm), both 0..1.

    This is contextual evidence only.  It MUST NOT hard-reject a physically
    strong edge hit.  Explicit game-object context is stronger than this generic
    spatial prior.
    """
    x0, y0, w, h = (float(v) for v in rect_xywh)
    if w <= 1e-9 or h <= 1e-9:
        return 0.5, 0.0
    nx = (float(screen_x) - x0) / w
    ny = (float(screen_y) - y0) / h
    edge = max(0.0, min(1.0, min(nx, 1.0 - nx, ny, 1.0 - ny) * 2.0))
    dx = (nx - 0.5) / 0.5
    dy = (ny - 0.5) / 0.5
    radius = min(1.0, math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))
    centre = max(0.0, min(1.0, 1.0 - radius))
    return centre, edge


@dataclass(frozen=True)
class HitRegionV2223:
    object_id: str
    polygon_screen: tuple[tuple[float, float], ...]
    priority: float = 1.0
    enabled: bool = True
    owner: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_mapping(value: Mapping[str, Any]) -> "HitRegionV2223 | None":
        object_id = str(value.get("object_id", value.get("id", ""))).strip()
        if not object_id:
            return None
        polygon = value.get("polygon", value.get("polygon_screen"))
        if polygon is None and "rect" in value:
            rect = value.get("rect")
            try:
                x, y, w, h = (float(v) for v in rect)
                polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            except Exception:
                polygon = None
        if polygon is None:
            return None
        clean: list[tuple[float, float]] = []
        try:
            for point in polygon:
                clean.append((float(point[0]), float(point[1])))
        except Exception:
            return None
        if len(clean) < 3:
            return None
        return HitRegionV2223(
            object_id=object_id,
            polygon_screen=tuple(clean),
            priority=max(0.0, _finite(value.get("priority", 1.0), 1.0)),
            enabled=bool(value.get("enabled", value.get("hittable", True))),
            owner=str(value.get("owner", "")),
            metadata=dict(value.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ObjectHitResultV2223:
    shot_id: int
    object_id: str
    hit: bool
    confidence: float
    camera_x: float | None
    camera_y: float | None
    screen_x: float | None
    screen_y: float | None
    local_x: float | None
    local_y: float | None
    candidate_score: float
    shot_novelty: float
    hole_likeness: float
    centre_prior: float
    edge_distance_norm: float
    candidate_rank: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectShotSnapshotV2223:
    shot_id: int
    peak_ts: float
    created_ts: float
    regions: tuple[HitRegionV2223, ...]
    results: dict[str, ObjectHitResultV2223] = field(default_factory=dict)


class ObjectHitRegistryV2223:
    """Thread-safe registry + per-shot frozen snapshots.

    Games may either register regions directly or implement
    ``scene.get_hit_regions()`` returning mapping-compatible region objects.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._regions: dict[str, HitRegionV2223] = {}
        self._snapshots: dict[int, ObjectShotSnapshotV2223] = {}
        self._latest_shot_id = 0

    def register_polygon(
        self,
        object_id: str,
        polygon_screen: Sequence[Sequence[float]],
        *,
        priority: float = 1.0,
        enabled: bool = True,
        owner: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> HitRegionV2223:
        region = HitRegionV2223.from_mapping({
            "object_id": str(object_id),
            "polygon": polygon_screen,
            "priority": float(priority),
            "enabled": bool(enabled),
            "owner": str(owner),
            "metadata": dict(metadata or {}),
        })
        if region is None:
            raise ValueError("invalid hit region")
        with self._lock:
            self._regions[region.object_id] = region
        return region

    def register_rect(
        self,
        object_id: str,
        rect_xywh: Sequence[float],
        **kwargs: Any,
    ) -> HitRegionV2223:
        x, y, w, h = (float(v) for v in rect_xywh)
        return self.register_polygon(
            object_id,
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            **kwargs,
        )

    def unregister(self, object_id: str) -> None:
        with self._lock:
            self._regions.pop(str(object_id), None)

    def clear(self, *, owner: str | None = None) -> None:
        with self._lock:
            if owner is None:
                self._regions.clear()
                return
            key = str(owner)
            self._regions = {k: v for k, v in self._regions.items() if v.owner != key}

    def _scene_regions(self, scene: Any) -> list[HitRegionV2223]:
        provider = getattr(scene, "get_hit_regions", None)
        if not callable(provider):
            return []
        try:
            values = provider() or []
        except Exception:
            return []
        result: list[HitRegionV2223] = []
        for value in values:
            if isinstance(value, HitRegionV2223):
                region = value
            elif isinstance(value, Mapping):
                region = HitRegionV2223.from_mapping(value)
            else:
                region = None
            if region is not None:
                result.append(region)
        return result

    def snapshot(self, shot_id: int, peak_ts: float, scene: Any = None) -> ObjectShotSnapshotV2223:
        with self._lock:
            merged = dict(self._regions)
            if scene is not None:
                for region in self._scene_regions(scene):
                    merged[region.object_id] = region
            regions = tuple(v for v in merged.values() if v.enabled)
            snap = ObjectShotSnapshotV2223(
                shot_id=int(shot_id),
                peak_ts=float(peak_ts),
                created_ts=time.time(),
                regions=regions,
            )
            self._snapshots[int(shot_id)] = snap
            self._latest_shot_id = max(self._latest_shot_id, int(shot_id))
            # Keep only a short diagnostic history.
            for old in sorted(self._snapshots)[:-32]:
                self._snapshots.pop(old, None)
            return snap

    def snapshot_for_shot(self, shot_id: int) -> ObjectShotSnapshotV2223 | None:
        with self._lock:
            return self._snapshots.get(int(shot_id))

    def game_context(self, shot_id: int) -> dict[str, Any] | None:
        with self._lock:
            snap = self._snapshots.get(int(shot_id))
            if snap is None:
                return None
            return {
                "schema": SCHEMA_VERSION,
                "shot_id": snap.shot_id,
                "peak_ts": snap.peak_ts,
                "hit_regions": [
                    {
                        "object_id": r.object_id,
                        "polygon_screen": [list(p) for p in r.polygon_screen],
                        "priority": r.priority,
                        "metadata": dict(r.metadata),
                    }
                    for r in snap.regions
                ],
            }

    def evaluate_candidates(
        self,
        shot_id: int,
        candidates: Sequence[Mapping[str, Any]],
        *,
        camera_to_screen,
        viewport_rect_xywh: Sequence[float] | None = None,
    ) -> list[ObjectHitResultV2223]:
        with self._lock:
            snap = self._snapshots.get(int(shot_id))
            if snap is None or not snap.regions:
                return []

        projected: list[tuple[int, Mapping[str, Any], float, float]] = []
        for rank, cand in enumerate(candidates, start=1):
            try:
                cx = float(cand.get("camera_x"))
                cy = float(cand.get("camera_y"))
                sx, sy = camera_to_screen(cx, cy)
                projected.append((rank, cand, float(sx), float(sy)))
            except Exception:
                continue

        results: list[ObjectHitResultV2223] = []
        for region in snap.regions:
            best = None
            best_key = None
            xs = [p[0] for p in region.polygon_screen]
            ys = [p[1] for p in region.polygon_screen]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            for rank, cand, sx, sy in projected:
                if not _point_in_polygon(sx, sy, region.polygon_screen):
                    continue
                score = max(0.0, _finite(cand.get("score", 0.0), 0.0))
                novelty = max(0.0, _finite(cand.get("pre_shot_change", cand.get("center_change", 0.0)), 0.0))
                likeness = max(
                    0.0,
                    _finite(cand.get("local_contrast_gain", cand.get("local_contrast", 0.0)), 0.0)
                    + 0.5 * _finite(cand.get("center_darkening", 0.0), 0.0),
                )
                # Shadow score intentionally simple and uncalibrated.  It exists
                # to validate the object API, not to grant game authority.
                evidence = score + min(10.0, novelty) * 0.35 + min(10.0, likeness) * 0.15
                key = (evidence, -rank)
                if best is None or key > best_key:
                    best = (rank, cand, sx, sy, score, novelty, likeness, evidence)
                    best_key = key

            if best is None:
                result = ObjectHitResultV2223(
                    shot_id=int(shot_id), object_id=region.object_id, hit=False,
                    confidence=0.0, camera_x=None, camera_y=None,
                    screen_x=None, screen_y=None, local_x=None, local_y=None,
                    candidate_score=0.0, shot_novelty=0.0, hole_likeness=0.0,
                    centre_prior=0.0, edge_distance_norm=0.0, candidate_rank=None,
                    metadata=dict(region.metadata),
                )
            else:
                rank, cand, sx, sy, score, novelty, likeness, evidence = best
                width = max(1e-9, max_x - min_x)
                height = max(1e-9, max_y - min_y)
                lx = (sx - min_x) / width
                ly = (sy - min_y) / height
                centre, edge = (0.5, 0.0)
                if viewport_rect_xywh is not None:
                    centre, edge = viewport_center_prior(sx, sy, viewport_rect_xywh)
                # Uncalibrated confidence; kept conservative in shadow mode.
                confidence = 1.0 - math.exp(-max(0.0, evidence) / 12.0)
                result = ObjectHitResultV2223(
                    shot_id=int(shot_id), object_id=region.object_id,
                    hit=True, confidence=max(0.0, min(1.0, confidence)),
                    camera_x=_finite(cand.get("camera_x")),
                    camera_y=_finite(cand.get("camera_y")),
                    screen_x=sx, screen_y=sy, local_x=lx, local_y=ly,
                    candidate_score=score, shot_novelty=novelty,
                    hole_likeness=likeness, centre_prior=centre,
                    edge_distance_norm=edge, candidate_rank=int(rank),
                    metadata=dict(region.metadata),
                )
            results.append(result)

        with self._lock:
            live = self._snapshots.get(int(shot_id))
            if live is not None:
                live.results = {r.object_id: r for r in results}
        return results

    def result(self, object_id: str, shot_id: int | None = None) -> ObjectHitResultV2223 | None:
        with self._lock:
            sid = self._latest_shot_id if shot_id is None else int(shot_id)
            snap = self._snapshots.get(sid)
            return None if snap is None else snap.results.get(str(object_id))

    def was_hit(self, object_id: str, shot_id: int | None = None, *, min_confidence: float = 0.0) -> bool:
        result = self.result(object_id, shot_id)
        return bool(result is not None and result.hit and result.confidence >= float(min_confidence))


object_hit_registry_v2223 = ObjectHitRegistryV2223()

__all__ = [
    "SCHEMA_VERSION",
    "HitRegionV2223",
    "ObjectHitResultV2223",
    "ObjectShotSnapshotV2223",
    "ObjectHitRegistryV2223",
    "object_hit_registry_v2223",
    "viewport_center_prior",
]
