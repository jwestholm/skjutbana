"""Object/game hit-context foundation.

V2.22.3 introduced a SHADOW-only object-hit registry.  V2.24.0 keeps that
runtime contract intact but adds the game-ready API that future games should
use:

* games expose simple viewport-local AABBs (x, y, width, height),
* shot-critical runtime snapshots those regions before scene update,
* the engine transforms the four AABB corners to camera space,
* the scanner receives simple camera AABBs, never game objects/meshes,
* missing/invalid game context falls back to the ordinary global detector.

The legacy screen-polygon API remains for backward compatibility only.  New
games should import ``HitRegion`` from ``src.engine.input.hit_regions``.

IMPORTANT: this module remains context/shadow infrastructure.  A hit region is
where the detector may search first; it is never permission to invent a hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA_VERSION = "2.24.0"
LEGACY_SCHEMA_VERSION = "2.22.3"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _rect_xywh(value: Any) -> tuple[float, float, float, float] | None:
    """Accept pygame.Rect-like objects or 4-tuples without importing pygame."""
    if value is None:
        return None
    try:
        if all(hasattr(value, attr) for attr in ("x", "y", "w", "h")):
            return (
                float(value.x), float(value.y), float(value.w), float(value.h)
            )
        if all(hasattr(value, attr) for attr in ("x", "y", "width", "height")):
            return (
                float(value.x), float(value.y), float(value.width), float(value.height)
            )
        if len(value) == 4:
            x, y, w, h = value
            return float(x), float(y), float(w), float(h)
    except Exception:
        pass
    return None


def _point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Legacy screen-polygon helper; boundary counts as inside."""
    pts = list(polygon)
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
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


def viewport_center_prior(
    screen_x: float, screen_y: float, rect_xywh: Sequence[float]
) -> tuple[float, float]:
    """Return (centre_prior, edge_distance_norm), both 0..1.

    Context only.  Strong physical evidence must always beat this prior.
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


@dataclass(frozen=True, slots=True)
class GameHitRegionV240:
    """Fast game/viewport-local hit-search rectangle.

    x/y are relative to the top-left of the configured game viewport.  The
    rectangle is deliberately approximate; exact game collision still uses the
    final HitEvent.game_x/game_y returned by the normal hit pipeline.
    """

    object_id: str
    x: float
    y: float
    width: float
    height: float
    role: str = "target"
    priority: float = 1.0
    enabled: bool = True
    owner: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_space: str = "game",
    ) -> "GameHitRegionV240 | None":
        space = str(
            value.get("space", value.get("coordinate_space", default_space))
        ).strip().lower()
        if space not in ("game", "viewport", "local", "game_local", "viewport_local"):
            return None
        object_id = str(value.get("object_id", value.get("id", ""))).strip()
        if not object_id:
            return None
        rect = value.get("rect")
        if rect is not None:
            parsed = _rect_xywh(rect)
        else:
            parsed = _rect_xywh(
                (
                    value.get("x"),
                    value.get("y"),
                    value.get("width", value.get("w")),
                    value.get("height", value.get("h")),
                )
            )
        if parsed is None:
            return None
        x, y, w, h = parsed
        if not all(math.isfinite(v) for v in parsed) or w <= 0.0 or h <= 0.0:
            return None
        role = str(value.get("role", value.get("kind", "target")) or "target").strip()
        return cls(
            object_id=object_id,
            x=float(x),
            y=float(y),
            width=float(w),
            height=float(h),
            role=role or "target",
            priority=max(0.0, _finite(value.get("priority", 1.0), 1.0)),
            enabled=bool(value.get("enabled", value.get("hittable", True))),
            owner=str(value.get("owner", "")),
            metadata=dict(value.get("metadata", {}) or {}),
        )

    @classmethod
    def from_screen_rect(
        cls,
        object_id: str,
        rect: Any,
        viewport: Any,
        **kwargs: Any,
    ) -> "GameHitRegionV240":
        """Convenience for games that currently render in absolute screen XY."""
        parsed = _rect_xywh(rect)
        vp = _rect_xywh(viewport)
        if parsed is None or vp is None:
            raise ValueError("rect and viewport must be rect-like")
        x, y, w, h = parsed
        vx, vy, _, _ = vp
        return cls(
            object_id=str(object_id),
            x=float(x - vx),
            y=float(y - vy),
            width=float(w),
            height=float(h),
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class CameraHitRegionV240:
    object_id: str
    x: float
    y: float
    width: float
    height: float
    role: str = "target"
    priority: float = 1.0
    owner: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        m = max(0.0, float(margin))
        return (
            self.x - m <= float(x) <= self.right + m
            and self.y - m <= float(y) <= self.bottom + m
        )


@dataclass(frozen=True)
class HitRegionV2223:
    """Legacy screen-space polygon.  Keep for compatibility; do not use in new games."""

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
    # Legacy, screen-space geometry retained for V2.22.3 shadow evaluator/AI.
    regions: tuple[HitRegionV2223, ...]
    results: dict[str, ObjectHitResultV2223] = field(default_factory=dict)
    # V2.24 canonical game-ready geometry.
    game_regions: tuple[GameHitRegionV240, ...] = ()
    camera_regions: tuple[CameraHitRegionV240, ...] = ()
    transform_available: bool = False
    transform_method: str = "none"
    viewport_xywh: tuple[float, float, float, float] | None = None


def game_rect_to_screen_polygon(
    region: GameHitRegionV240,
    viewport_xywh: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    vx, vy, _, _ = (float(v) for v in viewport_xywh)
    x0 = vx + region.x
    y0 = vy + region.y
    x1 = x0 + region.width
    y1 = y0 + region.height
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def camera_aabb_from_points(
    object_id: str,
    points: Iterable[Sequence[float]],
    *,
    role: str = "target",
    priority: float = 1.0,
    owner: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CameraHitRegionV240 | None:
    clean: list[tuple[float, float]] = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except Exception:
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        clean.append((x, y))
    if not clean:
        return None
    xs = [p[0] for p in clean]
    ys = [p[1] for p in clean]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    return CameraHitRegionV240(
        object_id=str(object_id),
        x=float(x0),
        y=float(y0),
        width=float(x1 - x0),
        height=float(y1 - y0),
        role=str(role or "target"),
        priority=max(0.0, float(priority)),
        owner=str(owner),
        metadata=dict(metadata or {}),
    )


def transform_game_rect_to_camera_aabb(
    region: GameHitRegionV240,
    viewport_xywh: Sequence[float],
    screen_to_camera: Callable[[float, float], tuple[float, float] | None],
) -> CameraHitRegionV240 | None:
    """Transform four game-rect corners and return a cheap camera-space AABB."""
    polygon = game_rect_to_screen_polygon(region, viewport_xywh)
    camera_points: list[tuple[float, float]] = []
    for sx, sy in polygon:
        point = screen_to_camera(float(sx), float(sy))
        if point is None:
            return None
        camera_points.append((float(point[0]), float(point[1])))
    return camera_aabb_from_points(
        region.object_id,
        camera_points,
        role=region.role,
        priority=region.priority,
        owner=region.owner,
        metadata=region.metadata,
    )


def transform_camera_point_to_game(
    camera_x: float,
    camera_y: float,
    viewport_xywh: Sequence[float],
    camera_to_screen: Callable[[float, float], tuple[float, float] | None],
) -> tuple[float, float] | None:
    point = camera_to_screen(float(camera_x), float(camera_y))
    if point is None:
        return None
    vx, vy, _, _ = (float(v) for v in viewport_xywh)
    return float(point[0] - vx), float(point[1] - vy)


class ObjectHitRegistryV2223:
    """Thread-safe legacy registry + V2.24 shot-time game-context snapshots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._regions: dict[str, HitRegionV2223] = {}
        self._snapshots: dict[int, ObjectShotSnapshotV2223] = {}
        self._latest_shot_id = 0

    # ------------------------------------------------------------------
    # Legacy explicit registration API.  New games should use get_hit_regions().
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # V2.24 game provider parsing.
    # ------------------------------------------------------------------
    @staticmethod
    def _scene_values(scene: Any) -> list[Any]:
        """Resolve the optional provider through normal scene wrappers.

        The canonical path is OverlayScene.get_hit_regions() ->
        GameScene.get_hit_regions() -> game.get_hit_regions().  The small
        wrapper traversal below is a defensive fallback so hit context keeps
        working if another Scene decorator is introduced later.
        """
        current = scene
        seen: set[int] = set()
        for _depth in range(6):
            if current is None or id(current) in seen:
                break
            seen.add(id(current))

            provider = getattr(current, "get_hit_regions", None)
            if callable(provider):
                try:
                    return list(provider() or [])
                except Exception as exc:
                    print(f"[V2.24.0 GAME-CONTEXT] get_hit_regions failed: {exc}")
                    return []

            inner = getattr(current, "inner", None)
            if inner is not None and inner is not current:
                current = inner
                continue

            game = getattr(current, "game", None)
            if game is not None and game is not current:
                current = game
                continue

            break
        return []

    def _scene_regions(
        self, scene: Any
    ) -> tuple[list[GameHitRegionV240], list[HitRegionV2223]]:
        game_regions: list[GameHitRegionV240] = []
        legacy_regions: list[HitRegionV2223] = []
        for value in self._scene_values(scene):
            if isinstance(value, GameHitRegionV240):
                game_regions.append(value)
                continue
            if isinstance(value, HitRegionV2223):
                legacy_regions.append(value)
                continue
            if isinstance(value, Mapping):
                # New game contract defaults mappings to game-local AABBs.
                game_region = GameHitRegionV240.from_mapping(value, default_space="game")
                if game_region is not None:
                    game_regions.append(game_region)
                    continue
                # Explicit screen/polygon mappings remain supported.
                legacy = HitRegionV2223.from_mapping(value)
                if legacy is not None:
                    legacy_regions.append(legacy)
                continue
            # pygame.Rect-like values are accepted as anonymous game-local
            # rectangles for debugging only.  Normal games should provide IDs.
        return game_regions, legacy_regions

    @staticmethod
    def _load_viewport_xywh() -> tuple[float, float, float, float] | None:
        try:
            from src.engine.settings import load_viewport_rect
            return _rect_xywh(load_viewport_rect())
        except Exception:
            return None

    @staticmethod
    def _camera_transformers(
    ) -> tuple[
        str,
        Callable[[float, float], tuple[float, float] | None] | None,
        Callable[[float, float], tuple[float, float] | None] | None,
    ]:
        """Return (method, screen->camera, camera->screen) without identity guesses."""
        try:
            from src.engine.input.hit_input import hit_input
            from src.engine.settings import load_scanport_rect, load_viewport_rect

            viewport = load_viewport_rect()
            scanport = load_scanport_rect()
            have_scanport = bool(
                viewport is not None
                and scanport is not None
                and float(getattr(viewport, "w", 0)) > 0
                and float(getattr(viewport, "h", 0)) > 0
                and float(getattr(scanport, "w", 0)) > 0
                and float(getattr(scanport, "h", 0)) > 0
            )
            have_h = getattr(hit_input, "homography", None) is not None
            have_inv = getattr(hit_input, "inverse", None) is not None
            prefers_h = False
            try:
                prefers_h = bool(hit_input._prefers_homography())
            except Exception:
                prefers_h = False

            if prefers_h and have_h and have_inv:
                return (
                    "homography",
                    hit_input._screen_to_camera_via_homography,
                    hit_input._camera_to_screen_via_homography,
                )
            if have_scanport:
                return (
                    "scanport",
                    hit_input._screen_to_camera_via_scanport,
                    hit_input._camera_to_screen_via_scanport,
                )
            if have_h and have_inv:
                return (
                    "homography_fallback",
                    hit_input._screen_to_camera_via_homography,
                    hit_input._camera_to_screen_via_homography,
                )
        except Exception:
            pass
        return "none", None, None

    @classmethod
    def game_to_camera_point(cls, x: float, y: float) -> tuple[float, float] | None:
        vp = cls._load_viewport_xywh()
        if vp is None:
            return None
        method, screen_to_camera, _ = cls._camera_transformers()
        if method == "none" or screen_to_camera is None:
            return None
        sx, sy = vp[0] + float(x), vp[1] + float(y)
        return screen_to_camera(sx, sy)

    @classmethod
    def camera_to_game_point(cls, x: float, y: float) -> tuple[float, float] | None:
        vp = cls._load_viewport_xywh()
        if vp is None:
            return None
        method, _, camera_to_screen = cls._camera_transformers()
        if method == "none" or camera_to_screen is None:
            return None
        return transform_camera_point_to_game(x, y, vp, camera_to_screen)

    @classmethod
    def game_rect_to_camera_aabb(
        cls, region: GameHitRegionV240
    ) -> CameraHitRegionV240 | None:
        vp = cls._load_viewport_xywh()
        if vp is None:
            return None
        method, screen_to_camera, _ = cls._camera_transformers()
        if method == "none" or screen_to_camera is None:
            return None
        return transform_game_rect_to_camera_aabb(region, vp, screen_to_camera)

    # ------------------------------------------------------------------
    # Shot-time snapshot.  Existing V2.22.3 shot-critical runtime calls this
    # before scene update, so moving targets are frozen at the shot frame.
    # ------------------------------------------------------------------
    def snapshot(
        self,
        shot_id: int,
        peak_ts: float,
        scene: Any = None,
    ) -> ObjectShotSnapshotV2223:
        with self._lock:
            merged_legacy = dict(self._regions)

        game_regions: list[GameHitRegionV240] = []
        if scene is not None:
            scene_game, scene_legacy = self._scene_regions(scene)
            for region in scene_game:
                if region.enabled:
                    game_regions.append(region)
            for region in scene_legacy:
                if region.enabled:
                    merged_legacy[region.object_id] = region

        viewport = self._load_viewport_xywh()
        if viewport is not None:
            for region in game_regions:
                polygon = game_rect_to_screen_polygon(region, viewport)
                metadata = dict(region.metadata)
                metadata.setdefault("role", region.role)
                metadata.setdefault("v240_game_local", True)
                merged_legacy[region.object_id] = HitRegionV2223(
                    object_id=region.object_id,
                    polygon_screen=polygon,
                    priority=region.priority,
                    enabled=region.enabled,
                    owner=region.owner,
                    metadata=metadata,
                )

        legacy_regions = tuple(v for v in merged_legacy.values() if v.enabled)
        method, screen_to_camera, _ = self._camera_transformers()
        camera_regions: list[CameraHitRegionV240] = []
        if viewport is not None and screen_to_camera is not None:
            # Canonical new regions: transform exactly four corners.
            for region in game_regions:
                cam = transform_game_rect_to_camera_aabb(region, viewport, screen_to_camera)
                if cam is not None:
                    camera_regions.append(cam)

            # Also expose camera AABBs for old explicitly registered screen
            # polygons so V2.24.1 can consume one uniform interface.
            known = {r.object_id for r in camera_regions}
            for legacy in legacy_regions:
                if legacy.object_id in known:
                    continue
                points: list[tuple[float, float]] = []
                ok = True
                for sx, sy in legacy.polygon_screen:
                    point = screen_to_camera(float(sx), float(sy))
                    if point is None:
                        ok = False
                        break
                    points.append(point)
                if not ok:
                    continue
                cam = camera_aabb_from_points(
                    legacy.object_id,
                    points,
                    role=str(legacy.metadata.get("role", "target")),
                    priority=legacy.priority,
                    owner=legacy.owner,
                    metadata=legacy.metadata,
                )
                if cam is not None:
                    camera_regions.append(cam)

        snap = ObjectShotSnapshotV2223(
            shot_id=int(shot_id),
            peak_ts=float(peak_ts),
            created_ts=time.time(),
            regions=legacy_regions,
            game_regions=tuple(game_regions),
            camera_regions=tuple(camera_regions),
            transform_available=(method != "none" and bool(camera_regions or not legacy_regions)),
            transform_method=method,
            viewport_xywh=viewport,
        )

        with self._lock:
            self._snapshots[int(shot_id)] = snap
            self._latest_shot_id = max(self._latest_shot_id, int(shot_id))
            for old in sorted(self._snapshots)[:-32]:
                self._snapshots.pop(old, None)

        if snap.game_regions or snap.regions:
            print(
                f"[V2.24.0 GAME-CONTEXT] shot={snap.shot_id} "
                f"game={len(snap.game_regions)} camera={len(snap.camera_regions)} "
                f"transform={snap.transform_method}"
            )
        return snap

    def snapshot_for_shot(self, shot_id: int) -> ObjectShotSnapshotV2223 | None:
        with self._lock:
            return self._snapshots.get(int(shot_id))

    def game_regions_for_shot(self, shot_id: int) -> tuple[GameHitRegionV240, ...]:
        snap = self.snapshot_for_shot(shot_id)
        return () if snap is None else snap.game_regions

    def camera_regions_for_shot(self, shot_id: int) -> tuple[CameraHitRegionV240, ...]:
        snap = self.snapshot_for_shot(shot_id)
        return () if snap is None else snap.camera_regions

    def latest_snapshot(self) -> ObjectShotSnapshotV2223 | None:
        with self._lock:
            return self._snapshots.get(self._latest_shot_id)

    def game_context(self, shot_id: int) -> dict[str, Any] | None:
        snap = self.snapshot_for_shot(shot_id)
        if snap is None:
            return None
        return {
            # Preserve the legacy schema value consumed by the existing V2.22/V2.23 AI bridge.
            "schema": LEGACY_SCHEMA_VERSION,
            "game_hit_context_schema": SCHEMA_VERSION,
            "shot_id": snap.shot_id,
            "peak_ts": snap.peak_ts,
            "transform_available": snap.transform_available,
            "transform_method": snap.transform_method,
            "viewport_xywh": list(snap.viewport_xywh) if snap.viewport_xywh else None,
            # Legacy field retained for V2.22/V2.23 AI consumers.
            "hit_regions": [
                {
                    "object_id": r.object_id,
                    "polygon_screen": [list(p) for p in r.polygon_screen],
                    "priority": r.priority,
                    "metadata": dict(r.metadata),
                }
                for r in snap.regions
            ],
            # New stable AABB context.
            "game_hit_regions": [
                {
                    "object_id": r.object_id,
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "role": r.role,
                    "priority": r.priority,
                    "metadata": dict(r.metadata),
                }
                for r in snap.game_regions
            ],
            "camera_hit_regions": [
                {
                    "object_id": r.object_id,
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "role": r.role,
                    "priority": r.priority,
                    "metadata": dict(r.metadata),
                }
                for r in snap.camera_regions
            ],
        }

    # ------------------------------------------------------------------
    # Existing V2.22.3 candidate-backed SHADOW evaluator.  V2.24.0 keeps it
    # unchanged in authority terms while allowing a cheaper camera-AABB filter.
    # ------------------------------------------------------------------
    def evaluate_candidates(
        self,
        shot_id: int,
        candidates: Sequence[Mapping[str, Any]],
        *,
        camera_to_screen,
        viewport_rect_xywh: Sequence[float] | None = None,
    ) -> list[ObjectHitResultV2223]:
        snap = self.snapshot_for_shot(int(shot_id))
        if snap is None or not snap.regions:
            return []

        camera_by_id = {r.object_id: r for r in snap.camera_regions}
        results: list[ObjectHitResultV2223] = []

        for region in snap.regions:
            best = None
            best_key = None
            xs = [p[0] for p in region.polygon_screen]
            ys = [p[1] for p in region.polygon_screen]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cam_region = camera_by_id.get(region.object_id)

            for rank, cand in enumerate(candidates, start=1):
                try:
                    cx = float(cand.get("camera_x"))
                    cy = float(cand.get("camera_y"))
                except Exception:
                    continue

                # New fast pre-filter when a valid game->camera transform exists.
                if cam_region is not None and not cam_region.contains(cx, cy):
                    continue

                try:
                    sx, sy = camera_to_screen(cx, cy)
                    sx, sy = float(sx), float(sy)
                except Exception:
                    continue
                if not _point_in_polygon(sx, sy, region.polygon_screen):
                    continue

                score = max(0.0, _finite(cand.get("score", 0.0), 0.0))
                novelty = max(
                    0.0,
                    _finite(
                        cand.get("pre_shot_change", cand.get("center_change", 0.0)),
                        0.0,
                    ),
                )
                likeness = max(
                    0.0,
                    _finite(
                        cand.get("local_contrast_gain", cand.get("local_contrast", 0.0)),
                        0.0,
                    )
                    + 0.5 * _finite(cand.get("center_darkening", 0.0), 0.0),
                )
                evidence = score + min(10.0, novelty) * 0.35 + min(10.0, likeness) * 0.15
                key = (evidence, -rank)
                if best is None or key > best_key:
                    best = (rank, cand, sx, sy, score, novelty, likeness, evidence)
                    best_key = key

            metadata = dict(region.metadata)
            if best is None:
                result = ObjectHitResultV2223(
                    shot_id=int(shot_id),
                    object_id=region.object_id,
                    hit=False,
                    confidence=0.0,
                    camera_x=None,
                    camera_y=None,
                    screen_x=None,
                    screen_y=None,
                    local_x=None,
                    local_y=None,
                    candidate_score=0.0,
                    shot_novelty=0.0,
                    hole_likeness=0.0,
                    centre_prior=0.0,
                    edge_distance_norm=0.0,
                    candidate_rank=None,
                    metadata=metadata,
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
                confidence = 1.0 - math.exp(-max(0.0, evidence) / 12.0)
                result = ObjectHitResultV2223(
                    shot_id=int(shot_id),
                    object_id=region.object_id,
                    hit=True,
                    confidence=max(0.0, min(1.0, confidence)),
                    camera_x=_finite(cand.get("camera_x")),
                    camera_y=_finite(cand.get("camera_y")),
                    screen_x=sx,
                    screen_y=sy,
                    local_x=lx,
                    local_y=ly,
                    candidate_score=score,
                    shot_novelty=novelty,
                    hole_likeness=likeness,
                    centre_prior=centre,
                    edge_distance_norm=edge,
                    candidate_rank=int(rank),
                    metadata=metadata,
                )
            results.append(result)

        with self._lock:
            live = self._snapshots.get(int(shot_id))
            if live is not None:
                live.results = {r.object_id: r for r in results}
        return results

    def result(
        self, object_id: str, shot_id: int | None = None
    ) -> ObjectHitResultV2223 | None:
        with self._lock:
            sid = self._latest_shot_id if shot_id is None else int(shot_id)
            snap = self._snapshots.get(sid)
            return None if snap is None else snap.results.get(str(object_id))

    def was_hit(
        self,
        object_id: str,
        shot_id: int | None = None,
        *,
        min_confidence: float = 0.0,
    ) -> bool:
        result = self.result(object_id, shot_id)
        return bool(
            result is not None
            and result.hit
            and result.confidence >= float(min_confidence)
        )


object_hit_registry_v2223 = ObjectHitRegistryV2223()

# Stable friendly name for new code.  The versioned name stays because the
# V2.22.3 runtime imports it directly.
HitRegion = GameHitRegionV240
CameraHitRegion = CameraHitRegionV240

__all__ = [
    "SCHEMA_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "HitRegion",
    "CameraHitRegion",
    "GameHitRegionV240",
    "CameraHitRegionV240",
    "HitRegionV2223",
    "ObjectHitResultV2223",
    "ObjectShotSnapshotV2223",
    "ObjectHitRegistryV2223",
    "object_hit_registry_v2223",
    "viewport_center_prior",
    "game_rect_to_screen_polygon",
    "camera_aabb_from_points",
    "transform_game_rect_to_camera_aabb",
    "transform_camera_point_to_game",
]
