"""Pure game-space geometry for V2.25 GameObjects.

All coordinates in this module are viewport-local/game-local.  The hit engine
still owns screen/camera transforms.  HitRegion uses an AABB for fast physical
search, while these shape snapshots are used later for exact gameplay collision.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "2.25.0"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


@dataclass(slots=True)
class ObjectGeometry:
    """Projected 2D geometry in viewport-local/game-local pixels.

    ``rotation_deg`` is stored for future render/projection use.  V2.25.0 exact
    hit shapes are rect/ellipse/circle/polygon snapshots and do not silently
    rotate an AABB.  Games needing rotated collision should use polygon points.
    """

    x: float
    y: float
    width: float
    height: float
    rotation_deg: float = 0.0

    @property
    def right(self) -> float:
        return float(self.x + self.width)

    @property
    def bottom(self) -> float:
        return float(self.y + self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.x + self.width * 0.5), float(self.y + self.height * 0.5))

    @property
    def aabb(self) -> tuple[float, float, float, float]:
        return float(self.x), float(self.y), float(self.width), float(self.height)

    def move(self, dx: float, dy: float) -> None:
        self.x = float(self.x + dx)
        self.y = float(self.y + dy)


@dataclass(frozen=True, slots=True)
class WorldPlacement:
    """Optional semantic/world information; never used as camera coordinates.

    This deliberately does not prescribe one 3D engine.  Range-projection games
    may store virtual distance/height here while the authoritative hit geometry
    remains projected game-local XY.
    """

    distance_m: float | None = None
    world_x_m: float | None = None
    world_y_m: float | None = None
    world_z_m: float | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class HitShapeSpec:
    """Object-local exact collision shape.

    Coordinates are normalized to the object's geometry unless otherwise noted:
    local_x/local_y/local_width/local_height are fractions of object width/height.
    Polygon points are normalized object-local points as well.
    """

    kind: str = "rect"
    local_x: float = 0.0
    local_y: float = 0.0
    local_width: float = 1.0
    local_height: float = 1.0
    points: tuple[tuple[float, float], ...] = ()

    @classmethod
    def rect(cls, *, x: float = 0.0, y: float = 0.0, width: float = 1.0, height: float = 1.0) -> "HitShapeSpec":
        return cls("rect", float(x), float(y), float(width), float(height), ())

    @classmethod
    def ellipse(cls, *, x: float = 0.0, y: float = 0.0, width: float = 1.0, height: float = 1.0) -> "HitShapeSpec":
        return cls("ellipse", float(x), float(y), float(width), float(height), ())

    @classmethod
    def circle(cls, *, center_x: float = 0.5, center_y: float = 0.5, diameter: float = 1.0) -> "HitShapeSpec":
        d = float(diameter)
        return cls("circle", float(center_x - d * 0.5), float(center_y - d * 0.5), d, d, ())

    @classmethod
    def polygon(cls, points: Iterable[Sequence[float]]) -> "HitShapeSpec":
        clean = tuple((float(p[0]), float(p[1])) for p in points)
        if len(clean) < 3:
            raise ValueError("polygon requires at least three points")
        return cls("polygon", 0.0, 0.0, 1.0, 1.0, clean)

    def snapshot(self, geometry: ObjectGeometry) -> dict[str, Any]:
        kind = str(self.kind).lower()
        if kind == "polygon":
            pts = [
                [
                    float(geometry.x + px * geometry.width),
                    float(geometry.y + py * geometry.height),
                ]
                for px, py in self.points
            ]
            return {"schema": SCHEMA_VERSION, "kind": "polygon", "points": pts}

        x = float(geometry.x + self.local_x * geometry.width)
        y = float(geometry.y + self.local_y * geometry.height)
        w = float(self.local_width * geometry.width)
        h = float(self.local_height * geometry.height)
        if kind not in {"rect", "ellipse", "circle"}:
            kind = "rect"
        return {
            "schema": SCHEMA_VERSION,
            "kind": kind,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        }


def shape_bounds(snapshot: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    kind = str(snapshot.get("kind", "rect")).lower()
    if kind == "polygon":
        try:
            pts = [(float(p[0]), float(p[1])) for p in snapshot.get("points", ())]
        except Exception:
            return None
        if len(pts) < 3:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    x = _finite(snapshot.get("x"))
    y = _finite(snapshot.get("y"))
    w = _finite(snapshot.get("width"))
    h = _finite(snapshot.get("height"))
    if w <= 0.0 or h <= 0.0:
        return None
    return x, y, w, h


def _point_in_polygon(x: float, y: float, points: Sequence[tuple[float, float]]) -> bool:
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        dx = xj - xi
        dy = yj - yi
        cross = (x - xi) * dy - (y - yi) * dx
        if abs(cross) <= 1e-7:
            dot = (x - xi) * dx + (y - yi) * dy
            if -1e-7 <= dot <= dx * dx + dy * dy + 1e-7:
                return True
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def shape_contains(snapshot: Mapping[str, Any], x: float, y: float) -> bool:
    """Exact point collision against a frozen shape snapshot."""
    kind = str(snapshot.get("kind", "rect")).lower()
    px = float(x)
    py = float(y)

    if kind == "polygon":
        try:
            pts = tuple((float(p[0]), float(p[1])) for p in snapshot.get("points", ()))
        except Exception:
            return False
        return _point_in_polygon(px, py, pts)

    bounds = shape_bounds(snapshot)
    if bounds is None:
        return False
    x0, y0, w, h = bounds
    if kind == "rect":
        return x0 <= px <= x0 + w and y0 <= py <= y0 + h

    # ellipse/circle share the same normalized ellipse equation.  A circle
    # snapshot normally has equal width/height but does not rely on it.
    cx = x0 + w * 0.5
    cy = y0 + h * 0.5
    rx = max(1e-9, w * 0.5)
    ry = max(1e-9, h * 0.5)
    nx = (px - cx) / rx
    ny = (py - cy) / ry
    return nx * nx + ny * ny <= 1.0 + 1e-9


__all__ = [
    "SCHEMA_VERSION",
    "ObjectGeometry",
    "WorldPlacement",
    "HitShapeSpec",
    "shape_bounds",
    "shape_contains",
]
