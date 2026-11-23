"""Uniform-grid spatial index for exact nearest/within-radius loops (2026-11-18).

With a cell size >= the query radius, a 3x3 cell query returns every point that can lie
within the radius, so callers keep their exact distance test (math.hypot) and tie order
(near() returns ascending insertion indices).  Used to replace O(n*m) Python loops in the
detector worker without changing results (see automation/*equivalence_selftest.py).
"""
from __future__ import annotations

import math

class GridIndex:
    """Uniform-grid spatial index (cell >= the query radius => a 3x3 cell query is exact).

    Used only to skip far-away comparisons; every surviving comparison uses the same
    math.hypot and the same tie order (ascending insertion index) as the reference loops.
    """

    def __init__(self, cell: float) -> None:
        self.cell = float(max(1e-6, cell))
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.where: dict[int, tuple[int, int]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def insert(self, idx: int, x: float, y: float) -> None:
        key = self._key(x, y)
        self.cells.setdefault(key, []).append(idx)
        self.where[idx] = key

    def move(self, idx: int, x: float, y: float) -> None:
        key = self._key(x, y)
        old = self.where.get(idx)
        if old == key:
            return
        if old is not None:
            bucket = self.cells.get(old)
            if bucket is not None:
                bucket.remove(idx)
        self.cells.setdefault(key, []).append(idx)
        self.where[idx] = key

    def near(self, x: float, y: float) -> list[int]:
        kx, ky = self._key(x, y)
        out: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.extend(self.cells.get((kx + dx, ky + dy), ()))
        out.sort()
        return out
