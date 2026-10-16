"""V3.10.0: shot occlusion against the SAME static triangles drawn by ModernGL.

No new collision volumes are invented. Solid triangles are taken directly from
MeshBuilder's (x,y,z,nx,ny,nz,u,v,layer) array, which also uploads to the GPU.
GPU-wind layers (canvas / leafy shrubs) have no ballistic collision here.
This module is pure NumPy so the intersection kernel is regression-testable
without pygame, OpenGL or physical shooting hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Texture ID is NOT material physics in general: a crate can have canvas texture.
# 6 = leaf planes, 11 = decorative flames, 12 = puddle decal, 26 = wind-animated
# thin fabric/net. 9 = canvas texture ALSO used on real boxes, so it must NOT
# be blindly excluded. 19 = sandbags with cloth surface: remains solid.
NON_BALLISTIC_LAYERS = frozenset((6, 11, 12, 26))


def cover_name(layer: int) -> str:
    if layer in (0, 1, 2, 3, 4, 20, 21, 22):
        return 'JORD'
    if layer in (7, 17):
        return 'STEN'
    if layer in (5, 8, 13, 14, 15, 16, 18):
        return 'TRÄ'
    if layer == 19:
        return 'SANDSÄCK'
    if layer == 10:
        return 'METALL'
    return 'SKYDD'


@dataclass(frozen=True)
class ShotIntersection:
    distance: float
    layer: int


class WorldShotMesh:
    """Immutable triangle soup (one chunk), stored while the chunk is visible.

    Per-shot broadphase is a vectorized segment-AABB filter, then a two-sided
    Möller–Trumbore triangle ray test. Geometry preparation is done ONCE on
    chunk upload, not each game frame. No scipy / trimesh / BVH dependency.
    """
    def __init__(self, vertex_array):
        verts = np.asarray(vertex_array, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 9 or len(verts) % 3:
            raise ValueError('Expected N×9 triangle vertices from MeshBuilder')
        tris = verts.reshape(-1, 3, 9)
        layers = np.rint(tris[:, 0, 8]).astype(np.int16)
        solid = ~np.isin(layers, tuple(NON_BALLISTIC_LAYERS))
        positions = tris[solid, :, :3]
        self.layers = np.ascontiguousarray(layers[solid])
        self.p0 = np.ascontiguousarray(positions[:, 0, :])
        self.edge1 = np.ascontiguousarray(positions[:, 1, :] - positions[:, 0, :])
        self.edge2 = np.ascontiguousarray(positions[:, 2, :] - positions[:, 0, :])
        self.mins = np.ascontiguousarray(positions.min(axis=1)) if len(positions) else np.empty((0, 3), 'f4')
        self.maxs = np.ascontiguousarray(positions.max(axis=1)) if len(positions) else np.empty((0, 3), 'f4')
        if len(positions):
            self.bounds_min = self.mins.min(axis=0)
            self.bounds_max = self.maxs.max(axis=0)
        else:
            self.bounds_min = np.full(3, np.inf, 'f4')
            self.bounds_max = np.full(3, -np.inf, 'f4')

    def first_hit(self, origin, direction, max_distance, *, start_epsilon=0.035,
                  target_epsilon=0.030):
        """First SOLID intersection BEFORE target surface; None means clear.

        start_epsilon avoids false self-hits from triangles touching camera.
        target_epsilon avoids treating a tangent triangle at the opponent's
        hit sphere boundary as cover. Neither expands the enemy hit sphere.
        """
        if not len(self.layers) or max_distance <= start_epsilon + target_epsilon:
            return None
        o = np.asarray(origin, dtype=np.float32)
        d = np.asarray(direction, dtype=np.float32)
        magnitude = float(np.linalg.norm(d))
        if magnitude < 1e-10:
            raise ValueError('Zero-length ray direction')
        d = d / magnitude
        end = o + d * np.float32(max_distance - target_epsilon)
        lo = np.minimum(o, end) - 0.0005
        hi = np.maximum(o, end) + 0.0005
        if np.any(self.bounds_max < lo) or np.any(self.bounds_min > hi):
            return None
        eligible = np.all((self.maxs >= lo) & (self.mins <= hi), axis=1)
        if not np.any(eligible):
            return None
        a, b, c = self.p0[eligible], self.edge1[eligible], self.edge2[eligible]
        layers = self.layers[eligible]
        # Batch Möller–Trumbore: no Python triangle loops / no per-frame work.
        p = np.cross(np.broadcast_to(d, b.shape), c)
        det = np.einsum('ij,ij->i', b, p)
        valid = np.abs(det) > 1.e-9
        if not np.any(valid):
            return None
        inv = np.zeros_like(det)
        inv[valid] = 1. / det[valid]
        tv = o - a
        u = np.einsum('ij,ij->i', tv, p) * inv
        valid &= (u >= -1.e-6) & (u <= 1.000001)
        if not np.any(valid):
            return None
        q = np.cross(tv, b)
        v = np.einsum('j,ij->i', d, q) * inv
        valid &= (v >= -1.e-6) & (u + v <= 1.000001)
        if not np.any(valid):
            return None
        distances = np.einsum('ij,ij->i', c, q) * inv
        valid &= (distances > start_epsilon) & (distances < max_distance - target_epsilon)
        if not np.any(valid):
            return None
        best = np.flatnonzero(valid)[np.argmin(distances[valid])]
        return ShotIntersection(float(distances[best]), int(layers[best]))


def trace_world(origin, direction, target_distance, visible_chunk_meshes):
    """Closest solid hit across the chunks of the LAST PRESENTED world frame."""
    best = None
    for mesh in visible_chunk_meshes:
        hit = mesh.first_hit(origin, direction,
                             target_distance if best is None else min(best.distance, target_distance))
        if hit is not None and (best is None or hit.distance < best.distance):
            best = hit
    return best
