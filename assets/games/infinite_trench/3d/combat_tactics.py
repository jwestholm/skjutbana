"""Deterministic arcade difficulty, cover semantics and sightline presentation.

This module has no renderer/physics/Skjutbanan dependency. Gameplay staging may
change with progress but must NEVER bypass world occlusion or shot-frozen hit XY.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CombatDifficulty:
    level: int
    peek_cycles: int
    scan_seconds: float
    locate_seconds: float
    rush_speed: float
    reaction_grace: float
    warning_shots: int
    accuracy_scale: float


def difficulty_at(distance):
    """Distance-based arcade progression, not hitbox/aim-assist scaling."""
    distance=max(0.0, float(distance))
    if distance < 180:
        return CombatDifficulty(0, 2, .72, .68, 1.65, 2.25, 2, .60)
    if distance < 450:
        return CombatDifficulty(1, 2, .62, .60, 1.92, 1.95, 1, .79)
    if distance < 850:
        return CombatDifficulty(2, 2, .50, .48, 2.18, 1.65, 1, 1.04)
    return CombatDifficulty(3, 1, .36, .35, 2.42, 1.40, 1, 1.20)


@dataclass(frozen=True)
class CoverMaterial:
    kind: str
    hides: bool
    bullet_stopping: float
    splash: str


MATERIALS = {
    'earth': CoverMaterial('earth', True, 1.0, 'earth'),
    'rock': CoverMaterial('rock', True, 1.0, 'stone'),
    'rubble': CoverMaterial('rubble', True, .93, 'stone'),
    'sandbags': CoverMaterial('sandbags', True, .95, 'earth'),
    'crate': CoverMaterial('crate', True, .46, 'wood'),
    'timber': CoverMaterial('timber', True, .58, 'wood'),
    'canvas': CoverMaterial('canvas', True, .04, 'cloth'),
    'net': CoverMaterial('net', False, .03, 'cloth'),
    'smoke': CoverMaterial('smoke', True, 0.0, 'none'),
}


def projected_visible(point, matrix, width, height, *, margin=.07):
    """Whether a point is actually in the LAST DRAWN viewport and in front.

    Does not establish world line of sight: combine with geometric visibility.
    Accepts ndarray or nested tuple; numpy is already a rendering dependency.
    """
    import numpy as np
    if matrix is None or width<=0 or height<=0:
        return False
    clip=np.asarray(matrix,dtype=float) @ np.asarray((*point,1.0),dtype=float)
    if clip[3]<=1e-6:
        return False
    xyz=clip[:3]/clip[3]
    return bool(-1+margin < xyz[0] < 1-margin and
                -1+margin < xyz[1] < 1-margin and
                -1 <= xyz[2] <= 1)


def encounter_lateral(enemy_id):
    """Stable lateral body position (metres), never a screen-coordinate shift."""
    index=int(enemy_id)
    if index<0:raise ValueError('enemy_id must be non-negative')
    magnitude=.30+.34*(.5+.5*math.sin((index+17)*2.399963229728653))
    side=1 if math.sin(index*1.823+.37)>0.0 else -1
    return side*magnitude


def tactical_cover_lane(enemy_lateral, route_m, wall_width):
    """World-space cover lane opposite a known enemy; leave real body clearance."""
    enemy_lateral=float(enemy_lateral)
    if not math.isfinite(enemy_lateral):return 0.0
    side=-1 if enemy_lateral>=0 else 1
    clearance=max(0.0,float(wall_width(float(route_m),side))-.49)
    return side*min(.51,clearance)*.82
