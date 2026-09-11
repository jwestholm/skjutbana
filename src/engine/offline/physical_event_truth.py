"""Minimal evaluation-only impact contract; native physical labels stay immutable."""
from __future__ import annotations

import math

TRUTH_SCHEMA = 'physical-event-truth-1'
STATES = {'SINGLE_IMPACT', 'NO_PHYSICAL_SHOT', 'AMBIGUOUS', 'UNKNOWN', 'MULTI_IMPACT'}


def event_truth(state, impacts=None, *, reason='', caveats=()):
    if state not in STATES:
        raise ValueError('Unknown physical truth state')
    points = [] if impacts is None else [dict(p) for p in impacts]
    for point in points:
        if not all(math.isfinite(float(point[k])) for k in ('camera_x', 'camera_y')):
            raise ValueError('Impact coordinates must be finite camera pixels')
    if state == 'SINGLE_IMPACT' and len(points) != 1:
        raise ValueError('Single-impact truth requires one impact')
    if state == 'NO_PHYSICAL_SHOT' and points:
        raise ValueError('Nonphysical truth cannot have impact coordinates')
    if state == 'MULTI_IMPACT' and len(points) < 2:
        raise ValueError('Multi-impact truth requires at least two impacts')
    return dict(schema=TRUTH_SCHEMA, state=state, impacts=points, reason=reason,
                caveats=list(caveats), coordinate_space='camera')


def score_event(truth, selected, radii=(5, 10, 20, 42)):
    """0/1 compatibility. Future multi-impact truth is representable, never flattened."""
    state = truth['state']
    if state in ('AMBIGUOUS', 'UNKNOWN', 'MULTI_IMPACT'):
        return dict(evaluable=False, reason=state, emitted=selected is not None)
    if state == 'NO_PHYSICAL_SHOT':
        return dict(evaluable=True, physical=False, emitted=selected is not None,
                    true_negative=selected is None, false_emission=selected is not None)
    if state != 'SINGLE_IMPACT' or len(truth['impacts']) != 1:
        raise ValueError('Invalid truth contract')
    gt = truth['impacts'][0]
    error = None if selected is None else math.hypot(selected['camera_x'] - gt['camera_x'],
                                                   selected['camera_y'] - gt['camera_y'])
    return dict(evaluable=True, physical=True, emitted=selected is not None, error_px=error,
                false_rejection=selected is None,
                hits={str(r): error is not None and error <= r for r in radii})
