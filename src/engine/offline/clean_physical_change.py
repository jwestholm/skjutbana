"""Experimental physical change channels. Never imported by the live detector.

Inputs are an existing causal EvidenceContext, with globally registered POST.
No labels, source scores, known-hole coordinates or absolute XY enter a map.
The split model uses two crop halves, not a calibrated physical panel seam.
"""
from __future__ import annotations

import time
import cv2
import numpy as np

from src.engine.offline.accuracy_verifier import align_to_reference, patch

VARIANTS = ('raw', 'local_background', 'persistent', 'pre_variability', 'piecewise', 'flow')
CONTRACT = dict(status='OFFLINE_RESEARCH_ONLY', reference='recorded immediate PRE snapshot',
    causality='PRE before event; recorded POST at/before decision and before next event',
    units='grayscale residual magnitude; variability is a soft attenuation, not a threshold',
    geometry='recorded detector crop; split is only a two-half motion proxy',
    features='source-independent; no GT, detector/source score, timing-after-prior-shot or old-hole veto',
    availability='camera capture-time causality only; no asynchronous worker/latency parity claim')


def local_residual(pre, frames):
    residual = pre[None] - frames
    return np.stack([frame - cv2.GaussianBlur(frame, (0, 0), 12) for frame in residual])


def motion_compensate(pre, frames, mode):
    """Broad-support compensation; fitted before seeing impact coordinates."""
    if mode not in ('piecewise', 'flow'):
        raise ValueError('Unknown motion model')
    output, metadata = [], []
    h, w = pre.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for frame in frames:
        if mode == 'piecewise':
            fields, evidence = [], []
            # Overlapping halves avoid a discontinuous hard seam/edge rejection.
            margin = min(64, w // 8)
            for xa, xb in ((0, w//2+margin), (w//2-margin, w)):
                _, reg = align_to_reference(pre[:, xa:xb], frame[:, xa:xb])
                fields.append((reg['dx'], reg['dy']))
                evidence.append(reg)
            blend = np.clip((xx-(w//2-margin))/max(1, 2*margin), 0, 1)
            dx = fields[0][0]*(1-blend) + fields[1][0]*blend
            dy = fields[0][1]*(1-blend) + fields[1][1]*blend
            metadata.append(evidence)
        else:
            size = (max(8, w//4), max(8, h//4))
            a = cv2.resize(pre, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            field = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 31, 3, 7, 1.5, 0)
            field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
            dx = np.clip(field[..., 0]*(w/size[0]), -4, 4)
            dy = np.clip(field[..., 1]*(h/size[1]), -4, 4)
            metadata.append(dict(dx_median=float(np.median(dx)), dy_median=float(np.median(dy)),
                                 magnitude_p95=float(np.percentile(np.hypot(dx, dy), 95))))
        output.append(cv2.remap(frame, xx+dx, yy+dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return np.stack(output), metadata


def change_maps(context):
    """Fixed sequential ablations: background, persistence, PRE noise, motion."""
    start = time.perf_counter()
    raw = np.mean(np.abs(context.pre[None]-context.post), axis=0)
    local = local_residual(context.pre, context.post)
    persistent = np.abs(np.median(local, axis=0))
    history = np.stack([align_to_reference(context.pre, im)[0] for im in context.history])
    before = local_residual(context.pre, history)
    # PRE fluctuations are evidence of instability, not no-impact ground truth.
    noise = np.sqrt(np.mean(before**2, axis=0))
    noise = cv2.GaussianBlur(noise, (0, 0), 1)
    noise = .75 + noise
    attenuate = lambda magnitude: magnitude*magnitude/(magnitude+noise)
    maps = dict(raw=raw, local_background=np.mean(np.abs(local), axis=0),
                persistent=persistent, pre_variability=attenuate(persistent))
    times = dict(base_and_pre_variability=(time.perf_counter()-start)*1000)
    motion = {}
    for mode in ('piecewise', 'flow'):
        t0 = time.perf_counter()
        compensated, motion[mode] = motion_compensate(context.pre, context.post, mode)
        maps[mode] = attenuate(np.abs(np.median(local_residual(context.pre, compensated), axis=0)))
        times[mode] = (time.perf_counter()-t0)*1000
    edge = np.hypot(cv2.Sobel(context.pre, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(context.pre, cv2.CV_32F, 0, 1, ksize=3)) / 8
    valid = context.roi > 0
    for name, array in maps.items():
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite {name} map')
        maps[name] = np.maximum(array, 0)*valid
    return maps, dict(noise=noise, edge=edge, motion=motion, timing_ms=times)


def response_map(magnitude):
    """Connected multiscale inner-versus-surround support; retains both polarities."""
    channels = []
    for sigma in (1., 2., 4.):
        inner = cv2.GaussianBlur(magnitude, (0, 0), sigma)
        ring = cv2.GaussianBlur(magnitude, (0, 0), 3*sigma)
        channels.append(np.maximum(inner-ring, 0))
    return np.max(channels, axis=0)


def centered_pre_variability(context):
    """Isolated offline ablation: temporal variation versus PRE-reference bias.

    The existing RMS channel includes persistent disagreement with the snapshot.
    Centering PRE residuals measures variation around their own temporal median.
    It is not known to improve accuracy; the baseline function stays unchanged.
    """
    local = local_residual(context.pre, context.post)
    persistent = np.abs(np.median(local, axis=0))
    history = np.stack([align_to_reference(context.pre, im)[0] for im in context.history])
    before = local_residual(context.pre, history)
    centered = before-np.median(before, axis=0)
    noise = .75 + cv2.GaussianBlur(np.sqrt(np.mean(centered**2, axis=0)), (0, 0), 1)
    magnitude = persistent*persistent/(persistent+noise)
    magnitude *= context.roi > 0
    edge = np.hypot(cv2.Sobel(context.pre, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(context.pre, cv2.CV_32F, 0, 1, ksize=3))/8
    return magnitude, dict(noise=noise, edge=edge)


def map_proposals(context, magnitude, budget=256):
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive integer proposal budget required')
    score = response_map(magnitude)
    h, w = score.shape
    valid = context.roi > 0
    valid[:25] = False
    valid[-25:] = False
    valid[:, :25] = False
    valid[:, -25:] = False
    maxima = (score == cv2.dilate(score, np.ones((7, 7), np.uint8))) & valid & (score > 0)
    cells = []
    for gy in range(4):
        for gx in range(4):
            ya, yb, xa, xb = gy*h//4, (gy+1)*h//4, gx*w//4, (gx+1)*w//4
            ys, xs = np.where(maxima[ya:yb, xa:xb])
            ys, xs = ys+ya, xs+xa
            order = np.lexsort((xs, ys, -score[ys, xs]))
            cells.append([(int(xs[i]), int(ys[i])) for i in order[:budget]])
    selected = []
    depth = 0
    while len(selected) < budget:
        layer = [cell[depth] for cell in cells if len(cell) > depth]
        if not layer:
            break
        for x, y in layer:
            if any((x-px)**2+(y-py)**2 < 64 for px, py in selected):
                continue
            selected.append((x, y))
            if len(selected) == budget:
                break
        depth += 1
    # Independent repeated-frame readiness, identical for every map variant.
    yy, xx = np.mgrid[-24:25, -24:25]
    radius = np.hypot(xx, yy)
    inner, ring = radius <= 4, (radius >= 8) & (radius <= 12)
    candidates = []
    for x, y in selected:
        xy = (x+context.origin[0], y+context.origin[1])
        delta = patch(context.pre, xy, context.origin)[None] - np.stack([
            patch(im, xy, context.origin) for im in context.post])
        contrast = delta[:, inner].mean(axis=1)-delta[:, ring].mean(axis=1)
        support = [t for t, value in zip(context.post_times, contrast) if abs(value) > .5]
        candidates.append(dict(camera_x=float(xy[0]), camera_y=float(xy[1]),
            ready=len(support) >= 3 and max(support)-min(support) >= .09,
            support_frames=len(support), proposal_response=float(score[y, x])))
    return candidates, dict(maxima=int(maxima.sum()), candidates=len(candidates),
                            ready=sum(c['ready'] for c in candidates))


def clean_features(context, magnitude, auxiliaries, xy):
    im = patch(magnitude, xy, context.origin)
    noise = patch(auxiliaries['noise'], xy, context.origin)
    edge = patch(auxiliaries['edge'], xy, context.origin)
    yy, xx = np.mgrid[-24:25, -24:25]
    radius = np.hypot(xx, yy)
    values, names = [], []
    for r in (2, 4, 8):
        inner = radius <= r
        ring = (radius >= 1.5*r) & (radius <= 2.5*r)
        parts = dict(mean=im[inner].mean(), peak=im[inner].max(),
                     inner_ring=im[inner].mean()/(.75+im[ring].mean()),
                     noise=noise[inner].mean(), edge=edge[inner].mean())
        for key, value in parts.items():
            names.append(f'clean_r{r}_{key}')
            values.append(value)
    return names, np.log1p(np.asarray(values, np.float32))


def map_metrics(context, magnitude, raw, gt):
    """GT is used only here to measure evidence survival, never to construct it."""
    valid = context.roi > 0
    result = dict(raw_mass=float(raw[valid].sum()), mass=float(magnitude[valid].sum()),
                  retained_mass=float(magnitude[valid].sum()/max(1e-9, raw[valid].sum())))
    if gt is None:
        return result
    x, y = gt['camera_x']-context.origin[0], gt['camera_y']-context.origin[1]
    yy, xx = np.mgrid[:raw.shape[0], :raw.shape[1]]
    d2 = (xx-x)**2+(yy-y)**2
    inner, background = valid & (d2 <= 8**2), valid & (d2 > 42**2)
    if not inner.any() or not background.any():
        raise ValueError('GT/background outside ROI')
    mean = float(magnitude[inner].mean())
    bg = magnitude[background]
    result.update(gt_mean=mean, gt_max=float(magnitude[inner].max()),
        gt_retained_mass=float(magnitude[inner].sum()/max(1e-9, raw[inner].sum())),
        background_mean=float(bg.mean()), signal_background=mean/max(1e-9, float(bg.mean())),
        gt_percentile=float(np.mean(bg < mean)),
        gt_evidence_nonzero=bool(magnitude[inner].max() > 0),
        severe_gt_attenuation=bool(magnitude[inner].sum() < .25*raw[inner].sum()))
    return result
