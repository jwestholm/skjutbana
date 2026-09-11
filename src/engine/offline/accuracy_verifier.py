"""Source-independent causal multiscale/temporal research evidence. Never live authority.

No detector scores, source identity, camera XY, labels or future frames are
features. Spatial normalization is local; PRE/POST remain camera grayscale.
Captured timestamps establish image-time causality, not worker delivery parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

RADII = (2, 4, 8, 16)
RADIUS = 24
FEATURE_CONTRACT = {
    'source': 'recorded grayscale PRE, PRE history and POST camera patches only',
    'timestamp': 'per-event pre_timestamps and post_timestamps in dataset metadata',
    'causal_availability': 'PRE < audio peak; POST <= decision cutoff and < next event peak',
    'coordinate_system': 'full camera XY translated once by recorded crop origin; 49x49 patches',
    'normalization': 'per-frame ring median offset; per-patch ring/MAD and PRE temporal noise',
    'live_available': 'same frame buffer inputs possible; capture-time replay does not prove worker delivery or latency parity',
    'source_independent': True,
    'labels': 'evaluation/training targets only; never input to context, features or proposals',
    'reference': 'explicit snapshot or fixed earlier PRE-history median intervention',
}


@dataclass
class EvidenceContext:
    pre: np.ndarray
    history: np.ndarray
    post: np.ndarray
    peak: float
    cutoff: float
    pre_times: tuple
    post_times: tuple
    origin: tuple
    roi: np.ndarray
    registration: list

    def __post_init__(self):
        if not np.isfinite([self.peak, self.cutoff, *self.pre_times, *self.post_times]).all():
            raise ValueError('Nonfinite evidence timestamp')
        if not self.post_times or len(self.post) != len(self.post_times):
            raise ValueError('POST timestamps must match image count')
        if len(self.history) != len(self.pre_times) or not self.pre_times:
            raise ValueError('PRE timestamps must match history count')
        if max(self.pre_times) >= self.peak or min(self.post_times) < self.peak or max(self.post_times) > self.cutoff:
            raise ValueError('Noncausal evidence rejected')
        if self.pre.ndim != 2 or self.history.shape[1:] != self.pre.shape or self.post.shape[1:] != self.pre.shape:
            raise ValueError('Camera crop shape mismatch')
        if self.roi.shape != self.pre.shape:
            raise ValueError('ROI shape mismatch')


def align_to_reference(reference, image):
    """Global camera translation only; estimate outside any GT-selected geometry."""
    scale = 2 if max(reference.shape) > 512 else 1
    a = reference[::scale, ::scale].astype(np.float32)
    b = image[::scale, ::scale].astype(np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    dx *= scale
    dy *= scale
    applied = bool(np.isfinite([dx, dy, response]).all() and response >= .1 and abs(dx) <= 4 and abs(dy) <= 4)
    if not applied:
        dx = dy = 0.
    aligned = cv2.warpAffine(image.astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]),
                             (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, dict(dx=float(dx), dy=float(dy), response=float(response), applied=applied)


def patch(image, xy, origin, radius=RADIUS):
    x, y = float(xy[0] - origin[0]), float(xy[1] - origin[1])
    if min(x, y) < radius or x + radius >= image.shape[1] or y + radius >= image.shape[0]:
        raise ValueError('Patch outside captured crop')
    return cv2.getRectSubPix(image.astype(np.float32, copy=False), (2*radius+1, 2*radius+1), (x, y))


def features(context, xy):
    pre = patch(context.pre, xy, context.origin)
    hist = np.stack([patch(im, xy, context.origin) for im in context.history])
    post = np.stack([patch(im, xy, context.origin) for im in context.post])
    yy, xx = np.mgrid[-RADIUS:RADIUS+1, -RADIUS:RADIUS+1]
    radius = np.hypot(xx, yy)
    outer = (radius >= 18) & (radius <= 24)
    residual = pre[None] - post
    residual -= np.median(residual[:, outer], axis=1)[:, None, None]
    before = pre[None] - hist
    before -= np.median(before[:, outer], axis=1)[:, None, None]
    noise = max(.75, float(np.median(np.std(before, axis=0))),
                1.4826 * float(np.median(np.abs(residual[:, outer] - np.median(residual[:, outer])))))
    residual /= noise
    before /= noise
    late = np.median(residual[max(0, len(residual)//2):], axis=0)
    v = {}
    for r in RADII:
        inside = radius <= r
        ring = (radius >= r*1.5) & (radius <= min(r*2.5, 24))
        center = residual[:, inside].mean(axis=1)
        surround = residual[:, ring].mean(axis=1)
        contrast = center - surround
        old = before[:, inside].mean(axis=1) - before[:, ring].mean(axis=1)
        name = f'r{r}_'
        v.update({name+'signed_late': float(np.mean(contrast[len(contrast)//2:])),
                  name+'abs_late': float(np.abs(late[inside]).mean()),
                  name+'dark_late': float(np.maximum(late[inside], 0).mean()),
                  name+'bright_late': float(np.maximum(-late[inside], 0).mean()),
                  name+'contrast_peak': float(np.max(np.abs(contrast))),
                  name+'contrast_std': float(np.std(contrast)),
                  name+'pre_std': float(np.std(old)),
                  name+'pre_peak': float(np.max(np.abs(old))),
                  name+'onset_gain': float(np.max(np.abs(contrast))-np.max(np.abs(old))),
                  name+'persistence': float(np.mean(np.abs(contrast) > 2)),
                  name+'polarity_agreement': float(abs(np.mean(np.sign(contrast)))),
                  name+'early_late_agreement': float(contrast[0]*np.mean(contrast[len(contrast)//2:])),
                  name+'spatial_consistency': float(np.mean(np.abs(np.mean(np.sign(residual[:, inside]), axis=0)))),
                  name+'center_ring_ratio': float(np.abs(late[inside]).mean()/(.5+np.abs(late[ring]).mean())),
                  name+'pre_texture': float(np.std(pre[inside])/(1+np.std(pre[ring]))),
                  name+'fraction_changed': float(np.mean(np.abs(late[inside]) > 2))})
    mass = np.maximum(np.abs(late) - 1., 0) * (radius <= 16)
    total = float(mass.sum()) + 1e-6
    mx, my = float((mass*xx).sum()/total), float((mass*yy).sum()/total)
    covariance = np.array([[(mass*(xx-mx)**2).sum(), (mass*(xx-mx)*(yy-my)).sum()],
                           [(mass*(xx-mx)*(yy-my)).sum(), (mass*(yy-my)**2).sum()]])/total
    eigen = np.linalg.eigvalsh(covariance)
    v.update(morph_offset=float(np.hypot(mx, my)), morph_elongation=float((eigen[-1]+1)/(eigen[0]+1)),
             morph_spread=float(eigen.sum()), morphology_mass=float(total),
             polarity_mass=float(np.sum(late*(radius <= 16))/(1e-6+np.sum(np.abs(late)*(radius <= 16)))))
    # Shift robustness, measured identically for every candidate; no best-XY oracle.
    disk = (radius <= 4).astype(np.float32)
    blurred = cv2.filter2D(late, -1, disk/disk.sum())
    jitter = blurred[RADIUS-1:RADIUS+2, RADIUS-1:RADIUS+2]
    v.update(jitter_min_abs=float(np.abs(jitter).min()), jitter_mean_abs=float(np.abs(jitter).mean()),
             jitter_std=float(jitter.std()))
    names = tuple(sorted(v))
    values = np.asarray([v[k] for k in names], np.float32)
    # Signed log is fixed robust compression; never fit on challenge-session data.
    values = np.sign(values)*np.log1p(np.abs(values))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite physical feature')
    return names, values


def proposals(context, budget=256):
    """Bounded, source-independent multiscale persistent-change maxima.

    No GT or detector candidates are consumed. Quotas cover space and both
    polarities; each output carries multi-frame support/readiness diagnostics.
    """
    if not isinstance(budget, int) or budget < 1:
        raise ValueError('Positive integer proposal budget required')
    if not np.any(context.roi):
        raise ValueError('Empty detector ROI')
    pre = context.pre
    residual = pre[None] - context.post
    for i in range(len(residual)):
        residual[i] -= np.median(residual[i][context.roi > 0])
    before_noise = np.std(context.history, axis=0)
    smooth_noise = cv2.GaussianBlur(before_noise, (0, 0), 4)
    noise = np.maximum(1., smooth_noise)
    maps = []
    for sigma in (1., 2., 4.):
        temporal = np.stack([cv2.GaussianBlur(im, (0, 0), sigma)-cv2.GaussianBlur(im, (0, 0), 3*sigma) for im in residual])
        for polarity in (1., -1.):
            # Third strongest frame requires repeated evidence, while retaining onset.
            support = np.sort(polarity*temporal, axis=0)
            score = support[-min(3, len(support))]/noise
            maps.append((sigma, polarity, score, temporal))
    border = np.zeros(pre.shape, bool)
    border[RADIUS+1:-RADIUS-1, RADIUS+1:-RADIUS-1] = True
    valid = (context.roi > 0) & border
    selected, candidates = [], []
    height, width = pre.shape
    per_cell = max(1, int(np.ceil(budget/(6*4*4))))
    for channel, (sigma, polarity, response, temporal) in enumerate(maps):
        maxima = (response >= cv2.dilate(response, np.ones((7, 7), np.uint8))) & valid & (response > .5)
        for gy in range(4):
            for gx in range(4):
                ya, yb = gy*height//4, (gy+1)*height//4
                xa, xb = gx*width//4, (gx+1)*width//4
                ys, xs = np.where(maxima[ya:yb, xa:xb])
                if not len(xs):
                    continue
                ys += ya
                xs += xa
                order = np.lexsort((xs, ys, -response[ys, xs]))[:per_cell]
                for i in order:
                    x, y = int(xs[i]), int(ys[i])
                    evidence = polarity*temporal[:, y, x]/noise[y, x]
                    timestamps = [t for t, z in zip(context.post_times, evidence) if z > .5]
                    candidates.append(dict(camera_x=float(x+context.origin[0]), camera_y=float(y+context.origin[1]),
                        proposal_response=float(response[y, x]), scale=sigma, polarity=polarity, channel=channel,
                        support_frames=len(timestamps), support_span=(max(timestamps)-min(timestamps)) if timestamps else 0.,
                        ready=len(timestamps)>=3 and max(timestamps)-min(timestamps)>=.09,
                        evidence_timestamp=max(timestamps) if timestamps else None))
    # Round-robin source-independent channel/cell selection happened above; sort
    # only the bounded union. Response is a proposal priority, never verifier score.
    for candidate in sorted(candidates, key=lambda c:(-c['proposal_response'], c['channel'], c['camera_y'], c['camera_x'])):
        if any((candidate['camera_x']-c['camera_x'])**2+(candidate['camera_y']-c['camera_y'])**2 < 8**2 for c in selected):
            continue
        selected.append(candidate)
        if len(selected) >= budget:
            break
    return selected
