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


def local_contrast(patches):
    """One fixed r4 disc minus r8..12 annulus, in grayscale levels.

    Patches are 49x49 at an observed full-camera coordinate, using the existing
    accuracy_verifier.patch sampler. Polarity is unrestricted. These are repeated
    measurements of ONE channel, not independent feature-family votes.
    """
    values = np.asarray(patches, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (49, 49) or not np.isfinite(values).all():
        raise ValueError('Finite 49x49 camera patches required')
    y, x = np.mgrid[-24:25, -24:25]
    r2 = x*x + y*y
    return values[:, r2 <= 16].mean(axis=1) - values[:, (r2 >= 64) & (r2 <= 144)].mean(axis=1)


def causal_local_onset(signal, timestamps, *, cutoff, sequences=None):
    """Conservative OFFLINE observed-step/PRE proposal; never impact truth.

    Caller supplies only the event's uncontaminated, bounded camera history.
    Read timestamps are not exposure times. No audio offset, GT or object input.
    Three adjacent stable samples precede three changed samples; PRE is the last
    actual before sample, never a temporal blend. Multiple steps, reversion,
    sparse history and unstable signals cannot manufacture a safe reference.
    Constants are prespecified engineering criteria, not calibrated significance
    probabilities or NO_PHYSICAL_HIT thresholds.
    """
    x = np.asarray(signal, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if x.ndim != 1 or t.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(t).all() or not np.isfinite(cutoff):
        raise ValueError('Finite matching signal/timestamps required')
    if np.any(np.diff(t) <= 0) or (len(t) and t[-1] > cutoff):
        raise ValueError('Strictly ordered observations at/before cutoff required')
    if sequences is not None:
        seq = np.asarray(sequences)
        if seq.shape != t.shape or not np.isfinite(seq).all() or np.any(np.diff(seq) <= 0):
            raise ValueError('Camera sequences must be strictly increasing in one stream')
    result = dict(state='INSUFFICIENT_HISTORY', physical_state='UNCERTAIN',
                  pre_index=None, onset_index=None, bracket=None, support_available_ts=None,
                  noise=None, signed_step=None, candidates=[])
    if len(x) < 6 or np.any(np.diff(t) > .25):
        return result
    choices = []
    for k in range(3, len(x)-2):
        before, after = x[k-3:k], x[k:k+3]
        a, b = np.median(before), np.median(after)
        noise = max(.75, 1.4826*float(np.median(np.abs(before-a))),
                    1.4826*float(np.median(np.abs(after-b))))
        delta = float(b-a)
        sign = np.sign(delta)
        if (abs(delta) >= 3*noise and np.max(np.abs(before-a)) <= 2*noise
                and np.min(sign*(after-x[k-1])) >= 3*noise):
            choices.append(dict(index=k, noise=noise, signed_step=delta,
                persists=bool(np.min(sign*(x[k:]-x[k-1])) >= 3*noise)))
    result['candidates'] = choices
    if len(choices) != 1 or not choices[0]['persists']:
        result['state'] = 'NO_SUPPORTED_ONSET' if np.ptp(x) < 2.25 else 'ONSET_AMBIGUOUS'
        return result
    c = choices[0]; k = c['index']
    result.update(state='ONSET_FOUND', pre_index=k-1, onset_index=k,
        bracket=[float(t[k-1]), float(t[k])], support_available_ts=float(t[k+2]),
        noise=c['noise'], signed_step=c['signed_step'])
    return result


def persistent_local_support(reference, post, noise):
    """Weakest same-polarity response of the SAME three POST observations.

    Reference/post are the fixed local-contrast scalar, not fitted features.
    Zero support is uncertainty, never proof that no impact happened.
    """
    delta = np.asarray(post, dtype=float) - float(reference)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(noise) or noise < .75:
        raise ValueError('Three finite POST values and noise >= .75 required')
    signed = float(np.median(delta))
    return dict(signed_change=signed, support=float(max(0., np.min(np.sign(signed)*delta))/noise))


# Same channel and reduction order as local_contrast; no additional evidence.
_Y, _X = np.mgrid[-24:25, -24:25]
_R2 = _X*_X + _Y*_Y
_LOCAL_MASKS = (_R2 <= 16, (_R2 >= 64) & (_R2 <= 144))


def local_contrast_batch(patches):
    """T x candidates x 49 x 49 -> T x candidates, exactly as scalar sampling.

    Only mask pixels are promoted to float64. Keep the pixel reduction on a
    non-contiguous axis, matching the reference's boolean-indexed layout and
    summation order (a fast-axis reduction can change floating-point results).
    """
    p = np.asarray(patches)
    if p.ndim != 4 or p.shape[2:] != (49, 49) or not np.isfinite(p).all():
        raise ValueError('Finite T x candidates x 49 x 49 patches required')
    means = []
    for mask in _LOCAL_MASKS:
        values = np.ascontiguousarray(p[:, :, mask].transpose(2, 0, 1), dtype=np.float64)
        means.append(values.mean(axis=0))
    return means[0] - means[1]


def _observed_step_choices_batch(x):
    """Shared arithmetic only; callers retain clocked/ordinal state semantics."""
    choices = [[] for _ in range(x.shape[1])]
    if len(x) < 6 or not x.shape[1]:
        return choices
    # window[k] = x[k:k+3], with the sample axis last for medians.
    window = np.lib.stride_tricks.sliding_window_view(x, 3, axis=0)
    before, after = window[:-3], window[3:]
    a, b = np.median(before, axis=2), np.median(after, axis=2)
    deviation = np.abs(before-a[:, :, None])
    noise = np.maximum(.75, np.maximum(1.4826*np.median(deviation, axis=2),
        1.4826*np.median(np.abs(after-b[:, :, None]), axis=2)))
    delta = b-a
    sign = np.sign(delta)
    qualifies = ((np.abs(delta) >= 3*noise) & (np.max(deviation, axis=2) <= 2*noise)
        & (np.min(sign[:, :, None]*(after-x[2:-3, :, None]), axis=2) >= 3*noise))
    for split, j in zip(*np.nonzero(qualifies)):
        k = int(split)+3
        choices[j].append(dict(index=k, noise=float(noise[split, j]),
            signed_step=float(delta[split, j]),
            persists=bool(np.min(sign[split, j]*(x[k:, j]-x[k-1, j])) >= 3*noise[split, j])))
    return choices


def causal_local_onset_batch(signals, timestamps, *, cutoff, sequences=None):
    """Batch dispatch of unchanged causal_local_onset criteria and clocks."""
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite T x candidates signals required')
    base = causal_local_onset(np.zeros(len(x)), timestamps, cutoff=cutoff, sequences=sequences)
    results = [dict(base, candidates=[]) for _ in range(x.shape[1])]
    if base['state'] == 'INSUFFICIENT_HISTORY':
        return results
    for r, choices in zip(results, _observed_step_choices_batch(x)):
        r['candidates'] = choices
    for j, r in enumerate(results):
        choices = r['candidates']
        if len(choices) != 1 or not choices[0]['persists']:
            r['state'] = 'NO_SUPPORTED_ONSET' if np.ptp(x[:, j]) < 2.25 else 'ONSET_AMBIGUOUS'
            continue
        c = choices[0]; k = c['index']
        r.update(state='ONSET_FOUND', pre_index=k-1, onset_index=k,
            bracket=[float(timestamps[k-1]), float(timestamps[k])],
            support_available_ts=float(timestamps[k+2]), noise=c['noise'], signed_step=c['signed_step'])
    return results


def causal_local_evidence(times, patches, snapshot, cutoff, early_times=()):
    """Exact batched equivalent of the frozen research infer, including diagnostics.

    All input patches must come from a cutoff-bounded camera context. The old
    snapshot and historical early median are comparison diagnostics, never a
    manufactured causal PRE. Outputs describe observed change, not impact truth.
    """
    signal = local_contrast_batch(patches)
    rows = causal_local_onset_batch(signal, times, cutoff=cutoff)
    old = local_contrast_batch(snapshot[1][None])[0] if snapshot else None
    early = [i for i, t in enumerate(times) if t in early_times]
    early_pixels = early_signal = None
    if early and len(early) == len(early_times):
        early_pixels = np.median(patches[early], axis=0)
        early_signal = local_contrast_batch(early_pixels[None])[0]
    for j, r in enumerate(rows):
        images = patches[:, j]; values = signal[:, j]
        r.update(signal=values.tolist(), pre_status='UNAVAILABLE', old=None, causal=None, history_early=None)
        noise = r['noise'] or max(.75, 1.4826*float(np.median(np.abs(values[:3]-np.median(values[:3])))))
        if early_signal is not None:
            r['history_early'] = persistent_local_support(early_signal[j], values[-3:], noise)
            r['history_early']['absolute_pixel_change'] = float(np.abs(images[-3:]-early_pixels[j]).mean())
        if old is not None and len(times) >= 3:
            r['old'] = persistent_local_support(old[j], values[-3:], noise)
            r['old']['absolute_pixel_change'] = float(np.abs(images[-3:]-snapshot[1][j]).mean())
        if r['state'] == 'ONSET_FOUND':
            k = r['pre_index']
            r['causal'] = persistent_local_support(values[k], values[-3:], noise)
            r['causal']['absolute_pixel_change'] = float(np.abs(images[-3:]-images[k]).mean())
            if snapshot:
                r['pre_status'] = ('PRE_UNCHANGED_SAFE' if snapshot[0]['timestamp'] <= times[k]
                    else 'PRE_CONTAMINATED' if snapshot[0]['timestamp'] >= times[k+1] else 'ONSET_AMBIGUOUS')
    return rows


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


def observed_local_step(signal):
    """Frozen three-before/three-after amplitude criteria, WITHOUT time claims.

    Kept here as a measurement adapter because feeding fabricated timestamps to
    the causal API would silently certify unknown frame gaps. Tests compare the
    amplitude result with the reference on fixtures with genuine test timing.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('finite one-dimensional signal required')
    choices = []
    for k in range(3, len(x)-2):
        before, after = x[k-3:k], x[k:k+3]
        a, b = np.median(before), np.median(after)
        noise = max(.75, 1.4826*np.median(np.abs(before-a)),
                    1.4826*np.median(np.abs(after-b)))
        delta = float(b-a)
        if (abs(delta) >= 3*noise and np.max(np.abs(before-a)) <= 2*noise
                and np.min(np.sign(delta)*(after-x[k-1])) >= 3*noise):
            choices.append(dict(index=k, signed_step=delta, noise=float(noise),
                persists=bool(np.min(np.sign(delta)*(x[k:]-x[k-1])) >= 3*noise)))
    state = ('INSUFFICIENT_SAMPLES' if len(x) < 6 else
             'UNIQUE_PERSISTENT_OBSERVED_STEP' if len(choices) == 1 and choices[0]['persists'] else
             'NO_SUPPORTED_OBSERVED_STEP' if np.ptp(x) < 2.25 else 'AMBIGUOUS')
    return dict(state=state, candidates=choices, causal_certification='UNAVAILABLE')



def temporal_step_evidence(signal, times=None, sequences=None):
    """Same amplitude rule everywhere; actual D02 clocks additionally validate gaps."""
    if times is None:
        r=observed_local_step(signal)
        supported=r['state']=='UNIQUE_PERSISTENT_OBSERVED_STEP'
        choice=r['candidates'][0] if supported else None
        state=r['state']; certification='ORDINAL_ONLY_TIMING_UNKNOWN'
    else:
        r=causal_local_onset(signal,times,cutoff=times[-1],sequences=sequences)
        supported=r['state']=='ONSET_FOUND'
        choice=r['candidates'][0] if supported else None
        state=r['state'];certification='RECORDED_RECEIPT_ORDER_NOT_EXPOSURE'
    out=dict(state=state,supported=supported,choices=r['candidates'],certification=certification,
             support=0.,onset_ordinal=None,pre_ordinal=None)
    if supported:
        k=choice['index'];e=persistent_local_support(signal[k-1],signal[-3:],choice['noise'])
        out.update(support=e['support'],signed_change=e['signed_change'],
                   onset_ordinal=k,pre_ordinal=k-1,noise=choice['noise'])
    return out


def temporal_step_evidence_batch(signals, times=None, sequences=None):
    """Same frozen evidence schema for T x bounded candidate signals.

    Ordinal observations use the existing batch amplitude kernel directly; no
    fabricated clocks are passed to the receipt-causal API.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite T x candidates signals required')
    if times is None:
        rows = []
        for j, choices in enumerate(_observed_step_choices_batch(x)):
            state = ('INSUFFICIENT_SAMPLES' if len(x) < 6 else
                'UNIQUE_PERSISTENT_OBSERVED_STEP' if len(choices) == 1 and choices[0]['persists'] else
                'NO_SUPPORTED_OBSERVED_STEP' if np.ptp(x[:, j]) < 2.25 else 'AMBIGUOUS')
            rows.append(dict(state=state, candidates=choices))
        supported_state = 'UNIQUE_PERSISTENT_OBSERVED_STEP'
        certification = 'ORDINAL_ONLY_TIMING_UNKNOWN'
    else:
        rows = causal_local_onset_batch(x, times, cutoff=times[-1], sequences=sequences)
        supported_state = 'ONSET_FOUND'
        certification = 'RECORDED_RECEIPT_ORDER_NOT_EXPOSURE'
    output = []
    for j, r in enumerate(rows):
        supported = r['state'] == supported_state
        out = dict(state=r['state'], supported=supported, choices=r['candidates'],
            certification=certification, support=0., onset_ordinal=None, pre_ordinal=None)
        if supported:
            choice = r['candidates'][0]; k = choice['index']
            evidence = persistent_local_support(x[k-1, j], x[-3:, j], choice['noise'])
            out.update(support=evidence['support'], signed_change=evidence['signed_change'],
                onset_ordinal=k, pre_ordinal=k-1, noise=choice['noise'])
        output.append(out)
    return output
