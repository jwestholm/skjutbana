"""Per-peak features + small logistic scorer for dense OBJECT candidates (SHADOW research, default off).

One implementation shared by the offline dataset tool (automation/can_peak_dataset.py) and the live
OBJECT dense query, so offline and live scores are computed by the same code.  Tiles are per frozen region:
camera AABB + 36 px margin, grey via detector_gray (pointwise), PRE = recorded PRE frames, POST = the dense
POST frames.  The model never decides gameplay here; the live query only logs what it would have done.

Enable live shadow scoring with SKJUTBANA_OBJECT_DENSE_MODEL=<model.json> (from automation/can_peak_model.py).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from src.engine.shot_dense_h15 import _core_bipolar, dense_top

RADII = (0, 1, 2, 3, 4, 6, 8)
FEATURES = (["s", "core", "bipolar", "rank", "ratio_next", "s_over_tile_median"]
            + [f"prof{r}" for r in RADII] + [f"prof{r}_z" for r in RADII]
            + ["pre_tstd", "post_tstd", "post_min_dk", "post_max_dk", "pre_max_dk", "consist", "edge", "local_level"])
TILE_MARGIN = 36


def ring_mean(a, x, y, r0, r1, d=None):
    if d is None:
        h, w = a.shape[-2:]
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot(xx - x, yy - y)
    m = (d >= r0) & (d < r1)
    return a[..., m].mean(axis=-1) if m.any() else np.zeros(a.shape[:-2])


def tile_bounds(camera_aabb):
    x, y, w, h = camera_aabb
    return (max(int(x) - TILE_MARGIN, 0), max(int(y) - TILE_MARGIN, 0),
            int(math.ceil(x + w)) + TILE_MARGIN, int(math.ceil(y + h)) + TILE_MARGIN)


def tile_peaks(pre_s, post_s, x0, y0):
    """pre_s/post_s: float32 stacks (n, h, w) of one tile.  Returns [(cx, cy, s, feature_vector)]."""
    pre_m, post_m = np.median(pre_s, 0), np.median(post_s, 0)
    tops = dense_top(pre_m, post_m, top_n=6)
    ss = [t[2] for t in tops]
    med_s = float(np.median(ss)) if ss else 1.0
    out = []
    for k, (px, py, s) in enumerate(tops):
        xi, yi = int(round(px)), int(round(py))
        win = (slice(max(yi - 12, 0), yi + 13), slice(max(xi - 12, 0), xi + 13))
        pre_w, post_w = pre_s[(slice(None),) + win], post_s[(slice(None),) + win]
        cx_l, cy_l = px - win[1].start, py - win[0].start
        pm, qm = np.median(pre_w, 0), np.median(post_w, 0)
        hh, ww = pm.shape
        yy, xx = np.mgrid[0:hh, 0:ww]
        d = np.hypot(xx - cx_l, yy - cy_l)          # one distance grid per peak
        diff = pm - qm
        outer = ring_mean(diff, cx_l, cy_l, 9, 12.5, d)
        prof = [float(ring_mean(diff, cx_l, cy_l, r, r + 1, d) - outer) for r in RADII]
        tstd = float(np.median(np.abs(pre_w - pm), axis=0).mean()) + 0.5

        def spot(a):   # darkening proxy: surrounding ring minus centre
            return ring_mean(a, cx_l, cy_l, 4, 7, d) - ring_mean(a, cx_l, cy_l, 0, 2, d)
        dk_post, dk_pre = spot(post_w), spot(pre_w)
        core, bip = _core_bipolar(pre_m, post_m, px, py)
        g = np.hypot(*np.gradient(pm))
        nxt = ss[k + 1] if k + 1 < len(ss) else 0.0
        f = [s, core, bip, k, s / nxt if nxt > 0 else 3.0, s / max(med_s, 1e-3)] + prof + [p / tstd for p in prof] + [
            tstd, float(np.median(np.abs(post_w - qm), axis=0).mean()), float(-dk_post.max()), float(-dk_post.min()),
            float(-dk_pre.min()), float(-dk_post.max() + dk_pre.min()),
            float(g[max(int(cy_l) - 4, 0):int(cy_l) + 5, max(int(cx_l) - 4, 0):int(cx_l) + 5].mean()), float(pm.mean())]
        out.append((x0 + px, y0 + py, float(s), f))
    return out


def dedupe(rows, key_xy=lambda r: (r[0], r[1]), key_s=lambda r: r[2]):
    uniq = []
    for r in sorted(rows, key=lambda r: -key_s(r)):
        if all(math.hypot(key_xy(r)[0] - key_xy(q)[0], key_xy(r)[1] - key_xy(q)[1]) > 3 for q in uniq):
            uniq.append(r)
    return uniq


class PeakModel:
    def __init__(self, path):
        m = json.load(open(path))
        self.path = path
        self.kind = m.get("kind", "logreg")
        if self.kind == "stumps":
            self.trees = [tuple(t) for t in m["trees"]]
        elif self.kind == "mlp":
            self.W1, self.b1 = np.array(m["W1"]), np.array(m["b1"])
            self.W2, self.b2 = np.array(m["W2"]), float(m["b2"])
        else:
            self.w, self.b = np.array(m["w"]), float(m["b"])
        self.mu, self.sd = np.array(m["mu"]), np.array(m["sd"])
        self.tau = float(m["tau"])
        self.ambig_p = float(m.get("ambig_p", 0.85))
        assert list(m["features"]) == list(FEATURES), "feature list mismatch"

    def p(self, feats):
        X = (np.asarray(feats, np.float64) - self.mu) / self.sd
        X = np.atleast_2d(X)
        if self.kind == "stumps":
            z = np.zeros(len(X))
            for j, t, vl, vr in self.trees:
                z += np.where(X[:, int(j)] <= t, vl, vr)
        elif self.kind == "mlp":
            z = np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2
        else:
            z = X @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


_MODEL = None


def live_model():
    global _MODEL
    path = os.environ.get("SKJUTBANA_OBJECT_DENSE_MODEL")
    if not path:
        return None
    if _MODEL is None or _MODEL.path != path:
        _MODEL = PeakModel(path)
    return _MODEL
