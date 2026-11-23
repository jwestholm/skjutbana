"""Universal (whole-viewport) candidate-peak dataset from recorded source sessions with camera-XY truth.

For every truth shot that is isolated in time (previous shot >= 0.5 s before, next shot >= 0.7 s after):
PRE = 3 frames before the shot, POST = 3 frames from +0.35 s (before the next shot).  The search area is
the bounding box of the session's truth points + 120 px, tiled 240x240 with 24 px overlap; every tile's dense
dark peaks get the shared features (src/engine/camera/dense_peak_model.py).  Labels: within 5 camera px of
the shot's truth = positive, > 12 px = negative (3 px dedupe across tiles).  Measurement only.

    python3 -m automation.global_peak_dataset OUT.npz SESSION_DIR [SESSION_DIR ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from automation.can_dense_offline import Frames
from src.engine.camera.dense_peak_model import FEATURES, dedupe, tile_peaks

TILE, OVER = 240, 24


def truth_rows(d: Path):
    rows = [json.loads(l) for l in open(d / "ground_truth.jsonl")]
    return sorted((r for r in rows if r.get("camera_x") is not None and r.get("kind") != "no_impact"),
                  key=lambda r: r["wall_ts"])


def main():
    out = sys.argv[1]
    X, meta, manifest = [], [], []
    for sd in sys.argv[2:]:
        d = Path(sd)
        T = truth_rows(d)
        if not T:
            continue
        F = Frames(d)
        xs = [t["camera_x"] for t in T]; ys = [t["camera_y"] for t in T]
        x0r, y0r = max(int(min(xs)) - 120, 0), max(int(min(ys)) - 120, 0)
        x1r, y1r = int(max(xs)) + 120, int(max(ys)) + 120
        for k, t in enumerate(T):
            pk = t["wall_ts"]
            prev = T[k - 1]["wall_ts"] if k else -math.inf
            nxt = T[k + 1]["wall_ts"] if k + 1 < len(T) else math.inf
            if pk - prev < 0.5 or nxt - pk < 0.7:
                continue
            pre_i = list(np.where((F.ts < pk - 0.02) & (F.ts > prev + 0.45))[0][-3:])
            post_i = list(np.where((F.ts >= pk + 0.35) & (F.ts < nxt))[0][:3])
            if len(pre_i) < 2 or len(post_i) < 3:
                continue
            pre = [F.gray(i) for i in pre_i]; post = [F.gray(i) for i in post_i]
            rows = []
            for ty in range(y0r, y1r, TILE - OVER):
                for tx in range(x0r, x1r, TILE - OVER):
                    a = np.stack([p[ty:ty + TILE, tx:tx + TILE] for p in pre]).astype(np.float32)
                    b = np.stack([p[ty:ty + TILE, tx:tx + TILE] for p in post]).astype(np.float32)
                    if a.shape[1] < 40 or a.shape[2] < 40:
                        continue
                    rows += tile_peaks(a, b, tx, ty)
            rows = dedupe(rows)
            manifest.append(dict(session=d.name, sid=k, background=t.get("background"),
                                 truth_source=t.get("truth_source"), n_candidates=len(rows)))
            for cx, cy, s, f in rows:
                dist = math.hypot(cx - t["camera_x"], cy - t["camera_y"])
                X.append(f)
                meta.append(dict(session=d.name, sid=k, dist=dist, s=float(s), cx=float(cx), cy=float(cy),
                                 background=t.get("background"), truth_source=t.get("truth_source")))
        print(d.name, len(manifest), "shots so far", flush=True)
    np.savez_compressed(out, X=np.array(X, np.float32), features=np.array(FEATURES))
    with open(out.replace(".npz", "_meta.jsonl"), "w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    with open(out.replace(".npz", "_shots.jsonl"), "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    print(out, len(X), "rows", len(manifest), "shots")


if __name__ == "__main__":
    main()
