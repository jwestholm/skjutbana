"""Offline real-stamp augmentation for the dense OBJECT peak model (measurement/training only).

For every MISS shot of a projected-harness session (the drawn hole is outside all objects), a real
airgun hole stamp (hole_library clean_stamps, multiplicative D, see src/engine/offline/hole_stamps.py) is
rendered into every camera frame from the shot onwards at a random camera point whose game position lies
inside one object's exact shape (>= 6 game px from the edge test is not required: edges are realistic).
The depth is scaled by U(lo, hi) (default 0.3-1.0 -> core ~0.05-0.3, like real holes).  The PRE frames
stay untouched.  Candidates are then extracted with exactly the dataset code (shot_rows), and labelled
against the pasted position.  Each miss shot is also emitted unchanged as a control (truth_object None).

    python3 -m automation.can_peak_augment OUT.npz SESSION RUN.log TRUTH.jsonl [...]
Env: AUG_SEED (1), AUG_DEPTH=lo,hi (0.3,1.0), AUG_PER_SHOT (2), AUG_SCALE=lo,hi (0.8,1.2)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from automation.can_dense_offline import Frames, load_shots, mapper
from automation.can_peak_dataset import BRIGHT, shot_rows
from src.engine.camera.dense_peak_model import FEATURES, FEATURES_V2
from src.engine.camera.object_physical_query import _dense_dx
from src.engine.game_objects.geometry import shape_contains
from src.engine.offline.hole_stamps import StampSet


class AugFrames:
    """Frames view that renders one stamp into every frame at/after t0 (gray, multiplicative)."""

    def __init__(self, base: Frames):
        self.base, self.ts = base, base.ts
        self.t0, self.stamp, self.cx, self.cy = math.inf, None, 0, 0

    def gray(self, i):
        g = self.base.gray(i)
        if self.stamp is None or self.ts[i] < self.t0:
            return g
        g = np.array(g, dtype=np.float32, copy=True)
        h, w = self.stamp.shape
        x0, y0 = int(round(self.cx)) - w // 2, int(round(self.cy)) - h // 2
        ys, xs = slice(max(y0, 0), min(y0 + h, g.shape[0])), slice(max(x0, 0), min(x0 + w, g.shape[1]))
        d = self.stamp[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        g[ys, xs] *= (1.0 - d)
        return g


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(int(os.environ.get("AUG_SEED", "1")))
    dlo, dhi = map(float, os.environ.get("AUG_DEPTH", "0.3,1.0").split(","))
    slo, shi = map(float, os.environ.get("AUG_SCALE", "0.8,1.2").split(","))
    per_shot = int(os.environ.get("AUG_PER_SHOT", "2"))
    bank = StampSet()
    to_game = mapper()
    X, meta, manifest = [], [], []
    args = sys.argv[2:]
    for i in range(0, len(args), 3):
        session, log, truth_path = args[i:i + 3]
        sname = session.rstrip("/").split("/")[-1]
        truth = {}
        for l in open(truth_path):
            r = json.loads(l)
            if r.get("kind") == "shot" and r.get("shot_id") is not None:
                truth[int(r["shot_id"])] = r
        shots = load_shots(log)
        frames = AugFrames(Frames(session))
        peaks = sorted(s["peak_ts"] for s in shots.values())
        n_aug = 0
        for sid in sorted(shots):
            t = truth.get(sid)
            if t is None or t.get("truth_object") is not None:
                continue
            shot = shots[sid]
            shapes = {p["object_id"]: p.get("shape") for p in shot.get("per_object", [])}
            regs = [r for r in shot["frozen_regions"] if shapes.get(r.get("object_id"))] or shot["frozen_regions"]
            pk = shot["peak_ts"]
            nxt = min([p for p in peaks if p > pk + 0.15] or [math.inf])
            variants = list(range(per_shot))   # unchanged miss shots are already in the base dataset
            for v in variants:
                frames.stamp = None
                tgt_obj, tgt_xy = None, None
                if v is not None:
                    for _ in range(200):
                        reg = regs[rng.integers(len(regs))]
                        x0, y0, bw, bh = reg["camera_aabb"]; x1, y1 = x0 + bw, y0 + bh
                        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
                        gx, gy = to_game(cx, cy); gx += _dense_dx(gx)
                        hit = [o for o, sh in shapes.items() if sh and shape_contains(sh, gx, gy)]
                        if len(hit) == 1:
                            tgt_obj, tgt_xy = hit[0], (gx, gy)
                            break
                    if tgt_obj is None:
                        continue
                    frames.stamp = (bank.patch(rng, float(rng.uniform(slo, shi)), extra_blur_px=0.4)[0]
                                    * float(rng.uniform(dlo, dhi))).astype(np.float32)
                    frames.cx, frames.cy, frames.t0 = cx, cy, pk + 0.02
                    n_aug += 1
                rows = shot_rows(shot, frames, to_game, nxt)
                vsid = sid if v is None else 100000 * (v + 1) + sid
                manifest.append(dict(session=sname, sid=vsid, truth=tgt_obj, n_candidates=len(rows), augmented=v is not None))
                for row in rows:
                    inside = [o for o, sh in shapes.items() if sh and shape_contains(sh, *row["game_xy"])]
                    if v is None:
                        tx, ty = (t.get("truth_game_xy") or [math.nan, math.nan])[:2]
                    else:
                        tx, ty = tgt_xy
                    X.append(row["f"])
                    meta.append(dict(bright=row.get("bright", False), session=sname, sid=vsid, truth=tgt_obj, exact=True,
                                     dist=math.hypot(row["game_xy"][0] - tx, row["game_xy"][1] - ty),
                                     inside=inside, s=row["s"], game_xy=row["game_xy"], augmented=v is not None))
        frames.stamp = None
        print(sname, "augmented", n_aug, "total shots", len(manifest), flush=True)
    np.savez_compressed(out, X=np.array(X, np.float32), features=np.array(FEATURES_V2 if BRIGHT else FEATURES))
    with open(out.replace(".npz", "_meta.jsonl"), "w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    with open(out.replace(".npz", "_shots.jsonl"), "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    print(out, len(X), "rows", len(manifest), "shots")


if __name__ == "__main__":
    main()
