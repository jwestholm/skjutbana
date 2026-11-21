"""Candidate-peak dataset from recorded Can Shootout sessions (offline, measurement only).

For every shot: the dense dark peaks (top 6 per tile, same tiles as can_dense_offline / live) with a feature
vector that describes the local signal more richly than (s, core, bipolar): radial darkening profile in
PRE/POST medians, per-frame stability, PRE temporal noise, local gradient, rank/ratio within the shot.
Labels come only from the projected-hole harness truth (exact drawn position) or, for real sessions, an
operator truth file (object level); they are never used by the features.

    python3 -m automation.can_peak_dataset OUT.npz SESSION RUN.log TRUTH.jsonl [SESSION RUN.log TRUTH.jsonl ...]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from automation.can_dense_offline import Frames, env, load_shots, mapper
import os
from src.engine.camera.dense_peak_model import FEATURES, FEATURES_V2, tile_bounds, tile_peaks

BRIGHT = os.environ.get("CAN_PEAK_BRIGHT") == "1"
from src.engine.camera.object_physical_query import _dense_dx
from src.engine.game_objects.geometry import shape_contains

def shot_rows(shot, frames, to_game, next_peak=math.inf):
    age = env("SKJUTBANA_OBJECT_DENSE_POST_AGE_S", 0.35)
    need = int(env("SKJUTBANA_OBJECT_DENSE_POST_FRAMES", 3))
    pk = shot["peak_ts"]
    pre_i = []
    for t in shot.get("pre_frame_ts") or []:
        i = int(np.argmin(np.abs(frames.ts - t)))
        if abs(frames.ts[i] - t) <= 0.002:          # exact recorded frame only
            pre_i.append(i)
    end = min(pk + env("SKJUTBANA_OBJECT_DENSE_DEADLINE_S", 1.2), next_peak)   # causal: stop at the next shot
    post_i = [i for i in np.where((frames.ts >= pk + age) & (frames.ts < end))[0][:need]]
    if not pre_i or len(post_i) < need:
        return []
    pre = [frames.gray(i) for i in pre_i]
    post = [frames.gray(i) for i in post_i]
    rows = []
    for reg in shot["frozen_regions"]:
        x0, y0, x1, y1 = tile_bounds(reg["camera_aabb"])
        pre_s = np.stack([p[y0:y1, x0:x1] for p in pre]).astype(np.float32)
        post_s = np.stack([p[y0:y1, x0:x1] for p in post]).astype(np.float32)
        for bright in ((False, True) if BRIGHT else (False,)):
            for cx, cy, s, f in tile_peaks(pre_s, post_s, x0, y0, bright=bright):
                gxy = to_game(cx, cy)
                gxy = (gxy[0] + _dense_dx(gxy[0]), gxy[1])
                rows.append(dict(f=f + ([1.0 if bright else 0.0] if BRIGHT else []), cx=cx, cy=cy,
                                 game_xy=[float(gxy[0]), float(gxy[1])], s=float(s), bright=bright))
    uniq = []
    for r in sorted(rows, key=lambda r: -r["s"]):
        if all(math.hypot(r["cx"] - q["cx"], r["cy"] - q["cy"]) > 3 or r.get("bright") != q.get("bright") for q in uniq):
            uniq.append(r)
    return uniq


def main():
    out = sys.argv[1]
    X, meta, manifest = [], [], []
    to_game = mapper()
    args = sys.argv[2:]
    for i in range(0, len(args), 3):
        session, log, truth_path = args[i:i + 3]
        truth = {}
        for l in open(truth_path):
            r = json.loads(l)
            if r.get("kind") == "shot" and r.get("shot_id") is not None:
                truth[int(r["shot_id"])] = r
        shots, frames = load_shots(log), Frames(session)
        peaks_sorted = sorted(s["peak_ts"] for s in shots.values())
        for sid in sorted(shots):
            t = truth.get(sid)
            if t is None:
                continue
            shapes = {p["object_id"]: p.get("shape") for p in shots[sid].get("per_object", [])}
            pk = shots[sid]["peak_ts"]
            nxt = min([p for p in peaks_sorted if p > pk + 0.15] or [math.inf])
            rows = shot_rows(shots[sid], frames, to_game, nxt)
            manifest.append(dict(session=session.rstrip("/").split("/")[-1], sid=sid, truth=t["truth_object"],
                                 n_candidates=len(rows)))
            for row in rows:
                inside = [o for o, sh in shapes.items() if sh and shape_contains(sh, *row["game_xy"])]
                tx, ty = (t.get("truth_game_xy") or [None, None])[:2]
                exact = t.get("source") != "operator_visual"
                dist = math.hypot(row["game_xy"][0] - tx, row["game_xy"][1] - ty) if exact else None
                X.append(row["f"])
                meta.append(dict(bright=row.get("bright", False), session=session.rstrip("/").split("/")[-1], sid=sid, truth=t["truth_object"],
                                 exact=exact, dist=dist, inside=inside, s=row["s"], game_xy=row["game_xy"]))
    np.savez_compressed(out, X=np.array(X, np.float32), features=np.array(FEATURES_V2 if BRIGHT else FEATURES))
    with open(out.replace(".npz", "_meta.jsonl"), "w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    with open(out.replace(".npz", "_shots.jsonl"), "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    print(out, len(X), "rows", len(manifest), "shots", sum(1 for m in manifest if m["n_candidates"] == 0), "without candidates")


if __name__ == "__main__":
    main()
