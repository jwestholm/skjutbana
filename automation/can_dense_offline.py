"""Offline replay of the Can Shootout dense OBJECT decision on a recorded source session (measurement only).

Inputs: a source session (SKJUTBANA_SOURCE_RECORD=1) plus the run's log ([OBJECT-HIT] lines give peak_ts,
PRE frame timestamps, frozen camera AABBs and exact game shapes) and, optionally, the projected-hole harness
jsonl (truth per shot_id).  Re-runs dense_top + the acceptance rule from object_physical_query on the recorded
frames so rule/timing changes can be tried in seconds and then verified live on the projector.

This is an approximation of the live tiles (AABB + margin, no per-pixel mask); `agree` reports how often it
matches the live decision on the same shots.  Rule knobs: the same SKJUTBANA_OBJECT_DENSE_* env variables.

    python3 -m automation.can_dense_offline SESSION_DIR RUN.log [HARNESS.jsonl]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from src.engine import lens_model
from src.engine.camera.detector_luma import detector_gray
from src.engine.camera.camera_manager import apply_frame_transform
from src.engine.game_objects.geometry import shape_contains
from src.engine.shot_dense_h15 import _core_bipolar, dense_top
from src.engine.camera.object_physical_query import _dense_dx


def env(name, default):
    return float(os.environ.get(name, default))


MARGIN = int(env("CAN_OFF_MARGIN", 36))   # live margin 36 px; smaller tiles lose edge peaks (dense border)


def load_shots(log_path):
    shots, bad = {}, 0
    dec = json.JSONDecoder()
    for line in open(log_path, errors="replace"):
        if line.startswith("[OBJECT-HIT] "):
            try:
                d, _end = dec.raw_decode(line[13:])   # tolerate text appended by another thread's print
            except ValueError:
                bad += 1
                continue
            shots[int(d["shot_id"])] = d
    if bad:
        print(f"[can_dense_offline] {log_path}: skipped {bad} unparsable [OBJECT-HIT] lines", file=sys.stderr)
    return shots


def mapper(settings="content/settings.json"):
    s = json.load(open(settings))
    cal = s["camera_calibration"]
    vx, vy = s["viewport"][:2]
    lens = lens_model.parse(cal)
    if lens is not None:
        return lambda x, y: tuple(np.subtract(lens_model.camera_to_screen(lens, x, y), (vx, vy)))
    H = np.linalg.inv(np.array(cal["homography"], float))   # screen->camera stored; invert

    def f(x, y):
        p = H @ np.array([x, y, 1.0])
        return p[0] / p[2] - vx, p[1] / p[2] - vy
    return f


class Frames:
    def __init__(self, session):
        self.dir = Path(session)
        self.rows = [json.loads(l) for l in open(self.dir / "camera_index.jsonl")]
        self.ts = np.array([r["acquisition"]["receive_wall_ts"] for r in self.rows])
        self.fh = open(self.dir / "camera.bin", "rb")
        self.cache = {}

    def gray(self, i):
        if i not in self.cache:
            r = self.rows[i]
            self.fh.seek(int(r["offset"]))
            px = cv2.imdecode(np.frombuffer(self.fh.read(int(r["length"])), np.uint8), cv2.IMREAD_COLOR)
            a = r["acquisition"]
            px = apply_frame_transform(px, int(a.get("rotation", 0) or 0), bool(a.get("mirror_horizontal")),
                                       bool(a.get("mirror_vertical")))
            self.cache = {i: detector_gray(px).astype(np.float32)} if len(self.cache) > 40 else {**self.cache, i: detector_gray(px).astype(np.float32)}
        return self.cache[i]


def decide(shot, frames, to_game):
    age = env("SKJUTBANA_OBJECT_DENSE_POST_AGE_S", 0.35)
    need = int(env("SKJUTBANA_OBJECT_DENSE_POST_FRAMES", 3))
    deadline = env("SKJUTBANA_OBJECT_DENSE_DEADLINE_S", 1.2)
    pk = shot["peak_ts"]
    pre_i = [int(np.argmin(np.abs(frames.ts - t))) for t in shot.get("pre_frame_ts") or []]
    post_i = [i for i in np.where((frames.ts >= pk + age) & (frames.ts <= pk + deadline))[0][:need]]
    if not pre_i or len(post_i) < need:
        return None, [], "insufficient_frames"
    pre = [frames.gray(i) for i in pre_i]
    post = [frames.gray(i) for i in post_i]
    peaks = []
    shapes = {p["object_id"]: p.get("shape") for p in shot.get("per_object", [])}
    for reg in shot["frozen_regions"]:
        x, y, w, h = reg["camera_aabb"]
        x0, y0 = max(int(x) - MARGIN, 0), max(int(y) - MARGIN, 0)
        x1, y1 = int(math.ceil(x + w)) + MARGIN, int(math.ceil(y + h)) + MARGIN
        pre_s = np.stack([p[y0:y1, x0:x1] for p in pre]); post_s = np.stack([p[y0:y1, x0:x1] for p in post])
        pre_m = np.median(pre_s, 0).astype(np.float32)
        post_m = np.median(post_s, 0).astype(np.float32)
        cand = [(px, py, s, 'dark') for px, py, s in dense_top(pre_m, post_m, top_n=6)]
        if os.environ.get("CAN_OFF_BRIGHT"):   # hypothesis: some real holes appear BRIGHT (light behind)
            cand += [(px, py, s, 'bright') for px, py, s in dense_top(post_m, pre_m, top_n=6)]
        for px, py, s, kind in cand:
            core, bip = (_core_bipolar(pre_m, post_m, px, py) if kind == 'dark' else _core_bipolar(post_m, pre_m, px, py))
            cx, cy = x0 + px, y0 + py
            yi, xi = int(round(py)), int(round(px))
            win = pre_m[max(yi - 4, 0):yi + 5, max(xi - 4, 0):xi + 5]
            edge = float(np.hypot(*np.gradient(win)).mean()) if win.size >= 9 else 0.0
            def spot(a):   # 3x3 spot minus 9x9 ring (local darkening proxy)
                c = a[..., max(yi - 1, 0):yi + 2, max(xi - 1, 0):xi + 2].mean(axis=(-1, -2))
                r = a[..., max(yi - 5, 0):yi + 6, max(xi - 5, 0):xi + 6].mean(axis=(-1, -2))
                return r - c
            dk_post = spot(post_s.astype(np.float32)); dk_pre = spot(pre_s.astype(np.float32))
            consist = float(dk_post.min() - dk_pre.max())   # every POST frame darker than every PRE frame
            peaks.append(dict(s=float(s), core=float(core), bipolar=float(bip), cx=cx, cy=cy, edge=edge, consist=consist, kind=kind,
                              game_xy=[float(v) + (_dense_dx(float(v)) if k == 0 else 0.0)
                                       for k, v in enumerate(to_game(cx, cy))]))
    # overlapping tiles can report the same peak twice
    uniq = []
    for p in sorted(peaks, key=lambda d: -d["s"]):
        if all(math.hypot(p["cx"] - q["cx"], p["cy"] - q["cy"]) > 3 for q in uniq):
            uniq.append(p)
    peaks = uniq
    holey = [p for p in peaks if p["core"] >= env("SKJUTBANA_OBJECT_DENSE_MIN_CORE", 0.15)
             and (p["core"] >= env("SKJUTBANA_OBJECT_DENSE_STRONG_CORE", 0.4)
                  or p["bipolar"] <= env("SKJUTBANA_OBJECT_DENSE_MAX_BIPOLAR", 1.8))]
    top = None
    if holey:
        s2 = holey[1]["s"] if len(holey) > 1 else 0.0
        if holey[0]["s"] >= env("SKJUTBANA_OBJECT_DENSE_MIN_S", 11) and (
                s2 <= 0 or holey[0]["s"] / s2 >= env("SKJUTBANA_OBJECT_DENSE_MIN_RATIO", 1.15)):
            top = holey[0]
    if top is None and peaks:
        s2 = peaks[1]["s"] if len(peaks) > 1 else 0.0
        if peaks[0]["s"] >= env("SKJUTBANA_OBJECT_DENSE_ISOLATED_S", 18) and (
                s2 <= 0 or peaks[0]["s"] / s2 >= env("SKJUTBANA_OBJECT_DENSE_ISOLATED_RATIO", 1.5)):
            top = peaks[0]
    mc = os.environ.get("SKJUTBANA_OBJECT_DENSE_MIN_CONSIST")
    if top is not None and mc is not None and top["consist"] < float(mc):
        return None, peaks, "inconsistent_over_frames"
    if top is None:
        return None, peaks, "dense_no_accepted_dot"
    inside = [oid for oid, sh in shapes.items() if sh and shape_contains(sh, *top["game_xy"])]
    return (inside[0] if len(inside) == 1 else None), peaks, ("hit" if len(inside) == 1 else "outside_or_ambiguous")


def main():
    session, log = sys.argv[1], sys.argv[2]
    truth = {}
    if len(sys.argv) > 3:
        for l in open(sys.argv[3]):
            r = json.loads(l)
            if r.get("kind") == "shot" and r.get("shot_id") is not None:
                truth[int(r["shot_id"])] = r
    shots, frames, to_game = load_shots(log), Frames(session), mapper()
    n = ok = wrong = false = agree = with_truth = 0
    for sid in sorted(shots):
        obj, peaks, why = decide(shots[sid], frames, to_game)
        t = truth.get(sid)
        live = shots[sid].get("object_id")
        agree += (obj == live)
        n += 1
        if t is not None:
            with_truth += 1
            if t["truth_object"]:
                ok += obj == t["truth_object"]
                wrong += obj is not None and obj != t["truth_object"]
            else:
                false += obj is not None
        if os.environ.get("CAN_OFF_DUMP"):
            with open(os.environ["CAN_OFF_DUMP"], "a") as fh:
                fh.write(json.dumps(dict(sid=sid, truth=t and t["truth_object"], truth_xy=t and t.get("truth_game_xy"),
                                         peaks=peaks[:4], shapes={p["object_id"]: p.get("shape") for p in shots[sid].get("per_object", [])})) + "\n")
        if os.environ.get("CAN_OFF_VERBOSE"):
            print(sid, "truth", t and t["truth_object"], "live", live, "off", obj, why,
                  [(round(p["s"], 1), round(p["core"], 2)) for p in peaks[:3]])
    on = sum(1 for t in truth.values() if t["truth_object"])
    print(json.dumps(dict(shots=n, agree_with_live=agree, on_object=on, correct=ok, wrong_object=wrong,
                          false_on_miss=false, misses=with_truth - on)))


if __name__ == "__main__":
    main()
