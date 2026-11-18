"""The grid-indexed V2.4 shot accumulator must equal the reference implementation exactly.

  python3 -m automation.v24_accumulator_equivalence_selftest
"""
from __future__ import annotations

import copy
import json
import random
import types
import unittest

from src.engine.camera import detector_v24_extension as v24


def _frames(seed: int, n_frames: int = 6, n: int = 220):
    rng = random.Random(seed)
    base = [(rng.uniform(0, 600), rng.uniform(0, 400)) for _ in range(n)]
    frames = []
    for f in range(n_frames):
        cands = []
        for i, (x, y) in enumerate(base):
            if rng.random() < 0.7:
                c = dict(camera_x=round(x + rng.gauss(0, 1.5), 1 if i % 3 else 0),
                         camera_y=round(y + rng.gauss(0, 1.5), 1),
                         score=rng.uniform(3, 30), detector_v1=float(rng.random() < 0.3),
                         detector_v2=float(rng.random() < 0.8), v24_tile_probe=float(rng.random() < 0.2),
                         v2_absdiff=rng.uniform(0, 12), v24_patch_prior=rng.uniform(0, 1))
                if rng.random() < 0.05:
                    c["candidate_bank_carried"] = 1.0
                cands.append(c)
        # exact duplicates and ties on purpose
        cands += [dict(cands[0]), dict(cands[1], camera_x=cands[1]["camera_x"] + 3.0)]
        frames.append((1000.0 + 0.04 * f, cands))
    return frames


class AccumulatorEquivalence(unittest.TestCase):
    def test_identical(self):
        cfg = {}
        for seed in range(6):
            a, b = types.SimpleNamespace(), types.SimpleNamespace()
            for ts, cands in _frames(seed):
                out_ref = v24._update_shot_accumulator_reference(a, shot_id=7, candidates=copy.deepcopy(cands), frame_ts=ts, cfg=cfg)
                out_new = v24._update_shot_accumulator(b, shot_id=7, candidates=copy.deepcopy(cands), frame_ts=ts, cfg=cfg)
                self.assertEqual(json.dumps(out_ref, sort_keys=True, default=str),
                                 json.dumps(out_new, sort_keys=True, default=str))
                self.assertEqual(json.dumps(a._v24_shot_accumulators, sort_keys=True, default=str),
                                 json.dumps(b._v24_shot_accumulators, sort_keys=True, default=str))


if __name__ == "__main__":
    unittest.main()
