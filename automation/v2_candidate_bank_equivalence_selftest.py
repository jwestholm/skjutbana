"""The grid-indexed V2 candidate bank must equal the reference implementation exactly.

  python3 -m automation.v2_candidate_bank_equivalence_selftest
"""
from __future__ import annotations

import copy
import json
import random
import unittest

from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2


def _frames(seed: int, n_frames: int = 8, n: int = 200):
    rng = random.Random(seed)
    base = [(rng.uniform(0, 500), rng.uniform(0, 300)) for _ in range(n)]
    frames = []
    for f in range(n_frames):
        cands = []
        for x, y in base:
            if rng.random() < 0.65:
                cands.append(dict(camera_x=round(x + rng.gauss(0, 1.2), 1), camera_y=round(y + rng.gauss(0, 1.2), 1),
                                  score=rng.uniform(3, 30), detector_v1=float(rng.random() < 0.3),
                                  detector_v2=1.0, v2_primary_peak=float(rng.random() < 0.6),
                                  detector_agreement=float(rng.random() < 0.1)))
        cands.append(dict(cands[0]))
        frames.append((2000.0 + 0.04 * f, cands))
    return frames


class BankEquivalence(unittest.TestCase):
    def test_identical(self):
        for seed in range(5):
            for limit in (200, 260):
                a = CandidateGeneratorV2()
                b = CandidateGeneratorV2()
                cfg = {"candidate_bank_output_limit": limit}
                for ts, cands in _frames(seed):
                    r = a._update_candidate_bank_reference(shot_id=5, frame_candidates=copy.deepcopy(cands), frame_ts=ts, cfg=cfg)
                    n = b._update_candidate_bank(shot_id=5, frame_candidates=copy.deepcopy(cands), frame_ts=ts, cfg=cfg)
                    self.assertEqual(json.dumps(r, sort_keys=True), json.dumps(n, sort_keys=True))
                    self.assertEqual(json.dumps(a._candidate_banks, sort_keys=True, default=str),
                                     json.dumps(b._candidate_banks, sort_keys=True, default=str))


if __name__ == "__main__":
    unittest.main()
