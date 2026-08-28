from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from src.engine.offline.temporal_local_v2212 import LocalTemporalConfigV2212, propose_local_temporal_v2212


def _assert(condition: bool, text: str) -> None:
    if not condition:
        raise AssertionError(text)
    print(f"[PASS] {text}")


def main() -> int:
    h, w = 360, 640
    yy, xx = np.mgrid[:h, :w]
    pre = (185 + 18 * np.sin(xx / 45.0) + 12 * np.cos(yy / 37.0)).astype(np.float32)
    pre = np.clip(pre, 0, 255).astype(np.uint8)
    post1 = pre.copy()
    post2 = pre.copy()
    gt = (333, 177)
    # Compact persistent dark mark plus mild camera/global nuisance.
    for frame, global_offset in ((post1, 3), (post2, -2)):
        tmp = np.clip(frame.astype(np.int16) + global_offset, 0, 255).astype(np.uint8)
        cv2.circle(tmp, gt, 4, 65, -1)
        cv2.circle(tmp, gt, 7, 135, 1)
        frame[:] = tmp

    direct = propose_direct_v221([pre], [post1, post2], config=DirectProposalConfigV221())
    current = [
        {"camera_x": float(gt[0] + 31), "camera_y": float(gt[1] - 18), "score": 50.0},
        {"camera_x": 110.0, "camera_y": 90.0, "score": 60.0},
    ]
    local = propose_local_temporal_v2212(
        current,
        direct.maps,
        direct.fused,
        config=LocalTemporalConfigV2212(search_radius_px=48),
    )
    nearest = min(np.hypot(row["camera_x"] - gt[0], row["camera_y"] - gt[1]) for row in local)
    _assert(nearest <= 8.0, "local temporal evidence refines an anchored <=42px candidate toward the new mark")
    _assert(all("gt" not in row for row in local), "local proposal rows contain no GT leakage")
    _assert(len(local) > 0, "local temporal proposal set is non-empty")
    _assert(max(float(row.get("local_shift_px", 0.0)) for row in local) <= 48.01, "local proposals obey configured search radius")

    # Deterministic for identical input.
    local2 = propose_local_temporal_v2212(current, direct.maps, direct.fused, config=LocalTemporalConfigV2212(search_radius_px=48))
    sig1 = [(round(r["camera_x"], 3), round(r["camera_y"], 3), r["evidence_source"]) for r in local]
    sig2 = [(round(r["camera_x"], 3), round(r["camera_y"], 3), r["evidence_source"]) for r in local2]
    _assert(sig1 == sig2, "local temporal proposal generation is deterministic")

    print("\nAll V2.21.2 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
