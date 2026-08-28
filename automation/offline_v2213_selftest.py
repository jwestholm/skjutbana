from __future__ import annotations

import numpy as np

from src.engine.offline.temporal_consensus_v2213 import (
    MaskedDirectConfigV2213,
    TemporalConsensusConfigV2213,
    candidate_target_mask_v2213,
    propose_masked_direct_v2213,
    propose_temporal_consensus_v2213,
)


def _nearest(rows, xy):
    if not rows:
        return 9999.0
    x, y = xy
    return min(((float(r["camera_x"]) - x) ** 2 + (float(r["camera_y"]) - y) ** 2) ** 0.5 for r in rows)


def main() -> int:
    h, w = 180, 260
    current = [{"camera_x": 128.0, "camera_y": 92.0, "score": 1.0}]

    # Build a deliberately hostile plateau: a long saturated line plus one
    # compact multi-source blob.  V2.21.2 pixel-level top-K could select
    # arbitrary positions on the line; V2.21.3 should collapse/filter it.
    maps = {}
    for name in ("blackhat_gain", "tophat_gain", "persistent_abs", "gradient_gain", "persistent_dark"):
        arr = np.zeros((h, w), dtype=np.float32)
        arr[72:75, 82:178] = 1.0  # elongated nuisance plateau
        yy, xx = np.ogrid[:h, :w]
        blob = (xx - 146) ** 2 + (yy - 105) ** 2 <= 4 ** 2
        arr[blob] = 0.96
        maps[name] = arr

    cfg = TemporalConsensusConfigV2213(
        search_radius_px=50,
        threshold_percentile=88.0,
        components_per_source=8,
        top_per_anchor=2,
        component_max_area=220,
        component_min_compactness=0.12,
    )
    rows = propose_temporal_consensus_v2213(current, maps, config=cfg)
    assert rows, "consensus emitted no proposals"
    assert _nearest(rows, (146.0, 105.0)) <= 3.0, "compact multi-source blob was not recovered"
    best = min(rows, key=lambda r: (float(r["camera_x"]) - 146.0) ** 2 + (float(r["camera_y"]) - 105.0) ** 2)
    assert int(best.get("temporal_consensus_support", 0)) >= 3, "multi-source support was not preserved"
    print("[PASS] plateau-aware anchored consensus recovers compact multi-source evidence")

    cloud = []
    for y in (55, 80, 105, 130):
        for x in (70, 105, 140, 175, 210):
            cloud.append({"camera_x": float(x), "camera_y": float(y)})
    mask = candidate_target_mask_v2213(cloud, (h, w), margin_px=10)
    assert mask[95, 145]
    assert not mask[10, 10]
    print("[PASS] candidate-derived target mask includes target cloud and excludes room corner")

    # Stronger nuisance outside target mask must not beat an inside compact blob.
    direct_maps = {}
    for name in ("blackhat_gain", "tophat_gain", "persistent_abs", "gradient_gain"):
        arr = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.ogrid[:h, :w]
        arr[(xx - 146) ** 2 + (yy - 105) ** 2 <= 4 ** 2] = 0.90
        arr[(xx - 20) ** 2 + (yy - 20) ** 2 <= 4 ** 2] = 1.0
        direct_maps[name] = arr
    direct, direct_mask = propose_masked_direct_v2213(
        cloud,
        direct_maps,
        config=MaskedDirectConfigV2213(margin_px=10, threshold_percentile=90.0, proposal_limit=40),
    )
    assert direct, "masked direct emitted no proposals"
    assert _nearest(direct, (146.0, 105.0)) <= 3.0
    assert _nearest(direct, (20.0, 20.0)) > 10.0
    assert direct_mask[105, 146] and not direct_mask[20, 20]
    print("[PASS] masked direct keeps in-target evidence and rejects stronger room nuisance")

    # Determinism / no GT parameter contract.
    rows2 = propose_temporal_consensus_v2213(current, maps, config=cfg)
    a = [(round(float(r["camera_x"]), 4), round(float(r["camera_y"]), 4), round(float(r["score"]), 4)) for r in rows]
    b = [(round(float(r["camera_x"]), 4), round(float(r["camera_y"]), 4), round(float(r["score"]), 4)) for r in rows2]
    assert a == b
    print("[PASS] consensus proposal generation is deterministic and has no GT input")
    print("\nAll V2.21.3 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
