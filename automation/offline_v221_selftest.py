from __future__ import annotations

import inspect
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.candidate_pack_v216 import CandidateCaptureConfigV216, CandidatePackV216, CandidateShadowRecorderV216
from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221, propose_direct_v221
from src.engine.offline.domain_gap_v221 import DomainGapConfigV221, _domain_classifier_cv
from src.engine.offline.candidate_ranking_training_v218 import CandidateGroupV218


def _assert(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def _distance(candidate: dict, gt: tuple[float, float]) -> float:
    return math.hypot(float(candidate["camera_x"]) - gt[0], float(candidate["camera_y"]) - gt[1])


def _make_scene(seed: int = 221) -> tuple[np.ndarray, list[np.ndarray], tuple[float, float]]:
    rng = np.random.default_rng(seed)
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pre = np.zeros((h, w), dtype=np.float32)
    pre[:] = 132 + 25 * (xx / w) + 18 * (yy / h)
    texture = cv2.resize(rng.normal(0, 1, size=(30, 48)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    pre += texture * 6
    # Old holes exist before and after and must not be mistaken for temporal novelty.
    pre_u8 = np.clip(pre, 0, 255).astype(np.uint8)
    for pt in ((105, 80), (285, 230), (470, 76), (525, 275)):
        cv2.circle(pre_u8, pt, 4, 40, -1, cv2.LINE_AA)
        cv2.circle(pre_u8, pt, 7, 210, 1, cv2.LINE_AA)
    gt = (382.0, 176.0)
    posts = []
    for index in range(3):
        post = pre_u8.astype(np.float32) * (1.0 + (index - 1) * 0.002) + (index - 1) * 0.7
        noise = rng.normal(0, 0.75, size=(h, w)).astype(np.float32)
        post += noise
        post = np.clip(post, 0, 255).astype(np.uint8)
        cv2.circle(post, (int(gt[0]), int(gt[1])), 4, 22, -1, cv2.LINE_AA)
        cv2.circle(post, (int(gt[0]), int(gt[1])), 7, 228, 1, cv2.LINE_AA)
        cv2.circle(post, (int(gt[0] + 2), int(gt[1] - 1)), 1, 245, -1, cv2.LINE_AA)
        posts.append(post)
    return pre_u8, posts, gt


def _test_direct_proposals() -> None:
    pre, posts, gt = _make_scene()
    cfg = DirectProposalConfigV221(proposal_limit=120, proposals_per_source=60)
    a = propose_direct_v221([pre], posts, config=cfg)
    b = propose_direct_v221([pre], posts, config=cfg)
    _assert(bool(a.candidates), "direct proposal engine produced no candidates")
    best = min(_distance(row, gt) for row in a.candidates)
    _assert(best <= 8.0, f"direct proposal engine did not cover the synthetic new hole; nearest={best:.2f}px")
    sig_a = [(round(float(r['camera_x']), 3), round(float(r['camera_y']), 3), round(float(r['score']), 6)) for r in a.candidates[:40]]
    sig_b = [(round(float(r['camera_x']), 3), round(float(r['camera_y']), 3), round(float(r['score']), 6)) for r in b.candidates[:40]]
    _assert(sig_a == sig_b, "direct proposals are not deterministic for frozen inputs")
    _assert("gt" not in inspect.signature(propose_direct_v221).parameters, "GT leaked into direct proposal API")


def _test_storage_aware_capture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = CandidateCaptureConfigV216(
            enabled=True, data_root=tmp, patch_size=32, max_post_frames=3, max_candidates=8,
            include_raw_extras=True, save_gt_patches=True, save_full_frames=True,
            full_frame_post_count=2, save_full_reference_pre=False, save_full_recent_pre=True, compress=True,
        )
        rec = CandidateShadowRecorderV216(cfg, session_id="v221_selftest")
        pre, posts, gt = _make_scene(222)
        candidates = [{"camera_x": gt[0], "camera_y": gt[1], "score": 1.0, "rank": 1}]
        result = rec.capture_shot(
            round_id=1, raw_candidates=candidates, ranked_candidates=candidates,
            pre_gray=pre, recent_pre_gray=pre, recent_pre_timestamp=1.0,
            post_gray=posts[-1], post_frames=[(x, 1.1 + i * 0.03) for i, x in enumerate(posts)],
            gt_camera_xy=gt,
        )
        _assert(bool(result.get("saved")), f"candidate recorder failed: {result}")
        pack = CandidatePackV216.load(Path(result["json_path"]))
        _assert(pack.full_pre_frame is None, "storage-aware V2.21 capture unexpectedly saved reference PRE")
        _assert(isinstance(pack.full_recent_pre_frame, np.ndarray), "V2.21 capture did not save recent PRE")
        _assert(isinstance(pack.full_post_frames, np.ndarray) and len(pack.full_post_frames) == 2, "V2.21 capture did not save requested full POST count")
        _assert("v221_full_frame_direct" in list(pack.metadata.get("capture_extensions") or []), "V2.21 capture provenance missing")


def _mock_group(domain: int, index: int) -> CandidateGroupV218:
    rng = np.random.default_rng(1000 + domain * 100 + index)
    n = 40
    shift = 0.0 if domain == 0 else 2.0
    embedding = rng.normal(shift, 0.7, size=(n, 12)).astype(np.float32)
    scalars = rng.normal(0.4 + domain * 0.65, 0.12, size=(n, 8)).astype(np.float32)
    return CandidateGroupV218(
        session_id=f"d{domain}_{index}", round_id=index, json_path="",
        embedding=embedding,
        base_probability=np.clip(rng.normal(0.25 + domain * 0.50, 0.10, size=n), 0, 1).astype(np.float32),
        base_offsets=rng.normal(0, 2, size=(n, 2)).astype(np.float32),
        temporal_scalars=scalars,
        distances=np.linspace(0, 100, n).astype(np.float32),
        candidate_xy=np.zeros((n, 2), dtype=np.float32), target_offsets=np.zeros((n, 2), dtype=np.float32),
        current_rank=np.arange(1, n + 1, dtype=np.int32), in_ranked_pool=np.ones(n, dtype=bool),
        known_hole_distance=np.full(n, np.nan, dtype=np.float32),
    )


def _test_domain_classifier_diagnostic() -> None:
    synth = [_mock_group(0, i) for i in range(15)]
    physical = [_mock_group(1, i) for i in range(15)]
    report = _domain_classifier_cv(synth, physical, DomainGapConfigV221(folds=5, classifier_steps=350, classifier_lr=0.08))
    _assert(float(report["auc"]) >= 0.95, f"domain-gap diagnostic failed to detect obvious shift: {report}")


def main() -> int:
    print("V2.21 SELFTEST")
    print("==============")
    _test_direct_proposals(); print("[PASS] direct proposal engine covers new hole without GT input and is deterministic")
    _test_storage_aware_capture(); print("[PASS] storage-aware full-frame shadow capture saves recent PRE + bounded POST stack")
    _test_domain_classifier_diagnostic(); print("[PASS] group-level domain-gap diagnostic detects a deliberately shifted domain")
    print("\nAll V2.21 selftests passed. Live hit authority/order remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
