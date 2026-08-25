from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.archive import classify_filename, discover_shot_cases
from src.engine.offline.evidence import EvidenceConfig, build_evidence, extract_overlay_candidates, merge_candidate_sources
from src.engine.offline.metrics import ReplayMetrics, nearest_distance
from src.engine.offline.replay import ReplaySettings, replay_case
from src.engine.offline.shot_case import GroundTruth, ShotCase, read_manifest, write_manifest


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_base(seed: int, width: int = 360, height: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    base = 205.0 + 22.0 * x + 9.0 * y
    base += rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    # Old holes: present in both before and after, therefore not novel.
    image = np.clip(base, 0, 255).astype(np.uint8)
    for px, py in ((80, 80), (165, 145), (280, 65), (265, 185)):
        cv2.circle(image, (px, py), 3, 28, -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), 5, 232, 1, cv2.LINE_AA)
    return image


def _frames_for_shot(seed: int, gt: tuple[int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = _make_base(seed)
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    for _ in range(3):
        noise = rng.normal(0.0, 0.7, base.shape).astype(np.float32)
        pre.append(np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    for index in range(4):
        frame = base.copy()
        # Persistent new hole.
        cv2.circle(frame, gt, 3, 22, -1, cv2.LINE_AA)
        cv2.circle(frame, gt, 5, 238, 1, cv2.LINE_AA)
        # Strong but moving distractor: should lose to temporal consensus.
        x = 35 + index * 37
        cv2.rectangle(frame, (x, 202), (x + 8, 210), 25, -1)
        noise = rng.normal(0.0, 0.8, base.shape).astype(np.float32)
        post.append(np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return pre, post


def _test_filename_classifier() -> None:
    cases = {
        "shot_0042_pre.png": ("pre", "shot_0042"),
        "shot_0042_post_03.png": ("post", "shot_0042"),
        "0042_before.jpg": ("before", "0042"),
        "0042_after.jpg": ("after", "0042"),
        "shot_2_diff.png": ("diff", "shot_2"),
        "round-9-without-hole.png": ("without_hole", "round-9"),
        "round-9-with-hole.png": ("with_hole", "round-9"),
    }
    for name, expected in cases.items():
        actual = classify_filename(Path(name))
        _assert(actual == expected, f"classifier {name}: expected {expected}, got {actual}")


def _test_archive_and_manifest(temp: Path) -> ShotCase:
    session = temp / "session_A"
    session.mkdir(parents=True)
    gt = (223, 117)
    pre, post = _frames_for_shot(11, gt)
    for index, frame in enumerate(pre):
        cv2.imwrite(str(session / f"shot_0001_pre_{index:02d}.png"), frame)
    for index, frame in enumerate(post):
        cv2.imwrite(str(session / f"shot_0001_post_{index:02d}.png"), frame)
    (session / "shot_0001.json").write_text(
        json.dumps({"gt_camera_x": gt[0], "gt_camera_y": gt[1], "note": "selftest"}),
        encoding="utf-8",
    )
    # A diff image must never become a false pre/post shot.
    cv2.imwrite(str(session / "shot_0001_diff.png"), cv2.absdiff(pre[-1], post[-1]))

    discovered, summary = discover_shot_cases(temp, inspect_images=True)
    _assert(len(discovered) == 1, f"expected one paired shot, got {len(discovered)}")
    _assert(summary.labelled_shots == 1, "ground truth was not discovered")
    case = discovered[0]
    _assert(case.ground_truth is not None, "missing ground truth")
    _assert(math.hypot(case.ground_truth.x - gt[0], case.ground_truth.y - gt[1]) < 0.01, "wrong GT")

    manifest = temp / "manifest.jsonl"
    _assert(write_manifest(manifest, discovered) == 1, "manifest write count")
    loaded = read_manifest(manifest)
    _assert(len(loaded) == 1 and loaded[0].shot_id == case.shot_id, "manifest roundtrip")
    return loaded[0]


def _test_overlay(temp: Path, case: ShotCase) -> None:
    pre = [cv2.imread(str(temp / value), cv2.IMREAD_GRAYSCALE) for value in case.pre_images]
    post = [cv2.imread(str(temp / value), cv2.IMREAD_GRAYSCALE) for value in case.post_images]
    cfg = EvidenceConfig(candidate_min_score=0.20, candidate_limit=200)
    bundle = build_evidence(pre, post, config=cfg)
    _assert(set(bundle.overlays) == {"temporal_consensus", "persistent_zscore", "local_contrast", "darkening", "absdiff"}, "overlay set")
    candidates = extract_overlay_candidates(bundle.fused, config=cfg)
    _assert(candidates, "overlay produced zero candidates")
    gt = case.ground_truth.as_xy() if case.ground_truth else (0.0, 0.0)
    distance = nearest_distance(candidates, gt)
    _assert(distance is not None and distance <= 5.0, f"overlay missed synthetic GT: {distance}")

    fake_v2 = [{"camera_x": 20.0, "camera_y": 20.0, "score": 30.0, "detector_v2": 1.0}]
    union = merge_candidate_sources((("current_detector", fake_v2), ("overlay", candidates)), limit=300)
    union_distance = nearest_distance(union, gt)
    _assert(union_distance is not None and union_distance <= 5.0, "union did not preserve overlay rescue")

    metrics = ReplayMetrics()
    metrics.add(
        {
            "ground_truth": case.ground_truth.to_dict() if case.ground_truth else None,
            "sources": {
                "current_detector": {"candidate_count": 1, "nearest_gt_distance": nearest_distance(fake_v2, gt)},
                "overlay": {"candidate_count": len(candidates), "nearest_gt_distance": distance},
                "union": {"candidate_count": len(union), "nearest_gt_distance": union_distance},
            },
        }
    )
    summary = metrics.to_dict()
    _assert(summary["complementarity"]["overlay_rescues_current_detector"]["20.0"] == 1, "rescue metric")




def _test_single_pair_mode() -> None:
    """Historical archives may contain exactly one before and one after image."""
    gt = (144, 96)
    pre, post = _frames_for_shot(27, gt)
    cfg = EvidenceConfig(candidate_min_score=0.18, candidate_limit=220)
    bundle = build_evidence([pre[-1]], [post[-1]], config=cfg)
    candidates = extract_overlay_candidates(bundle.fused, config=cfg)
    distance = nearest_distance(candidates, gt)
    _assert(distance is not None and distance <= 6.0, f"single-pair replay missed GT: {distance}")

def _test_live_v2_adapter_if_available(temp: Path, case: ShotCase, require_v2: bool) -> str:
    try:
        from src.engine.camera.hit_scanner import HitScanner  # noqa: F401
        from src.engine.camera.candidate_generator_v2 import install_candidate_generator_v2  # noqa: F401
    except Exception as exc:
        if require_v2:
            raise AssertionError(f"Current live detector import failed: {exc}") from exc
        return f"SKIPPED ({type(exc).__name__}: {exc})"

    result = replay_case(
        case,
        root=temp,
        settings=ReplaySettings(use_current_detector=True, use_overlay=True, candidate_limit=500),
    )
    _assert("current_detector" in result["sources"] and "union" in result["sources"], "live detector adapter payload")
    # The selftest's direct-image overlay is guaranteed to recover GT; current live detector
    # may or may not on this deliberately artificial camera frame.  This test is
    # about API compatibility, not forcing detector behaviour on synthetic pixels.
    union_distance = result["sources"]["union"]["nearest_gt_distance"]
    _assert(union_distance is not None and union_distance <= 5.0, f"live replay union missed GT: {union_distance}")
    return "PASSED"


def main() -> int:
    parser = argparse.ArgumentParser(description="Selftest Detector V2.12 offline replay foundation")
    parser.add_argument(
        "--require-v2",
        action="store_true",
        help="Fail unless the current repo's live HitScanner V1->V2 hybrid can be imported and replayed",
    )
    args = parser.parse_args()

    print("V2.12 OFFLINE SELFTEST")
    print("======================")
    _test_filename_classifier()
    print("[PASS] filename pairing classifier")
    with tempfile.TemporaryDirectory(prefix="skjutbana_v212_") as raw:
        temp = Path(raw)
        case = _test_archive_and_manifest(temp)
        print("[PASS] archive discovery + GT metadata + JSONL manifest")
        _test_overlay(temp, case)
        print("[PASS] temporal-consensus physical overlay + candidate union + rescue metric")
        _test_single_pair_mode()
        print("[PASS] single before/after compatibility for older archives")
        status = _test_live_v2_adapter_if_available(temp, case, args.require_v2)
        print(f"[{status if status == 'PASSED' else 'INFO'}] current live V1->V2 replay adapter: {status}")
    print("\nAll mandatory V2.12 selftests passed.")
    if not args.require_v2:
        print("Run inside the full skjutbana repo with --require-v2 for live V1->V2 hybrid verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
