from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from src.engine.offline.evidence import EvidenceConfig, build_evidence, extract_overlay_candidates
from src.engine.offline.metrics import nearest_distance


def make_case(rng: np.random.Generator, width: int, height: int, old_holes: int, post_frames: int):
    # Mixed procedural background: gradient + texture + geometric projector-like content.
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    base = 120.0 + 75.0 * x + 25.0 * y + rng.normal(0.0, 6.0, (height, width)).astype(np.float32)
    base = np.clip(base, 0, 255).astype(np.uint8)
    for _ in range(8):
        x0 = int(rng.integers(0, width - 20))
        y0 = int(rng.integers(0, height - 20))
        x1 = min(width - 1, x0 + int(rng.integers(10, 90)))
        y1 = min(height - 1, y0 + int(rng.integers(8, 60)))
        value = int(rng.integers(25, 235))
        cv2.rectangle(base, (x0, y0), (x1, y1), value, -1)
    for _ in range(old_holes):
        px = int(rng.integers(8, width - 8))
        py = int(rng.integers(8, height - 8))
        cv2.circle(base, (px, py), int(rng.integers(2, 5)), int(rng.integers(5, 55)), -1, cv2.LINE_AA)

    gt = (int(rng.integers(8, width - 8)), int(rng.integers(8, height - 8)))
    pre = [np.clip(base.astype(np.float32) + rng.normal(0, 1.0, base.shape), 0, 255).astype(np.uint8) for _ in range(3)]
    post = []
    for frame_index in range(post_frames):
        frame = base.copy()
        radius = int(rng.integers(2, 5))
        cv2.circle(frame, gt, radius, int(rng.integers(5, 45)), -1, cv2.LINE_AA)
        if rng.random() < 0.65:
            # Moving distractor that should not form stable consensus.
            dx = int((frame_index * 31 + rng.integers(0, 20)) % max(1, width - 12))
            dy = int(rng.integers(0, height - 12))
            cv2.rectangle(frame, (dx, dy), (min(width - 1, dx + 7), min(height - 1, dy + 7)), int(rng.integers(10, 245)), -1)
        post.append(np.clip(frame.astype(np.float32) + rng.normal(0, 1.1, base.shape), 0, 255).astype(np.uint8))
    return pre, post, gt


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware-free V2.12 temporal overlay stress loop")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=212)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--post-frames", type=int, default=4)
    args = parser.parse_args()

    iterations = max(1, args.iterations)
    rng = np.random.default_rng(args.seed)
    cfg = EvidenceConfig(candidate_limit=220, candidate_min_score=0.22)
    hit20 = 0
    hit42 = 0
    distances: list[float] = []
    candidate_total = 0
    start = time.perf_counter()
    for index in range(iterations):
        pre, post, gt = make_case(
            rng,
            max(80, args.width),
            max(60, args.height),
            old_holes=int(rng.integers(0, 70)),
            post_frames=max(2, args.post_frames),
        )
        bundle = build_evidence(pre, post, config=cfg)
        candidates = extract_overlay_candidates(bundle.fused, config=cfg)
        candidate_total += len(candidates)
        distance = nearest_distance(candidates, gt)
        if distance is not None:
            distances.append(distance)
            hit20 += int(distance <= 20.0)
            hit42 += int(distance <= 42.0)
        if (index + 1) % 1000 == 0:
            print(f"{index + 1}/{iterations} ...")
    elapsed = time.perf_counter() - start
    rate = iterations / max(elapsed, 1e-9)
    print("V2.12 SYNTHETIC OFFLINE STRESS")
    print("==============================")
    print(f"Iterations       : {iterations}")
    print(f"Elapsed          : {elapsed:.3f} s")
    print(f"Rate             : {rate:.1f} shots/s")
    print(f"Recall <=20 px   : {100.0 * hit20 / iterations:.2f}%")
    print(f"Recall <=42 px   : {100.0 * hit42 / iterations:.2f}%")
    print(f"Avg candidates   : {candidate_total / iterations:.1f}")
    if distances:
        print(f"Median GT dist   : {float(np.median(distances)):.2f} px")
        print(f"P95 GT dist      : {float(np.percentile(distances, 95)):.2f} px")
    print("\nThis is a plumbing/performance stress test, NOT a real-world accuracy benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
