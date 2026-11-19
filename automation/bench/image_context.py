"""Measure a grid of brightness, texture and temporal variation in supplied images."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np


def image_context(frames: Sequence[np.ndarray], *, rows: int = 4, columns: int = 6) -> dict:
    """Use the last image for brightness and texture, and the full series for variation."""
    start = perf_counter_ns()
    if len(frames) == 0:
        raise ValueError("Supply at least one image")
    current = frames[-1]
    if any(frame.ndim != 2 or frame.dtype != np.uint8 or frame.shape != current.shape
           for frame in frames):
        raise ValueError("Images must be equally sized 8-bit grayscale arrays")
    height, width = current.shape
    if not 1 <= rows <= height or not 1 <= columns <= width:
        raise ValueError("Grid dimensions must be positive and fit inside the image")

    # A unit ramp has derivative 1 away from the image border.
    dx = cv2.Sobel(current, cv2.CV_32F, 1, 0, scale=1 / 8, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(current, cv2.CV_32F, 0, 1, scale=1 / 8, borderType=cv2.BORDER_REPLICATE)
    edges = np.hypot(dx, dy)
    regions = []
    for row in range(rows):
        top, bottom = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            left, right = column * width // columns, (column + 1) * width // columns
            tile = np.s_[top:bottom, left:right]
            motion = None
            if len(frames) > 1:
                samples = np.stack([frame[tile] for frame in frames])
                motion = float(np.std(samples, axis=0, dtype=np.float32).mean())
            regions.append(dict(xyxy=[left, top, right, bottom], ctx_brightness=float(current[tile].mean()),
                                ctx_texture=float(edges[tile].mean()), ctx_motion=motion))
    return dict(rows=rows, columns=columns, frame_count=len(frames), regions=regions,
                motion_missing_reason="single_frame" if len(frames) == 1 else None,
                cost_ms=(perf_counter_ns() - start) / 1e6)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Image files, oldest to newest")
    parser.add_argument("--grid", nargs=2, metavar=("ROWS", "COLUMNS"), type=int, default=(4, 6))
    args = parser.parse_args(argv)
    frames = []
    for path in args.images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            parser.error(f"Could not decode {path}")
        frames.append(gray)
    try:
        result = image_context(frames, rows=args.grid[0], columns=args.grid[1])
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
