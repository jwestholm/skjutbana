"""Board snapshot + wear map (white projection, camera) — measurement only.

  python3 -m automation.bench board-snapshot [--output DIR] [--compare EARLIER_DIR]

Run while the game is NOT running (one process owns the camera). Projects plain white
and grey at the calibrated window position, stores median camera frames, detects dark
hole blobs, maps them to screen coordinates (saved calibration) and writes:
  board_white.png / board_gray.png / board_white_color.jpg   camera evidence
  wear.json      holes in screen coords, 8x6 viewport grid counts, centre shares
  board_wear_heatmap.png   screen-space board image + density heatmap + grid counts
With --compare, also lists blobs new since an earlier snapshot (review evidence for
labelling a physical session — NOT automatic ground truth: lighting, projection and
merged damage confound diffs). Torn/merged areas are undercounted by blob counting.
Wear is context for spawn placement and maintenance only, never a hit veto.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

GRID = (8, 6)


def capture(out: Path, window_pos=None, settle_s: float = 2.0, frames: int = 12) -> None:
    import cv2
    import pygame
    from config import SCREEN_HEIGHT, SCREEN_WIDTH
    from automation.bench.calibration_check import _window_position
    from src.engine.camera.camera_manager import camera_manager
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    from pygame._sdl2.video import Window
    Window.from_display_module().position = window_pos or _window_position()
    pygame.display.set_caption("Skjutbana board snapshot")
    if not camera_manager.start():
        raise SystemExit(f"Camera unavailable: {camera_manager.last_error}")
    try:
        for name, colour in (("white", (255, 255, 255)), ("gray", (160, 160, 160))):
            deadline = time.monotonic() + settle_s
            while time.monotonic() < deadline:
                pygame.event.pump(); screen.fill(colour); pygame.display.flip(); time.sleep(0.03)
            stack = []
            for _ in range(frames):
                pygame.event.pump(); screen.fill(colour); pygame.display.flip(); time.sleep(0.08)
                camera_manager.update()
                frame = camera_manager.get_latest_frame()
                if frame is not None:
                    stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
                    last = frame
            if not stack:
                raise SystemExit("no camera frames")
            cv2.imwrite(str(out / f"board_{name}.png"), np.median(np.stack(stack), axis=0).astype(np.uint8))
            cv2.imwrite(str(out / f"board_{name}_color.jpg"), last)
    finally:
        camera_manager.stop()
        pygame.quit()


def detect_holes(gray: np.ndarray, homography: np.ndarray, screen_wh) -> np.ndarray:
    """Dark compact blobs inside the projected screen -> camera xy (N,2)."""
    import cv2
    w, h = screen_wh
    corners = np.float64([[[0, 0], [w, 0], [w, h], [0, h]]])
    cam = cv2.perspectiveTransform(corners, np.linalg.inv(homography))[0]
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, cam.astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((25, 25), np.uint8))
    g = gray.astype(np.float32)
    bg = cv2.medianBlur(gray.astype(np.uint8), 31).astype(np.float32)
    holes = (((bg - g) > 35) & (mask > 0)).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, centres = cv2.connectedComponentsWithStats(holes)
    keep = [centres[i] for i in range(1, n) if 4 <= stats[i, cv2.CC_STAT_AREA] <= 400]
    return np.array(keep, dtype=np.float64).reshape(-1, 2)


def wear_stats(screen_xy: np.ndarray, viewport) -> dict:
    x0, y0, vw, vh = viewport
    gx, gy = GRID
    counts = np.zeros((gy, gx), int)
    inside = []
    for x, y in screen_xy:
        if x0 <= x < x0 + vw and y0 <= y < y0 + vh:
            counts[int((y - y0) / vh * gy), int((x - x0) / vw * gx)] += 1
            inside.append((x, y))
    inside = np.array(inside).reshape(-1, 2)
    out = dict(viewport=[x0, y0, vw, vh], holes_in_viewport=int(len(inside)), grid_counts=counts.tolist(),
               holes_screen=screen_xy.tolist())
    if len(inside):
        rx = np.abs(inside[:, 0] - (x0 + vw / 2)) / (vw / 2)
        ry = np.abs(inside[:, 1] - (y0 + vh / 2)) / (vh / 2)
        out["centre_share"] = {"box_25pct_side_6pct_area": float(np.mean((rx < 0.25) & (ry < 0.25))),
                               "box_50pct_side_25pct_area": float(np.mean((rx < 0.5) & (ry < 0.5)))}
    return out


def heatmap(out: Path, homography: np.ndarray, stats: dict, screen_wh, scale: int = 2) -> None:
    import cv2
    w, h = screen_wh
    T = np.diag([scale, scale, 1.0]) @ homography
    warp = cv2.warpPerspective(cv2.imread(str(out / "board_white_color.jpg")), T, (w * scale, h * scale))
    dens = np.zeros((h * scale, w * scale), np.float32)
    pts = np.array(stats["holes_screen"]).reshape(-1, 2)
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            dens[int(y * scale), int(x * scale)] += 1
    dens = cv2.GaussianBlur(dens, (0, 0), 15 * scale)
    dens /= max(float(dens.max()), 1e-9)
    over = cv2.addWeighted(warp, 0.55, cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.45, 0)
    for x, y in pts:
        cv2.circle(over, (int(x * scale), int(y * scale)), 3, (0, 0, 0), -1)
    x0, y0, vw, vh = stats["viewport"]
    cv2.rectangle(over, (x0 * scale, y0 * scale), ((x0 + vw) * scale, (y0 + vh) * scale), (255, 255, 255), 2)
    grid = np.array(stats["grid_counts"])
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            p = (int((x0 + (c + 0.5) * vw / grid.shape[1]) * scale) - 15, int((y0 + (r + 0.5) * vh / grid.shape[0]) * scale) + 10)
            cv2.putText(over, str(grid[r, c]), p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(over, str(grid[r, c]), p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    cv2.imwrite(str(out / "board_wear_heatmap.png"), over)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m automation.bench board-snapshot")
    parser.add_argument("--output", default=None, help="default /data/skjutbana/board_snapshots/<UTC>")
    parser.add_argument("--from-dir", default=None, help="analyse an existing snapshot dir (no camera)")
    parser.add_argument("--compare", default=None, help="earlier snapshot dir: list new blobs (review only)")
    args = parser.parse_args(argv)
    import cv2
    from config import SCREEN_HEIGHT, SCREEN_WIDTH
    from src.engine.input.hit_input import hit_input
    from src.engine.settings import load_viewport_rect
    if args.from_dir:
        out = Path(args.from_dir)
    else:
        out = Path(args.output or f"/data/skjutbana/board_snapshots/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        out.mkdir(parents=True, exist_ok=False)
        capture(out)
    H = np.array(hit_input.calibration_data["homography"], dtype=np.float64)
    gray = cv2.imread(str(out / "board_white.png"), 0)
    cam_xy = detect_holes(gray, H, (SCREEN_WIDTH, SCREEN_HEIGHT))
    screen_xy = cv2.perspectiveTransform(cam_xy[None], H)[0] if len(cam_xy) else np.zeros((0, 2))
    vp = load_viewport_rect()
    stats = wear_stats(screen_xy, (vp.x, vp.y, vp.w, vp.h))
    stats["holes_camera"] = cam_xy.tolist()
    stats["evidence"] = "WHITE_PROJECTION_CAMERA_SNAPSHOT (context only; torn areas undercounted)"
    if args.compare:
        earlier = json.loads((Path(args.compare) / "wear.json").read_text())
        old = np.array(earlier.get("holes_camera") or []).reshape(-1, 2)
        new = [p for p in cam_xy if not len(old) or np.min(np.hypot(*(old - p).T)) > 6.0]
        stats["new_since"] = dict(compare=str(args.compare), new_holes_camera=[list(map(float, p)) for p in new])
    (out / "wear.json").write_text(json.dumps(stats, indent=1))
    heatmap(out, H, stats, (SCREEN_WIDTH, SCREEN_HEIGHT))
    print(json.dumps({k: stats[k] for k in ("holes_in_viewport", "grid_counts", "centre_share") if k in stats}))
    if "new_since" in stats:
        print(f"new blobs since {args.compare}: {len(stats['new_since']['new_holes_camera'])}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
