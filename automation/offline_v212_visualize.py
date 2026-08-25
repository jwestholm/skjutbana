from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.evidence import EvidenceConfig, build_evidence, extract_overlay_candidates
from src.engine.offline.replay import ReplaySettings, load_gray, replay_case
from src.engine.offline.shot_case import read_manifest


def _heatmap(values: np.ndarray) -> np.ndarray:
    unit = np.clip(values.astype(np.float32), 0.0, 1.0)
    image = np.clip(unit * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def _annotate_overlay(image: np.ndarray, candidates: list[dict], gt_xy: tuple[float, float] | None) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for rank, candidate in enumerate(candidates[:80], start=1):
        x = int(round(float(candidate.get("camera_x", 0))))
        y = int(round(float(candidate.get("camera_y", 0))))
        radius = 5 if rank <= 10 else 3
        cv2.circle(canvas, (x, y), radius, (0, 255, 255), 1, cv2.LINE_AA)
        if rank <= 10:
            cv2.putText(canvas, str(rank), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    if gt_xy is not None:
        gx, gy = int(round(gt_xy[0])), int(round(gt_xy[1]))
        cv2.drawMarker(canvas, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    return canvas


def _comparison_canvas(
    image: np.ndarray,
    detector: list[dict],
    overlay: list[dict],
    union: list[dict],
    gt_xy: tuple[float, float] | None,
) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    # current detector = cyan; direct-image overlay = yellow; union top points = magenta.
    for candidate in detector[:120]:
        x, y = int(round(float(candidate.get("camera_x", 0)))), int(round(float(candidate.get("camera_y", 0))))
        cv2.circle(canvas, (x, y), 4, (255, 255, 0), 1, cv2.LINE_AA)
    for candidate in overlay[:120]:
        x, y = int(round(float(candidate.get("camera_x", 0)))), int(round(float(candidate.get("camera_y", 0))))
        cv2.circle(canvas, (x, y), 3, (0, 255, 255), 1, cv2.LINE_AA)
    for candidate in union[:20]:
        x, y = int(round(float(candidate.get("camera_x", 0)))), int(round(float(candidate.get("camera_y", 0))))
        cv2.drawMarker(canvas, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
    if gt_xy is not None:
        gx, gy = int(round(gt_xy[0])), int(round(gt_xy[1]))
        cv2.drawMarker(canvas, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.putText(canvas, "cyan=current detector  yellow=overlay  magenta=union(top20)  red=GT", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "cyan=current detector  yellow=overlay  magenta=union(top20)  red=GT", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Render V2.12 component evidence maps for one saved shot")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0, help="0-based shot index in manifest")
    parser.add_argument("--out-dir", type=Path, default=Path("content/ai/offline/v212/visualize"))
    parser.add_argument("--no-detector", action="store_true", help="Do not attempt current live detector comparison")
    args = parser.parse_args()

    cases = read_manifest(args.manifest)
    if not cases:
        print("ERROR: empty manifest")
        return 2
    index = int(args.index)
    if index < 0 or index >= len(cases):
        print(f"ERROR: --index must be 0..{len(cases)-1}")
        return 2
    case = cases[index]
    root = args.root.expanduser().resolve() if args.root else None
    pre = [load_gray(path) for path in case.resolved_pre_paths(root)]
    post = [load_gray(path) for path in case.resolved_post_paths(root)]
    cfg = EvidenceConfig.from_file()
    bundle = build_evidence(pre, post, config=cfg)
    candidates = extract_overlay_candidates(bundle.fused, config=cfg)
    gt_xy = case.ground_truth.as_xy() if case.ground_truth else None

    out = args.out_dir / f"shot_{index:05d}"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "00_reference.png"), bundle.reference)
    cv2.imwrite(str(out / "01_post_last.png"), post[-1])
    for number, (name, overlay) in enumerate(bundle.overlays.items(), start=10):
        cv2.imwrite(str(out / f"{number:02d}_{name}.png"), _heatmap(overlay.values))
    fused_heat = _heatmap(bundle.fused.values)
    cv2.imwrite(str(out / "90_physical_fusion_v212.png"), fused_heat)
    cv2.imwrite(str(out / "91_overlay_candidates_and_gt.png"), _annotate_overlay(fused_heat, candidates, gt_xy))

    detector_status = "not requested"
    if not args.no_detector:
        try:
            replay = replay_case(
                case,
                root=root,
                settings=ReplaySettings(use_current_detector=True, use_overlay=True),
            )
            detector_candidates = replay["candidates"]["current_detector"]
            overlay_candidates = replay["candidates"]["overlay"]
            union_candidates = replay["candidates"]["union"]
            cv2.imwrite(
                str(out / "92_current_detector_vs_overlay.png"),
                _comparison_canvas(post[-1], detector_candidates, overlay_candidates, union_candidates, gt_xy),
            )
            detector_status = (
                f"ok ({len(detector_candidates)} detector / {len(overlay_candidates)} overlay / "
                f"{len(union_candidates)} union candidates)"
            )
        except Exception as exc:
            detector_status = f"skipped ({type(exc).__name__}: {exc})"

    np.savez_compressed(
        out / "evidence_maps.npz",
        **{name: overlay.values for name, overlay in bundle.overlays.items()},
        physical_fusion_v212=bundle.fused.values,
    )

    print("V2.12 EVIDENCE VISUALISATION")
    print("============================")
    print(f"Shot       : {case.shot_id}")
    print(f"Pre frames : {len(pre)}")
    print(f"Post frames: {len(post)}")
    print(f"Candidates : {len(candidates)}")
    print(f"GT         : {gt_xy if gt_xy is not None else 'unknown'}")
    print(f"Detector   : {detector_status}")
    print(f"Output     : {out}")
    print("Maps       : absdiff, darkening, persistent_zscore, temporal_consensus, local_contrast, fusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
