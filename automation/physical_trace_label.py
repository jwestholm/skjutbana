"""Interactive human ground-truth labeling for physical trace sessions.

The UI stores the exact camera-space click separately from detector output.  It
uses only captured ``.npy`` frames and imports pygame lazily, so the pure data
helpers remain usable in headless tests.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.engine.physical_trace import TRACE_SCHEMA, _json
from automation.physical_trace_target import (
    TargetEvidenceError,
    camera_to_target,
    load_calibration,
    target_evidence_view,
    target_to_camera,
)


class AnnotationDataError(ValueError):
    """A trace cannot be displayed or safely labeled."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationDataError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnnotationDataError(f"Trace is not an object: {path}")
    return value


def annotation_path(shot_dir: Path) -> Path:
    return shot_dir / "ground_truth.json"


def status_path(shot_dir: Path) -> Path:
    return shot_dir / "ground_truth_status.json"


def load_existing_annotation(shot_dir: Path) -> dict[str, Any] | None:
    path = annotation_path(shot_dir)
    if not path.exists():
        return None
    value = _read_json(path)
    if value.get("schema_version") != TRACE_SCHEMA or value.get("space") != "camera":
        raise AnnotationDataError(f"Invalid ground truth schema: {path}")
    try:
        x, y = float(value["camera_x"]), float(value["camera_y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationDataError(f"Invalid ground truth coordinates: {path}") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise AnnotationDataError(f"Non-finite ground truth coordinates: {path}")
    quality = str(value.get("quality", "precise"))
    if quality not in {"precise", "approximate"}:
        raise AnnotationDataError(f"Invalid ground truth quality: {path}")
    radius = value.get("uncertainty_radius_px")
    if quality == "approximate":
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(float(radius)) or float(radius) < 0:
            raise AnnotationDataError(f"Approximate labels require uncertainty_radius_px: {path}")
        value["uncertainty_radius_px"] = float(radius)
    value["quality"] = quality
    return {**value, "camera_x": x, "camera_y": y}


def discover_shots(root: Path, *, include_labeled: bool = False, include_skipped: bool = False) -> list[dict[str, Any]]:
    shots: list[dict[str, Any]] = []
    shot_dirs = sorted(p for p in (root / "shots").glob("shot_*") if p.is_dir())
    for shot_dir in shot_dirs:
        trace_path = shot_dir / "trace.json"
        if trace_path.exists():
            trace = _read_json(trace_path)
        else:
            # A crash can leave immutable frame files without the final JSON.
            # Reconstruct only display metadata; never infer detector output.
            try:
                shot_id = int(shot_dir.name.rsplit("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise AnnotationDataError(f"Unrecognized shot directory: {shot_dir}") from exc
            frames = []
            for path in sorted((shot_dir / "frames").glob("*.npy")):
                kind = "post" if path.name.startswith("post_") else "pre_snapshot" if path.name.startswith("pre_snapshot_") else "pre_history"
                try:
                    seq = int(path.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    seq = len(frames)
                frames.append({"kind": kind, "timestamp": float(seq), "path": str(path.relative_to(shot_dir))})
            trace = {"schema_version": TRACE_SCHEMA, "shot_id": shot_id, "session_id": root.name,
                     "coordinate_space": "camera", "frames": frames, "stages": [],
                     "context": {"calibration": "UNAVAILABLE"},
                     "provenance": {"status": "trace_json_unavailable"}}
        if trace.get("schema_version") != TRACE_SCHEMA:
            raise AnnotationDataError(f"Unsupported trace schema: {trace_path}")
        annotation = load_existing_annotation(shot_dir)
        skipped = status_path(shot_dir).exists()
        if (annotation is not None and not include_labeled) or (skipped and not include_skipped):
            continue
        shots.append({"trace_path": trace_path, "shot_dir": shot_dir, "trace": trace,
                      "annotation": annotation, "skipped": skipped})
    return shots


def frame_entries(trace: dict[str, Any], kind: str | None = None) -> list[dict[str, Any]]:
    frames = [f for f in trace.get("frames", []) if isinstance(f, dict)]
    if kind is not None:
        return sorted([f for f in frames if f.get("kind") == kind], key=lambda f: (float(f.get("timestamp", 0.0)), str(f.get("path", ""))))
    posts = [f for f in frames if f.get("kind") == "post"]
    selected = posts or [f for f in frames if f.get("kind") in {"pre_snapshot", "pre_history"}]
    return sorted(selected, key=lambda f: (float(f.get("timestamp", 0.0)), str(f.get("path", ""))))


def load_frame(shot: dict[str, Any], index: int = -1, kind: str | None = None) -> tuple[np.ndarray, dict[str, Any], int]:
    entries = frame_entries(shot["trace"], kind)
    if not entries:
        raise AnnotationDataError(f"No annotation frames in {shot['trace_path']}")
    actual = len(entries) + index if index < 0 else index
    if actual < 0 or actual >= len(entries):
        raise AnnotationDataError(f"Frame index out of range for {shot['trace_path']}")
    candidates = range(actual, -1, -1) if index < 0 else (actual,)
    last_error = None
    for candidate_index in candidates:
        entry = entries[candidate_index]
        path = shot["shot_dir"] / str(entry.get("path", ""))
        try:
            image = np.load(path, allow_pickle=False)
            if image.ndim not in (2, 3) or image.shape[0] < 1 or image.shape[1] < 1:
                raise ValueError(f"unsupported shape {image.shape}")
            return image, entry, candidate_index
        except (OSError, ValueError) as exc:
            last_error = f"Cannot load frame {path}: {exc}"
    raise AnnotationDataError(last_error or f"Cannot load frame for {shot['trace_path']}")


def load_best_pre_frame(shot: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], int]:
    """Use the latest stored PRE history, falling back to the snapshot."""
    if frame_entries(shot["trace"], "pre_history"):
        return load_frame(shot, -1, "pre_history")
    return load_frame(shot, -1, "pre_snapshot")


def fit_display(shape: tuple[int, ...], max_size: tuple[int, int]) -> tuple[tuple[float, float], tuple[int, int], tuple[int, int]]:
    """Return actual (x_scale,y_scale), displayed size, and centered origin."""
    height, width = int(shape[0]), int(shape[1])
    scale = min(float(max_size[0]) / width, float(max_size[1]) / height)
    if not math.isfinite(scale) or scale <= 0:
        raise AnnotationDataError(f"Invalid display size for frame {shape}")
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    origin = ((max_size[0] - size[0]) // 2, (max_size[1] - size[1]) // 2)
    return (size[0] / width, size[1] / height), size, origin


def display_to_camera(click: tuple[float, float], origin: tuple[int, int], scale: float | tuple[float, float],
                      shape: tuple[int, ...]) -> tuple[float, float] | None:
    """Map a displayed click to original full-camera pixel coordinates."""
    if isinstance(scale, tuple):
        sx, sy = map(float, scale)
    else:
        sx = sy = float(scale)
    if sx <= 0 or sy <= 0:
        raise AnnotationDataError("Display scale must be positive")
    x = (float(click[0]) - origin[0]) / sx
    y = (float(click[1]) - origin[1]) / sy
    if x < 0 or y < 0 or x > shape[1] - 1 or y > shape[0] - 1:
        return None
    return x, y


def save_annotation(shot_dir: Path, shot_id: int, camera_xy: tuple[float, float], *, label_source: str = "manual_click",
                    quality: str = "precise", uncertainty_radius_px: float | None = None) -> Path:
    x, y = map(float, camera_xy)
    if not all(math.isfinite(v) for v in (x, y)):
        raise AnnotationDataError("Ground truth click must be finite")
    if quality not in {"precise", "approximate"}:
        raise AnnotationDataError("Ground truth quality must be precise or approximate")
    if quality == "approximate":
        if uncertainty_radius_px is None or not math.isfinite(float(uncertainty_radius_px)) or float(uncertainty_radius_px) < 0:
            raise AnnotationDataError("Approximate labels require a nonnegative uncertainty radius")
    path = annotation_path(shot_dir)
    payload = {"schema_version": TRACE_SCHEMA, "shot_id": int(shot_id), "space": "camera",
               "camera_x": x, "camera_y": y, "label_source": label_source, "quality": quality,
               "attached_at": time.time()}
    if quality == "approximate":
        payload["uncertainty_radius_px"] = float(uncertainty_radius_px)
    path.write_text(_json(payload), encoding="utf-8")
    status_path(shot_dir).unlink(missing_ok=True)
    return path


def save_skip(shot_dir: Path, shot_id: int, reason: str = "unresolved") -> Path:
    path = status_path(shot_dir)
    path.write_text(_json({"schema_version": TRACE_SCHEMA, "shot_id": int(shot_id),
                           "status": "unresolved", "reason": str(reason), "updated_at": time.time()}), encoding="utf-8")
    return path


def _pygame_surface(pygame: Any, image: np.ndarray, size: tuple[int, int]) -> Any:
    arr = np.asarray(image)
    if arr.ndim == 2:
        work = arr.astype(np.float32)
        lo, hi = float(np.nanmin(work)), float(np.nanmax(work))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            work = np.zeros_like(work)
        else:
            work = (work - lo) * (255.0 / (hi - lo))
        rgb = np.repeat(np.clip(work, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    else:
        rgb = np.clip(arr[..., :3], 0, 255).astype(np.uint8)
    surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    return pygame.transform.smoothscale(surface, size)


def _run_ui(root: Path, shots: list[dict[str, Any]], max_size: tuple[int, int], show_candidates: bool) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise AnnotationDataError("pygame is required for interactive labeling") from exc
    pygame.init()
    screen = pygame.display.set_mode(max_size)
    pygame.display.set_caption("Physical trace ground truth")
    font = pygame.font.Font(None, 26)
    index, frame_index, pending, show = 0, -1, None, bool(show_candidates)
    running = True
    while running and shots:
        shot = shots[index]
        try:
            image, entry, actual = load_frame(shot, frame_index)
            scale, size, origin = fit_display(image.shape, max_size)
        except AnnotationDataError as exc:
            print(f"Skipping shot {shot['trace'].get('shot_id')}: {exc}")
            save_skip(shot["shot_dir"], int(shot["trace"]["shot_id"]), "missing_or_malformed_image")
            shots.pop(index)
            if shots:
                index %= len(shots)
            continue
        screen.fill((24, 24, 24))
        screen.blit(_pygame_surface(pygame, image, size), origin)
        if show:
            for candidate in (shot["trace"].get("stages", [])[-1].get("candidates", []) if shot["trace"].get("stages") else []):
                if "camera_x" in candidate and "camera_y" in candidate:
                    cx = int(origin[0] + float(candidate["camera_x"]) * scale[0])
                    cy = int(origin[1] + float(candidate["camera_y"]) * scale[1])
                    pygame.draw.circle(screen, (255, 220, 0), (cx, cy), 7, 1)
        if pending is not None:
            px = int(origin[0] + pending[0] * scale[0])
            py = int(origin[1] + pending[1] * scale[1])
            pygame.draw.line(screen, (0, 255, 0), (px - 12, py), (px + 12, py), 2)
            pygame.draw.line(screen, (0, 255, 0), (px, py - 12), (px, py + 12), 2)
        text = f"shot_id {shot['trace'].get('shot_id')} | {index + 1}/{len(shots)} | frame {actual + 1}/{len(frame_entries(shot['trace']))}"
        controls = "click hole  Enter accept  R retry  S skip  arrows shot/frame  C candidates  Q quit"
        screen.blit(font.render(text, True, (255, 255, 255)), (12, 10))
        screen.blit(font.render(controls, True, (190, 190, 190)), (12, max_size[1] - 32))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and pending is not None:
                    save_annotation(shot["shot_dir"], int(shot["trace"]["shot_id"]), pending)
                    shots.pop(index)
                    if shots:
                        index %= len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_r:
                    pending = None
                elif event.key == pygame.K_s:
                    save_skip(shot["shot_dir"], int(shot["trace"]["shot_id"]))
                    shots.pop(index)
                    if shots:
                        index %= len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_c:
                    show = not show
                elif event.key == pygame.K_LEFT:
                    index = (index - 1) % len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_RIGHT:
                    index = (index + 1) % len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_UP:
                    frame_index = max(-len(frame_entries(shot["trace"])), frame_index - 1)
                elif event.key == pygame.K_DOWN:
                    frame_index = min(-1, frame_index + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pending = display_to_camera(event.pos, origin, scale, image.shape)
    pygame.quit()


def _run_target_ui(shots: list[dict[str, Any]], max_size: tuple[int, int], show_candidates: bool,
                   calibration: dict[str, Any] | None, target_size: tuple[int, int],
                   uncertainty_radius_px: float = 42.0) -> None:
    """Interactive UI with target-space evidence and explicit raw-image modes."""
    try:
        import pygame
    except ImportError as exc:
        raise AnnotationDataError("pygame is required for interactive labeling") from exc
    homography = None
    if calibration is not None:
        from automation.physical_trace_target import homography_from_calibration
        homography = homography_from_calibration(calibration)
    pygame.init()
    screen = pygame.display.set_mode(max_size)
    pygame.display.set_caption("Physical trace ground truth")
    font = pygame.font.Font(None, 26)
    index, frame_index, pending, mode, show = 0, -1, None, "target" if homography is not None else "post", bool(show_candidates)
    running = True
    while running and shots:
        shot = shots[index]
        try:
            post, post_entry, post_actual = load_frame(shot, frame_index, "post")
            pre, _, _ = load_best_pre_frame(shot)
            if homography is not None:
                target_base, target_overlay = target_evidence_view(pre, post, homography, target_size)
            else:
                target_base = target_overlay = None
            if mode == "target" and target_overlay is not None:
                image, click_space, image_shape = target_overlay, "target", target_overlay.shape
            elif mode == "diff":
                from automation.physical_trace_target import enhanced_difference
                image, click_space, image_shape = enhanced_difference(pre, post), "camera", post.shape
            elif mode == "pre":
                image, click_space, image_shape = pre, "camera", pre.shape
            else:
                image, click_space, image_shape = post, "camera", post.shape
            scale, size, origin = fit_display(image.shape, max_size)
        except (AnnotationDataError, TargetEvidenceError) as exc:
            print(f"Skipping shot {shot['trace'].get('shot_id')}: {exc}")
            save_skip(shot["shot_dir"], int(shot["trace"]["shot_id"]), "missing_or_malformed_evidence")
            shots.pop(index)
            if shots:
                index %= len(shots)
            continue
        screen.fill((24, 24, 24))
        screen.blit(_pygame_surface(pygame, image, size), origin)
        if show:
            for candidate in (shot["trace"].get("stages", [])[-1].get("candidates", []) if shot["trace"].get("stages") else []):
                if "camera_x" not in candidate or "camera_y" not in candidate:
                    continue
                point = (float(candidate["camera_x"]), float(candidate["camera_y"]))
                if click_space == "target" and homography is not None:
                    point = camera_to_target(point, homography)
                cx = int(origin[0] + point[0] * scale[0])
                cy = int(origin[1] + point[1] * scale[1])
                pygame.draw.circle(screen, (255, 220, 0), (cx, cy), 7, 1)
        if pending is not None:
            point = pending
            if click_space == "target" and homography is not None:
                point = camera_to_target(pending, homography)
            px = int(origin[0] + point[0] * scale[0])
            py = int(origin[1] + point[1] * scale[1])
            pygame.draw.line(screen, (0, 255, 0), (px - 12, py), (px + 12, py), 2)
            pygame.draw.line(screen, (0, 255, 0), (px, py - 12), (px, py + 12), 2)
        text = f"shot_id {shot['trace'].get('shot_id')} | {index + 1}/{len(shots)} | mode {mode} | POST {post_actual + 1}"
        controls = "click hole  P precise  A approximate  Enter precise  R retry  S unresolved  arrows shot/frame  1 target 2 diff 3 POST 4 PRE  C candidates  Q quit"
        screen.blit(font.render(text, True, (255, 255, 255)), (12, 10))
        screen.blit(font.render(controls, True, (190, 190, 190)), (12, max_size[1] - 32))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_p) and pending is not None:
                    save_annotation(shot["shot_dir"], int(shot["trace"]["shot_id"]), pending, quality="precise")
                    shots.pop(index)
                    if shots:
                        index %= len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_a and pending is not None:
                    save_annotation(shot["shot_dir"], int(shot["trace"]["shot_id"]), pending,
                                    quality="approximate", uncertainty_radius_px=uncertainty_radius_px)
                    shots.pop(index)
                    if shots:
                        index %= len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_r:
                    pending = None
                elif event.key == pygame.K_s:
                    save_skip(shot["shot_dir"], int(shot["trace"]["shot_id"]))
                    shots.pop(index)
                    if shots:
                        index %= len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_c:
                    show = not show
                elif event.key == pygame.K_1 and homography is not None:
                    mode = "target"
                elif event.key == pygame.K_2:
                    mode = "diff"
                elif event.key == pygame.K_3:
                    mode = "post"
                elif event.key == pygame.K_4:
                    mode = "pre"
                elif event.key == pygame.K_LEFT:
                    index = (index - 1) % len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_RIGHT:
                    index = (index + 1) % len(shots)
                    pending, frame_index = None, -1
                elif event.key == pygame.K_UP:
                    frame_index = max(-len(frame_entries(shot["trace"], "post")), frame_index - 1)
                elif event.key == pygame.K_DOWN:
                    frame_index = min(-1, frame_index + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                from automation.physical_trace_label import display_to_camera
                point = display_to_camera(event.pos, origin, scale, image_shape)
                if point is not None:
                    pending = target_to_camera(point, homography) if click_space == "target" and homography is not None else point
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-labeled", action="store_true", help="include existing labels for relabeling")
    parser.add_argument("--include-skipped", action="store_true", help="include previously skipped shots")
    parser.add_argument("--show-candidates", action="store_true", help="show yellow candidate overlays (off by default)")
    parser.add_argument("--calibration", type=Path, default=Path("content/settings.json"), help="settings JSON containing camera_calibration")
    parser.add_argument("--target-width", type=int, default=760)
    parser.add_argument("--target-height", type=int, default=580)
    parser.add_argument("--uncertainty-radius-px", type=float, default=42.0,
                        help="default radius saved when accepting an approximate label")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    shots = discover_shots(args.root, include_labeled=args.include_labeled, include_skipped=args.include_skipped)
    total = len([p for p in (args.root / "shots").glob("shot_*") if p.is_dir()])
    print(f"Shots total: {total}; to label: {len(shots)}")
    if shots:
        calibration = load_calibration(args.calibration)
        _run_target_ui(shots, (args.width, args.height), args.show_candidates, calibration,
                       (args.target_width, args.target_height), args.uncertainty_radius_px)
    shot_dirs = [p for p in (args.root / "shots").glob("shot_*") if p.is_dir()]
    labeled = sum(annotation_path(p).exists() for p in shot_dirs)
    skipped = sum(status_path(p).exists() for p in shot_dirs)
    print(f"Shots total: {total}; labeled: {labeled}; skipped/unresolved: {skipped}; remaining: {total - labeled - skipped}")


if __name__ == "__main__":
    main()
