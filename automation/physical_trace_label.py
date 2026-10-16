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
from src.engine.offline.causal_candidates import overlay_candidates
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
    peaks: list[float] = []
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
        if "peak_ts" in trace:
            peaks.append(float(trace["peak_ts"]))
        annotation = load_existing_annotation(shot_dir)
        skipped = status_path(shot_dir).exists()
        if (annotation is not None and not include_labeled) or (skipped and not include_skipped):
            continue
        shots.append({"trace_path": trace_path, "shot_dir": shot_dir, "trace": trace,
                      "annotation": annotation, "skipped": skipped})
    # Find boundaries from all events, including already labelled/false events.
    peaks.sort()
    for shot in shots:
        peak = shot["trace"].get("peak_ts", float("inf"))
        shot["next_audio_peak"] = next((p for p in peaks if p > peak), None)
        shot["causal_overlay"] = list(overlay_candidates(shot["trace"], include_later=True, next_peak=shot["next_audio_peak"]))
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


def _pygame_surface(pygame: Any, image: np.ndarray, size: tuple[int, int], *, original: bool = False) -> Any:
    arr = np.asarray(image)
    if original:
        if arr.dtype != np.uint8 or not (arr.ndim==2 or (arr.ndim==3 and arr.shape[2]==3)):
            raise AnnotationDataError("Original viewer requires saved uint8 gray/BGR camera pixels")
        rgb = np.repeat(arr[..., None], 3, axis=2) if arr.ndim==2 else arr[...,::-1].copy()
    elif arr.ndim == 2:
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
    return (pygame.transform.scale if original else pygame.transform.smoothscale)(surface, size)


def _run_ui(root: Path, shots: list[dict[str, Any]], max_size: tuple[int, int], show_candidates: bool) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise AnnotationDataError("pygame is required for interactive labeling") from exc
    pygame.init()
    screen = pygame.display.set_mode(max_size)
    pygame.display.set_caption("Physical trace ground truth")
    font = pygame.font.Font(None, 26)
    include_later = False
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
            for candidate, category, color in (item for item in shot["causal_overlay"] if include_later or item[1] == "CAUSALLY_AVAILABLE"):
                if "camera_x" in candidate and "camera_y" in candidate:
                    cx = int(origin[0] + float(candidate["camera_x"]) * scale[0])
                    cy = int(origin[1] + float(candidate["camera_y"]) * scale[1])
                    pygame.draw.circle(screen, color, (cx, cy), 7, 1)
        if pending is not None:
            px = int(origin[0] + pending[0] * scale[0])
            py = int(origin[1] + pending[1] * scale[1])
            pygame.draw.line(screen, (0, 255, 0), (px - 12, py), (px + 12, py), 2)
            pygame.draw.line(screen, (0, 255, 0), (px, py - 12), (px, py + 12), 2)
        text = f"shot_id {shot['trace'].get('shot_id')} | {index + 1}/{len(shots)} | frame {actual + 1}/{len(frame_entries(shot['trace']))}"
        controls = "click hole  Enter accept  R retry  S skip  arrows shot/frame  C candidates  T later evidence  Q quit"
        screen.blit(font.render(text, True, (255, 255, 255)), (12, 10))
        if show:
            legend = "green: causal | orange: post-decision | magenta: cross-event | grey: unknown" if include_later else "Causal candidates only (green); T includes later evidence"
            screen.blit(font.render(legend, True, (230, 230, 230)), (12, 34))
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
                elif event.key == pygame.K_t:
                    include_later = not include_later
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
                   uncertainty_radius_px: float = 42.0, *, paired=None, independent_observation=None) -> None:
    """Interactive UI with target-space evidence and explicit raw-image modes."""
    try:
        import pygame
    except ImportError as exc:
        raise AnnotationDataError("pygame is required for interactive labeling") from exc
    if paired is not None:
        if show_candidates or calibration is not None:
            raise AnnotationDataError("Paired original-pixel labeling forbids overlays/calibration")
        from automation import physical_trace_label_crops as crops
        manifest, label_path = paired
        crops.assert_inputs(manifest)
        document = crops.validate_document(_read_json(label_path), manifest) if label_path.exists() else crops.new_document(manifest)
        tile_index, pre_index, zoom, pan, blink = 0, -1, 1., (0., 0.), False
        inspected_pre, image_cache = {}, {}
        message = "Inspect PRE, return to POST, click independently; UNKNOWN is valid."
        if independent_observation is not None:
            if manifest.get('exact_gt_schema') != crops.REVIEW_LABEL_SCHEMA:
                raise AnnotationDataError('independent observation is GT-review only')
            message = 'Independent physical observation mode; click original settled hole. PRE causality is NOT certified.'
    homography = None
    if calibration is not None:
        from automation.physical_trace_target import homography_from_calibration
        homography = homography_from_calibration(calibration)
    pygame.init()
    screen = pygame.display.set_mode(max_size)
    pygame.display.set_caption("Physical trace ground truth")
    font = pygame.font.Font(None, 22 if paired is not None else 26)
    if paired is not None and manifest.get('exact_gt_schema') == crops.REVIEW_LABEL_SCHEMA:
        driver = pygame.display.get_driver()
        print(f"[GT-REVIEW] UI initialized: SDL={driver}, window={max_size[0]}x{max_size[1]}", flush=True)
        if driver in ('dummy', 'offscreen'):
            print(f"[GT-REVIEW] WARNING: SDL {driver} driver has no visible window; run from a graphical desktop terminal without a headless SDL_VIDEODRIVER override.", flush=True)
    include_later = False
    index, frame_index, pending, mode, show = 0, -1, None, "target" if homography is not None else "post", bool(show_candidates)
    running = True
    clock = pygame.time.Clock()
    while running and shots:
        shot = shots[index]
        if paired is not None:
            post_entry, pre_entry, post_count, pre_count = crops.ui_frames(shot, tile_index, frame_index, pre_index)
            entry = pre_entry if mode == "pre" or (blink and int(time.monotonic()*3)%2 == 0) else post_entry
            image = None
            if entry is not None:
                if entry['path'] not in image_cache:
                    if len(image_cache)>=6:image_cache.pop(next(iter(image_cache)))
                    image_cache[entry['path']] = crops.source_image(manifest, entry)
                image = image_cache[entry['path']]
                if entry['kind'] == 'pre':
                    key = (shot['trace']['shot_id'], tuple(entry['camera_bbox']))
                    inspected_pre.setdefault(key, {})[entry['path']] = entry
                scale, size, origin = crops.view_transform(entry, (max_size[0], max_size[1]-110), zoom, pan)
                origin = (origin[0], origin[1]+70)
            screen.fill((24,24,24))
            if image is not None:
                screen.set_clip(pygame.Rect(0,70,max_size[0],max_size[1]-110))
                screen.blit(_pygame_surface(pygame,image,size,original=True),origin)
                if pending and pending['source']['path'] == entry['path']:
                    px,py = [round(origin[i]+pending['crop_local_xy'][i]*scale[i]) for i in (0,1)]
                    pygame.draw.line(screen,(0,255,0),(px-10,py),(px+10,py),1)
                    pygame.draw.line(screen,(0,255,0),(px,py-10),(px,py+10),1)
                screen.set_clip(None)
            sid=shot['trace']['shot_id']
            saved=next(r['physical_truth_status'] for r in document['shots'] if r['shot_id']==sid)
            tiles=shot['paired']['tiles']
            title=f"SHOT {sid} ({index+1}/{len(shots)}) | tile {tile_index%len(tiles)+1 if tiles else 0}/{len(tiles)} | {entry['kind'].upper() if entry else 'IMAGE DATA UNAVAILABLE'} | POST {frame_index%post_count+1 if post_count else 0}/{post_count} PRE {pre_index%pre_count+1 if pre_count else 0}/{pre_count} | {saved}"
            if entry and entry.get('role'):
                title=f"SHOT {sid} ({index+1}/{len(shots)}) | {entry['role']} {entry['projector_state']} | frame {frame_index%post_count+1 if post_count else 0}/{post_count} | {saved}"
                if shot['paired'].get('missing'):title+=' | MISSING '+','.join(shot['paired']['missing'])
                if entry['role']=='PRE':
                    checks=shot['paired'].get('pre_causality',[])
                    title+=' | PRE '+('clock-guarded; pre-damage UNVERIFIED' if checks and all(p['pre_cutoff_verified'] for p in checks) else 'UNQUALIFIED')
            controls="3 POST  4 PRE  B blink  [ ] tile  , . PRE  up/down POST  left/right shot  +/- zoom  right-drag pan  0 fit"
            footer="Enter save exact  U/S UNKNOWN  N no visible hole (UNKNOWN)  R cancel  Q quit"
            for text,y,color in [(title,8,(255,255,255)),(controls,32,(210,210,210)),(message,52,(240,220,130)),(footer,max_size[1]-28,(210,210,210))]:
                screen.blit(font.render(text,True,color),(8,y))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running=False
                elif event.type == pygame.KEYDOWN:
                    key=event.key
                    if key in (pygame.K_q,pygame.K_ESCAPE):running=False
                    elif key in (pygame.K_c,pygame.K_t,pygame.K_1,pygame.K_2):
                        message='Candidate/target/difference overlays are disabled in paired GT mode.'
                    elif key in (pygame.K_RETURN,pygame.K_u,pygame.K_s,pygame.K_n):
                        row=pending if key==pygame.K_RETURN else crops.unknown_row(sid, 'NO_VISIBLE_HOLE' if key==pygame.K_n else (shot['paired']['reason'] or 'NOT_LABELABLE'))
                        if row is not None:
                            try:
                                document=crops.save_row(label_path,manifest,row,document['revision'])
                                message=f"Saved shot {sid}: {row['physical_truth_status']} (revision {document['revision']})"
                                pending=None
                            except (ValueError,OSError) as exc:
                                message=f"NOT SAVED: {exc}"
                    elif key==pygame.K_r:pending=None
                    elif key in (pygame.K_3,pygame.K_4):mode='post' if key==pygame.K_3 else 'pre';blink=False
                    elif key==pygame.K_b:blink=not blink;mode='post'
                    elif key in (pygame.K_PLUS,pygame.K_EQUALS,pygame.K_KP_PLUS):zoom=min(8.,zoom*1.5)
                    elif key in (pygame.K_MINUS,pygame.K_KP_MINUS):zoom=max(.25,zoom/1.5)
                    elif key==pygame.K_0:zoom=1.;pan=(0.,0.)
                    elif key in (pygame.K_LEFT,pygame.K_RIGHT):
                        index=(index+(-1 if key==pygame.K_LEFT else 1))%len(shots)
                        tile_index,frame_index,pre_index,pending,zoom,pan=0,-1,-1,None,1.,(0.,0.)
                        break  # A queued click must never bind to a newly selected unseen event.
                    elif key in (pygame.K_LEFTBRACKET,pygame.K_RIGHTBRACKET):
                        tile_index+=-1 if key==pygame.K_LEFTBRACKET else 1
                        pending=None;zoom=1.;pan=(0.,0.);break
                    elif key in (pygame.K_UP,pygame.K_DOWN):
                        frame_index+=-1 if key==pygame.K_UP else 1
                        pending=None;break
                    elif key in (pygame.K_COMMA,pygame.K_PERIOD):
                        pre_index+=-1 if key==pygame.K_COMMA else 1
                        mode='pre';blink=False;break
                elif event.type == pygame.MOUSEMOTION and event.buttons[2]:
                    pan=(pan[0]+event.rel[0],pan[1]+event.rel[1])
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
                    if entry is None or mode!='post' or blink or entry['kind']!='post' or not 70<=event.pos[1]<max_size[1]-40:
                        message='Click only on original POST, with blink OFF.';continue
                    try:
                        click=crops.crop_click(event.pos,origin,scale,entry)
                        seen=list(inspected_pre.get((sid,tuple(entry['camera_bbox'])),{}).values())
                        pending=crops.make_label(manifest,shot['paired'],entry,click,seen,
                                                 independent_observation=independent_observation)
                        message=f"Pending camera XY {pending['camera_xy']} — Enter to confirm NEW physical hole, R cancels"
                    except ValueError as exc:message=str(exc)
            clock.tick(30)
            continue
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
            for candidate, category, color in (item for item in shot["causal_overlay"] if include_later or item[1] == "CAUSALLY_AVAILABLE"):
                if "camera_x" not in candidate or "camera_y" not in candidate:
                    continue
                point = (float(candidate["camera_x"]), float(candidate["camera_y"]))
                if click_space == "target" and homography is not None:
                    point = camera_to_target(point, homography)
                cx = int(origin[0] + point[0] * scale[0])
                cy = int(origin[1] + point[1] * scale[1])
                pygame.draw.circle(screen, color, (cx, cy), 7, 1)
        if pending is not None:
            point = pending
            if click_space == "target" and homography is not None:
                point = camera_to_target(pending, homography)
            px = int(origin[0] + point[0] * scale[0])
            py = int(origin[1] + point[1] * scale[1])
            pygame.draw.line(screen, (0, 255, 0), (px - 12, py), (px + 12, py), 2)
            pygame.draw.line(screen, (0, 255, 0), (px, py - 12), (px, py + 12), 2)
        text = f"shot_id {shot['trace'].get('shot_id')} | {index + 1}/{len(shots)} | mode {mode} | POST {post_actual + 1}"
        controls = "click hole  P precise  A approximate  Enter precise  R retry  S unresolved  arrows shot/frame  1 target 2 diff 3 POST 4 PRE  C candidates  T later evidence  Q quit"
        screen.blit(font.render(text, True, (255, 255, 255)), (12, 10))
        if show:
            legend = "green: causal | orange: post-decision | magenta: cross-event | grey: unknown" if include_later else "Causal candidates only (green); T includes later evidence"
            screen.blit(font.render(legend, True, (230, 230, 230)), (12, 34))
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
                elif event.key == pygame.K_t:
                    include_later = not include_later
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
    parser.add_argument("--root", type=Path)
    parser.add_argument("--capture", type=Path, help="explicit paired play-can source, read only")
    parser.add_argument("--paired-object-crops", action="store_true")
    parser.add_argument("--gt-review", action="store_true", help="Separate post-decision original full-frame WHITE/BLACK/MID_GRAY inspection")
    parser.add_argument("--independent-hole-observation", metavar="NOTE",
                        help="GT review only: operator's independent physical observation; allows settled-image labeling without PRE; never certifies PRE causality")
    parser.add_argument("--output", type=Path, help="separate exact_camera_ground_truth.json (resumable)")
    parser.add_argument("--review-package", type=Path, help="new candidate-free contact-sheet directory")
    parser.add_argument("--prepare-only", action="store_true", help="inventory/contact sheets only, no UI or labels")
    parser.add_argument("--include-labeled", action="store_true", help="include existing labels for relabeling")
    parser.add_argument("--include-skipped", action="store_true", help="include previously skipped shots")
    parser.add_argument("--show-candidates", action="store_true", help="show causal candidate overlays (off by default; T adds later evidence)")
    parser.add_argument("--calibration", type=Path, default=Path("content/settings.json"), help="settings JSON containing camera_calibration")
    parser.add_argument("--target-width", type=int, default=760)
    parser.add_argument("--target-height", type=int, default=580)
    parser.add_argument("--uncertainty-radius-px", type=float, default=42.0,
                        help="default radius saved when accepting an approximate label")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    if args.independent_hole_observation is not None and (not args.gt_review or not args.independent_hole_observation.strip()):
        parser.error('--independent-hole-observation requires --gt-review and a non-empty human observation note')
    if args.paired_object_crops and args.gt_review:
        parser.error('choose one original-image adapter')
    if args.paired_object_crops or args.gt_review:
        if not args.capture or args.root or args.show_candidates:
            parser.error("paired mode requires --capture, forbids --root/--show-candidates")
        if not args.prepare_only and not args.output:
            parser.error("paired UI requires separate --output exact_camera_ground_truth.json")
        from automation import physical_trace_label_crops as crops
        if args.gt_review:
            print(f"[GT-REVIEW] Labeler: {Path(__file__).resolve()}", flush=True)
            print(f"[GT-REVIEW] Loading: {args.capture / 'gt_review' / 'manifest.json'}", flush=True)
        manifest = crops.load_gt_review(args.capture) if args.gt_review else crops.load_capture(args.capture)
        if args.gt_review:
            if not manifest['shots']:
                parser.error("GT review manifest contains no shots; no labeling UI can be opened")
            for shot in manifest['shots']:
                counts = {role: sum(f['role'] == role for f in shot['frames'])
                          for role in ('PRE', 'LIVE_POST', 'GT_SETTLED')}
                print(f"[GT-REVIEW] Shot {shot['shot_id']}: {shot['availability']} "
                      + ' '.join(f'{role}={count}' for role, count in counts.items()), flush=True)
                if counts['PRE']:
                    guarded=all(p['pre_cutoff_verified'] for p in shot.get('pre_causality',[])) and bool(shot.get('pre_causality'))
                    print(f"[GT-REVIEW] WARNING: shot {shot['shot_id']} PRE "
                          + ('passes receipt guard; physical pre-damage causality remains unverified.' if guarded else
                             'is UNQUALIFIED; before-audio timestamp alone does not prove before-damage truth.'),flush=True)
        if args.review_package:
            print(crops.prepare_package(manifest,args.review_package))
        if args.prepare_only:
            if not args.review_package:parser.error("--prepare-only requires --review-package")
            return
        output=crops.checked(args.output)
        if (not output.is_relative_to(Path('/data/skjutbana')) or output.is_relative_to(Path(manifest['capture_path']))
                or output in [Path(p) for p in manifest['input_hashes']]):
            parser.error("GT output must be outside source capture and under /data/skjutbana")
        _run_target_ui(crops.label_shots(manifest),(args.width,args.height),False,None,
                       (args.target_width,args.target_height),paired=(manifest,output),
                       independent_observation=args.independent_hole_observation)
        print(f"Labels (only if saved): {output}")
        return
    if not args.root or args.capture or args.output or args.prepare_only or args.review_package:
        parser.error("normal full-frame mode requires --root; paired options require --paired-object-crops")
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
