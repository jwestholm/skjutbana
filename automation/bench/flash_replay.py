"""Replay the precision-flash measurement offline from a recorded flash session.

  python3 -m automation.bench flash-replay SESSION_DIR [...] --output DIR [--set min_darkening=0.03 ...]

Uses the recorded camera bytes and the recorded flash windows/peaks
(flash_results.jsonl), and re-runs exactly the engine's measurement functions
(select_plateau / average_gray / find_new_holes) with optionally changed
FlashConfig fields.  Flash *timing* is not re-simulated (the projector showed
what it showed); only the image analysis is.  Writes flash_hits.jsonl +
score per session under --output (never into the session directory).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def replay_session(session: Path, out: Path, overrides: dict) -> dict:
    import numpy as np
    from automation.bench.flash_score import score_session
    from src.engine.precision_flash import FlashConfig, average_gray, find_new_holes, select_plateau
    from src.engine.settings import load_scanport_rect
    from src.engine.source_recording import SourceSessionReader

    cfg = FlashConfig()
    for key, value in overrides.items():
        setattr(cfg, key, type(getattr(cfg, key))(value))
    results = [json.loads(l) for l in (session / "flash_results.jsonl").read_text().splitlines() if l.strip()]
    rect_obj = load_scanport_rect()
    rect = (rect_obj.x, rect_obj.y, rect_obj.w, rect_obj.h)
    windows = [(r["flash_on_ts"] - 0.6, r["flash_off_ts"] + cfg.capture_wait_s) for r in results]
    frames_by_window = [[] for _ in results]
    reader = SourceSessionReader(session)
    for row, pixels in reader.iter_camera_frames(decode=False):
        ts = float((row.get("acquisition") or {}).get("receive_wall_ts") or 0.0)
        hit = [i for i, (a, b) in enumerate(windows) if a <= ts <= b]
        if not hit:
            continue
        import cv2
        from src.engine.camera.camera_manager import apply_frame_transform
        img = cv2.imdecode(np.frombuffer(pixels, dtype=np.uint8), cv2.IMREAD_COLOR)
        acq = row.get("acquisition") or {}
        img = apply_frame_transform(img, int(acq.get("rotation", 0) or 0), bool(acq.get("mirror_horizontal")),
                                    bool(acq.get("mirror_vertical")))
        for i in hit:
            frames_by_window[i].append((ts, img))
    all_peaks = sorted(p for r in results for p in r.get("peaks", []))
    reference, hits, rows = None, [], []
    for r, frames in zip(results, frames_by_window):
        on, off = r["flash_on_ts"], r["flash_off_ts"]
        late = [p for p in all_peaks if on <= p <= off + cfg.capture_wait_s]
        cutoff = min(late) if late else None
        before = [img for ts, img in frames if ts < on]
        during = [(ts, img) for ts, img in frames if ts >= on and (cutoff is None or ts < cutoff)]
        baseline = float(before[-1][rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].mean()) if before else None
        idx, info = select_plateau([img for _, img in during], rect, baseline=baseline, cfg=cfg)
        row = dict(flash_on_ts=on, reference_flash=r.get("reference_flash"), valid=bool(idx))
        if idx:
            image = average_gray([during[i][1] for i in idx], rect)
            if reference is not None and not r.get("reference_flash"):
                n = max(1, len([p for p in r.get("peaks", []) if p < on]))
                holes, stats = find_new_holes(reference, image, cfg)
                for h in holes[:n]:
                    hits.append(dict(wall_ts=off, camera_x=h["x"] + rect[0], camera_y=h["y"] + rect[1]))
                row.update(expected=n, holes=holes[:n], extra_holes=holes[n:], threshold=stats["threshold"])
            reference = image
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "flash_hits.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hits))
    (out / "flash_results.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    for name in ("ground_truth.jsonl", "optical_holes.json"):
        if (session / name).exists():
            shutil.copy(session / name, out / name)
    score = score_session(out)
    score.update(session=session.name, overrides=overrides)
    (out / "score.json").write_text(json.dumps(score, indent=2))
    return score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m automation.bench flash-replay")
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[], help="FlashConfig field=value")
    args = parser.parse_args(argv)
    overrides = dict(item.split("=", 1) for item in args.set)
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} exists and is not empty (never overwrite a run)")
    for s in args.sessions:
        print(json.dumps(replay_session(Path(s), root / Path(s).name, overrides)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
