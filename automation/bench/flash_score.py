"""Score a precision-flash optical session against its ground truth.

  python3 -m automation.bench flash-score SESSION_DIR [SESSION_DIR ...] [--tol-px 10]

Matches every flash hit to the nearest unused truth dot (camera px) and
reports detected / missed / false and the localisation error.
OPTICAL != PROJECTILE: projected dots, not paper damage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def score_session(directory: Path, tol_px: float = 10.0) -> dict:
    truth = [r for r in _rows(directory / "ground_truth.jsonl") if "camera_x" in r]
    hits = _rows(directory / "flash_hits.jsonl")
    results = _rows(directory / "flash_results.jsonl")
    pairs = sorted(((math.hypot(h["camera_x"] - t["camera_x"], h["camera_y"] - t["camera_y"]), i, j)
                    for i, h in enumerate(hits) for j, t in enumerate(truth)))
    used_h, used_t, errors, vectors = set(), set(), [], []
    for dist, i, j in pairs:
        if dist > tol_px or i in used_h or j in used_t:
            continue
        used_h.add(i); used_t.add(j); errors.append(dist)
        vectors.append((hits[i]["camera_x"] - truth[j]["camera_x"], hits[i]["camera_y"] - truth[j]["camera_y"]))
    errors.sort()
    # Truth is calibration-predicted: a common shift is calibration drift, not
    # measurement error.  Report it and the residual after removing it.
    shift, residual = None, []
    if vectors:
        sx = sorted(v[0] for v in vectors)[len(vectors) // 2]
        sy = sorted(v[1] for v in vectors)[len(vectors) // 2]
        shift = [round(sx, 2), round(sy, 2)]
        residual = sorted(math.hypot(v[0] - sx, v[1] - sy) for v in vectors)
    shot_results = [r for r in results if not r.get("reference_flash")]
    meta = json.loads((directory / "optical_holes.json").read_text()) if (directory / "optical_holes.json").exists() else {}
    return dict(session=directory.name, background=meta.get("background"), truth=len(truth),
                detected=len(used_t), missed=len(truth) - len(used_t), false=len(hits) - len(used_h),
                flashes=meta.get("flash_count"), invalid_flashes=sum(1 for r in results if not r.get("valid")),
                extra_candidates=sum(len(r.get("extra_holes") or []) for r in shot_results),
                median_err_px=round(errors[len(errors) // 2], 2) if errors else None,
                max_err_px=round(errors[-1], 2) if errors else None,
                common_shift_px=shift,
                residual_median_px=round(residual[len(residual) // 2], 2) if residual else None,
                residual_max_px=round(residual[-1], 2) if residual else None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m automation.bench flash-score")
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--tol-px", type=float, default=10.0)
    args = parser.parse_args(argv)
    for s in args.sessions:
        print(json.dumps(score_session(Path(s), args.tol_px)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
