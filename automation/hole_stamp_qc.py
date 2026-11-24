"""Create an immutable visual/structural review pack for projected hole stamps.

This is a measurement-only helper. It never edits the source stamp libraries or
trains a model. The contact sheet renders the exact darkening maps, including
rotated/flipped/gain-scaled variants like the projector harness uses.

    python3 -m automation.hole_stamp_qc
    python3 -m automation.hole_stamp_qc --output /data/skjutbana/evaluation_runs/my_review
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data/skjutbana/hole_library")
SOURCES = (
    # These two PNG directories are what SyntheticHoleOverlay actually loads.
    ("runtime_airgun_png", ROOT / "stamps_png", "png_alpha"),
    ("runtime_ar15_png", ROOT / "ar15_co2_20261118" / "stamps_png", "png_alpha"),
    # Preserve review of the other archived stamp arrays too; they are not the
    # active F2 co2 library unless the configured profile/library is changed.
    ("co2_20261122_16", ROOT / "clean_stamps_co2_20261122.npz", "npz"),
    ("air_rifle_r7_5", ROOT / "clean_stamps_air_rifle_r7_20261122.npz", "npz"),
    ("co2_r7_1", ROOT / "clean_stamps_co2_r7_20261122.npz", "npz"),
)


def _components(image: np.ndarray) -> dict:
    peak = float(np.max(image))
    if not math.isfinite(peak) or peak <= 0:
        return {"peak": peak, "components_at_20pct": 0, "largest_component_fraction": 0.0,
                "second_component_pixels": 0, "review_flag": "EMPTY_OR_INVALID"}
    mask = np.asarray(image >= peak * 0.20, dtype=np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    total = sum(areas)
    second = areas[1] if len(areas) > 1 else 0
    largest_fraction = (areas[0] / total) if total else 0.0
    flag = "REVIEW_MULTIPLE_LOBES" if second >= 2 else "REVIEW_BROAD_BLOB" if total > 900 else "OK_SINGLE_LOBE"
    return {"peak": peak, "components_at_20pct": len(areas), "largest_component_fraction": largest_fraction,
            "second_component_pixels": second, "threshold_pixels": total, "review_flag": flag}


def _variant(image: np.ndarray, angle: float, flip: bool, gain: float) -> np.ndarray:
    src = np.fliplr(image) if flip else image
    h, w = src.shape
    mat = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    out = cv2.warpAffine(np.asarray(src, np.float32), mat, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.clip(out * gain, 0.0, 0.95)


def _tile(image: np.ndarray, label: str, width: int = 420, height: int = 180) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, np.uint8)
    h, w = image.shape
    scale = min(5.0, (width - 18) / w, (height - 38) / h)
    # Show darkening on a neutral projected surface, not as a nearly black alpha map.
    preview = 0.72 * (1.0 - np.clip(image, 0, 0.95))
    shown = cv2.resize(np.clip(preview * 255, 0, 255).astype(np.uint8),
                       (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_NEAREST)
    x = (width - shown.shape[1]) // 2
    y = 28 + (height - 28 - shown.shape[0]) // 2
    canvas[y:y + shown.shape[0], x:x + shown.shape[1]] = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, label[:46], (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="new output directory; must not already exist")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or Path("/data/skjutbana/evaluation_runs") / f"hole_stamp_qc_{stamp}"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing review output: {out}")
    out.mkdir(parents=True)

    rows, tiles_by_group = [], {}
    for group, path, source_kind in SOURCES:
        if not path.exists():
            continue
        if source_kind == "png_alpha":
            items = []
            for png_path in sorted(path.glob("*.png")):
                rgba = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
                if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                    items.append((png_path.stem, rgba[..., 3].astype(np.float32) / 255.0, png_path))
        else:
            data = np.load(path, allow_pickle=False)
            if "stamps" not in data.files:
                continue
            stamps = data["stamps"]
            ids = data["id"] if "id" in data.files else np.arange(len(stamps))
            items = [(str(stamp_id), np.asarray(image, dtype=np.float32), path)
                     for stamp_id, image in zip(ids, stamps)]
        for index, (stamp_id, image, item_path) in enumerate(items):
            base = _components(image)
            row = {"source_group": group, "source_file": str(item_path), "index": index,
                   "stamp_id": str(stamp_id), "shape": list(image.shape), **base}
            rows.append(row)
            tiles = tiles_by_group.setdefault(group, [])
            tiles.append(_tile(image, f"{group} #{index} {stamp_id} ORIGINAL"))
            # Representative, deterministic augmentation preview: both flips, rotations, and alpha gains.
            for angle, flip, gain in ((37, False, 0.72), (143, True, 0.90)):
                aug = _variant(image, angle, flip, gain)
                aug_stats = _components(aug)
                row.setdefault("augmentation_checks", []).append({"angle_deg": angle, "horizontal_flip": flip,
                                                                     "gain": gain, **aug_stats})
                tiles.append(_tile(aug, f"{group} #{index} {stamp_id} {angle}deg flip={int(flip)} a={gain:.2f}"))

    if not rows:
        raise SystemExit(f"No stamp arrays found under {ROOT}")
    for group, tiles in tiles_by_group.items():
        columns = 3
        cell_w, cell_h = tiles[0].shape[1], tiles[0].shape[0]
        rows_n = (len(tiles) + columns - 1) // columns
        sheet = np.full((rows_n * cell_h, columns * cell_w, 3), 218, np.uint8)
        for i, tile in enumerate(tiles):
            y, x = divmod(i, columns)
            sheet[y * cell_h:(y + 1) * cell_h, x * cell_w:(x + 1) * cell_w] = tile
        cv2.imwrite(str(out / f"{group}_review.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "purpose": "VISUAL_REVIEW_ONLY_NOT_TRAINING",
              "sources": [dict(group=name, path=str(path), loader=kind, present=path.exists())
                          for name, path, kind in SOURCES],
              "n_stamps": len(rows), "n_review_flags": sum(r["review_flag"] != "OK_SINGLE_LOBE" for r in rows),
              "component_rule": "threshold at 20% of each stamp peak; flag secondary connected lobe >=2 pixels; broad-map flag >900 pixels",
              "rows": rows}
    (out / "stamp_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "contact_sheets": [str(out / f"{g}_review.jpg") for g in tiles_by_group],
                      "report": str(out / "stamp_review.json"), "stamps": len(rows),
                      "review_flags": report["n_review_flags"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
