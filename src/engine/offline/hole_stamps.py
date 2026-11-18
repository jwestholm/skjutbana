"""Real-hole stamps: background-free darkening maps cut from labelled real shots.

Source: ``/data/skjutbana/hole_library/stamps.npz`` (ratio = post/pre around
real pellet holes, see that directory's README).  ``build_clean`` turns each
ratio patch into a pure hole map ``D = 1 - ratio`` that is non-zero ONLY on the
hole itself (core + soft penumbra) and exactly 0 elsewhere: projected edges,
older holes, game content and noise around it are removed.  Patches where the
hole is too weak (e.g. it was already present in the PRE image), off-centre, or
merged with background structure are rejected with a reason.

Rendering is multiplicative (``frame *= 1 - D``), as measured: a hole darkens
the local light to ~0.7-0.8x, it is never "black".  ``scale`` shrinks airgun
holes to AR15 size (~0.5); extra blur keeps the optical blur realistic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LIBRARY = Path("/data/skjutbana/hole_library")
CLEAN_NAME = "clean_stamps.npz"


def _component_at_centre(mask: np.ndarray, cy: int, cx: int, search: int = 2) -> np.ndarray | None:
    import cv2
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    best = None
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            lab = labels[cy + dy, cx + dx]
            if lab:
                best = lab
                break
        if best:
            break
    return None if best is None else labels == best


def clean_one(ratio: np.ndarray, noise: float, *, min_depth: float = 0.10, max_radius: float = 9.0,
              tear_radius: float = 18.0) -> tuple[np.ndarray | None, dict]:
    """Return (D, info). D is float32, 0 outside the hole; None when rejected."""
    import cv2
    d = (1.0 - ratio.astype(np.float32))
    h, w = d.shape
    cy, cx = h // 2, w // 2
    smooth = cv2.GaussianBlur(d, (0, 0), 1.0)
    peak = float(smooth[cy - 3:cy + 4, cx - 3:cx + 4].max())
    info = dict(peak_smoothed=round(peak, 4), noise=round(float(noise), 4))
    if peak < min_depth:
        return None, dict(info, reason="TOO_WEAK (hole already in PRE or not visible)")
    level = max(0.25 * peak, 3.0 * float(noise), 0.03)
    raw = (smooth >= level).astype(np.uint8)
    # remove thin (1-2 px) protrusions: projected lines/edges passing the hole
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    comp = _component_at_centre(opened, cy, cx) if opened.any() else None
    if comp is None:
        comp = _component_at_centre(raw, cy, cx)
    if comp is None:
        return None, dict(info, reason="NO_CENTRAL_BLOB")
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - cy, xx - cx)
    reach = float(rr[comp].max())
    area = int(comp.sum())
    # a tear may be long but thin; a blob that spreads wide is background leakage
    if reach > tear_radius or (reach > max_radius and area > 4.0 * reach * 3.0):
        return None, dict(info, reason=f"LEAKS_INTO_BACKGROUND (reach {reach:.1f}px area {area})")
    # outside structure near the hole that is as strong as the hole = contaminated context
    ring = (rr > reach + 3) & (rr < reach + 10)
    if ring.any() and float(np.abs(smooth[ring]).max()) > 0.6 * peak:
        return None, dict(info, reason="STRONG_STRUCTURE_NEXT_TO_HOLE")
    # mask = component + 1 px penumbra, feathered; everything else exactly 0
    mask = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    feather = cv2.GaussianBlur(mask, (0, 0), 0.8)
    out = np.clip(d, 0.0, None) * feather
    out[feather < 0.02] = 0.0
    # re-centre on the darkness centroid (the hole, not the label)
    total = float(out.sum())
    my, mx = float((out * yy).sum() / total), float((out * xx).sum() / total)
    shift = np.float32([[1, 0, cx - mx], [0, 1, cy - my]])
    out = cv2.warpAffine(out, shift, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    info.update(reason="OK", reach_px=round(reach, 1), area_px=area, depth_max=round(float(out.max()), 3),
                tear=bool(reach > max_radius))
    return out.astype(np.float32), info


def build_clean(library: Path = LIBRARY, out_name: str = CLEAN_NAME) -> dict:
    src = np.load(library / "stamps.npz", allow_pickle=True)
    kept, meta, report = [], [], []
    for i in range(len(src["stamps"])):
        clean, info = clean_one(src["stamps"][i], float(src["noise"][i]))
        row = dict(id=str(src["id"][i]), kind=str(src["kind"][i]), bg_gray=float(src["bg_gray"][i]), **info)
        report.append(row)
        if clean is not None:
            kept.append(clean)
            meta.append(row)
    np.savez_compressed(library / out_name, stamps=np.stack(kept),
                        id=np.array([m["id"] for m in meta]), kind=np.array([m["kind"] for m in meta]),
                        tear=np.array([m["tear"] for m in meta]),
                        depth_max=np.array([m["depth_max"] for m in meta], dtype=np.float32))
    summary = dict(source=str(library / "stamps.npz"), kept=len(kept), rejected=len(report) - len(kept),
                   rejected_reasons={}, rows=report)
    for r in report:
        if r["reason"] != "OK":
            key = r["reason"].split(" ")[0]
            summary["rejected_reasons"][key] = summary["rejected_reasons"].get(key, 0) + 1
    (library / "clean_stamps.json").write_text(json.dumps(summary, indent=2))
    _contact_sheet(src, report, kept, library / "clean_stamps_vs_source.png")
    _export_png(kept, meta, library)
    return summary


def _export_png(kept, meta, library: Path) -> None:
    """Each stamp as RGBA PNG: black, alpha = darkening D, fully transparent elsewhere.

    Alpha-blending black with alpha D over any background gives bg * (1 - D),
    i.e. exactly the multiplicative hole model.  The sheet shows the stamps
    alone on a transparency checkerboard (no source background).
    """
    import cv2
    out = library / "stamps_png"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    tiles = []
    for k, (d, m) in enumerate(zip(kept, meta)):
        rgba = np.zeros(d.shape + (4,), np.uint8)
        rgba[..., 3] = np.clip(d * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{k:03d}.png"), rgba)
        big = cv2.resize(d, (123, 123), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.mgrid[0:123, 0:123]
        checker = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 235.0, 200.0).astype(np.float32)
        tile = cv2.cvtColor(np.clip(checker * (1.0 - big), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile, (0, 0), (122, 122), (90, 90, 90), 1)
        tiles.append(tile)
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(library / "clean_stamps.png"), np.vstack(rows))


def _contact_sheet(src, report, kept, path: Path) -> None:
    import cv2
    tiles, k = [], 0
    for i, row in enumerate(report):
        ratio = src["stamps"][i]
        a = np.clip((ratio - 0.6) / 0.55 * 255, 0, 255).astype(np.uint8)
        if row["reason"] == "OK":
            clean = kept[k]; k += 1
            b = np.clip((1.0 - clean - 0.6) / 0.55 * 255, 0, 255).astype(np.uint8)
        else:
            b = np.zeros_like(a)
        pair = np.hstack([cv2.resize(a, (123, 123), interpolation=cv2.INTER_NEAREST), np.full((123, 3), 255, np.uint8),
                          cv2.resize(b, (123, 123), interpolation=cv2.INTER_NEAREST)])
        pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
        colour = (0, 160, 0) if row["reason"] == "OK" else (0, 0, 200)
        cv2.rectangle(pair, (0, 0), (pair.shape[1] - 1, pair.shape[0] - 1), colour, 2)
        tiles.append(pair)
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


class StampSet:
    """Clean real-hole stamps for rendering."""

    def __init__(self, path: Path | str = LIBRARY / CLEAN_NAME) -> None:
        data = np.load(path, allow_pickle=True)
        self.stamps = data["stamps"].astype(np.float32)
        self.ids = [str(x) for x in data["id"]]

    def __len__(self) -> int:
        return len(self.stamps)

    def patch(self, rng: np.random.Generator, scale: float, *, extra_blur_px: float = 0.0,
              index: int | None = None) -> tuple[np.ndarray, int]:
        """A darkening map D (0..1) for one hole: random stamp, rotation, mirror and scale."""
        import cv2
        i = int(rng.integers(len(self.stamps))) if index is None else int(index)
        d = self.stamps[i]
        if rng.random() < 0.5:
            d = d[:, ::-1]
        h, w = d.shape
        angle = float(rng.uniform(0.0, 360.0))
        m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, float(scale))
        d = cv2.warpAffine(np.ascontiguousarray(d), m, (w, h), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                           borderValue=0.0)
        if extra_blur_px > 0:
            d = cv2.GaussianBlur(d, (0, 0), extra_blur_px)
        # crop to the non-zero support (+1 px) so rendering stays cheap
        ys, xs = np.nonzero(d > 1e-4)
        if len(ys) == 0:
            return np.zeros((1, 1), np.float32), i
        cy, cx = h // 2, w // 2
        r = int(max(abs(ys - cy).max(), abs(xs - cx).max())) + 1
        return np.clip(d[cy - r:cy + r + 1, cx - r:cx + r + 1], 0.0, 0.95).astype(np.float32), i


def main(argv=None) -> int:
    summary = build_clean()
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
