"""Hole-likeness re-ranking of exported candidates — MEASUREMENT ONLY (no live behaviour).

  python3 -m automation.bench hole-likeness SESSION_DIR REPLAY_DIR [--age 0.15] [--out OUT.json]

REPLAY_DIR must come from `bench replay --export-candidates` on SESSION_DIR (which has
ground_truth.jsonl). For every replayed shot with a GT hole, each exported candidate (and
the GT point itself, for diagnostics) gets features from the multiplicative change image
D = 1 - post/pre (robust to static texture):
  mf      matched filter: <D, K> with K = mean real-stamp radial profile (zero-mean in r<=14)
  snr     mf / robust std of D in the annulus 9..20 px (texture/motion noise)
  bipolar strongest bright (negative D) response within r<=10, relative to the dark core
  persist core darkening in a later frame window / first window (holes stay, animation moves)
and a combined `hole_likeness = snr * purity * persistence`. Reports, per background, how often
the GT is among the candidates (within --match-px) and its rank under the current V2 score and
under hole_likeness. Stamps: real airgun + real AR15 libraries (same ceiling camera).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

R = 20
LIB = Path("/data/skjutbana/hole_library")


def _radial_kernel() -> np.ndarray:
    stamps = []
    for p in (LIB / "clean_stamps.npz", LIB / "ar15_co2_20261118" / "clean_stamps.npz"):
        try:
            stamps.extend(np.load(p, allow_pickle=True)["stamps"].astype(np.float32))
        except Exception:
            pass
    if not stamps:
        raise SystemExit("no stamp libraries")
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    prof = np.zeros(15)
    for s in stamps:
        s = s / max(float(s.max()), 1e-6)
        for r in range(15):
            prof[r] += s[(rr >= r - 0.5) & (rr < r + 0.5)].mean()
    prof /= len(stamps)
    k = np.interp(rr, np.arange(15), prof, right=0.0)
    inside = rr <= 14
    k[inside] -= k[inside].mean()
    k[~inside] = 0.0
    return (k / np.sqrt((k ** 2).sum())).astype(np.float32), rr


class FrameStore:
    def __init__(self, session: Path):
        import cv2
        self.cv2 = cv2
        from src.engine.camera.camera_manager import apply_frame_transform
        self.tf = apply_frame_transform
        self.session = session
        self.rows = [json.loads(l) for l in open(session / "camera_index.jsonl")]
        self.ts = np.array([r["acquisition"]["receive_wall_ts"] for r in self.rows])
        self.fh = open(session / "camera.bin", "rb")
        self.cache: dict[int, np.ndarray] = {}

    def frame(self, i: int) -> np.ndarray:
        if i not in self.cache:
            r = self.rows[i]
            self.fh.seek(int(r["offset"]))
            px = self.cv2.imdecode(np.frombuffer(self.fh.read(int(r["length"])), np.uint8), self.cv2.IMREAD_COLOR)
            a = r["acquisition"]
            px = self.tf(px, int(a.get("rotation", 0) or 0), bool(a.get("mirror_horizontal")), bool(a.get("mirror_vertical")))
            self.cache[i] = self.cv2.cvtColor(px, self.cv2.COLOR_BGR2GRAY).astype(np.float32)
            if len(self.cache) > 60:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def median(self, t0: float, t1: float) -> np.ndarray | None:
        idx = np.where((self.ts > t0) & (self.ts < t1))[0]
        if len(idx) == 0:
            return None
        return np.median(np.stack([self.frame(int(i)) for i in idx]), axis=0)


def features(pre, post, late, x, y, kernel, rr) -> dict | None:
    x, y = int(round(x)), int(round(y))
    h, w = pre.shape
    if not (R <= x < w - R and R <= y < h - R):
        return None
    sl = (slice(y - R, y + R + 1), slice(x - R, x + R + 1))
    p = np.maximum(pre[sl], 1.0)
    d = 1.0 - post[sl] / p
    ann = (rr >= 9) & (rr <= 20)
    d = d - np.median(d[ann])
    mf = float((d * kernel).sum())
    mad = float(np.median(np.abs(d[ann] - np.median(d[ann])))) * 1.4826 + 1e-3
    snr = mf / mad
    core = float(d[rr <= 2].mean())
    bright = float(max(0.0, -d[rr <= 10].min()))
    bipolar = bright / max(core, 1e-3) if core > 0 else 9.0
    purity = 1.0 / (1.0 + max(0.0, bipolar - 0.3))
    persist = 1.0
    if late is not None:
        dl = 1.0 - late[sl] / p
        dl = dl - np.median(dl[ann])
        core_l = float(dl[rr <= 2].mean())
        persist = float(np.clip(core_l / core, 0.0, 1.5)) if core > 1e-3 else 0.0
    hl = max(0.0, snr) * purity * min(1.0, persist)
    return dict(mf=mf, snr=snr, core=core, bipolar=bipolar, persist=persist, hole_likeness=hl)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m automation.bench hole-likeness")
    ap.add_argument("session")
    ap.add_argument("replay")
    ap.add_argument("--age", type=float, default=0.15, help="first post window start after the peak")
    ap.add_argument("--match-px", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    session, replay = Path(args.session), Path(args.replay)
    kernel, rr = _radial_kernel()
    gt = [json.loads(l) for l in open(session / "ground_truth.jsonl") if l.strip()]
    gt = [g for g in gt if "camera_x" in g]
    groups = collections.defaultdict(list)
    for line in open(replay / "candidates.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "candidate":
            groups[r["group"]].append(r["candidate"])
    store = FrameStore(session)
    per_bg = collections.defaultdict(lambda: collections.Counter())
    rows = []
    used = set()
    for g, cands in groups.items():
        t_first = min(float(c["timestamp"]) for c in cands)
        # the GT shot whose peak precedes the first candidate frame by 0.05..0.6 s
        match = [i for i, s in enumerate(gt) if i not in used and 0.0 < t_first - float(s["wall_ts"]) < 0.6]
        if not match:
            continue
        i = min(match, key=lambda j: t_first - float(gt[j]["wall_ts"]))
        used.add(i)
        s = gt[i]
        peak = float(s["wall_ts"])
        pre = store.median(peak - 0.4, peak - 0.02)
        post = store.median(peak + args.age, peak + args.age + 0.12)
        late = store.median(peak + args.age + 0.15, peak + args.age + 0.35)
        if pre is None or post is None:
            continue
        bg = s.get("background") or s.get("weapon") or "physical"
        gx, gy = float(s["camera_x"]), float(s["camera_y"])
        scored = []
        for c in cands:
            f = features(pre, post, late, c["camera_x"], c["camera_y"], kernel, rr)
            if f is None:
                continue
            dist = float(np.hypot(c["camera_x"] - gx, c["camera_y"] - gy))
            scored.append((c, f, dist))
        fg = features(pre, post, late, gx, gy, kernel, rr)
        present = [k for k, (_, _, dist) in enumerate(scored) if dist <= args.match_px]
        st = per_bg[bg]
        st["shots"] += 1
        rec = dict(group=g, background=bg, n_cand=len(scored), gt_features=fg)
        if present:
            st["present"] += 1
            for key, name in ((lambda t: t[0].get("score", 0.0), "v2"), (lambda t: t[1]["hole_likeness"], "hl")):
                order = sorted(range(len(scored)), key=lambda k: -key(scored[k]))
                rank = min(order.index(k) for k in present) + 1
                rec[f"rank_{name}"] = rank
                st[f"top1_{name}"] += rank == 1
                st[f"top3_{name}"] += rank <= 3
        rows.append(rec)
    total = collections.Counter()
    print(f"{'background':18s} shots present | top1 v2  hl | top3 v2  hl")
    for bg, st in sorted(per_bg.items()):
        total.update(st)
        print(f"{bg:18s} {st['shots']:5d} {st['present']:7d} | {st['top1_v2']:7d} {st['top1_hl']:3d} | {st['top3_v2']:7d} {st['top3_hl']:3d}")
    print(f"{'TOTAL':18s} {total['shots']:5d} {total['present']:7d} | {total['top1_v2']:7d} {total['top1_hl']:3d} | {total['top3_v2']:7d} {total['top3_hl']:3d}")
    out = Path(args.out) if args.out else replay / "hole_likeness.json"
    out.write_text(json.dumps(dict(per_background={k: dict(v) for k, v in per_bg.items()}, total=dict(total),
                                   age=args.age, match_px=args.match_px, rows=rows,
                                   evidence="OFFLINE_RERANK_OF_EXPORTED_CANDIDATES (measurement only)"), indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Dense mode: the matched filter itself as a proposal source (recall + ranking)
# ---------------------------------------------------------------------------
def dense_peaks(pre, post, late, kernel, rr, roi, top_n: int = 40) -> list[tuple[float, float, dict]]:
    """Matched-filter response on D = 1 - post/pre inside roi (x0, y0, x1, y1), normalised by the
    local robust noise of D; the top_n local maxima are re-scored with features()."""
    import cv2
    x0, y0, x1, y1 = roi
    p = np.maximum(pre[y0:y1, x0:x1], 1.0)
    d = 1.0 - post[y0:y1, x0:x1] / p
    d = d - cv2.blur(d, (41, 41))
    mf = cv2.filter2D(d, cv2.CV_32F, kernel)
    noise = np.sqrt(np.maximum(cv2.blur(d * d, (31, 31)), 1e-8)) + 1e-3
    s = mf / noise
    dil = cv2.dilate(s, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero((s >= dil) & (s > 0))
    order = np.argsort(-s[ys, xs])[: top_n * 3]
    out = []
    for i in order:
        cx, cy = float(xs[i] + x0), float(ys[i] + y0)
        f = features(pre, post, late, cx, cy, kernel, rr)
        if f is not None:
            f["dense_s"] = float(s[ys[i], xs[i]])
            out.append((cx, cy, f))
        if len(out) >= top_n:
            break
    return out


def dense_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m automation.bench hole-likeness-dense")
    ap.add_argument("session")
    ap.add_argument("--age", type=float, default=0.15)
    ap.add_argument("--post-len", type=float, default=0.12)
    ap.add_argument("--late-len", type=float, default=0.13, help="late window right after the post window")
    ap.add_argument("--match-px", type=float, default=8.0)
    ap.add_argument("--rank", default="hole_likeness", help="feature to rank by (hole_likeness | dense_s | snr)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from src.engine.input.hit_input import hit_input
    from src.engine.settings import load_viewport_rect
    session = Path(args.session)
    kernel, rr = _radial_kernel()
    gt = [json.loads(l) for l in open(session / "ground_truth.jsonl") if l.strip()]
    gt = [g for g in gt if "camera_x" in g]
    vp = load_viewport_rect()
    corners = [hit_input.screen_to_camera(x, y) for x, y in ((vp.x, vp.y), (vp.right, vp.y), (vp.right, vp.bottom), (vp.x, vp.bottom))]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    roi = (int(max(R, min(xs))), int(max(R, min(ys))), int(max(xs)), int(max(ys)))
    store = FrameStore(session)
    per_bg = collections.defaultdict(collections.Counter)
    rows = []
    for k, s in enumerate(gt):
        peak = float(s["wall_ts"])
        nxt = min([float(g["wall_ts"]) for g in gt if float(g["wall_ts"]) > peak + 0.01] or [peak + 9])
        pre = store.median(peak - 0.4, peak - 0.02)
        # earlier holes of a burst are in pre; later shots must not leak into post/late
        p1 = min(peak + args.age + args.post_len, nxt + 0.10)
        post = store.median(peak + args.age, p1)
        late = store.median(p1, min(p1 + args.late_len, nxt + 0.10)) if p1 < nxt + 0.10 else None
        if pre is None or post is None:
            continue
        bg = s.get("background") or s.get("weapon") or "physical"
        gx, gy = float(s["camera_x"]), float(s["camera_y"])
        peaks = dense_peaks(pre, post, late, kernel, rr, roi)
        peaks.sort(key=lambda t: -t[2].get(args.rank, 0.0))
        hits = [i for i, (x, y, _) in enumerate(peaks) if np.hypot(x - gx, y - gy) <= args.match_px]
        st = per_bg[bg]
        st["shots"] += 1
        rank = hits[0] + 1 if hits else None
        st["present40"] += rank is not None
        st["top1"] += rank == 1
        st["top3"] += rank is not None and rank <= 3
        err = float(np.hypot(peaks[0][0] - gx, peaks[0][1] - gy)) if peaks else None
        rows.append(dict(i=k, background=bg, rank=rank, top1_err_px=err,
                         top1=dict(x=peaks[0][0], y=peaks[0][1], **{a: round(b, 3) for a, b in peaks[0][2].items()}) if peaks else None))
    total = collections.Counter()
    print(f"{'background':18s} shots | top1 top3 in-top40   (rank by {args.rank}, post {args.age}+{args.post_len}s, late +{args.late_len}s)")
    for bg, st in sorted(per_bg.items()):
        total.update(st)
        print(f"{bg:18s} {st['shots']:5d} | {st['top1']:4d} {st['top3']:4d} {st['present40']:8d}")
    print(f"{'TOTAL':18s} {total['shots']:5d} | {total['top1']:4d} {total['top3']:4d} {total['present40']:8d}")
    out = Path(args.out) if args.out else Path("/data/skjutbana/evaluation_runs/holelike_20261118") / f"dense_{session.name}.json"
    out.write_text(json.dumps(dict(per_background={a: dict(b) for a, b in per_bg.items()}, total=dict(total), args=vars(args),
                                   rows=rows, evidence="OFFLINE_DENSE_MATCHED_FILTER (measurement only)"), indent=1))
    print(f"written {out}")
    return 0


# ---------------------------------------------------------------------------
# Emission check: hole-likeness at each emitted XY (correct vs false) — abstention study
# ---------------------------------------------------------------------------
def emissions_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m automation.bench hole-likeness-emissions")
    ap.add_argument("session")
    ap.add_argument("replay")
    ap.add_argument("--age", type=float, default=0.15)
    ap.add_argument("--tol-px", type=float, default=10.0)
    args = ap.parse_args(argv)
    session, replay = Path(args.session), Path(args.replay)
    kernel, rr = _radial_kernel()
    rows_gt = [json.loads(l) for l in open(session / "ground_truth.jsonl") if l.strip()]
    events = sorted(float(g["wall_ts"]) for g in rows_gt)
    holes = [g for g in rows_gt if "camera_x" in g]
    em = json.load(open(replay / "replay.json"))["emissions"]
    store = FrameStore(session)
    out = []
    for e in em:
        t = float(e["wall_ts"])
        prev = [p for p in events if p < t - 0.05]
        if not prev:
            continue
        peak = prev[-1]
        pre = store.median(peak - 0.4, peak - 0.02)
        post = store.median(peak + args.age, min(t, peak + args.age + 0.12) + 1e-3)
        if pre is None or post is None:
            continue
        f = features(pre, post, None, float(e["camera_x"]), float(e["camera_y"]), kernel, rr)
        if f is None:
            continue
        near = [h for h in holes if abs(float(h["wall_ts"]) - peak) < 0.02]
        ok = any(np.hypot(float(h["camera_x"]) - e["camera_x"], float(h["camera_y"]) - e["camera_y"]) <= args.tol_px for h in near)
        out.append(dict(correct=bool(ok), snr=f["snr"], core=f["core"], bipolar=f["bipolar"]))
    good = sorted(r["snr"] for r in out if r["correct"])
    bad = sorted(r["snr"] for r in out if not r["correct"])
    print(f"emissions {len(out)}: correct {len(good)} false {len(bad)}")
    for thr in (0, 2, 4, 6, 8, 10, 15, 20, 30):
        print(f"  abstain if snr<{thr:3d}: keep correct {sum(v >= thr for v in good):3d}/{len(good)}  "
              f"keep false {sum(v >= thr for v in bad):3d}/{len(bad)}")
    (replay / "emission_hole_likeness.json").write_text(json.dumps(out, indent=1))
    return 0
