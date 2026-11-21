"""Offline audio shot-signature study (measurement only; never changes engine behaviour).

Reads recorded source sessions (``audio.pcm`` + ``audio_index.jsonl``), replays the EXISTING
``AudioPeakDetector._process_chunk`` chunk by chunk (wall clock faked to each block's receipt time,
which equals the SKJUTBANA_AUDIO_RECEIPT_ANCHOR=1 anchoring; the live default anchors on processing
time, a few ms later), and for every trigger extracts waveform / spectral descriptors:

    envelope (1 ms RMS) rise time, decay times, energy concentration, pre-trigger SNR,
    spectral centroid / band fractions / flatness over 0-32 ms, secondary envelope peaks in 0-250 ms
    (sees the CO2 double report even when the 80 ms cooldown hides it), inter-trigger gaps.

Labels (out-of-band, truth only used here):
    GT sessions with real weapons: trigger nearest each ground-truth shot = ``shot``; trigger within
    PAIR_S after a shot trigger and not itself nearest a GT shot = ``follower``; else ``unmatched``.
    Sessions with synthetic audio only (projected tests): every trigger is ``background``.

Also evaluates candidate merge/reject rules (see ``RULES``) and prints counts per class.

    python3 automation/audio_shot_signature.py [--min-abs 0.56] OUT_DIR [SESSION_DIR ...]
    python3 automation/audio_shot_signature.py --live-logs OUT.jsonl LOG ...   (trigger gaps in game logs)
    python3 automation/audio_shot_signature.py --selftest
Default sessions: every directory under /data/skjutbana/source_sessions.  Input data is read-only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 16000
SOURCE_ROOT = Path("/data/skjutbana/source_sessions")
LIVE_ABS = 0.56            # audio_peak.threshold recorded in every session.json 2026-11-17..20
GT_MATCH_S = (-0.35, 0.35)  # trigger - GT wall_ts window (GT ts are camera-diff/operator estimates)
PAIR_S = 0.15


class _FakeTime:
    """Stands in for the detector module's ``time`` so _process_chunk sees recorded block times."""

    def __init__(self) -> None:
        self.wall = 0.0
        self.mono = 0.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


def replay_detector(blocks, *, min_abs: float = LIVE_ABS, cooldown_s: float | None = None):
    """Run the real AudioPeakDetector._process_chunk over recorded blocks.

    blocks: iterable of (wall_end_ts, mono_end_ts, int16 ndarray).  Returns (samples float32,
    sample_wall float64 per sample, triggers list of dict(ts, peak, rms, noise, chunk_start, idx)).
    """
    import importlib
    apd = importlib.import_module("src.engine.audio.audio_peak_detector")
    apd = sys.modules["src.engine.audio.audio_peak_detector"]  # package re-exports an instance of the same name
    det = apd.AudioPeakDetector()
    det.min_abs_peak = float(min_abs)
    if cooldown_s is not None:
        det.cooldown_s = float(cooldown_s)
    det._current_acquisition = None
    fake, real = _FakeTime(), apd.time
    apd.time = fake
    samples, walls, triggers = [], [], []
    pos = 0
    try:
        for wall_end, mono_end, pcm in blocks:
            n = pcm.size
            fake.wall, fake.mono = float(wall_end), float(mono_end)
            noise_before = det.noise_floor
            det._process_chunk(pcm.astype("<i2").tobytes())
            with det._lock:
                fresh = list(det._pending_dispatch)
                det._pending_dispatch.clear()
                det._sample_history.clear()
            for ev in fresh:
                idx = pos + n - 1 - int(round((wall_end - ev.timestamp) * SR))
                triggers.append(dict(ts=float(ev.timestamp), peak=float(ev.peak), rms=float(ev.rms),
                                     noise=float(det.noise_floor), noise_before=float(noise_before),
                                     chunk_start=pos, idx=int(idx)))
            samples.append(pcm.astype(np.float32) / 32768.0)
            walls.append(wall_end - (n - 1 - np.arange(n)) / SR)
            pos += n
    finally:
        apd.time = real
    if not samples:
        return np.zeros(0, np.float32), np.zeros(0), []
    return np.concatenate(samples), np.concatenate(walls), triggers


def load_blocks(session: Path):
    from src.engine.source_recording import SourceSessionReader
    reader = SourceSessionReader(session)
    for row, data in reader.iter_audio_blocks():
        b = row["block"]
        if not data:
            continue
        yield float(b["receive_wall_ts"]), float(b.get("read_complete_monotonic", 0.0)), \
            np.frombuffer(data, dtype="<i2")


def envelope(x: np.ndarray, win: int = 16) -> np.ndarray:
    """1 ms RMS envelope at sample resolution of the window (one value per `win` samples)."""
    m = x.size // win
    if m == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(np.square(x[:m * win].reshape(m, win)), axis=1))


def trigger_features(x: np.ndarray, idx: int) -> dict:
    """Waveform/spectral descriptors around trigger sample `idx` (1 ms envelope bins)."""
    ms = SR // 1000
    lo, hi = max(0, idx - 500 * ms), min(x.size, idx + 300 * ms)
    seg = x[lo:hi]
    t0 = (idx - lo) // ms                       # trigger bin
    env = envelope(seg, ms)
    if env.size == 0 or t0 >= env.size:
        return {}
    search = env[max(0, t0 - 5): t0 + 15]
    pk_bin = max(0, t0 - 5) + int(np.argmax(search))
    pk = float(env[pk_bin]) + 1e-9
    pre = env[max(0, t0 - 500): max(1, t0 - 20)]
    bg = float(np.median(pre)) if pre.size else 0.0
    # rise: from the envelope minimum in the 30 ms before the peak to the first bin >= 90 % of peak
    lo_b = max(0, pk_bin - 30)
    min_b = lo_b + int(np.argmin(env[lo_b: pk_bin + 1]))
    first90 = min_b + int(np.flatnonzero(env[min_b: pk_bin + 1] >= 0.9 * pk)[0])
    rise_ms = float(first90 - min_b)
    pre_min_rel = float(env[min_b] / pk)
    after = env[pk_bin:]

    def fall(frac):
        below = np.flatnonzero(after < frac * pk)
        return float(below[0]) if below.size else float(after.size)
    e = np.square(env[t0:])
    e10, e30, e100, e250 = (float(np.sum(e[:k])) for k in (10, 30, 100, 250))
    # secondary peaks: local maxima of a 3 ms smoothed envelope after the first decay (>=20 ms lag)
    sm = np.convolve(env, np.ones(3) / 3, mode="same")
    sec = []
    for b in range(pk_bin + 20, min(env.size - 1, pk_bin + 250)):
        if sm[b] >= sm[b - 1] and sm[b] > sm[b + 1] and sm[b] >= 0.25 * pk and \
                sm[b] >= np.max(sm[max(pk_bin + 15, b - 15): b + 16]):
            sec.append((b - pk_bin, round(float(sm[b] / pk), 3)))
    # spectrum of 32 ms from 2 ms before trigger
    s0 = idx - 2 * ms
    w = x[max(0, s0): s0 + 512]
    spec = {}
    if w.size == 512:
        mag = np.abs(np.fft.rfft(w * np.hanning(512))) ** 2
        f = np.fft.rfftfreq(512, 1 / SR)
        tot = float(np.sum(mag)) + 1e-12
        spec = dict(centroid_hz=float(np.sum(f * mag) / tot),
                    band_lt500=float(np.sum(mag[f < 500]) / tot),
                    band_500_2k=float(np.sum(mag[(f >= 500) & (f < 2000)]) / tot),
                    band_2k_4k=float(np.sum(mag[(f >= 2000) & (f < 4000)]) / tot),
                    band_gt4k=float(np.sum(mag[f >= 4000]) / tot),
                    flatness=float(np.exp(np.mean(np.log(mag[1:] + 1e-12))) / (np.mean(mag[1:]) + 1e-12)))
    w30 = x[idx: idx + 30 * ms]
    zcr = float(np.mean(np.abs(np.diff(np.sign(w30))) > 0)) if w30.size > 1 else 0.0
    return dict(env_peak=pk, bg_rms=bg, snr_db=float(20 * np.log10(pk / (bg + 1e-6))),
                rise_ms=rise_ms, pre_min_rel=pre_min_rel, peak_lag_ms=float(pk_bin - t0),
                decay50_ms=fall(0.5), decay20_ms=fall(0.2), decay10_ms=fall(0.1),
                conc_10_100=e10 / (e100 + 1e-12), conc_30_250=e30 / (e250 + 1e-12),
                secondary=sec, zcr_30ms=zcr, abs_max_30ms=float(np.max(np.abs(w30))) if w30.size else 0.0,
                **spec)


# Discharges NOT in ground_truth.jsonl but documented by the operator (independent of this audio study):
#  04f1be81: operator note "9th triple shot 251.22 unlabelled".
#  ar15cut:  GT covers 13 of 23 trigger pulls; operator protocol = row1 5 singles, row2 3 doubles,
#            row3 2 triples, then empty magazine.  GT notes (aim7 second_shot_of_double 76.396, aim9
#            second_shot_of_triple 81.356, aim10 first_shot_of_triple 82.579) pin the rhythm; the 10 missing
#            pulls are the leading triggers of aims 5, 6, 7, 8, 9 (x2), 10 (x2).  INFERRED, not observed.
OPERATOR_EXTRA = {
    "20261118T155321Z_04f1be81": [1795017251.22],
    "20261118T135339Z_da06f87a_ar15cut": [1795010273.42, 1795010275.11, 1795010275.355, 1795010276.16,
                                          1795010277.25, 1795010277.525, 1795010281.09, 1795010281.68,
                                          1795010282.845, 1795010283.09],
}
NONSHOT_NOTES = ("unknown_sound",)


def load_truth(session: Path) -> tuple[list[dict], str]:
    """Returns (truth rows [{ts, cls, note}], kind).  cls: discharge | discharge_inferred | nonshot."""
    gt = session / "ground_truth.jsonl"
    syn = session / "synthetic_audio.jsonl"
    rows = [json.loads(l) for l in open(gt)] if gt.exists() else []
    weapons = {r.get("weapon") for r in rows if r.get("weapon")}
    if syn.exists() and syn.stat().st_size > 0:
        return [], "synthetic_only"                 # projected test: real mic = room background
    if not rows:
        return [], "unlabelled"
    truth = []
    for r in rows:
        note = str(r.get("note") or "")
        cls = "nonshot" if any(k in note for k in NONSHOT_NOTES) else "discharge"
        truth.append(dict(ts=float(r["wall_ts"]), cls=cls, note=note))
    for ts in OPERATOR_EXTRA.get(session.name, []):
        truth.append(dict(ts=ts, cls="discharge_inferred", note="operator_protocol"))
    weapon = ",".join(sorted(w for w in weapons if w != "unlabelled")) or "unknown"
    return sorted(truth, key=lambda r: r["ts"]), "real:" + weapon


def label_triggers(trigs: list[dict], truth: list[dict], kind: str) -> None:
    """shot = trigger nearest a discharge (GT or operator-documented); nonshot = nearest a documented
    non-shot sound; follower = within PAIR_S after a shot trigger; unmatched = anything else."""
    if kind in ("synthetic_only", "unlabelled"):
        for t in trigs:
            t["label"] = "background" if kind == "synthetic_only" else "unlabelled"
        return
    for t in trigs:
        t["label"] = "unmatched"
    used = set()
    for g in truth:
        cand = [(abs(t["ts"] - g["ts"]), i) for i, t in enumerate(trigs)
                if GT_MATCH_S[0] <= t["ts"] - g["ts"] <= GT_MATCH_S[1] and i not in used]
        if cand:
            _d, i = min(cand)
            used.add(i)
            trigs[i]["label"] = "shot" if g["cls"].startswith("discharge") else "nonshot"
            trigs[i]["truth_cls"], trigs[i]["truth_note"], trigs[i]["gt_ts"] = g["cls"], g["note"], g["ts"]
        else:
            print(f"  WARN no trigger for truth {g}")
    for i, t in enumerate(trigs):
        if t["label"] == "unmatched":
            prev = [p for p in trigs[:i] if p["label"] == "shot" and 0 < t["ts"] - p["ts"] <= PAIR_S]
            if prev:
                t["label"] = "follower"


# ----------------------------------------------------------------------------- candidate rules
# Rules are hypotheses formulated on THIS data (the two physical AR15 sessions) - in-sample, not validation.
MERGE_S = 0.12              # == shot_dense_h15.CO2_PAIR_S; == SKJUTBANA_AUDIO_COOLDOWN_S=0.12
FOLLOWER_MIN_CENTROID = 2500.0
MIN_DECAY20_MS = 30.0


def rule_merge(trigs, *, window_s=MERGE_S, spectral_guard=False):
    """Merge a trigger into the previous KEPT trigger if it starts within window_s of it.

    spectral_guard: only merge when the candidate also looks like a follower (centroid >= 2.5 kHz);
    a low-centroid discharge-like sound inside the window is kept as its own shot."""
    kept, out = [], []
    for t in trigs:
        lead = kept[-1] if kept else None
        merge = lead is not None and 0 < t["ts"] - lead["ts"] <= window_s
        if merge and spectral_guard:
            merge = t.get("centroid_hz", 0.0) >= FOLLOWER_MIN_CENTROID
        out.append("merged" if merge else "kept")
        if not merge:
            kept.append(t)
    return out


def gate_sustained(t) -> bool:
    """Shot-like: envelope stays above 20 % of its peak for >= MIN_DECAY20_MS (discharges 43-114 ms)."""
    return t.get("decay20_ms", 0.0) >= MIN_DECAY20_MS


def evaluate_rules(all_trigs) -> dict:
    variants = {
        "baseline_live (cooldown 0.08 s)": dict(window_s=0.0, gate=None, guard=False),
        "A merge 120 ms": dict(window_s=MERGE_S, gate=None, guard=False),
        "A' merge 120 ms + follower centroid>=2.5k": dict(window_s=MERGE_S, gate=None, guard=True),
        "A+B merge 120 ms + decay20>=30 ms": dict(window_s=MERGE_S, gate=gate_sustained, guard=False),
    }
    by_sess: dict[str, list] = {}
    for t in all_trigs:
        by_sess.setdefault(t["session"], []).append(t)
    res = {}
    for name, v in variants.items():
        counts: dict[str, list] = {}
        for _s, ts in by_sess.items():
            dec = rule_merge(ts, window_s=v["window_s"], spectral_guard=v["guard"]) if v["window_s"] else ["kept"] * len(ts)
            for t, d in zip(ts, dec):
                emitted = d == "kept" and (v["gate"] is None or v["gate"](t))
                c = counts.setdefault(t["label"], [0, 0])
                c[0] += 1
                c[1] += int(emitted)
        res[name] = {k: dict(n=c[0], emitted=c[1]) for k, c in sorted(counts.items())}
    return res


def cooldown_sweep(session: Path, cooldowns=(0.08, 0.10, 0.12, 0.15, 0.20, 0.25)) -> list[dict]:
    """Live-path-equivalent: replay the detector with SKJUTBANA_AUDIO_COOLDOWN_S-style cooldowns."""
    truth, kind = load_truth(session)
    disc = [g["ts"] for g in truth if g["cls"].startswith("discharge")]
    rows = []
    for cd in cooldowns:
        _x, _w, trigs = replay_detector(load_blocks(session), cooldown_s=cd)
        ts = [t["ts"] for t in trigs]
        hit = sum(any(abs(t - g) <= 0.035 for t in ts) for g in disc)
        rows.append(dict(session=session.name, cooldown_s=cd, triggers=len(ts), discharges=len(disc),
                         discharges_with_trigger=hit, extra_triggers=len(ts) - hit))
    return rows


def analyse_session(session: Path, **kw) -> list[dict]:
    truth, kind = load_truth(session)
    x, _walls, trigs = replay_detector(load_blocks(session), **kw)
    for t in trigs:
        t.update(trigger_features(x, t["idx"]))
        t["session"] = session.name
        t["kind"] = kind
    for i, t in enumerate(trigs):
        t["gap_prev_ms"] = (t["ts"] - trigs[i - 1]["ts"]) * 1000 if i else None
        t["gap_next_ms"] = (trigs[i + 1]["ts"] - t["ts"]) * 1000 if i + 1 < len(trigs) else None
    label_triggers(trigs, truth, kind)
    summary = dict(session=session.name, kind=kind, audio_s=round(x.size / SR, 1),
                   n_truth=len(truth), n_truth_discharge=sum(g['cls'].startswith('discharge') for g in truth), n_trig=len(trigs),
                   max_abs=float(np.max(np.abs(x))) if x.size else 0.0,
                   n_chunks_over_abs=int(np.sum(np.max(np.abs(x[:x.size // 256 * 256]).reshape(-1, 256), axis=1)
                                                >= kw.get("min_abs", LIVE_ABS))) if x.size >= 256 else 0)
    return trigs, summary


def parse_live_log(path: Path) -> list[dict]:
    """[V2.22.6 AUDIO-RAW] TRIGGER chunk features + the following [V2.22.3 AUDIO-PRIORITY] peak ts."""
    import re
    lines = open(path, errors="replace").read().splitlines()
    out = []
    for i, line in enumerate(lines):
        if "AUDIO-RAW] TRIGGER" not in line:
            continue
        f = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", line.split("TRIGGER", 1)[1])}
        ts = None
        for k in range(i + 1, min(i + 12, len(lines))):
            m = re.search(r"AUDIO-PRIORITY\] peak=([\d.]+)", lines[k])
            if m:
                ts = float(m.group(1))
                break
            if "AUDIO-RAW] TRIGGER" in lines[k]:
                break
        out.append(dict(log=path.name, ts=ts, peak=f.get("peak"), rms=f.get("rms"), noise=f.get("noise")))
    return out


def live_log_summary(paths) -> list[dict]:
    rows = []
    for p in paths:
        ev = [e for e in parse_live_log(Path(p)) if e["ts"] is not None]
        ts = [e["ts"] for e in ev]
        gaps = [round((b - a) * 1000) for a, b in zip(ts, ts[1:])]
        row = dict(log=Path(p).name, triggers=len(ev), gaps_lt150_ms=sorted(g for g in gaps if g < 150),
                   gaps_150_1000_ms=sorted(g for g in gaps if 150 <= g < 1000))
        for w in (0.12, 0.15):
            dec = rule_merge(ev, window_s=w)
            row[f"after_merge_{int(w * 1000)}ms"] = dec.count("kept")
        rows.append(row)
    return rows


def _selftest() -> None:
    rng = np.random.default_rng(0)
    n = SR * 3
    x = (rng.normal(0, 0.003, n)).astype(np.float32)

    def burst(at, amp, tau_ms):
        k = np.arange(int(0.05 * SR))
        x[at:at + k.size] += (amp * np.exp(-k / (tau_ms * SR / 1000)) * rng.choice([-1, 1], k.size)).astype(np.float32)
    burst(SR, 0.9, 3.0)              # shot
    burst(SR + int(0.075 * SR), 0.7, 3.0)   # double report 75 ms later
    burst(2 * SR, 0.9, 3.0)          # separate shot 1 s later
    pcm = np.clip(x * 32768, -32768, 32767).astype("<i2")
    blocks = [(100.0 + (i + 256) / SR, 0.0, pcm[i:i + 256]) for i in range(0, n, 256)]
    xs, _w, tr = replay_detector(blocks)
    assert len(tr) == 3, tr
    gaps = np.diff([t["ts"] for t in tr]) * 1000
    assert 60 <= gaps[0] <= 90 and 900 <= gaps[1] <= 1100, gaps
    f = trigger_features(xs, tr[0]["idx"])
    assert f["decay20_ms"] < 15 and f["snr_db"] > 30, f
    assert any(60 <= lag <= 90 for lag, _r in f["secondary"]), f["secondary"]
    for t in tr:
        t.update(trigger_features(xs, t["idx"]))
        t["session"] = "s"
    assert rule_merge(tr, window_s=0.12) == ["kept", "merged", "kept"]
    print("audio_shot_signature selftest OK")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selftest":
        _selftest()
        return 0
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--live-logs":
        rows = live_log_summary(argv[2:])
        Path(argv[1]).write_text("".join(json.dumps(r) + "\n" for r in rows))
        for r in rows:
            print(json.dumps(r))
        return 0
    min_abs = LIVE_ABS
    if argv[0] == "--min-abs":
        min_abs, argv = float(argv[1]), argv[2:]
    out = Path(argv[0])
    sessions = [Path(a) for a in argv[1:]] or sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir())
    out.mkdir(parents=True, exist_ok=False if not os.environ.get("AUDIO_SIG_APPEND") else True)
    feats, summaries = [], []
    for s in sessions:
        trigs, summ = analyse_session(s, min_abs=min_abs)
        print(json.dumps(summ))
        summaries.append(summ)
        feats.extend(trigs)
    with open(out / "features.jsonl", "w") as fh:
        for t in feats:
            fh.write(json.dumps(t) + "\n")
    with open(out / "sessions.jsonl", "w") as fh:
        for s in summaries:
            fh.write(json.dumps(s) + "\n")
    metrics = dict(min_abs=min_abs, rules=evaluate_rules(feats))
    if min_abs == LIVE_ABS:
        metrics["cooldown_sweep"] = [r for s in sessions if load_truth(s)[1].startswith("real")
                                     for r in cooldown_sweep(s)]
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
