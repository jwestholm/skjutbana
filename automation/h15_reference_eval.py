"""H15-REF-1: frozen reference decision rule for the H15 dense late search, and the comparison harness that
every later H15 change is measured against (old chain kept as the second reference).

Rule H15-REF-1 (live rule in src/engine/shot_dense_h15.py, frozen 2026-11-20):
    top-1 MF peak accepted iff  s >= 16  and  s1/s2 >= 1.15  and  core <= 0.4  and  bipolar <= 1.0
    no fallback to the old chain, causal windows (PRE [peak-0.4, peak-0.02], POST [peak+0.15, submit]).

Input: replay run dirs produced with SKJUTBANA_H15_LOG (one ``h15_<session>.jsonl`` and one
``shadow_<session>/{replay.json,candidates.jsonl}`` per session).  Truth is used ONLY here, out-of-band.

Per run and split (DEV = sessions the rule was designed on, TODAY = held-out projected sessions,
NEW = anything else, e.g. future recordings), it reports:
    correct, wrong (accepted, > TOL px from truth), abstain, missed (= wrong + abstain),
    false alarms (acceptance on audio events without an impact), precision, and the old chain.
Physical and projected sessions are reported separately.

    python3 automation/h15_reference_eval.py RUN_DIR [CHALLENGER_RUN_DIR ...]
A challenger run is e.g. the same sessions replayed with SKJUTBANA_H15_MIN_PRE_DN=16 (dark mask).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

TOL = float(os.environ.get("EVAL_TOL", 8.0))
REF = dict(name="H15-REF-1", s_min=16.0, ratio_min=1.15, core_max=0.4, bipolar_max=1.0)
DEV = {"20261119T134333Z_4f975191", "20261119T133608Z_161c19db", "20261119T122749Z_7a0d5e51",
       "20261118T162349Z_66d659ce", "20261118T155321Z_04f1be81", "20261118T135339Z_da06f87a_ar15cut"}
TODAY_PREFIX = "20261120T"
PHYS = {"20261118T155321Z_04f1be81", "20261118T135339Z_da06f87a_ar15cut"}


def split(sess: str) -> str:
    return "DEV" if sess in DEV else ("TODAY" if sess.startswith(TODAY_PREFIX) else "NEW")


def load(R: Path, sess: str):
    rep = json.load(open(R / f"shadow_{sess}" / "replay.json"))
    off = rep["pulse_offset_s"]
    jobs = {}
    p = R / f"h15_{sess}.jsonl"
    for line in (open(p) if p.exists() else []):
        r = json.loads(line)
        if r["kind"] == "job":
            jobs[(r["shot_id"], r["peak_ts"])] = r
    shots, used = [], set()
    for t in rep["truth"]:
        if t.get("camera_x") is None:
            continue
        tp = t["wall_ts"] + off
        near = sorted((abs(k[1] - tp), k) for k in jobs if abs(k[1] - tp) <= 0.05)
        job = None
        for _d, k in near:                      # one job matches at most one truth shot
            if k in used or jobs[k].get("status") == "co2_pair_follower":
                continue
            job = jobs[k]; used.add(k); break
        if job is None:
            for _d, k in near:
                if k not in used:
                    used.add(k); break
        shots.append(dict(gt=(t["camera_x"], t["camera_y"]), job=job,
                          physical=t.get("background") is None or sess in PHYS))
    extra = [jobs[k] for k in jobs if k not in used and jobs[k].get("status") == "ok"]
    return rep, shots, extra


def ref_rule(job, rule=REF):
    if not job or job.get("status") != "ok" or not job.get("peaks"):
        return None
    pk = job["peaks"]
    x, y, s, core, bip = pk[0][:5]
    s2 = pk[1][2] if len(pk) > 1 else 0.0
    ok = (s >= rule["s_min"] and (s2 <= 0 or s / s2 >= rule["ratio_min"]) and core <= rule["core_max"]
          and bip <= rule["bipolar_max"])
    return (x, y) if ok else None


def hit(p, gt) -> bool:
    return math.hypot(p[0] - gt[0], p[1] - gt[1]) <= TOL


def evaluate(R: Path) -> dict:
    sessions = sorted(p.name[7:] for p in R.glob("shadow_*") if (p / "replay.json").exists())
    out = {}
    for sess in sessions:
        rep, shots, extra = load(R, sess)
        for sh in shots:
            key = (split(sess), "physical" if sh["physical"] else "projected")
            a = out.setdefault(key, dict(n=0, correct=0, wrong=0, abstain=0, false_alarm=0, chain_correct=0,
                                         chain_unmatched=0, sessions=set()))
            a["sessions"].add(sess)
            a["n"] += 1
            d = ref_rule(sh["job"])
            if d is None:
                a["abstain"] += 1
            elif hit(d, sh["gt"]):
                a["correct"] += 1
            else:
                a["wrong"] += 1
        for j in extra:
            key = (split(sess), "physical" if sess in PHYS else "projected")
            if key in out:
                out[key]["false_alarm"] += ref_rule(j) is not None
        key = (split(sess), "physical" if sess in PHYS else "projected")
        if key in out and sess not in out[key].get("_chain_done", set()):
            sc = rep.get("score", {})
            out[key]["chain_correct"] += sc.get("correct", 0)
            out[key]["chain_unmatched"] += sc.get("unmatched_emissions", 0)
            out[key].setdefault("_chain_done", set()).add(sess)
    return out


def report(R: Path) -> None:
    print(f"== {R} ({REF['name']}, TOL {TOL} px)")
    for (sp, kind), a in sorted(evaluate(R).items()):
        n = max(a["n"], 1)
        acc = a["correct"] + a["wrong"] + a["false_alarm"]
        print(f"  {sp:5s} {kind:9s} sessions {len(a['sessions']):2d} shots {a['n']:4d} | H15: correct {a['correct']:4d} "
              f"({100 * a['correct'] / n:5.1f}%) wrong {a['wrong']:3d} false_alarm {a['false_alarm']:3d} "
              f"abstain {a['abstain']:4d} missed {a['wrong'] + a['abstain']:4d} precision "
              f"{100 * a['correct'] / max(acc, 1):5.1f}% | old chain: correct {a['chain_correct']:4d} "
              f"unmatched emissions {a['chain_unmatched']:4d}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(Path(arg))
