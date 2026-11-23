"""Offline bench CLI.

  python3 -m automation.bench synth   --background FRAME --output DIR [--seeds 5] [--variant NAME]
  python3 -m automation.bench compare BASELINE_DIR CANDIDATE_DIR
  python3 -m automation.bench ab      --background FRAME --output DIR --candidate-env KEY=VAL [...]

`ab` runs the same seeds twice in separate processes (baseline env vs candidate
env) and prints BETTER / WORSE / NO_SIGNIFICANT_DIFFERENCE. Output dirs must be new.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_cases(directory: Path) -> tuple[dict, dict]:
    data = json.loads((Path(directory) / "results.json").read_text())
    return data["manifest"], {c["key"]: c for c in data["cases"]}


def compare(baseline: Path, candidate: Path, *, quiet: bool = False) -> dict:
    from src.engine.offline.paired_stats import grouped_compare
    man_a, a = _load_cases(baseline)
    man_b, b = _load_cases(candidate)
    from dataclasses import asdict
    from src.engine.offline.ar15_scenarios import PhysicsAssumptions
    defaults = json.loads(json.dumps(asdict(PhysicsAssumptions())))
    for man in (man_a, man_b):   # runs from before a field existed used its default
        if isinstance(man.get("physics"), dict):
            man["physics"] = {**defaults, **man["physics"]}
    for field in ("evidence_level", "physics", "tolerance_mm", "background_sha256", "seeds", "scenarios",
                  "settings_sha256", "ai_settings_sha256"):
        if man_a.get(field) != man_b.get(field):
            raise SystemExit(f"Refusing to compare: '{field}' differs between runs (not the same experiment)")
    groups = {k: c["scenario"] for k, c in a.items()}
    result = grouped_compare({k: c["correct"] for k, c in a.items()},
                             {k: c["correct"] for k, c in b.items()}, groups)
    result["baseline"] = dict(dir=str(baseline), variant=man_a.get("variant"), env=man_a.get("env"))
    result["candidate"] = dict(dir=str(candidate), variant=man_b.get("variant"), env=man_b.get("env"))
    if not quiet:
        o = result["overall"]
        print(f"\nVERDICT: {o['verdict']}   baseline {o['baseline_correct']}/{o['cases']} "
              f"-> candidate {o['candidate_correct']}/{o['cases']}  (p={o['p_value']:.3g}, "
              f"diff 95% CI [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}])")
        print(f"  fixed: {len(o['fixed'])}  broken: {len(o['broken'])}")
        for name, g in result["groups"].items():
            print(f"  {name:12s} {g['baseline_correct']:>3}/{g['cases']:<3} -> {g['candidate_correct']:>3}/{g['cases']:<3} {g['verdict']}")
    return result


def _synth_cmd(args, output: Path, variant: str) -> list[str]:
    cmd = [sys.executable, "-m", "automation.bench", "synth", "--background", args.background,
           "--output", str(output), "--seeds", str(args.seeds), "--seed-start", str(args.seed_start),
           "--tolerance-mm", str(args.tolerance_mm), "--worker-cost-s", str(args.worker_cost_s),
           "--settings", args.settings, "--variant", variant]
    if args.scenarios:
        cmd += ["--scenarios", args.scenarios]
    if args.physics:
        cmd += ["--physics", args.physics]
    return cmd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m automation.bench")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_synth_args(p):
        p.add_argument("--background", required=True, help="4K camera frame (.npy/.png/.jpg) used as static base")
        p.add_argument("--output", required=True)
        p.add_argument("--seeds", type=int, default=5)
        p.add_argument("--seed-start", type=int, default=1000)
        p.add_argument("--scenarios", default=None, help="comma list; default all")
        p.add_argument("--tolerance-mm", type=float, default=5.0)
        p.add_argument("--worker-cost-s", type=float, default=0.15)
        p.add_argument("--settings", default="content/settings.json")
        p.add_argument("--physics", default=None, help="JSON overrides for PhysicsAssumptions")

    p_synth = sub.add_parser("synth", help="run synthetic AR-15 scenarios through the live shot path")
    add_synth_args(p_synth)
    p_synth.add_argument("--variant", default="baseline")
    p_synth.add_argument("--export-candidates", action="store_true",
                         help="also write labelled candidates.jsonl (training data)")

    p_latency = sub.add_parser("latency-report", help="show observed replay job intervals in milliseconds")
    p_latency.add_argument("replay_dir", metavar="REPLAY_DIR", type=Path,
                           help="output directory from replay --wall-clock-profile")

    p_cmp = sub.add_parser("compare", help="paired comparison of two synth runs")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("candidate")

    p_ab = sub.add_parser("ab", help="run baseline and candidate env on identical seeds, then compare")
    add_synth_args(p_ab)
    p_ab.add_argument("--candidate-env", action="append", default=[], help="KEY=VALUE for the candidate arm")
    p_ab.add_argument("--baseline-env", action="append", default=[], help="KEY=VALUE for the baseline arm")

    p_cal = sub.add_parser("calibration-check", help="validate (or --save) projector/camera calibration", add_help=False)
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "calibration-check":
        from automation.bench.calibration_check import main as calibration_main
        return calibration_main(raw[1:])
    if raw and raw[0] == "evidence-bench":
        from automation.bench.evidence_bench import main as evidence_bench_main
        return evidence_bench_main(raw[1:])
    if raw and raw[0] == "hole-max":
        from automation.bench.hole_likeness import max_main
        return max_main(raw[1:])
    if raw and raw[0] == "shot-timing":
        from automation.bench.shot_timing import main as shot_timing_main
        return shot_timing_main(raw[1:])
    if raw and raw[0] == "hole-likeness-emissions":
        from automation.bench.hole_likeness import emissions_main
        return emissions_main(raw[1:])
    if raw and raw[0] == "hole-likeness-dense":
        from automation.bench.hole_likeness import dense_main
        return dense_main(raw[1:])
    if raw and raw[0] == "hole-likeness":
        from automation.bench.hole_likeness import main as hole_likeness_main
        return hole_likeness_main(raw[1:])
    if raw and raw[0] == "board-snapshot":
        from automation.bench.board_snapshot import main as board_snapshot_main
        return board_snapshot_main(raw[1:])
    if raw and raw[0] == "train":
        from automation.bench.train import main as train_main
        return train_main(raw[1:])
    if raw and raw[0] == "nightly":
        from automation.bench.nightly import main as nightly_main
        return nightly_main(raw[1:])
    if raw and raw[0] == "flash-replay":
        from automation.bench.flash_replay import main as flash_replay_main
        return flash_replay_main(raw[1:])
    if raw and raw[0] == "flash-score":
        from automation.bench.flash_score import main as flash_score_main
        return flash_score_main(raw[1:])
    if raw and raw[0] == "optical-holes":
        from automation.bench.optical_holes import main as optical_main
        return optical_main(raw[1:])
    if raw and raw[0] == "replay":
        rp = argparse.ArgumentParser(prog="python3 -m automation.bench replay")
        rp.add_argument("session")
        rp.add_argument("--output", required=True)
        rp.add_argument("--use-current-settings", action="store_true")
        rp.add_argument("--tolerance-px", type=float, default=8.0)
        rp.add_argument("--worker-cost-s", type=float, default=0.15)
        rp.add_argument("--wall-clock-profile", action="store_true",
                        help="save host-monotonic submission, start and finish times for offline jobs")
        rp.add_argument("--export-candidates", action="store_true")
        rp.add_argument("--pulse-offset-s", type=float, default=0.0,
                        help="shift synthetic_audio pulses (optical rig: ~+0.025)")
        ra = rp.parse_args(raw[1:])
        from automation.bench.session_replay import replay
        from automation import disk_guard
        # disk guard: refuse to start when < estimate + 20 GB would remain (reclaims declared temp first)
        with disk_guard.maybe_experiment(Path(ra.output), estimate_gb=1.5 if ra.export_candidates else 0.2,
                                         label=f"replay {Path(ra.session).name}"):
            result = replay(Path(ra.session), Path(ra.output), use_current_settings=ra.use_current_settings,
                            worker_cost_s=ra.worker_cost_s, tolerance_px=ra.tolerance_px,
                            pulse_offset_s=ra.pulse_offset_s, export_candidates=ra.export_candidates,
                            wall_clock_profile=ra.wall_clock_profile)
        score = result.get("score")
        print(json.dumps(dict(frames=result["frames"], pcm_blocks=result["pcm_blocks"],
                              audio_events=result["audio_events"], emissions=len(result["emissions"]),
                              score=None if not score else {k: score[k] for k in ("cases", "correct", "unmatched_emissions")}),
                         indent=2))
        return 0
    args = parser.parse_args(argv)
    if args.command == "latency-report":
        from automation.bench.latency_report import format_report
        try:
            print(format_report(args.replay_dir))
        except (OSError, ValueError) as exc:
            p_latency.error(str(exc))
        return 0
    if args.command == "synth":
        from automation.bench.synth_runner import run
        from automation import disk_guard
        with disk_guard.maybe_experiment(Path(args.output), estimate_gb=1.0, label=f"synth {args.variant}"):
            run(args)
        return 0
    if args.command == "compare":
        result = compare(Path(args.baseline), Path(args.candidate))
        return 0 if result["overall"]["verdict"] != "WORSE" else 2
    if args.command == "ab":
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        for arm, extra in (("baseline", args.baseline_env), ("candidate", args.candidate_env)):
            env = {k: v for k, v in os.environ.items() if not k.startswith("SKJUTBANA_")}
            for item in extra:
                key, _, value = item.partition("=")
                env[key] = value
            print(f"[bench ab] running {arm} {extra}", flush=True)
            subprocess.run(_synth_cmd(args, out / arm, arm), env=env, check=True)
        result = compare(out / "baseline", out / "candidate")
        (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["overall"]["verdict"] != "WORSE" else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
