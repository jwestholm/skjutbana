from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from automation.ai_training_results import AITrainingRunStore
from automation.autostart_ai_training import (
    WINDOW_X,
    WINDOW_Y,
    autostart_ai_training,
    parse_background,
)
from src.engine.communication.tcp_network_handler import (
    EventListener,
    TcpNetworkError,
    send_command,
)


BENCHMARK_CONTROL_PATH = Path("content/ai/benchmark_control.json")


def _write_benchmark_control(seed: int | None) -> None:
    BENCHMARK_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CONTROL_PATH.write_text(
        json.dumps(
            {
                "enabled": seed is not None,
                "seed": int(seed) if seed is not None else None,
                "updated_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_training_loop(
    background: int | str,
    runs: int,
    *,
    seed: int | None = None,
) -> dict:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    store = AITrainingRunStore(background, runs)
    seed_manifest = {
        "base_seed": seed,
        "background": str(background),
        "runs": runs,
        "run_seeds": [],
        "started_at": time.time(),
    }

    print("=" * 72)
    print("AI TRAINING LOOP")
    print("=" * 72)
    print(f"Background: {background}")
    print(f"Runs: {runs}")
    print(f"Window position: ({WINDOW_X}, {WINDOW_Y})")
    print(f"Results: {store.directory}")
    if seed is None:
        print("Benchmark seed: random")
    else:
        print(f"Benchmark seed: {seed} (run N uses seed + N - 1)")
    print("=" * 72)

    try:
        with EventListener() as listener:
            print("[LOOP] Event listener connected")
            send_command("setWindowPos", [WINDOW_X, WINDOW_Y])

            for run_number in range(1, runs + 1):
                run_seed = int(seed) + run_number - 1 if seed is not None else None
                _write_benchmark_control(run_seed)
                seed_manifest["run_seeds"].append(run_seed)

                print()
                print("-" * 72)
                print(f"RUN {run_number}/{runs}")
                if run_seed is not None:
                    print(f"Deterministic seed: {run_seed}")
                print("-" * 72)

                started = time.time()
                completed_event = autostart_ai_training(
                    background,
                    listener=listener,
                    move_window=False,
                    show_progress=True,
                )
                wall_duration = time.time() - started

                compact = store.save_run(
                    run_number,
                    completed_event,
                    wall_duration_seconds=wall_duration,
                )
                compact["benchmark_seed"] = run_seed

                print(
                    "[LOOP] Saved run "
                    f"{run_number:03d}: "
                    f"found={compact.get('found_pct')}% | "
                    f"AI={compact.get('ai_guess_correct_pct')}% | "
                    f"file={compact.get('file')}"
                )

        summary = store.finalize()
        seed_manifest["completed_at"] = time.time()
        (store.directory / "seed_manifest.json").write_text(
            json.dumps(seed_manifest, indent=2),
            encoding="utf-8",
        )

    except Exception as exc:
        store.mark_failed(str(exc))
        raise

    finally:
        # A later manual F2 run must not accidentally inherit a benchmark seed.
        try:
            _write_benchmark_control(None)
        except Exception:
            pass

    print()
    print("=" * 72)
    print("AI TRAINING LOOP COMPLETED")
    print("=" * 72)
    print(f"Completed runs: {summary.get('completed_runs')}/{runs}")
    print(f"Synthetic shots: {summary.get('total_synthetic_shots')}")

    aggregate = summary.get("aggregate", {})
    print(
        "Aggregate: "
        f"found={aggregate.get('found_pct')}% | "
        f"top1={aggregate.get('top1_pct')}% | "
        f"top3={aggregate.get('top3_pct')}% | "
        f"AI={aggregate.get('ai_guess_correct_pct')}%"
    )

    trend = summary.get("trend", {})
    print(
        "First -> last: "
        f"found delta={trend.get('first_to_last_found_pct_delta')} pp | "
        f"AI delta={trend.get('first_to_last_ai_correct_pct_delta')} pp"
    )
    print(f"Results directory: {store.directory}")
    print(f"Summary: {store.directory / 'summary.json'}")
    if seed is not None:
        print(f"Seed manifest: {store.directory / 'seed_manifest.json'}")
    print("=" * 72)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run many complete automated F2 AI-training sessions"
    )
    parser.add_argument(
        "background",
        type=parse_background,
        help="Background 1-8 or background name",
    )
    parser.add_argument(
        "runs",
        type=int,
        help="Number of complete F2 training runs (100 shots per run today)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional deterministic base seed. Run N uses seed + N - 1. "
            "Use the same seed when comparing detector/ranker versions."
        ),
    )
    args = parser.parse_args()

    try:
        run_training_loop(args.background, args.runs, seed=args.seed)
    except (TcpNetworkError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
