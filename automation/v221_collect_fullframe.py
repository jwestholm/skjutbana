from __future__ import annotations

"""V2.21.1 short full-frame projector/camera capture.

Starts the existing automation AI-training scene, but arms a one-shot control
file so the scene performs only the requested number of rounds and temporarily
freezes online learning.  Candidate/full-frame capture remains shadow-only.

The running Skjutbana application, projector and camera must already be active.
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from src.engine.communication.tcp_network_handler import (
    EventListener,
    TcpNetworkError,
    send_command,
)
from src.engine.offline.candidate_pack_v216 import CandidateCaptureConfigV216


WINDOW_X = 2130
WINDOW_Y = 50
CAPTURE_CONTROL_PATH = Path("content/ai/v221_capture_control.json")
BENCHMARK_CONTROL_PATH = Path("content/ai/benchmark_control.json")
REPORT_DIR = Path("content/ai/reports/v221")

BACKGROUND_NAMES = {
    1: "white",
    2: "white_grid",
    3: "coord_grid",
    4: "gray",
    5: "black",
    6: "checker",
    7: "checker_anim",
    8: "bubbles",
}


def parse_background(value: str) -> int | str:
    value = value.strip()
    try:
        number = int(value)
    except ValueError:
        number = None
    if number is not None:
        if number not in BACKGROUND_NAMES:
            raise argparse.ArgumentTypeError("Background number must be 1-8")
        return number
    normalized = value.lower()
    if normalized not in BACKGROUND_NAMES.values():
        valid = ", ".join(BACKGROUND_NAMES.values())
        raise argparse.ArgumentTypeError(
            f"Unknown background '{value}'. Use 1-8 or: {valid}"
        )
    return normalized


def _background_name(value: int | str) -> str:
    if isinstance(value, int):
        return BACKGROUND_NAMES.get(value, str(value))
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def _disable_control(path: Path, *, token: str, reason: str) -> None:
    payload: dict[str, Any] = {}
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = dict(raw)
    except Exception:
        payload = {}
    # Never disable another process' freshly armed request.
    existing = str(payload.get("token", ""))
    if existing and existing != token:
        return
    payload.update(
        {
            "enabled": False,
            "token": token,
            "disabled_at": time.time(),
            "disabled_reason": reason,
        }
    )
    _atomic_json(path, payload)


def _write_benchmark_control(seed: int | None) -> None:
    _atomic_json(
        BENCHMARK_CONTROL_PATH,
        {
            "enabled": seed is not None,
            "seed": None if seed is None else int(seed),
            "updated_at": time.time(),
            "purpose": "v221_fullframe_capture",
        },
    )


def _validate_capture_config() -> dict[str, Any]:
    config = CandidateCaptureConfigV216.load()
    problems: list[str] = []
    if not bool(config.enabled):
        problems.append("candidate capture is disabled")
    if not bool(config.save_full_frames):
        problems.append("save_full_frames must be true")
    if not bool(config.save_full_recent_pre):
        problems.append("save_full_recent_pre must be true")
    if int(config.full_frame_post_count) < 1:
        problems.append("full_frame_post_count must be >= 1")
    if problems:
        raise RuntimeError(
            "V2.21.1 full-frame capture config is not ready: " + "; ".join(problems)
        )
    return {
        "data_root": str(config.data_root),
        "save_full_frames": bool(config.save_full_frames),
        "save_full_recent_pre": bool(config.save_full_recent_pre),
        "save_full_reference_pre": bool(config.save_full_reference_pre),
        "full_frame_post_count": int(config.full_frame_post_count),
        "max_post_frames": int(config.max_post_frames),
        "max_candidates": int(config.max_candidates),
        "compress": bool(config.compress),
    }


def collect_fullframe_v221(
    background: int | str = "white",
    *,
    shots: int = 30,
    seed: int | None = 22101,
    freeze_learning: bool = True,
    move_window: bool = True,
) -> dict[str, Any]:
    if shots < 1 or shots > 100:
        raise ValueError("--shots must be in the range 1..100")

    capture_config = _validate_capture_config()
    token = uuid.uuid4().hex[:12]
    started_at = time.time()
    control = {
        "schema_version": "2.21.1",
        "enabled": True,
        "token": token,
        "shots": int(shots),
        "freeze_learning": bool(freeze_learning),
        "purpose": "v221_fullframe_direct",
        "created_at": started_at,
        # One-shot TTL prevents a stale control from changing a later manual run.
        "expires_at": started_at + 2.0 * 3600.0,
    }
    _atomic_json(CAPTURE_CONTROL_PATH, control)
    _write_benchmark_control(seed)

    print("=" * 72)
    print("V2.21.1 FULL-FRAME PROJECTOR/CAMERA CAPTURE")
    print("=" * 72)
    print(f"Background      : {_background_name(background)}")
    print(f"Rounds          : {shots}")
    print(f"Deterministic   : {'random' if seed is None else seed}")
    print(f"Freeze learning : {freeze_learning}")
    print(f"Full recent PRE : {capture_config['save_full_recent_pre']}")
    print(f"Full POST frames: {capture_config['full_frame_post_count']}")
    print("Authority       : shadow/offline only")
    print("=" * 72)

    training_started = False
    completed = False
    started_event: dict[str, Any] | None = None
    completed_event: dict[str, Any] | None = None
    session_summary: dict[str, Any] = {}

    try:
        with EventListener() as listener:
            print("[V2.21.1] Event listener connected")
            if move_window:
                print(f"[V2.21.1] Moving window to ({WINDOW_X}, {WINDOW_Y})")
                send_command("setWindowPos", [WINDOW_X, WINDOW_Y])

            print("[V2.21.1] Opening automation AI-training scene")
            send_command("startAITraining", [background])
            f2_sent = False

            while True:
                event = listener.wait_for_event()
                name = str(event.get("event", ""))
                data = event.get("data", {})
                if not isinstance(data, dict):
                    data = {}

                if name == "aiTraining.started":
                    print(
                        "[V2.21.1] Scene started | "
                        f"background={data.get('background')}"
                    )

                elif name == "aiTraining.calibrationStarted":
                    print(
                        "[V2.21.1] Calibration started | "
                        f"phase={data.get('phase')}"
                    )

                elif name == "aiTraining.calibrationDone":
                    print(
                        "[V2.21.1] Calibration done | "
                        f"attempts={data.get('attempts')}"
                    )

                elif name == "aiTraining.calibrationFailed":
                    raise RuntimeError(
                        "Calibration failed: " + str(data.get("result", "unknown"))
                    )

                elif name == "aiTraining.waitingForFirstShot":
                    if not f2_sent:
                        print("[V2.21.1] Ready - starting short F2 capture")
                        send_command("keyPress", ["F2"])
                        f2_sent = True

                elif name == "aiTraining.trainingStarted":
                    training_started = True
                    started_event = event
                    target = int(data.get("target_iterations", 0) or 0)
                    session_summary = dict(data.get("candidate_capture_v216") or {})
                    applied = dict(data.get("capture_control_v221") or {})
                    print(
                        "[V2.21.1] Capture started | "
                        f"target={target} session={session_summary.get('session_id')}"
                    )
                    if target != int(shots):
                        # Avoid accidentally filling disk with the legacy 100-round run.
                        try:
                            send_command("keyPress", ["F2"])
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Capture control was not applied: target={target}, expected={shots}. "
                            "Do not continue; verify V2.21.1 scene file is installed."
                        )
                    if freeze_learning and not bool(applied.get("freeze_learning", False)):
                        try:
                            send_command("keyPress", ["F2"])
                        except Exception:
                            pass
                        raise RuntimeError("freeze_learning was not acknowledged by the scene")
                    if not bool(session_summary.get("save_full_frames", False)):
                        try:
                            send_command("keyPress", ["F2"])
                        except Exception:
                            pass
                        raise RuntimeError("candidate recorder did not enable full-frame capture")

                elif name == "aiTraining.iterationCompleted":
                    iteration = int(data.get("iteration", 0) or 0)
                    target = int(data.get("target_iterations", shots) or shots)
                    print(f"[V2.21.1] {iteration}/{target}")

                elif name == "aiTraining.trainingStopped":
                    session_summary = dict(data.get("candidate_capture_v216") or {})
                    raise RuntimeError(
                        "Capture stopped before natural completion at iteration "
                        f"{data.get('iteration')}"
                    )

                elif name == "aiTraining.completed":
                    completed = True
                    completed_event = event
                    session_summary = dict(data.get("candidate_capture_v216") or {})
                    break

    except Exception:
        if training_started and not completed:
            # Best-effort cleanup if the listener/validation failed mid-run.
            try:
                send_command("keyPress", ["F2"])
            except Exception:
                pass
        raise
    finally:
        _disable_control(CAPTURE_CONTROL_PATH, token=token, reason="collector_finished")
        # Never let a later manual run inherit our deterministic benchmark seed.
        try:
            _write_benchmark_control(None)
        except Exception:
            pass

    finished_at = time.time()
    data = (completed_event or {}).get("data", {})
    if not isinstance(data, dict):
        data = {}
    iterations = int(data.get("iterations", 0) or 0)
    shots_saved = int(session_summary.get("shots_saved", 0) or 0)
    capture_errors = int(session_summary.get("capture_errors", 0) or 0)

    report = {
        "schema_version": "2.21.1",
        "purpose": "physical/projector-camera full-frame evidence for V2.21 AI_DIRECT",
        "background": _background_name(background),
        "requested_shots": int(shots),
        "iterations": iterations,
        "seed": seed,
        "freeze_learning": bool(freeze_learning),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(finished_at - started_at, 3),
        "capture_config": capture_config,
        "capture_session": session_summary,
        "checks": {
            "natural_completion": bool(completed),
            "iteration_count_matches": iterations == int(shots),
            "shots_saved_matches": shots_saved == int(shots),
            "capture_errors_zero": capture_errors == 0,
            "full_frame_capture_enabled": bool(session_summary.get("save_full_frames", False)),
            "full_frame_post_count_ge_1": int(session_summary.get("full_frame_post_count", 0) or 0) >= 1,
        },
        "training_started_event": started_event,
        "completed_event": completed_event,
        "shadow_only": True,
        "eligible_for_live_authority": False,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"fullframe_capture_v2211_{stamp}.json"
    latest_path = REPORT_DIR / "fullframe_capture_v2211_latest.json"
    _atomic_json(report_path, report)
    _atomic_json(latest_path, report)

    print()
    print("=" * 72)
    print("V2.21.1 CAPTURE COMPLETED")
    print("=" * 72)
    print(f"Iterations : {iterations}/{shots}")
    print(f"Packs saved: {shots_saved}/{shots}")
    print(f"Errors     : {capture_errors}")
    print(f"Session    : {session_summary.get('session_id')}")
    print(f"Root       : {session_summary.get('root')}")
    print(f"Report     : {report_path}")
    print()
    print("NEXT:")
    print("  python3 -m automation.physical_pack_v221_inspect --root content/ai/candidate_shadow_v216")
    print("  python3 -m automation.direct_proposal_v221_benchmark --root content/ai/candidate_shadow_v216")
    print("=" * 72)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a short V2.21.1 full-frame projector/camera F2 session "
            "for offline AI_DIRECT benchmarking"
        )
    )
    parser.add_argument(
        "background",
        nargs="?",
        default="white",
        type=parse_background,
        help="Background 1-8 or name (default: white)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=30,
        help="Number of capture rounds, 1-100 (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22101,
        help="Deterministic synthetic target seed (default: 22101)",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Disable deterministic benchmark seeding for this capture",
    )
    parser.add_argument(
        "--allow-learning",
        action="store_true",
        help="Allow normal online AI learning during capture (default freezes learning)",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Do not move the running game window to the configured projector position",
    )
    args = parser.parse_args()

    try:
        collect_fullframe_v221(
            args.background,
            shots=args.shots,
            seed=None if args.random_seed else args.seed,
            freeze_learning=not args.allow_learning,
            move_window=not args.no_move,
        )
    except (TcpNetworkError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
