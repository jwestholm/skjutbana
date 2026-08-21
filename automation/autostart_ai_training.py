from __future__ import annotations

import argparse

from src.engine.communication.tcp_network_handler import (
    EventListener,
    TcpNetworkError,
    send_command,
)


WINDOW_X = 2130
WINDOW_Y = 50

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


def autostart_ai_training(background: int | str = 1) -> dict:
    """
    Move the running game window, create a fresh AI training scene, wait for
    calibration, inject F2 through the game's command/event path and return
    the final training result event.
    """

    with EventListener() as listener:
        print(f"[AUTOMATION] Event listener connected")

        print(f"[AUTOMATION] Moving window to ({WINDOW_X}, {WINDOW_Y})")
        send_command("setWindowPos", [WINDOW_X, WINDOW_Y])

        print(f"[AUTOMATION] Starting AI training scene, background={background}")
        send_command("startAITraining", [background])

        f2_sent = False

        while True:
            event = listener.wait_for_event()
            event_name = str(event.get("event", ""))
            data = event.get("data", {})

            if event_name == "aiTraining.started":
                print(
                    "[AI TRAINING] Scene started | "
                    f"background={data.get('background')}"
                )

            elif event_name == "aiTraining.calibrationStarted":
                print(
                    "[AI TRAINING] Starting calibration | "
                    f"phase={data.get('phase')}"
                )

            elif event_name == "aiTraining.calibrationDone":
                print(
                    "[AI TRAINING] Calibration done | "
                    f"attempts={data.get('attempts')}"
                )

            elif event_name == "aiTraining.calibrationFailed":
                raise RuntimeError(
                    "Calibration failed: "
                    f"{data.get('result', 'unknown error')}"
                )

            elif event_name == "aiTraining.waitingForFirstShot":
                print("[AI TRAINING] Ready - sending F2")
                if not f2_sent:
                    send_command("keyPress", ["F2"])
                    f2_sent = True

            elif event_name == "aiTraining.trainingStarted":
                print(
                    "[AI TRAINING] F2 headless training started | "
                    f"target={data.get('target_iterations')}"
                )

            elif event_name == "aiTraining.iterationCompleted":
                iteration = data.get("iteration")
                target = data.get("target_iterations")
                print(f"[AI TRAINING] {iteration}/{target}")

            elif event_name == "aiTraining.trainingStopped":
                raise RuntimeError(
                    "AI training stopped before completion at iteration "
                    f"{data.get('iteration')}"
                )

            elif event_name == "aiTraining.completed":
                print()
                print("=" * 72)
                print("AI TRAINING COMPLETED")
                print("=" * 72)
                print(f"Background: {data.get('background')}")
                print(
                    f"Found: {data.get('found')}/{data.get('iterations')} | "
                    f"Top-1: {data.get('top1')} | "
                    f"Top-3: {data.get('top3')} | "
                    f"AI correct: {data.get('ai_guess_correct')}"
                )

                report = data.get("report", [])
                if isinstance(report, list) and report:
                    print()
                    for line in report:
                        print(line)

                print("=" * 72)
                return event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically start one F2 AI training run"
    )
    parser.add_argument(
        "background",
        nargs="?",
        default=1,
        type=parse_background,
        help="Background 1-8 or background name (default: 1 / white)",
    )
    args = parser.parse_args()

    try:
        autostart_ai_training(args.background)
    except (TcpNetworkError, RuntimeError) as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
