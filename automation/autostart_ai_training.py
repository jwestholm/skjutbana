from __future__ import annotations

import argparse
import time
from typing import Any

from src.engine.communication.tcp_network_handler import (
    TcpNetworkError,
    send_command,
)


WINDOW_X = 2130
WINDOW_Y = 50

POLL_INTERVAL_SECONDS = 0.5


BACKGROUND_NAMES = {
    1: "white",
    2: "white_grid",
    3: "gray",
    4: "black",
    5: "checker",
    6: "checker_anim",
    7: "bubbles",
}


def set_window_position() -> None:
    print(
        f"[AUTOMATION] Moving window to "
        f"({WINDOW_X}, {WINDOW_Y})"
    )

    send_command(
        "setWindowPos",
        [
            WINDOW_X,
            WINDOW_Y,
        ],
    )


def start_ai_training(
    background: int | str,
) -> dict[str, Any]:
    print(
        f"[AUTOMATION] Starting AI training "
        f"with background: {background}"
    )

    return send_command(
        "startAITraining",
        [
            background,
        ],
    )


def get_ai_training_status() -> dict[str, Any]:
    response = send_command(
        "getAITrainingStatus"
    )

    data = response.get(
        "data",
        {},
    )

    if not isinstance(
        data,
        dict,
    ):
        return {}

    return data


def wait_for_ai_training() -> dict[str, Any]:
    last_iteration = -1

    print(
        "[AUTOMATION] Waiting for AI training "
        "to finish..."
    )

    while True:
        status = get_ai_training_status()

        state = status.get(
            "state",
            "unknown",
        )

        iteration = int(
            status.get(
                "iteration",
                0,
            )
        )

        target = int(
            status.get(
                "target_iterations",
                0,
            )
        )

        background = status.get(
            "background",
            "unknown",
        )

        if iteration != last_iteration:
            if target > 0:
                print(
                    f"[AI TRAINING] "
                    f"{iteration}/{target} "
                    f"| background={background}"
                )

            last_iteration = iteration

        if state == "completed":
            print(
                "[AUTOMATION] AI training completed."
            )

            return status

        if state == "error":
            raise RuntimeError(
                status.get(
                    "error",
                    "AI training failed",
                )
            )

        if state == "not_running":
            raise RuntimeError(
                "AI training is no longer running "
                "and no completed report is available."
            )

        time.sleep(
            POLL_INTERVAL_SECONDS
        )


def print_report(
    result: dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print("AI TRAINING REPORT")
    print("=" * 70)

    print(
        "Background: "
        f"{result.get('background', 'unknown')}"
    )

    print(
        "Iterations: "
        f"{result.get('iteration', 0)}"
        "/"
        f"{result.get('target_iterations', 0)}"
    )

    report_lines = result.get(
        "report",
        [],
    )

    if (
        isinstance(report_lines, list)
        and report_lines
    ):
        print()

        for line in report_lines:
            print(line)

    print("=" * 70)


def autostart_ai_training(
    background: int | str = 1,
) -> dict[str, Any]:
    """
    Complete automation flow:

        1. Move the running Skjutbana window.
        2. Create a new AITrainingScene.
        3. Select requested background.
        4. Start F2 headless AI training.
        5. Wait until training is complete.
        6. Return the final report.
    """

    set_window_position()

    start_response = start_ai_training(
        background
    )

    start_data = start_response.get(
        "data",
        {},
    )

    if isinstance(
        start_data,
        dict,
    ):
        print(
            "[AUTOMATION] AI training scene started."
        )

        print(
            "[AUTOMATION] Background: "
            f"{start_data.get('background', background)}"
        )

        print(
            "[AUTOMATION] F2 event sent."
        )

    result = wait_for_ai_training()

    print_report(
        result
    )

    return result


def parse_background(
    value: str,
) -> int | str:
    value = value.strip()

    try:
        number = int(
            value
        )

        if number not in BACKGROUND_NAMES:
            raise argparse.ArgumentTypeError(
                "Background number must be 1-7."
            )

        return number

    except ValueError:
        pass

    lowered = value.lower()

    if lowered not in BACKGROUND_NAMES.values():
        valid_names = ", ".join(
            BACKGROUND_NAMES.values()
        )

        raise argparse.ArgumentTypeError(
            "Unknown background. "
            f"Use 1-7 or one of: {valid_names}"
        )

    return lowered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start an automated F2 AI training "
            "run in an already running Skjutbana."
        )
    )

    parser.add_argument(
        "background",
        nargs="?",
        default=1,
        type=parse_background,
        help=(
            "Background number 1-7 or name. "
            "Default: 1 (white)"
        ),
    )

    args = parser.parse_args()

    try:
        autostart_ai_training(
            args.background
        )

    except (
        TcpNetworkError,
        RuntimeError,
    ) as exc:
        print(
            f"ERROR: {exc}"
        )


if __name__ == "__main__":
    main()