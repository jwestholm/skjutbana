from __future__ import annotations

import json
import socket


HOST = "127.0.0.1"
PORT = 8765


# -------------------------------------------------
# CHANGE THESE VALUES WHEN TESTING
# -------------------------------------------------

WINDOW_X = 100
WINDOW_Y = 100


def send_command(message: dict) -> dict:
    with socket.create_connection(
        (HOST, PORT),
        timeout=5.0,
    ) as sock:

        payload = json.dumps(message) + "\n"

        sock.sendall(
            payload.encode("utf-8")
        )

        buffer = b""

        while b"\n" not in buffer:
            chunk = sock.recv(4096)

            if not chunk:
                raise ConnectionError(
                    "Skjutbana closed the connection "
                    "before replying"
                )

            buffer += chunk

    raw_response, _remainder = buffer.split(
        b"\n",
        1,
    )

    return json.loads(
        raw_response.decode("utf-8")
    )


def main() -> None:
    command = {
        "type": "command",
        "command": "setWindowPos",
        "args": {
            "x": WINDOW_X,
            "y": WINDOW_Y,
        },
    }

    print(
        f"Sending setWindowPos("
        f"{WINDOW_X}, {WINDOW_Y})..."
    )

    try:
        response = send_command(command)

    except (
        OSError,
        ConnectionError,
        json.JSONDecodeError,
    ) as exc:

        print(
            "ERROR: Could not communicate "
            f"with Skjutbana: {exc}"
        )

        return

    if response.get("success"):
        data = response.get("data", {})

        print(
            "OK: Skjutbana moved window to "
            f"({data.get('x')}, "
            f"{data.get('y')})"
        )

    else:
        print(
            "ERROR: "
            f"{response.get('error', 'Unknown error')}"
        )


if __name__ == "__main__":
    main()