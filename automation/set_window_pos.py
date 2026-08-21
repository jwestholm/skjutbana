from src.engine.communication.tcp_network_handler import (
    TcpNetworkError,
    send_command,
)


WINDOW_X = 2130
WINDOW_Y = 50


def main() -> None:
    try:
        response = send_command(
            "setWindowPos",
            [
                WINDOW_X,
                WINDOW_Y,
            ],
        )

        print(
            "setWindowPos successful: "
            f"({WINDOW_X}, {WINDOW_Y})"
        )

        print(
            f"Response: {response}"
        )

    except TcpNetworkError as exc:
        print(
            f"ERROR: {exc}"
        )


if __name__ == "__main__":
    main()