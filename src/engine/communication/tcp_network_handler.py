from __future__ import annotations

import json
import socket
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 5.0


class TcpNetworkError(Exception):
    """
    Raised when communication with Skjutbana fails.
    """


class TcpNetworkHandler:
    """
    TCP/JSON client used by external automation programs.

    The caller only needs to know the command name and arguments.

    Example:

        network = TcpNetworkHandler()

        network.send_command(
            "setWindowPos",
            [2150, 700],
        )
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def send_command(
        self,
        command: str,
        args: Any = None,
    ) -> dict[str, Any]:
        """
        Send one command to the running Skjutbana application.

        args may be either a list or dictionary.

        Examples:

            send_command(
                "setWindowPos",
                [2150, 700],
            )

            send_command(
                "someFutureCommand",
                {
                    "value": 123,
                },
            )
        """

        if not isinstance(
            command,
            str,
        ) or not command:
            raise ValueError(
                "command must be "
                "a non-empty string"
            )

        if args is None:
            args = []

        if not isinstance(
            args,
            (
                list,
                dict,
            ),
        ):
            raise ValueError(
                "args must be "
                "a list or dictionary"
            )

        message = {
            "type": "command",
            "command": command,
            "args": args,
        }

        try:
            with socket.create_connection(
                (
                    self.host,
                    self.port,
                ),
                timeout=self.timeout,
            ) as sock:
                sock.settimeout(
                    self.timeout
                )

                self._send_message(
                    sock,
                    message,
                )

                response = (
                    self._receive_message(
                        sock
                    )
                )

        except TcpNetworkError:
            raise

        except OSError as exc:
            raise TcpNetworkError(
                "Could not communicate "
                "with Skjutbana: "
                f"{exc}"
            ) from exc

        if not response.get(
            "success",
            False,
        ):
            raise TcpNetworkError(
                str(
                    response.get(
                        "error",
                        (
                            "Unknown error "
                            "from Skjutbana"
                        ),
                    )
                )
            )

        return response

    @staticmethod
    def _send_message(
        sock: socket.socket,
        message: dict[str, Any],
    ) -> None:
        """
        Send one JSON message.
        """

        payload = (
            json.dumps(
                message,
                ensure_ascii=False,
            )
            + "\n"
        )

        sock.sendall(
            payload.encode(
                "utf-8"
            )
        )

    @staticmethod
    def _receive_message(
        sock: socket.socket,
    ) -> dict[str, Any]:
        """
        Receive one newline-terminated JSON response.
        """

        buffer = b""

        while b"\n" not in buffer:
            try:
                chunk = sock.recv(
                    4096
                )

            except socket.timeout as exc:
                raise TcpNetworkError(
                    "Timed out waiting "
                    "for response from "
                    "Skjutbana"
                ) from exc

            if not chunk:
                raise TcpNetworkError(
                    "Skjutbana closed the "
                    "connection before "
                    "sending a response"
                )

            buffer += chunk

        (
            raw_response,
            _remaining,
        ) = buffer.split(
            b"\n",
            1,
        )

        try:
            response = json.loads(
                raw_response.decode(
                    "utf-8"
                )
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise TcpNetworkError(
                "Skjutbana returned "
                "invalid JSON"
            ) from exc

        if not isinstance(
            response,
            dict,
        ):
            raise TcpNetworkError(
                "Skjutbana returned "
                "an invalid response"
            )

        return response


#
# Default instance.
#
# Most automation scripts do not need to create their own
# TcpNetworkHandler. They can simply import send_command().
#
_default_handler = TcpNetworkHandler()


def send_command(
    command: str,
    args: Any = None,
) -> dict[str, Any]:
    """
    Send a command using the default localhost connection.

    Example:

        send_command(
            "setWindowPos",
            [2150, 700],
        )
    """

    return (
        _default_handler.send_command(
            command,
            args,
        )
    )