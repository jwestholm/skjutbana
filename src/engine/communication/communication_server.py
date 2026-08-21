from __future__ import annotations

import json
import queue
import socket
import threading
from dataclasses import dataclass, field
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_MESSAGE_BYTES = 64 * 1024


@dataclass
class AutomationCommand:
    """
    One parsed command received from an automation client.

    The communication thread receives the JSON message and places an
    AutomationCommand in the game's command queue.

    The game processes the command in its main thread and then replies
    through this object's response queue.
    """

    message: dict[str, Any]

    _response_queue: queue.Queue[dict[str, Any]] = field(
        default_factory=lambda: queue.Queue(maxsize=1),
        repr=False,
    )

    @property
    def command(self) -> str:
        return str(
            self.message.get(
                "command",
                "",
            )
        )

    @property
    def args(self) -> Any:
        return self.message.get(
            "args",
            [],
        )

    def reply_success(
        self,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._reply(
            {
                "type": "response",
                "command": self.command,
                "success": True,
                "data": data or {},
            }
        )

    def reply_error(
        self,
        error: str,
    ) -> None:
        self._reply(
            {
                "type": "response",
                "command": self.command,
                "success": False,
                "error": str(error),
            }
        )

    def wait_for_response(
        self,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        return self._response_queue.get(
            timeout=timeout
        )

    def _reply(
        self,
        response: dict[str, Any],
    ) -> None:
        try:
            self._response_queue.put_nowait(
                response
            )

        except queue.Full:
            pass


class CommunicationServer:
    """
    Minimal TCP/JSON communication server.

    Protocol:
        - TCP
        - localhost by default
        - one JSON object per line
        - UTF-8 encoding

    The networking thread never executes game or Pygame logic directly.

    Received commands are placed in a thread-safe queue. The game's
    main thread retrieves them with poll_commands().
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.host = host
        self.port = int(port)

        self._commands: queue.Queue[
            AutomationCommand
        ] = queue.Queue()

        self._stop_event = (
            threading.Event()
        )

        self._server_socket: (
            socket.socket | None
        ) = None

        self._server_thread: (
            threading.Thread | None
        ) = None

    def start(self) -> None:
        """
        Start the TCP server in a background thread.
        """

        if (
            self._server_thread
            and self._server_thread.is_alive()
        ):
            return

        self._stop_event.clear()

        self._server_thread = threading.Thread(
            target=self._serve,
            name="SkjutbanaCommunicationServer",
            daemon=True,
        )

        self._server_thread.start()

    def stop(self) -> None:
        """
        Stop the TCP server.
        """

        self._stop_event.set()

        if self._server_socket is not None:
            try:
                self._server_socket.close()

            except OSError:
                pass

            self._server_socket = None

        if (
            self._server_thread
            and self._server_thread.is_alive()
        ):
            self._server_thread.join(
                timeout=1.0
            )

    def poll_commands(
        self,
    ) -> list[AutomationCommand]:
        """
        Return all currently waiting commands.

        This function is intended to be called from the game's main
        thread.
        """

        commands: list[
            AutomationCommand
        ] = []

        while True:
            try:
                commands.append(
                    self._commands.get_nowait()
                )

            except queue.Empty:
                break

        return commands

    def _serve(self) -> None:
        """
        Main communication server loop.
        """

        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.settimeout(0.5)

        try:
            server.bind(
                (
                    self.host,
                    self.port,
                )
            )

            server.listen()

            self._server_socket = server

            print(
                "[Automation] Listening on "
                f"{self.host}:{self.port}"
            )

            while not (
                self._stop_event.is_set()
            ):
                try:
                    client, address = (
                        server.accept()
                    )

                except socket.timeout:
                    continue

                except OSError:
                    break

                threading.Thread(
                    target=self._handle_client,
                    args=(
                        client,
                        address,
                    ),
                    name=(
                        "SkjutbanaAutomationClient"
                    ),
                    daemon=True,
                ).start()

        except OSError as exc:
            if not (
                self._stop_event.is_set()
            ):
                print(
                    "[Automation] "
                    f"Server error: {exc}"
                )

        finally:
            try:
                server.close()

            except OSError:
                pass

            self._server_socket = None

    def _handle_client(
        self,
        client: socket.socket,
        address: tuple[str, int],
    ) -> None:
        """
        Handle one connected automation client.
        """

        del address

        with client:
            client.settimeout(10.0)

            buffer = b""

            while not (
                self._stop_event.is_set()
            ):
                try:
                    chunk = client.recv(
                        4096
                    )

                except socket.timeout:
                    continue

                except OSError:
                    return

                if not chunk:
                    return

                buffer += chunk

                if (
                    len(buffer)
                    > MAX_MESSAGE_BYTES
                ):
                    self._send_json(
                        client,
                        {
                            "type": "response",
                            "success": False,
                            "error": (
                                "Message too large"
                            ),
                        },
                    )

                    return

                while b"\n" in buffer:
                    (
                        raw_line,
                        buffer,
                    ) = buffer.split(
                        b"\n",
                        1,
                    )

                    if not raw_line.strip():
                        continue

                    response = (
                        self._process_line(
                            raw_line
                        )
                    )

                    self._send_json(
                        client,
                        response,
                    )

    def _process_line(
        self,
        raw_line: bytes,
    ) -> dict[str, Any]:
        """
        Parse one JSON command and wait for the game to process it.
        """

        try:
            message = json.loads(
                raw_line.decode(
                    "utf-8"
                )
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            return {
                "type": "response",
                "success": False,
                "error": (
                    f"Invalid JSON: {exc}"
                ),
            }

        if not isinstance(
            message,
            dict,
        ):
            return {
                "type": "response",
                "success": False,
                "error": (
                    "JSON message must "
                    "be an object"
                ),
            }

        message_type = message.get(
            "type",
            "command",
        )

        if message_type != "command":
            return {
                "type": "response",
                "success": False,
                "error": (
                    "Only messages of type "
                    "'command' are supported"
                ),
            }

        command_name = message.get(
            "command"
        )

        if (
            not isinstance(
                command_name,
                str,
            )
            or not command_name
        ):
            return {
                "type": "response",
                "success": False,
                "error": (
                    "Missing or invalid "
                    "'command'"
                ),
            }

        #
        # args intentionally supports both:
        #
        #   [1, 2, 3]
        #
        # and:
        #
        #   {"x": 1, "y": 2}
        #
        # This keeps the communication engine generic.
        #
        args = message.get(
            "args",
            [],
        )

        if not isinstance(
            args,
            (
                list,
                dict,
            ),
        ):
            return {
                "type": "response",
                "command": command_name,
                "success": False,
                "error": (
                    "'args' must be either "
                    "a JSON array or object"
                ),
            }

        command = AutomationCommand(
            message=message
        )

        self._commands.put(
            command
        )

        try:
            return (
                command.wait_for_response(
                    timeout=5.0
                )
            )

        except queue.Empty:
            return {
                "type": "response",
                "command": command_name,
                "success": False,
                "error": (
                    "Game did not process "
                    "command within 5 seconds"
                ),
            }

    @staticmethod
    def _send_json(
        client: socket.socket,
        message: dict[str, Any],
    ) -> None:
        """
        Send one newline-terminated JSON message.
        """

        try:
            payload = (
                json.dumps(
                    message,
                    ensure_ascii=False,
                )
                + "\n"
            )

            client.sendall(
                payload.encode(
                    "utf-8"
                )
            )

        except OSError:
            pass