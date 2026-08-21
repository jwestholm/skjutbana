from __future__ import annotations

import json
import socket
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 5.0


class TcpNetworkError(Exception):
    """Raised when communication with Skjutbana fails."""


class TcpNetworkHandler:
    """TCP/JSON command client used by external automation programs."""

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
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")

        if args is None:
            args = []
        if not isinstance(args, (list, dict)):
            raise ValueError("args must be a list or dictionary")

        message = {
            "type": "command",
            "command": command,
            "args": args,
        }

        try:
            with socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            ) as sock:
                sock.settimeout(self.timeout)
                self._send_message(sock, message)
                response = self._receive_message(sock)
        except TcpNetworkError:
            raise
        except OSError as exc:
            raise TcpNetworkError(
                f"Could not communicate with Skjutbana: {exc}"
            ) from exc

        if not response.get("success", False):
            raise TcpNetworkError(
                str(response.get("error", "Unknown error from Skjutbana"))
            )
        return response

    @staticmethod
    def _send_message(sock: socket.socket, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False) + "\n"
        sock.sendall(payload.encode("utf-8"))

    @staticmethod
    def _receive_message(sock: socket.socket) -> dict[str, Any]:
        buffer = b""
        while b"\n" not in buffer:
            try:
                chunk = sock.recv(4096)
            except socket.timeout as exc:
                raise TcpNetworkError(
                    "Timed out waiting for response from Skjutbana"
                ) from exc

            if not chunk:
                raise TcpNetworkError(
                    "Skjutbana closed the connection before sending a response"
                )
            buffer += chunk

        raw_response, _remaining = buffer.split(b"\n", 1)
        try:
            response = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TcpNetworkError("Skjutbana returned invalid JSON") from exc

        if not isinstance(response, dict):
            raise TcpNetworkError("Skjutbana returned an invalid response")
        return response


class EventListener:
    """Persistent TCP listener for events broadcast by the running game."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self._socket: socket.socket | None = None
        self._buffer = b""

    def connect(self) -> None:
        if self._socket is not None:
            return

        try:
            sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout,
            )
            sock.settimeout(self.connect_timeout)
            TcpNetworkHandler._send_message(sock, {"type": "subscribe"})
            self._socket = sock

            response = self._receive_next_message()
            if not response.get("success", False):
                raise TcpNetworkError(
                    str(response.get("error", "Event subscription failed"))
                )

            # Event waits should block unless the caller supplies a timeout.
            sock.settimeout(None)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        sock = self._socket
        self._socket = None
        self._buffer = b""
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def wait_for_event(
        self,
        event_name: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._socket is None:
            self.connect()

        assert self._socket is not None
        old_timeout = self._socket.gettimeout()
        self._socket.settimeout(timeout)

        try:
            while True:
                message = self._receive_next_message()
                if message.get("type") != "event":
                    continue
                if event_name is None or message.get("event") == event_name:
                    return message
        except socket.timeout as exc:
            wanted = event_name or "event"
            raise TcpNetworkError(f"Timed out waiting for {wanted}") from exc
        finally:
            if self._socket is not None:
                self._socket.settimeout(old_timeout)

    def _receive_next_message(self) -> dict[str, Any]:
        if self._socket is None:
            raise TcpNetworkError("Event listener is not connected")

        while b"\n" not in self._buffer:
            try:
                chunk = self._socket.recv(4096)
            except socket.timeout:
                raise
            except OSError as exc:
                raise TcpNetworkError(f"Event connection failed: {exc}") from exc

            if not chunk:
                raise TcpNetworkError("Skjutbana closed the event connection")
            self._buffer += chunk

        raw_line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            message = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TcpNetworkError("Skjutbana returned invalid event JSON") from exc

        if not isinstance(message, dict):
            raise TcpNetworkError("Skjutbana returned an invalid event message")
        return message

    def __enter__(self) -> "EventListener":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


_default_handler = TcpNetworkHandler()


def send_command(
    command: str,
    args: Any = None,
) -> dict[str, Any]:
    """Send a command using the default localhost connection."""
    return _default_handler.send_command(command, args)
