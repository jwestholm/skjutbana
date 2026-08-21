from __future__ import annotations

import json
import queue
import socket
import threading
from dataclasses import dataclass, field
from typing import Any

from src.engine.events.event_bus import event_bus


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_MESSAGE_BYTES = 64 * 1024


@dataclass
class AutomationCommand:
    """One parsed command received from an automation client."""

    message: dict[str, Any]
    _response_queue: queue.Queue[dict[str, Any]] = field(
        default_factory=lambda: queue.Queue(maxsize=1),
        repr=False,
    )

    @property
    def command(self) -> str:
        return str(self.message.get("command", ""))

    @property
    def args(self) -> Any:
        return self.message.get("args", [])

    def reply_success(self, data: dict[str, Any] | None = None) -> None:
        self._reply(
            {
                "type": "response",
                "command": self.command,
                "success": True,
                "data": data or {},
            }
        )

    def reply_error(self, error: str) -> None:
        self._reply(
            {
                "type": "response",
                "command": self.command,
                "success": False,
                "error": str(error),
            }
        )

    def wait_for_response(self, timeout: float = 5.0) -> dict[str, Any]:
        return self._response_queue.get(timeout=timeout)

    def _reply(self, response: dict[str, Any]) -> None:
        try:
            self._response_queue.put_nowait(response)
        except queue.Full:
            pass


@dataclass(eq=False)
class _EventSubscriber:
    socket: socket.socket
    send_lock: threading.Lock = field(default_factory=threading.Lock)


class CommunicationServer:
    """
    TCP/JSON communication server.

    Command clients send newline-delimited JSON and receive one response.
    Event clients send {"type": "subscribe"} and keep the connection open;
    every EventBus message is then broadcast to those clients.

    Networking threads never execute Pygame/game logic directly. Commands are
    queued and processed by the game main thread via poll_commands().
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._commands: queue.Queue[AutomationCommand] = queue.Queue()
        self._broadcast_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._broadcast_thread: threading.Thread | None = None
        self._subscribers: set[_EventSubscriber] = set()
        self._subscriber_lock = threading.RLock()

    def start(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            return

        self._stop_event.clear()
        event_bus.subscribe(self._queue_event_for_broadcast)

        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="SkjutbanaEventBroadcaster",
            daemon=True,
        )
        self._broadcast_thread.start()

        self._server_thread = threading.Thread(
            target=self._serve,
            name="SkjutbanaCommunicationServer",
            daemon=True,
        )
        self._server_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        event_bus.unsubscribe(self._queue_event_for_broadcast)

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        with self._subscriber_lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()

        for subscriber in subscribers:
            try:
                subscriber.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                subscriber.socket.close()
            except OSError:
                pass

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

        if self._broadcast_thread and self._broadcast_thread.is_alive():
            self._broadcast_thread.join(timeout=1.0)

    def poll_commands(self) -> list[AutomationCommand]:
        commands: list[AutomationCommand] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return commands

    def _queue_event_for_broadcast(self, message: dict[str, Any]) -> None:
        self._broadcast_queue.put(dict(message))

    def _broadcast_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._broadcast_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            with self._subscriber_lock:
                subscribers = list(self._subscribers)

            dead: list[_EventSubscriber] = []
            for subscriber in subscribers:
                try:
                    self._send_json_to_subscriber(subscriber, message)
                except OSError:
                    dead.append(subscriber)

            for subscriber in dead:
                self._remove_subscriber(subscriber)

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(0.5)

        try:
            server.bind((self.host, self.port))
            server.listen()
            self._server_socket = server
            print(f"[Automation] Listening on {self.host}:{self.port}")

            while not self._stop_event.is_set():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                threading.Thread(
                    target=self._handle_client,
                    args=(client, address),
                    name="SkjutbanaAutomationClient",
                    daemon=True,
                ).start()

        except OSError as exc:
            if not self._stop_event.is_set():
                print(f"[Automation] Server error: {exc}")
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
        del address
        subscriber: _EventSubscriber | None = None

        try:
            client.settimeout(1.0)
            buffer = b""

            while not self._stop_event.is_set():
                try:
                    chunk = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    return

                if not chunk:
                    return

                buffer += chunk
                if len(buffer) > MAX_MESSAGE_BYTES:
                    self._send_json(client, {"type": "response", "success": False, "error": "Message too large"})
                    return

                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    if not raw_line.strip():
                        continue

                    try:
                        message = self._decode_message(raw_line)
                    except ValueError as exc:
                        self._send_json(client, {"type": "response", "success": False, "error": str(exc)})
                        continue

                    if message.get("type") == "subscribe":
                        if subscriber is None:
                            subscriber = _EventSubscriber(client)
                            with self._subscriber_lock:
                                self._subscribers.add(subscriber)

                        self._send_json_to_subscriber(
                            subscriber,
                            {
                                "type": "response",
                                "success": True,
                                "subscription": "events",
                            },
                        )
                        continue

                    response = self._process_message(message)
                    self._send_json(client, response)

        finally:
            if subscriber is not None:
                self._remove_subscriber(subscriber)
            else:
                try:
                    client.close()
                except OSError:
                    pass

    def _remove_subscriber(self, subscriber: _EventSubscriber) -> None:
        with self._subscriber_lock:
            self._subscribers.discard(subscriber)
        try:
            subscriber.socket.close()
        except OSError:
            pass

    @staticmethod
    def _decode_message(raw_line: bytes) -> dict[str, Any]:
        try:
            message = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if not isinstance(message, dict):
            raise ValueError("JSON message must be an object")
        return message

    def _process_message(self, message: dict[str, Any]) -> dict[str, Any]:
        message_type = message.get("type", "command")
        if message_type != "command":
            return {
                "type": "response",
                "success": False,
                "error": "Only 'command' and 'subscribe' messages are supported",
            }

        command_name = message.get("command")
        if not isinstance(command_name, str) or not command_name:
            return {
                "type": "response",
                "success": False,
                "error": "Missing or invalid 'command'",
            }

        args = message.get("args", [])
        if not isinstance(args, (list, dict)):
            return {
                "type": "response",
                "command": command_name,
                "success": False,
                "error": "'args' must be either a JSON array or object",
            }

        command = AutomationCommand(message=message)
        self._commands.put(command)

        try:
            return command.wait_for_response(timeout=5.0)
        except queue.Empty:
            return {
                "type": "response",
                "command": command_name,
                "success": False,
                "error": "Game did not process command within 5 seconds",
            }

    @staticmethod
    def _send_json(client: socket.socket, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False) + "\n"
        client.sendall(payload.encode("utf-8"))

    @staticmethod
    def _send_json_to_subscriber(
        subscriber: _EventSubscriber,
        message: dict[str, Any],
    ) -> None:
        payload = json.dumps(message, ensure_ascii=False) + "\n"
        with subscriber.send_lock:
            subscriber.socket.sendall(payload.encode("utf-8"))
