from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

from .deltaco_tuya_driver import DeltacoTuyaDriver
from .led_types import LedConnectionConfig, LedMode, RgbColor

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Command:
    name: str
    args: tuple[Any, ...] = ()


class LedService:
    """
    Motorns publika LED-API.

    Viktigt:
    - start() startar bara worker-tråden
    - ingen nätverksanslutning vid programstart
    - anslutning sker först när ett LED-kommando skickas
    - LED-fel ska inte krascha resten av programmet
    """

    def __init__(self) -> None:
        self.config = LedConnectionConfig()
        self.driver = DeltacoTuyaDriver(self.config)

        self._queue: Queue[_Command] = Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_error = ""
        self._lock = threading.Lock()

        self._default_mode: LedMode = self.config.default_mode
        self._default_colour = self.config.default_colour
        self._default_brightness = self.config.default_brightness
        self._default_temperature = self.config.default_temperature

    def configure(self, config: LedConnectionConfig) -> None:
        with self._lock:
            try:
                self.driver.disconnect()
            except Exception:
                pass

            self.config = config
            self.driver = DeltacoTuyaDriver(self.config)
            self._default_mode = config.default_mode
            self._default_colour = config.default_colour
            self._default_brightness = config.default_brightness
            self._default_temperature = config.default_temperature

    def start(self) -> None:
        if self._running:
            return

        self._last_error = ""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        if not self._running:
            try:
                self.driver.disconnect()
            except Exception:
                pass
            return

        self._running = False
        self._queue.put(_Command("stop"))

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        try:
            self.driver.disconnect()
        except Exception:
            pass

    def reload(self, config: LedConnectionConfig) -> None:
        self.configure(config)

    def is_running(self) -> bool:
        return self._running

    def is_available(self) -> bool:
        try:
            return self.driver.is_connected()
        except Exception:
            return False

    def get_last_error(self) -> str:
        return self._last_error

    def turn_on(self) -> None:
        self._queue.put(_Command("turn_on"))

    def turn_off(self) -> None:
        self._queue.put(_Command("turn_off"))

    def show_color(self, color: RgbColor) -> None:
        self._queue.put(_Command("show_color", (color,)))

    def show_white(self, brightness: int = 1000, temperature: int = 500) -> None:
        self._queue.put(_Command("show_white", (brightness, temperature)))

    def flash(self, color: RgbColor, duration_s: float = 0.10) -> None:
        self._queue.put(_Command("flash", (color, float(duration_s))))

    def restore_default(self) -> None:
        self._queue.put(_Command("restore_default"))

    def show_red(self) -> None:
        self.show_color(RgbColor(255, 0, 0))

    def show_green(self) -> None:
        self.show_color(RgbColor(0, 255, 0))

    def show_blue(self) -> None:
        self.show_color(RgbColor(0, 0, 255))

    def _worker(self) -> None:
        while self._running:
            try:
                cmd = self._queue.get(timeout=0.25)
            except Empty:
                continue

            if cmd.name == "stop":
                break

            try:
                self._handle_command(cmd)
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                log.exception("LED command failed: %s", cmd.name)

    def _handle_command(self, cmd: _Command) -> None:
        if not self.config.enabled and cmd.name not in ("turn_off",):
            return

        try:
            if cmd.name == "turn_on":
                self.driver.turn_on()
                return

            if cmd.name == "turn_off":
                self.driver.turn_off()
                return

            if cmd.name == "show_color":
                (color,) = cmd.args
                self.driver.show_color(color)
                return

            if cmd.name == "show_white":
                brightness, temperature = cmd.args
                self.driver.show_white(brightness, temperature)
                return

            if cmd.name == "restore_default":
                self._apply_default()
                return

            if cmd.name == "flash":
                color, duration_s = cmd.args
                self.driver.show_color(color)
                time.sleep(max(0.0, float(duration_s)))
                self._apply_default()
                return
        except Exception as exc:
            self._last_error = str(exc)
            log.exception("LED runtime error")
            try:
                self.driver.disconnect()
            except Exception:
                pass

    def _apply_default(self) -> None:
        try:
            if self._default_mode == "white":
                self.driver.show_white(
                    brightness=self._default_brightness,
                    temperature=self._default_temperature,
                )
            elif self._default_mode == "colour":
                self.driver.show_color(self._default_colour)
            else:
                self.driver.turn_off()
        except Exception as exc:
            self._last_error = str(exc)
            log.exception("LED default apply failed")
            try:
                self.driver.disconnect()
            except Exception:
                pass


led_service = LedService()