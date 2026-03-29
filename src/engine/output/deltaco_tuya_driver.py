from __future__ import annotations

import logging

from .led_types import LedConnectionConfig, RgbColor

log = logging.getLogger(__name__)

try:
    import tinytuya
except Exception:  # pragma: no cover
    tinytuya = None


class DeltacoTuyaDriver:
    """
    Låg nivå: känner bara till TinyTuya/Tuya.
    Ingen spel- eller skjutbanalogik här.
    """

    def __init__(self, config: LedConnectionConfig) -> None:
        self.config = config
        self._device = None

    def connect(self) -> None:
        if tinytuya is None:
            raise RuntimeError(
                "tinytuya är inte installerat. Installera med: pip install tinytuya"
            )

        if not self.config.is_configured():
            raise RuntimeError("LED är inte komplett konfigurerad.")

        self._device = tinytuya.BulbDevice(
            self.config.device_id,
            self.config.ip_address,
            self.config.local_key,
        )
        self._device.set_version(float(self.config.version))
        self._device.set_socketPersistent(True)

    def disconnect(self) -> None:
        if self._device is None:
            return

        try:
            self._device.set_socketPersistent(False)
        except Exception:
            log.exception("Kunde inte stänga persistent socket för LED.")
        finally:
            self._device = None

    def is_connected(self) -> bool:
        return self._device is not None

    def turn_on(self) -> None:
        if self._device is None:
            return
        self._device.set_status(True)

    def turn_off(self) -> None:
        if self._device is None:
            return
        self._device.set_status(False)

    def show_color(self, color: RgbColor) -> None:
        if self._device is None:
            return

        c = color.clamped()
        self._device.set_status(True)
        self._device.set_mode("colour")
        self._device.set_colour(c.r, c.g, c.b)

    def show_white(self, brightness: int, temperature: int) -> None:
        if self._device is None:
            return

        brightness = max(10, min(1000, int(brightness)))
        temperature = max(0, min(1000, int(temperature)))

        self._device.set_status(True)
        self._device.set_mode("white")
        self._device.set_white(brightness, temperature)