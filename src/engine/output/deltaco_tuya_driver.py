from __future__ import annotations

import logging
from typing import Optional

from .led_types import LedConnectionConfig, RgbColor

log = logging.getLogger(__name__)

try:
    import tinytuya
except Exception:  # pragma: no cover
    tinytuya = None


class DeltacoTuyaDriver:
    """
    Deltaco SH-LS3M via generisk Tuya Device + DPS.

    Viktigt för denna enhet:
    - version 3.3 används
    - färger styrs via DPS
    - anslutning sker bara när ett kommando faktiskt skickas
    """

    DPS_SWITCH = "20"   # switch_led
    DPS_MODE = "21"     # work_mode
    DPS_BRIGHT = "22"   # bright_value
    DPS_TEMP = "23"     # temp_value
    DPS_COLOUR = "24"   # colour_data

    def __init__(self, config: LedConnectionConfig) -> None:
        self.config = config
        self._device: Optional["tinytuya.Device"] = None

    # ---------------------------------------------------------
    # connection
    # ---------------------------------------------------------

    def connect(self) -> None:
        if tinytuya is None:
            raise RuntimeError("tinytuya saknas. Installera med: pip install tinytuya")

        if not self.config.is_configured():
            raise RuntimeError("LED config saknar device_id / ip_address / local_key")

        if self._device is not None:
            return

        version = float(self.config.version or 3.3)

        log.info(
            "LED: ansluter till %s med version %.1f",
            self.config.ip_address,
            version,
        )

        device = tinytuya.Device(
            self.config.device_id,
            self.config.ip_address,
            self.config.local_key,
        )
        device.set_version(version)

        try:
            device.set_socketPersistent(True)
        except Exception:
            pass

        status = device.status()
        if not status:
            raise RuntimeError("Tom status från LED-enheten")

        self._device = device
        log.info("LED: ansluten OK")

    def disconnect(self) -> None:
        if self._device is None:
            return

        try:
            try:
                self._device.set_socketPersistent(False)
            except Exception:
                pass
        finally:
            self._device = None

    def is_connected(self) -> bool:
        return self._device is not None

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    def _ensure_device(self) -> bool:
        if self._device is not None:
            return True

        try:
            self.connect()
            return self._device is not None
        except Exception:
            log.exception("LED: kunde inte ansluta")
            self._device = None
            return False

    def _set_value(self, dps: str, value) -> None:
        if self._device is None:
            return
        self._device.set_value(dps, value)

    @staticmethod
    def _rgb_to_tuya_hsv_hex(color: RgbColor) -> str:
        """
        Tuya colour_data-format: HHHHSSSSVVVV (hex)
        H = 0..360
        S = 0..1000
        V = 0..1000
        """
        c = color.clamped()
        r = c.r / 255.0
        g = c.g / 255.0
        b = c.b / 255.0

        mx = max(r, g, b)
        mn = min(r, g, b)
        diff = mx - mn

        if diff == 0:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360

        s = 0 if mx == 0 else int(round((diff / mx) * 1000))
        v = int(round(mx * 1000))

        h_i = max(0, min(360, int(round(h))))
        s_i = max(0, min(1000, s))
        v_i = max(0, min(1000, v))

        return f"{h_i:04x}{s_i:04x}{v_i:04x}"

    # ---------------------------------------------------------
    # public control
    # ---------------------------------------------------------

    def turn_on(self) -> None:
        if not self._ensure_device():
            return
        try:
            self._set_value(self.DPS_SWITCH, True)
        except Exception:
            log.exception("LED: turn_on misslyckades")
            self._device = None

    def turn_off(self) -> None:
        if not self._ensure_device():
            return
        try:
            self._set_value(self.DPS_SWITCH, False)
        except Exception:
            log.exception("LED: turn_off misslyckades")
            self._device = None

    def show_color(self, color: RgbColor) -> None:
        if not self._ensure_device():
            return

        try:
            colour_data = self._rgb_to_tuya_hsv_hex(color)
            self._set_value(self.DPS_SWITCH, True)
            self._set_value(self.DPS_MODE, "colour")
            self._set_value(self.DPS_COLOUR, colour_data)
        except Exception:
            log.exception("LED: show_color misslyckades")
            self._device = None

    def show_white(self, brightness: int, temperature: int) -> None:
        if not self._ensure_device():
            return

        brightness = max(10, min(1000, int(brightness)))
        temperature = max(0, min(1000, int(temperature)))

        try:
            self._set_value(self.DPS_SWITCH, True)
            self._set_value(self.DPS_MODE, "white")
            self._set_value(self.DPS_BRIGHT, brightness)
            self._set_value(self.DPS_TEMP, temperature)
        except Exception:
            log.exception("LED: show_white misslyckades")
            self._device = None