from __future__ import annotations

import logging
from typing import Optional

from .led_types import LedConnectionConfig, RgbColor

log = logging.getLogger(__name__)

try:
    import tinytuya
except Exception:
    tinytuya = None


class DeltacoTuyaDriver:
    """
    Robust driver för Deltaco/Tuya LED.

    - Testar flera versioner automatiskt
    - Skyddar mot None-device
    - Ger tydlig debug output
    """

    def __init__(self, config: LedConnectionConfig) -> None:
        self.config = config
        self._device: Optional["tinytuya.BulbDevice"] = None
        self._connected_version: Optional[float] = None

    # ---------------------------------------------------------
    # Connection
    # ---------------------------------------------------------

    def connect(self) -> None:
        if tinytuya is None:
            raise RuntimeError("tinytuya saknas. Kör: pip install tinytuya")

        if not self.config.is_configured():
            raise RuntimeError("LED config saknar device_id / ip / key")

        versions_to_try = [
            float(self.config.version),
            3.3,
            3.4,
            3.5,
        ]

        log.info("LED: försöker ansluta till %s", self.config.ip_address)

        for version in dict.fromkeys(versions_to_try):  # remove duplicates
            try:
                device = tinytuya.BulbDevice(
                    self.config.device_id,
                    self.config.ip_address,
                    self.config.local_key,
                )

                device.set_version(version)
                device.set_socketPersistent(True)

                # 🔥 TESTA anslutning
                status = device.status()

                if status:
                    self._device = device
                    self._connected_version = version
                    log.info("LED: ansluten med version %s", version)
                    return

            except Exception as exc:
                log.warning("LED: version %s failade: %s", version, exc)

        # Om vi hamnar här → inget funkade
        self._device = None
        raise RuntimeError("Kunde inte ansluta till LED (alla versioner misslyckades)")

    def disconnect(self) -> None:
        if self._device is None:
            return

        try:
            self._device.set_socketPersistent(False)
        except Exception:
            log.exception("LED: kunde inte stänga socket")

        self._device = None

    def is_connected(self) -> bool:
        return self._device is not None

    # ---------------------------------------------------------
    # Internal safe call
    # ---------------------------------------------------------

    def _ensure_device(self) -> bool:
        if self._device is None:
            log.warning("LED: device ej ansluten")
            return False
        return True

    # ---------------------------------------------------------
    # Public control
    # ---------------------------------------------------------

    def turn_on(self) -> None:
        if not self._ensure_device():
            return
        try:
            self._device.set_status(True)
        except Exception:
            log.exception("LED turn_on failed")

    def turn_off(self) -> None:
        if not self._ensure_device():
            return
        try:
            self._device.set_status(False)
        except Exception:
            log.exception("LED turn_off failed")

    def show_color(self, color: RgbColor) -> None:
        if not self._ensure_device():
            return

        c = color.clamped()

        try:
            self._device.set_status(True)

            # ⚠️ vissa enheter kräver inte set_mode
            try:
                self._device.set_mode("colour")
            except Exception:
                pass

            self._device.set_colour(c.r, c.g, c.b)

        except Exception:
            log.exception("LED show_color failed")

    def show_white(self, brightness: int, temperature: int) -> None:
        if not self._ensure_device():
            return

        brightness = max(10, min(1000, int(brightness)))
        temperature = max(0, min(1000, int(temperature)))

        try:
            self._device.set_status(True)

            # ⚠️ fallback om set_mode inte finns
            try:
                self._device.set_mode("white")
            except Exception:
                pass

            self._device.set_white(brightness, temperature)

        except Exception:
            log.exception("LED show_white failed")