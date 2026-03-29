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
    Robust driver för Deltaco/Tuya LED-strip.

    Mål:
    - tåla att enheten tappar anslutning
    - testa flera protokollversioner
    - inte krascha om white-mode saknas eller strular
    - kunna falla tillbaka till RGB-vit när white-mode inte fungerar
    """

    def __init__(self, config: LedConnectionConfig) -> None:
        self.config = config
        self._device: Optional["tinytuya.BulbDevice"] = None
        self._connected_version: Optional[float] = None

    # ---------------------------------------------------------
    # connection
    # ---------------------------------------------------------

    def connect(self) -> None:
        if tinytuya is None:
            raise RuntimeError("tinytuya saknas. Installera med: pip install tinytuya")

        if not self.config.is_configured():
            raise RuntimeError("LED config saknar device_id / ip_address / local_key")

        versions_to_try = [float(self.config.version), 3.3, 3.4, 3.5]
        seen = set()

        last_error: Exception | None = None

        for version in versions_to_try:
            if version in seen:
                continue
            seen.add(version)

            try:
                log.info(
                    "LED: försöker ansluta till %s med version %.1f",
                    self.config.ip_address,
                    version,
                )

                device = tinytuya.BulbDevice(
                    self.config.device_id,
                    self.config.ip_address,
                    self.config.local_key,
                )
                device.set_version(version)
                device.set_socketPersistent(True)

                status = device.status()
                if not status:
                    raise RuntimeError("Tom status från enheten")

                self._device = device
                self._connected_version = version
                log.info("LED: ansluten med version %.1f", version)
                return

            except Exception as exc:
                last_error = exc
                log.warning("LED: version %.1f misslyckades: %s", version, exc)

        self._device = None
        self._connected_version = None
        raise RuntimeError(
            f"Kunde inte ansluta till LED-enheten. Sista fel: {last_error}"
        )

    def disconnect(self) -> None:
        if self._device is None:
            return

        try:
            self._device.set_socketPersistent(False)
        except Exception:
            log.exception("LED: kunde inte stänga persistent socket")
        finally:
            self._device = None
            self._connected_version = None

    def is_connected(self) -> bool:
        return self._device is not None

    # ---------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------

    def _ensure_device(self) -> bool:
        if self._device is not None:
            return True

        try:
            self.connect()
            return self._device is not None
        except Exception:
            log.exception("LED: återanslutning misslyckades")
            self._device = None
            return False

    def _set_mode_if_supported(self, mode: str) -> None:
        if self._device is None:
            return

        try:
            maybe_result = self._device.set_mode(mode)
            # vissa implementationer kan returnera None utan att det är ett problem
            _ = maybe_result
        except AttributeError:
            log.info("LED: set_mode stöds inte av denna enhet, fortsätter ändå")
        except Exception:
            log.exception("LED: set_mode(%s) misslyckades", mode)

    # ---------------------------------------------------------
    # public control
    # ---------------------------------------------------------

    def turn_on(self) -> None:
        if not self._ensure_device():
            return

        try:
            self._device.set_status(True)
        except Exception:
            log.exception("LED: turn_on misslyckades")
            self._device = None

    def turn_off(self) -> None:
        if not self._ensure_device():
            return

        try:
            self._device.set_status(False)
        except Exception:
            log.exception("LED: turn_off misslyckades")
            self._device = None

    def show_color(self, color: RgbColor) -> None:
        if not self._ensure_device():
            return

        c = color.clamped()

        try:
            self._device.set_status(True)
            self._set_mode_if_supported("colour")
            self._device.set_colour(c.r, c.g, c.b)
        except Exception:
            log.exception("LED: show_color misslyckades")
            self._device = None

    def show_white(self, brightness: int, temperature: int) -> None:
        """
        Försöker äkta white-mode först.
        Om enheten inte stöder det eller anslutningen beter sig konstigt,
        faller vi tillbaka till vanlig RGB-vit.
        """
        brightness = max(10, min(1000, int(brightness)))
        temperature = max(0, min(1000, int(temperature)))

        if not self._ensure_device():
            return

        try:
            self._device.set_status(True)
            self._set_mode_if_supported("white")

            # Vissa enheter har inte set_white alls, eller beter sig annorlunda
            set_white_fn = getattr(self._device, "set_white", None)
            if callable(set_white_fn):
                set_white_fn(brightness, temperature)
                return

            log.info("LED: set_white saknas, fallback till RGB-vit")
            self.show_color(RgbColor(255, 255, 255))

        except Exception:
            log.exception("LED: show_white misslyckades, fallback till RGB-vit")
            # Tappa inte hela device direkt; prova fallback
            try:
                self.show_color(RgbColor(255, 255, 255))
            except Exception:
                log.exception("LED: fallback RGB-vit misslyckades")
                self._device = None