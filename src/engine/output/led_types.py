from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LedMode = Literal["off", "white", "colour"]


@dataclass(slots=True, frozen=True)
class RgbColor:
    r: int
    g: int
    b: int

    def clamped(self) -> "RgbColor":
        return RgbColor(
            r=max(0, min(255, int(self.r))),
            g=max(0, min(255, int(self.g))),
            b=max(0, min(255, int(self.b))),
        )

    def as_tuple(self) -> tuple[int, int, int]:
        c = self.clamped()
        return (c.r, c.g, c.b)


@dataclass(slots=True)
class LedConnectionConfig:
    enabled: bool = False
    device_id: str = ""
    ip_address: str = ""
    local_key: str = ""
    version: float = 3.3

    default_mode: LedMode = "white"
    default_brightness: int = 700
    default_temperature: int = 450
    default_colour: RgbColor = RgbColor(255, 255, 255)

    def is_configured(self) -> bool:
        return bool(
            str(self.device_id).strip()
            and str(self.ip_address).strip()
            and str(self.local_key).strip()
        )