from __future__ import annotations

from dataclasses import dataclass, field

import cv2


def _safe_get(cap: cv2.VideoCapture, prop: int, default=0.0):
    try:
        value = cap.get(prop)
        if value is None:
            return default
        return value
    except Exception:
        return default


def _safe_set(cap: cv2.VideoCapture, prop: int, value) -> bool:
    try:
        return bool(cap.set(prop, value))
    except Exception:
        return False


def _decode_fourcc(value: float) -> str:
    try:
        iv = int(value)
        chars = [
            chr(iv & 0xFF),
            chr((iv >> 8) & 0xFF),
            chr((iv >> 16) & 0xFF),
            chr((iv >> 24) & 0xFF),
        ]
        text = "".join(chars).strip("\x00").strip()
        return text or "unknown"
    except Exception:
        return "unknown"


@dataclass
class CameraCapabilities:
    backend_name: str = "unknown"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = "unknown"
    auto_exposure_value: float | None = None
    exposure_value: float | None = None
    auto_wb_value: float | None = None
    focus_value: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Backend: {self.backend_name}",
            f"Capture: {self.width}x{self.height} @ {self.fps:.1f} fps",
            f"FOURCC: {self.fourcc}",
        ]
        if self.auto_exposure_value is not None:
            lines.append(f"Auto exposure: {self.auto_exposure_value}")
        if self.exposure_value is not None:
            lines.append(f"Exposure: {self.exposure_value}")
        if self.auto_wb_value is not None:
            lines.append(f"Auto white balance: {self.auto_wb_value}")
        if self.focus_value is not None:
            lines.append(f"Focus: {self.focus_value}")
        return lines


def probe_camera_capabilities(cap: cv2.VideoCapture) -> CameraCapabilities:
    backend_name = "unknown"
    try:
        if hasattr(cap, "getBackendName"):
            backend_name = cap.getBackendName() or "unknown"
    except Exception:
        backend_name = "unknown"

    caps = CameraCapabilities(
        backend_name=backend_name,
        width=int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH, 0)),
        height=int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT, 0)),
        fps=float(_safe_get(cap, cv2.CAP_PROP_FPS, 0.0)),
        fourcc=_decode_fourcc(_safe_get(cap, cv2.CAP_PROP_FOURCC, 0.0)),
        auto_exposure_value=float(_safe_get(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)),
        exposure_value=float(_safe_get(cap, cv2.CAP_PROP_EXPOSURE, 0.0)),
        auto_wb_value=float(_safe_get(cap, cv2.CAP_PROP_AUTO_WB, 0.0)),
        focus_value=float(_safe_get(cap, cv2.CAP_PROP_FOCUS, 0.0)),
    )

    interesting = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "gain": cv2.CAP_PROP_GAIN,
        "sharpness": getattr(cv2, "CAP_PROP_SHARPNESS", -1),
        "zoom": cv2.CAP_PROP_ZOOM,
    }

    for name, prop in interesting.items():
        if prop < 0:
            continue
        caps.extra[name] = float(_safe_get(cap, prop, 0.0))

    return caps


def apply_preferred_camera_settings(
    cap: cv2.VideoCapture,
    preferred_width: int = 3840,
    preferred_height: int = 2160,
    preferred_fps: int = 30,
) -> dict[str, bool]:
    """
    Best effort.
    Vi försöker köra kameran i 3840x2160 först.
    Om kameran/backend inte stöder det kommer faktiskt negotiated läge
    att synas via probe_camera_capabilities().
    """
    results: dict[str, bool] = {}

    try:
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        results["fourcc_mjpg"] = _safe_set(cap, cv2.CAP_PROP_FOURCC, mjpg)
    except Exception:
        results["fourcc_mjpg"] = False

    results["width"] = _safe_set(cap, cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
    results["height"] = _safe_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)
    results["fps"] = _safe_set(cap, cv2.CAP_PROP_FPS, preferred_fps)

    results["auto_wb_off"] = _safe_set(cap, cv2.CAP_PROP_AUTO_WB, 0)

    autofocus_prop = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
    if autofocus_prop is not None:
        results["autofocus_off"] = _safe_set(cap, autofocus_prop, 0)
    else:
        results["autofocus_off"] = False

    results["focus"] = _safe_set(cap, cv2.CAP_PROP_FOCUS, 0)

    ok_manual = _safe_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    ok_manual_alt = _safe_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    results["auto_exposure_manual"] = ok_manual or ok_manual_alt

    return results