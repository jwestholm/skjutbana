"""Deterministic AR-15 style shot sequences (camera frames + PCM + truth).

Physical parameters are ASSUMPTIONS until calibrated against recorded source
sessions; they are stored with every run. Coordinates are canonical camera
pixels. Holes are placed inside the calibrated viewport and sized in mm using
the local calibration scale, so resolution changes are modelled honestly.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCENARIOS = ("single", "double", "string3", "echo", "false_audio", "silence")


@dataclass(frozen=True)
class PhysicsAssumptions:
    fps: float = 25.0
    exposure_s: float = 0.0157          # exposure_time_absolute=157 (x100 us)
    camera_receive_latency_s: float = 0.066  # measured V4L2 -> receive median
    camera_latency_jitter_s: float = 0.003
    shooter_distance_m: float = 6.3     # range_projection.wall_distance_m
    mic_to_board_m: float = 0.5
    bullet_speed_mps: tuple[float, float] = (150.0, 200.0)
    speed_of_sound_mps: float = 343.0
    muzzle_amplitude: float = 0.85
    impact_amplitude: float = 0.65
    noise_rms: float = 0.012
    pulse_decay_s: float = 0.004
    sample_rate: int = 16000
    chunk_samples: int = 256
    pcm_delivery_latency_s: tuple[float, float] = (0.008, 0.020)   # parec
    hole_diameter_mm: tuple[float, float] = (2.0, 3.5)              # AR-15 CO2, assumption
    hole_depth_gray: tuple[float, float] = (25.0, 55.0)   # darkening of a fully covered pixel, assumption
    camera_blur_sigma_px: float = 0.6
    # "disk" = modelled dark disk (subtractive, hole_depth_gray); "stamps" = real-hole
    # stamps (src/engine/offline/hole_stamps.py), multiplicative darkening, airgun holes
    # shrunk by hole_stamp_scale to CO2 AR15/pistol size (50-80 % smaller).
    hole_model: str = "disk"
    hole_stamp_scale: tuple[float, float] = (0.2, 0.5)   # CO2 AR15/pistol: 50-80 % smaller than airgun
    hole_stamp_extra_blur_px: float = 0.4
    hole_stamp_path: str = "/data/skjutbana/hole_library/clean_stamps.npz"
    # Real CO2 AR15 holes as they are: hole_model="stamps", hole_stamp_path=AR15_STAMP_PATH,
    # hole_stamp_scale=(1.0, 1.0), hole_stamp_extra_blur_px=0.0 (see AR15_REAL_STAMPS).
    sensor_noise_sigma: float = 1.5
    jpeg_quality: int | None = 80       # None disables MJPG compression artefacts
    pre_roll_s: float = 2.2
    post_roll_s: float = 4.5
    double_interval_s: tuple[float, float] = (0.15, 0.30)
    string_interval_s: tuple[float, float] = (0.25, 0.50)
    echo_delay_s: tuple[float, float] = (0.08, 0.20)
    echo_gain: tuple[float, float] = (0.6, 0.9)


@dataclass
class Shot:
    fire_t: float
    impact_t: float
    camera_x: float | None
    camera_y: float | None
    diameter_mm: float | None
    kind: str = "impact"      # impact | false_audio


@dataclass
class Scenario:
    name: str
    seed: int
    duration_s: float
    shots: list[Shot] = field(default_factory=list)
    audio_pulses: list[tuple[float, float]] = field(default_factory=list)   # (mic arrival t, amplitude)

    def to_dict(self) -> dict[str, Any]:
        return dict(name=self.name, seed=self.seed, duration_s=self.duration_s,
                    shots=[asdict(s) for s in self.shots], audio_pulses=list(self.audio_pulses))


class ViewportGeometry:
    """Camera-space viewport polygon and local px/mm from saved calibration."""

    def __init__(self, settings: dict[str, Any]) -> None:
        cal = settings["camera_calibration"]
        inv = np.array(cal["inverse_homography"], dtype=np.float64)
        x, y, w, h = settings["viewport"]
        proj = settings.get("range_projection") or {}
        self.width_mm = float(proj.get("viewport_physical_width_cm", 90.0)) * 10.0
        self.height_mm = float(proj.get("viewport_physical_height_cm", 70.0)) * 10.0
        self._inv = inv
        self._vp = (float(x), float(y), float(w), float(h))
        from src.engine import lens_model
        self._lens = lens_model.parse(cal)     # same lens-aware mapping as hit_input

    def to_camera(self, u: float, v: float) -> tuple[float, float]:
        """(u, v) in [0,1]^2 of the viewport -> camera pixels."""
        x, y, w, h = self._vp
        if self._lens is not None:
            from src.engine import lens_model
            return lens_model.screen_to_camera(self._lens, x + u * w, y + v * h)
        pt = np.float64([[[x + u * w, y + v * h]]])
        cam = cv2.perspectiveTransform(pt, self._inv)[0, 0]
        return float(cam[0]), float(cam[1])

    def px_per_mm(self, u: float, v: float) -> float:
        eps = 0.01
        x0 = np.array(self.to_camera(u, v))
        dx = np.linalg.norm(np.array(self.to_camera(min(1, u + eps), v)) - x0) / (eps * self.width_mm)
        dy = np.linalg.norm(np.array(self.to_camera(u, min(1, v + eps))) - x0) / (eps * self.height_mm)
        return float(math.sqrt(dx * dy))   # area-equivalent scale for a round hole


def generate(name: str, seed: int, geometry: ViewportGeometry, phys: PhysicsAssumptions) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name}")
    rng = np.random.default_rng(seed)
    t0 = phys.pre_roll_s + float(rng.uniform(0.0, 0.3))
    fires: list[float] = []
    if name in ("single", "echo", "false_audio"):
        fires = [t0]
    elif name == "double":
        fires = [t0, t0 + float(rng.uniform(*phys.double_interval_s))]
    elif name == "string3":
        fires = [t0]
        for _ in range(2):
            fires.append(fires[-1] + float(rng.uniform(*phys.string_interval_s)))
    scenario = Scenario(name, seed, duration_s=(fires[-1] if fires else t0) + phys.post_roll_s)
    muzzle_delay = (phys.shooter_distance_m - phys.mic_to_board_m) / phys.speed_of_sound_mps
    impact_sound_delay = phys.mic_to_board_m / phys.speed_of_sound_mps
    for fire in fires:
        if name == "false_audio":
            scenario.shots.append(Shot(fire, fire, None, None, None, kind="false_audio"))
            scenario.audio_pulses.append((fire, phys.muzzle_amplitude))
            continue
        flight = phys.shooter_distance_m / float(rng.uniform(*phys.bullet_speed_mps))
        u, v = float(rng.uniform(0.08, 0.92)), float(rng.uniform(0.10, 0.90))
        cx, cy = geometry.to_camera(u, v)
        diameter = float(rng.uniform(*phys.hole_diameter_mm))
        impact = fire + flight
        scenario.shots.append(Shot(fire, impact, cx, cy, diameter))
        scenario.audio_pulses.append((fire + muzzle_delay, phys.muzzle_amplitude))
        scenario.audio_pulses.append((impact + impact_sound_delay, phys.impact_amplitude))
        if name == "echo":
            scenario.audio_pulses.append((fire + muzzle_delay + float(rng.uniform(*phys.echo_delay_s)),
                                          phys.muzzle_amplitude * float(rng.uniform(*phys.echo_gain))))
    return scenario


def synth_pcm(scenario: Scenario, phys: PhysicsAssumptions, seed: int) -> np.ndarray:
    """Mono float PCM in [-1, 1] for the whole scenario."""
    rng = np.random.default_rng(seed + 7_000_003)
    n = int(math.ceil(scenario.duration_s * phys.sample_rate))
    pcm = rng.normal(0.0, phys.noise_rms, n)
    t = np.arange(int(phys.pulse_decay_s * 8 * phys.sample_rate)) / phys.sample_rate
    for arrival, amplitude in scenario.audio_pulses:
        start = int(round(arrival * phys.sample_rate))
        burst = amplitude * np.exp(-t / phys.pulse_decay_s) * np.sign(rng.normal(size=t.size))
        end = min(n, start + burst.size)
        if 0 <= start < n:
            pcm[start:end] += burst[: end - start]
    return np.clip(pcm, -1.0, 1.0)


class FrameRenderer:
    """Renders camera frames: static base + sensor noise + visible holes (+ MJPG).

    A hole is modelled in camera space as a dark disk of the physical diameter
    at the local calibration scale, rendered with 8x supersampled pixel coverage
    and a Gaussian optical blur. ``hole_depth_gray`` is the darkening of a fully
    covered pixel (light through the paper hole, assumption to calibrate).
    """

    SUPERSAMPLE = 8

    def __init__(self, base_gray: np.ndarray, scenario: Scenario, geometry: ViewportGeometry,
                 phys: PhysicsAssumptions, seed: int) -> None:
        self.base = base_gray.astype(np.float32)
        self.phys = phys
        self.rng = np.random.default_rng(seed + 9_000_011)
        # (impact_t, x0, y0, patch, multiplicative): disk patches are grey levels to
        # subtract, stamp patches are relative darkening D (frame *= 1 - D)
        self.holes: list[tuple[float, int, int, np.ndarray, bool]] = []
        self.stamp_ids: list[str] = []
        h, w = base_gray.shape[:2]
        stamps = _stamp_set(phys.hole_stamp_path) if phys.hole_model == "stamps" else None
        for shot in scenario.shots:
            if shot.camera_x is None:
                continue
            if stamps is not None:
                patch, x0, y0, sid = self._stamp_patch(stamps, shot.camera_x, shot.camera_y)
                self.stamp_ids.append(sid)
                multiplicative = True
            else:
                scale = geometry.px_per_mm(0.5, 0.5)
                radius = 0.5 * float(shot.diameter_mm) * scale
                depth = float(self.rng.uniform(*phys.hole_depth_gray))
                patch, x0, y0 = self._hole_patch(shot.camera_x, shot.camera_y, radius, depth)
                multiplicative = False
            if 0 <= x0 and 0 <= y0 and x0 + patch.shape[1] <= w and y0 + patch.shape[0] <= h:
                self.holes.append((shot.impact_t, x0, y0, patch, multiplicative))

    def _stamp_patch(self, stamps, cx: float, cy: float):
        """Real-hole stamp, randomly rotated/mirrored/scaled, placed with sub-pixel centre."""
        scale = float(self.rng.uniform(*self.phys.hole_stamp_scale))
        d, index = stamps.patch(self.rng, scale, extra_blur_px=self.phys.hole_stamp_extra_blur_px)
        r = d.shape[0] // 2
        fx, fy = cx - math.floor(cx), cy - math.floor(cy)
        shift = np.float32([[1, 0, fx - 0.5], [0, 1, fy - 0.5]])
        padded = np.pad(d, 1)
        d = cv2.warpAffine(padded, shift, padded.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=0.0)
        x0, y0 = int(math.floor(cx)) - r - 1, int(math.floor(cy)) - r - 1
        return d.astype(np.float32), x0, y0, stamps.ids[index]

    def _hole_patch(self, cx: float, cy: float, radius: float, depth: float):
        pad = int(math.ceil(radius + 4 * max(self.phys.camera_blur_sigma_px, 0.3))) + 2
        x0, y0 = int(math.floor(cx)) - pad, int(math.floor(cy)) - pad
        size = 2 * pad + 1
        ss = self.SUPERSAMPLE
        coords = (np.arange(size * ss) + 0.5) / ss
        xx, yy = np.meshgrid(coords + x0, coords + y0)
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius).astype(np.float32)
        coverage = disk.reshape(size, ss, size, ss).mean(axis=(1, 3))
        if self.phys.camera_blur_sigma_px > 0:
            coverage = cv2.GaussianBlur(coverage, (0, 0), self.phys.camera_blur_sigma_px)
        return (coverage * depth).astype(np.float32), x0, y0

    def render(self, exposure_end_t: float) -> np.ndarray:
        frame = self.base.copy()
        exposure_start = exposure_end_t - self.phys.exposure_s
        for impact_t, x0, y0, patch, multiplicative in self.holes:
            if impact_t >= exposure_end_t:
                continue
            visible = 1.0 if impact_t <= exposure_start else (exposure_end_t - impact_t) / self.phys.exposure_s
            hh, ww = patch.shape
            if multiplicative:
                frame[y0:y0 + hh, x0:x0 + ww] *= 1.0 - patch * visible
            else:
                frame[y0:y0 + hh, x0:x0 + ww] -= patch * visible
        if self.phys.sensor_noise_sigma > 0:
            frame += self.rng.normal(0.0, self.phys.sensor_noise_sigma, frame.shape).astype(np.float32)
        gray = np.clip(frame, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.phys.jpeg_quality:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(self.phys.jpeg_quality)])
            if ok:
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return bgr


_STAMP_CACHE: dict[str, Any] = {}


def _stamp_set(path: str):
    if path not in _STAMP_CACHE:
        from src.engine.offline.hole_stamps import StampSet
        _STAMP_CACHE[path] = StampSet(path)
    return _STAMP_CACHE[path]


def load_settings(path: Path | str = "content/settings.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["SCENARIOS", "PhysicsAssumptions", "Shot", "Scenario", "ViewportGeometry",
           "generate", "synth_pcm", "FrameRenderer", "load_settings"]


# Real CO2 AR15 holes at true size (same ceiling camera, 2026-11-18 physical session).
AR15_STAMP_PATH = "/data/skjutbana/hole_library/ar15_co2_20261118/clean_stamps.npz"
AR15_REAL_STAMPS = {"hole_model": "stamps", "hole_stamp_path": AR15_STAMP_PATH,
                    "hole_stamp_scale": (0.95, 1.0), "hole_stamp_extra_blur_px": 0.0}  # + random 0-360 rotation and mirror (StampSet.patch)
