"""Record a projected-hole session: real optics/camera/MJPG, known truth.

  python3 -m automation.bench optical-holes --background gray|pattern|moving|IMAGE [--shots 12]

Shows a background in the game window position on the projector, records a
continuous SOURCE session (raw MJPG + PCM), and draws small dark dots with
hole-like diameter that persist (like real holes) in singles, doubles and
rapid strings. Writes ground_truth.jsonl (calibration-predicted camera XY) and
synthetic_audio.jsonl (a pulse at each dot's flip time) into the session dir,
so `bench replay` runs it through the live shot path.

OPTICAL != PROJECTILE: projected dots are not paper damage. This measures the
optical/camera/detector/timing chain, never physical hit accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path


class _Background:
    """Precomputed backgrounds: the render loop must stay as light as the game's."""

    def __init__(self, kind: str, size):
        import numpy as np
        import pygame
        self.kind, self.size = kind, size
        w, h = size
        if kind == "gray":
            self.surface = pygame.Surface(size)
            self.surface.fill((200, 200, 200))
        elif kind == "dark":                 # dark game scene: holes nearly invisible
            self.surface = pygame.Surface(size)
            self.surface.fill((35, 35, 40))
        elif kind in ("pattern", "moving"):
            ww = 2 * w if kind == "moving" else w
            xx, yy = np.meshgrid(np.arange(ww), np.arange(h))
            val = 150 + 60 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
            rgb = np.dstack([val, val * 0.95 + 8, val * 0.85 + 20]).clip(0, 255).astype(np.uint8)
            self.surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1)).convert()
        else:
            self.surface = pygame.transform.smoothscale(pygame.image.load(kind).convert(), size)

    def blit(self, screen, t: float) -> None:
        if self.kind == "moving":
            offset = int(t * 40.0) % self.size[0]          # 40 screen px/s scroll
            screen.blit(self.surface, (-offset, 0))
        else:
            screen.blit(self.surface, (0, 0))


_STAMP_SPRITES: dict = {}


def _stamp_sprite(hole):
    """Black sprite with alpha = real-stamp D * optical gain (multiplicative darkening, like
    the AI test's real_stamp kinds; same sprite code as SyntheticHoleOverlay)."""
    import pygame
    key = hole["sprite_key"]
    if key not in _STAMP_SPRITES:
        from src.engine.synthetic.synthetic_hole_overlay import SyntheticHole, SyntheticHoleOverlay
        hole_spec = SyntheticHole(hole_id="h", x=0.0, y=0.0, radius_px=hole["r"], kind=hole["kind"],
                                  rotation_deg=hole["angle"], strength=hole["gain"], opacity=1.0,
                                  shadow_offset=(0.0, 0.0), seed=hole["seed"])
        rgba = SyntheticHoleOverlay(8, 8)._build_real_stamp_sprite(hole_spec)
        n = rgba.shape[0]
        arr = rgba.copy()
        arr[..., :3] = 0
        surf = pygame.image.frombuffer(arr.tobytes(), (n, n), "RGBA").copy()
        _STAMP_SPRITES[key] = surf
    return _STAMP_SPRITES[key]


def _draw_holes(screen, holes) -> None:
    """Holes persist like paper damage and darken whatever is projected there (also the flash)."""
    import pygame
    w, h = screen.get_size()
    for hole in holes:
        cx, cy = int(round(hole["x"])), int(round(hole["y"]))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        if hole.get("kind"):
            sprite = _stamp_sprite(hole)
            screen.blit(sprite, (cx - sprite.get_width() // 2, cy - sprite.get_height() // 2))
            continue
        base = screen.get_at((cx, cy))
        colour = tuple(int(c * hole["factor"]) for c in base[:3])
        r, elong = hole["r"], hole["elong"]
        if elong <= 1.01:
            pygame.draw.circle(screen, colour, (cx, cy), max(1, int(round(r))))
        else:
            dx = math.cos(hole["angle"]) * r * elong
            dy = math.sin(hole["angle"]) * r * elong
            pygame.draw.line(screen, colour, (cx - dx, cy - dy), (cx + dx, cy + dy), max(1, int(round(r * 1.4))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m automation.bench optical-holes")
    parser.add_argument("--background", default="gray", help="gray | pattern | moving | path to image")
    parser.add_argument("--shots", type=int, default=12)
    parser.add_argument("--hole-mm", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--root", default="/data/skjutbana/source_sessions")
    parser.add_argument("--freeze-on-shot-ms", type=float, default=0.0,
                        help="model a game that pauses its animation this long after each shot "
                             "(Skyttegrav pauses via begin_demo_shot); 0 = never freeze")
    parser.add_argument("--freeze-delay-ms", type=float, default=40.0,
                        help="delay from shot to the game actually freezing (dispatch + IPC + frame)")
    parser.add_argument("--hole-style", choices=("dot", "varied", "stamps_ar15", "stamps_airgun"), default="dot",
                        help="dot = black disc (2026-11-17 sessions); varied = size/elongation/orientation vary and "
                             "the hole only darkens what is projected there (factor 0.55-0.85, like real holes on "
                             "a lit board). A projector cannot darken ambient light, so on black backgrounds the "
                             "varied hole is still weaker than a real one.")
    parser.add_argument("--hole-factor-range", default="0.55,0.85",
                        help="varied holes: projected light kept in the hole (1 - darkening), uniform in lo,hi")
    parser.add_argument("--flash", action="store_true",
                        help="run the engine precision flash (src/engine/precision_flash.py) in the loop; "
                             "dots stay dark under the flash like real holes")
    args = parser.parse_args(argv)

    os.environ["SKJUTBANA_SOURCE_RECORD"] = "1"
    os.environ["SKJUTBANA_SOURCE_RECORD_ROOT"] = args.root
    import cv2
    import numpy as np
    import pygame
    from pygame._sdl2.video import Window
    from config import SCREEN_HEIGHT, SCREEN_WIDTH
    from src.engine.audio.audio_peak_detector import audio_peak_detector
    from src.engine.camera.camera_manager import camera_manager
    from src.engine.settings import load_camera_calibration, load_viewport_rect, load_range_projection_settings
    from src.engine.source_recording import get_source_recorder

    calibration = load_camera_calibration() or {}
    setup = calibration.get("setup_at_calibration") or {}
    pos = (setup.get("window") or {}).get("sdl_desktop_position") or [0, 0]
    inv = np.array(calibration["inverse_homography"], dtype=np.float64)
    vp = load_viewport_rect()
    try:
        width_mm = float(load_range_projection_settings().get("viewport_physical_width_cm", 90.0)) * 10.0
    except Exception:
        width_mm = 900.0
    radius_px = max(1, int(round(0.5 * args.hole_mm * vp.w / width_mm)))

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    Window.from_display_module().position = (int(pos[0]), int(pos[1]))
    pygame.display.set_caption("Skjutbana optical holes")
    if not camera_manager.start():
        print(f"Camera unavailable: {camera_manager.last_error}")
        return 2
    audio_peak_detector.start()
    recorder = get_source_recorder()

    background = _Background(args.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
    flash, flash_hits = None, []
    if args.flash:
        from src.engine.precision_flash import FlashConfig, PrecisionFlash

        class _ExactScene:
            hit_precision = "exact"

        flash = PrecisionFlash(FlashConfig.from_settings(), camera=camera_manager,
                               emit=lambda x, y: flash_hits.append(dict(wall_ts=time.time(), camera_x=float(x),
                                                                        camera_y=float(y))))
        flash.sync_policy(_ExactScene())
    clock = pygame.time.Clock()
    rng = random.Random(args.seed)
    pattern, shots_left = [], args.shots
    while shots_left > 0:          # singles, doubles, strings of three
        kind = rng.choice(("single", "double", "string3"))
        n = min(shots_left, dict(single=1, double=2, string3=3)[kind])
        gaps = [0.0] + [rng.uniform(0.15, 0.30) if kind == "double" else rng.uniform(0.25, 0.5) for _ in range(n - 1)]
        pattern.append(gaps)
        shots_left -= n
    holes, truth, pulses = [], [], []
    start = time.monotonic()
    schedule, t = [], args.warmup_s
    for group in pattern:
        for gap in group:
            t += gap
            schedule.append(t)
        t += 2.5
    end_t = t + 2.0 + (2.0 if args.flash else 0.0)
    next_index = 0
    anim_t = 0.0             # animation clock (stops while the game is frozen)
    last_now = 0.0
    freeze_from = freeze_until = -1.0
    while time.monotonic() - start < end_t:
        pygame.event.pump()
        now = time.monotonic() - start
        if not (freeze_from <= now < freeze_until):
            anim_t += now - last_now
        last_now = now
        background.blit(screen, anim_t)
        if flash is not None:
            flash.tick()
            flash.render(screen)
        new_hole = None
        if next_index < len(schedule) and now >= schedule[next_index]:
            gx = vp.x + rng.uniform(0.08, 0.92) * vp.w
            gy = vp.y + rng.uniform(0.10, 0.90) * vp.h
            new_hole = (gx, gy)
            if args.hole_style.startswith("stamps_"):
                # Real hole stamps exactly as the AI test projects them (profile co2 / airgun).
                from src.engine.scenes.ai_training_content import PROJECTED_PROFILE
                from src.engine.synthetic.synthetic_hole_overlay import STAMP_NOMINAL_RADIUS_CAM_PX
                from src.engine.scenes.ai_training_content import AITrainingContent
                prof = PROJECTED_PROFILE["co2" if args.hole_style == "stamps_ar15" else "airgun"]
                k = AITrainingContent.camera_px_per_screen_px(type("C", (), {"_cam_per_screen": None})())
                r = STAMP_NOMINAL_RADIUS_CAM_PX * rng.uniform(*prof["scale"]) / k
                seed = rng.randrange(2 ** 31)
                holes.append(dict(x=gx, y=gy, kind=prof["kind"], r=r, gain=rng.uniform(*prof["gain"]),
                                  angle=rng.uniform(0, 360), seed=seed, sprite_key=seed))
            elif args.hole_style == "varied":
                size_mm = rng.uniform(0.8, 1.5) * args.hole_mm
                elong = rng.choice((1.0, 1.0, 1.4, 2.2))          # some tears/oblique hits
                f_lo, f_hi = (float(v) for v in args.hole_factor_range.split(","))
                holes.append(dict(x=gx, y=gy, factor=rng.uniform(f_lo, f_hi),
                                  r=max(0.6, 0.5 * size_mm * vp.w / width_mm), elong=elong,
                                  angle=rng.uniform(0, math.pi)))
            else:
                holes.append(dict(x=gx, y=gy, factor=0.0, r=radius_px, elong=1.0, angle=0.0))
            next_index += 1
            if args.freeze_on_shot_ms > 0:
                freeze_from = now + args.freeze_delay_ms / 1000.0
                freeze_until = freeze_from + args.freeze_on_shot_ms / 1000.0
        _draw_holes(screen, holes)
        pygame.display.flip()
        if new_hole is not None:
            flip_wall = time.time()
            from src.engine.input.hit_input import hit_input
            cam_xy = hit_input.screen_to_camera(*new_hole)   # lens-aware canonical mapping
            truth.append(dict(wall_ts=flip_wall, camera_x=float(cam_xy[0]), camera_y=float(cam_xy[1]),
                              screen_xy=list(new_hole), truth_source="CALIBRATION_PREDICTED_PROJECTED_DOT"))
            pulses.append(dict(wall_ts=flip_wall, amplitude=0.85))
            if flash is not None:
                flash.on_peak(flip_wall)
        camera_manager.update()
        clock.tick(60)   # same pacing as the game loop (config.FPS)
    if flash is not None:                      # let the last measurement finish
        deadline = time.monotonic() + 3.0
        while flash.state != flash.IDLE and time.monotonic() < deadline:
            background.blit(screen, anim_t)
            flash.tick()
            flash.render(screen)
            _draw_holes(screen, holes)
            pygame.display.flip()
            camera_manager.update()
            clock.tick(60)
    audio_peak_detector.stop()
    camera_manager.stop()
    pygame.quit()
    if recorder is None or recorder.directory is None:
        print("Recording did not start")
        return 2
    directory = recorder.directory
    recorder.stop("optical_holes_done")
    (directory / "ground_truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in truth))
    (directory / "synthetic_audio.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pulses))
    if flash is not None:
        (directory / "flash_hits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in flash_hits))
        (directory / "flash_results.jsonl").write_text("".join(json.dumps(r, default=float) + "\n"
                                                              for r in flash.results))
    (directory / "optical_holes.json").write_text(json.dumps(dict(
        flash=bool(args.flash), flash_count=(flash.flash_count if flash else 0),
        background=args.background, hole_mm=args.hole_mm, freeze_on_shot_ms=args.freeze_on_shot_ms,
        freeze_delay_ms=args.freeze_delay_ms, radius_screen_px=radius_px, hole_style=args.hole_style,
        holes=holes, seed=args.seed,
        shots=len(truth), window_position=pos, disclaimer="OPTICAL != PROJECTILE"), indent=2))
    print(json.dumps(dict(session=str(directory), shots=len(truth), radius_screen_px=radius_px), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
