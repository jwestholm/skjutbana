from __future__ import annotations

import pygame
from pygame._sdl2.video import Window

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.communication.communication_server import (
    AutomationCommand,
    CommunicationServer,
)
from src.engine.output.led_service import led_service
from src.engine.output.led_types import LedConnectionConfig, RgbColor
from src.engine.scenes.automation_ai_training import AutomationAITrainingScene
from src.engine.scenes.loading import LoadingScene
from src.engine.settings import load_led_settings


AUTOMATION_EVENT = pygame.event.custom_type()


class App:
    def __init__(self) -> None:
        pygame.init()

        self.base_caption = "Skjutbana"
        pygame.display.set_caption(self.base_caption)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Create the SDL2 wrapper once and reuse it for window automation.
        self._automation_window = Window.from_display_module()

        self.clock = pygame.time.Clock()
        self.running = True

        self.communication_server = CommunicationServer()
        self.communication_server.start()

        camera_manager.start()
        audio_peak_detector.start()

        self._init_led_runtime()

        self.scene = LoadingScene()
        self.scene.on_enter()

        self._sync_runtime_services(force=True)
        self._apply_scene_led()
        self._update_window_caption()

    def _init_led_runtime(self) -> None:
        led_data = load_led_settings()
        config = LedConnectionConfig(
            enabled=bool(led_data.get("enabled", False)),
            device_id=str(led_data.get("device_id", "")),
            ip_address=str(led_data.get("ip_address", "")),
            local_key=str(led_data.get("local_key", "")),
            version=float(led_data.get("version", 3.3)),
            default_mode=str(led_data.get("default_mode", "colour")),
            default_brightness=int(led_data.get("default_brightness", 1000)),
            default_temperature=int(led_data.get("default_temperature", 500)),
            default_colour=RgbColor(
                *led_data.get("default_colour", [255, 255, 255])
            ),
        )

        try:
            led_service.configure(config)
            led_service.start()
        except Exception:
            pass

    def _apply_scene_led(self) -> None:
        try:
            scene_led_enabled = bool(
                getattr(self.scene, "scene_led_enabled", False)
            )
            scene_led_color = tuple(
                getattr(self.scene, "scene_led_color", (255, 255, 255))
            )

            if scene_led_enabled:
                led_service.show_color(RgbColor(*scene_led_color))
            else:
                led_service.turn_off()
        except Exception:
            # LED must never be able to crash the app.
            pass

    def quit(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            camera_manager.update()
            audio_peak_detector.update()

            # Convert external TCP/JSON commands into internal Pygame events.
            self._post_automation_events()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    break

                if event.type == AUTOMATION_EVENT:
                    self._handle_automation_event(event)
                    continue

                switch = self.scene.handle_event(event)
                if switch:
                    self._switch_to(switch.new_scene)
                    break

            if not self.running:
                break

            switch = self.scene.update(dt)
            if switch:
                self._switch_to(switch.new_scene)

            hit_scanner.update(dt)
            self._update_window_caption()
            self.scene.render(self.screen)
            pygame.display.flip()

        self.scene.on_exit()
        hit_scanner.disable()

        try:
            led_service.stop()
        except Exception:
            pass

        audio_peak_detector.stop()
        camera_manager.stop()
        self.communication_server.stop()
        pygame.quit()

    # =====================================================
    # AUTOMATION
    # =====================================================

    def _post_automation_events(self) -> None:
        """Move queued TCP commands into Pygame's main-thread event queue."""
        for command in self.communication_server.poll_commands():
            pygame.event.post(
                pygame.event.Event(
                    AUTOMATION_EVENT,
                    automation_command=command,
                )
            )

    def _handle_automation_event(self, event: pygame.event.Event) -> None:
        command: AutomationCommand = event.automation_command

        if command.command == "setWindowPos":
            self._handle_set_window_pos(command)
            return

        if command.command == "startAITraining":
            self._handle_start_ai_training(command)
            return

        if command.command == "keyPress":
            self._handle_key_press(command)
            return

        command.reply_error(f"Unknown command: {command.command}")

    def _handle_set_window_pos(self, command: AutomationCommand) -> None:
        """Handle send_command('setWindowPos', [x, y])."""
        args = command.args

        try:
            if isinstance(args, list):
                if len(args) != 2:
                    raise ValueError("Expected exactly two arguments")
                x = int(args[0])
                y = int(args[1])
            elif isinstance(args, dict):
                x = int(args["x"])
                y = int(args["y"])
            else:
                raise ValueError("Unsupported args format")
        except (KeyError, IndexError, TypeError, ValueError):
            command.reply_error(
                "setWindowPos requires two integer arguments: [x, y]"
            )
            return

        try:
            self._automation_window.position = (x, y)
        except Exception as exc:
            command.reply_error(f"Could not move Pygame window: {exc}")
            return

        print(f"[Automation] setWindowPos({x}, {y})")
        command.reply_success({"x": x, "y": y})

    def _handle_start_ai_training(self, command: AutomationCommand) -> None:
        """
        Create a fresh automation-enabled AI training scene.

        This only opens the scene and starts its normal calibration. F2 is NOT
        sent here. The external automation waits for
        aiTraining.waitingForFirstShot and then sends keyPress(F2).
        """
        args = command.args

        try:
            iterations = None
            if isinstance(args, list):
                if len(args) not in (1, 2):
                    raise ValueError("Expected background and optional iteration count")
                background_value = args[0]
                if len(args) == 2:
                    iterations = int(args[1])
            elif isinstance(args, dict):
                background_value = args["background"]
                if args.get("iterations") is not None:
                    iterations = int(args["iterations"])
            else:
                raise ValueError("Unsupported args format")

            background_index, background_name = self._resolve_ai_background(
                background_value
            )
            if iterations is not None and not 1 <= iterations <= 10000:
                raise ValueError("iterations must be between 1 and 10000")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            command.reply_error(
                "startAITraining requires a valid background and optional iteration count: "
                f"{exc}"
            )
            return

        try:
            new_scene = AutomationAITrainingScene()
            new_scene.bg_mode_index = background_index
            if iterations is not None:
                new_scene.auto_target_iterations = int(iterations)
            self._switch_to(new_scene)
        except Exception as exc:
            command.reply_error(f"Could not start AI training scene: {exc}")
            return

        print(
            f"[Automation] startAITraining({background_name}, "
            f"iterations={int(new_scene.auto_target_iterations)})"
        )
        command.reply_success(
            {
                "background_number": background_index + 1,
                "background": background_name,
                "iterations": int(new_scene.auto_target_iterations),
                "scene_started": True,
            }
        )

    def _handle_key_press(self, command: AutomationCommand) -> None:
        """Inject a normal Pygame KEYDOWN + KEYUP pair into the event queue."""
        args = command.args

        try:
            if isinstance(args, list):
                if len(args) != 1:
                    raise ValueError("Expected one key argument")
                key_name = str(args[0])
            elif isinstance(args, dict):
                key_name = str(args["key"])
            else:
                raise ValueError("Unsupported args format")

            key_code = self._resolve_pygame_key(key_name)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            command.reply_error(f"keyPress requires a valid key name: {exc}")
            return

        pygame.event.post(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=key_code,
                mod=pygame.KMOD_NONE,
                unicode="",
                automation=True,
            )
        )
        pygame.event.post(
            pygame.event.Event(
                pygame.KEYUP,
                key=key_code,
                mod=pygame.KMOD_NONE,
                automation=True,
            )
        )

        print(f"[Automation] keyPress({key_name.upper()})")
        command.reply_success(
            {
                "key": key_name.upper(),
                "key_code": int(key_code),
            }
        )

    @staticmethod
    def _resolve_pygame_key(key_name: str) -> int:
        normalized = key_name.strip().upper()
        key_map = {
            "F1": pygame.K_F1,
            "F2": pygame.K_F2,
            "F3": pygame.K_F3,
            "F4": pygame.K_F4,
            "F5": pygame.K_F5,
            "F6": pygame.K_F6,
            "F7": pygame.K_F7,
            "F8": pygame.K_F8,
            "F9": pygame.K_F9,
            "F10": pygame.K_F10,
            "F11": pygame.K_F11,
            "F12": pygame.K_F12,
            "ESC": pygame.K_ESCAPE,
            "ESCAPE": pygame.K_ESCAPE,
            "ENTER": pygame.K_RETURN,
            "RETURN": pygame.K_RETURN,
            "SPACE": pygame.K_SPACE,
            "TAB": pygame.K_TAB,
            "UP": pygame.K_UP,
            "DOWN": pygame.K_DOWN,
            "LEFT": pygame.K_LEFT,
            "RIGHT": pygame.K_RIGHT,
        }

        if normalized in key_map:
            return key_map[normalized]

        if len(normalized) == 1:
            attr_name = f"K_{normalized.lower()}"
            if hasattr(pygame, attr_name):
                return int(getattr(pygame, attr_name))

        raise ValueError(f"Unknown key '{key_name}'")

    @staticmethod
    def _resolve_ai_background(value) -> tuple[int, str]:
        mode_names = AutomationAITrainingScene.MODE_NAMES

        if isinstance(value, bool):
            raise ValueError("Boolean is not a valid background")

        if isinstance(value, int):
            if value < 1 or value > len(mode_names):
                raise ValueError(f"Background number must be 1-{len(mode_names)}")
            index = value - 1
            return index, mode_names[index]

        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                number = int(normalized)
            except ValueError:
                number = None

            if number is not None:
                return App._resolve_ai_background(number)

            if normalized in mode_names:
                index = mode_names.index(normalized)
                return index, mode_names[index]

        raise ValueError(f"Unknown background '{value}'")

    # =====================================================
    # EXISTING APP FUNCTIONALITY
    # =====================================================

    def _sync_runtime_services(self, force: bool = False) -> None:
        wants_scanning = bool(
            getattr(self.scene, "wants_hit_scanning", False)
        )

        if wants_scanning:
            if force or not hit_scanner.enabled:
                hit_scanner.enable()
        else:
            hit_scanner.disable()

    def _update_window_caption(self) -> None:
        if (
            hit_scanner.enabled
            and hit_scanner.state != hit_scanner.STATE_OFF
        ):
            caption = f"{self.base_caption} (Scanning)"
        else:
            caption = self.base_caption

        pygame.display.set_caption(caption)

    def _switch_to(self, new_scene) -> None:
        self.scene.on_exit()
        self.scene = new_scene
        self.scene.on_enter()
        self._sync_runtime_services(force=True)
        self._apply_scene_led()
        self._update_window_caption()
