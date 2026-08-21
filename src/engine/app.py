from __future__ import annotations

import pygame
from pygame._sdl2.video import Window

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS

from src.engine.audio.audio_peak_detector import (
    audio_peak_detector,
)
from src.engine.camera.camera_manager import (
    camera_manager,
)
from src.engine.camera.hit_scanner import (
    hit_scanner,
)

from src.engine.communication.communication_server import (
    AutomationCommand,
    CommunicationServer,
)

from src.engine.output.led_service import (
    led_service,
)
from src.engine.output.led_types import (
    LedConnectionConfig,
    RgbColor,
)
from src.engine.scenes.loading import (
    LoadingScene,
)
from src.engine.settings import (
    load_led_settings,
)


AUTOMATION_EVENT = pygame.event.custom_type()


class App:
    def __init__(self) -> None:
        pygame.init()

        self.base_caption = "Skjutbana"

        pygame.display.set_caption(
            self.base_caption
        )

        self.screen = pygame.display.set_mode(
            (
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
            )
        )

        # -------------------------------------------------
        # SDL2 window wrapper
        #
        # Create once and reuse for the entire lifetime
        # of the application.
        # -------------------------------------------------

        self._automation_window = (
            Window.from_display_module()
        )

        self.clock = pygame.time.Clock()
        self.running = True

        # -------------------------------------------------
        # Automation / external communication
        # -------------------------------------------------

        self.communication_server = (
            CommunicationServer()
        )

        self.communication_server.start()

        # -------------------------------------------------
        # Existing runtime services
        # -------------------------------------------------

        camera_manager.start()
        audio_peak_detector.start()

        self._init_led_runtime()

        self.scene = LoadingScene()
        self.scene.on_enter()

        self._sync_runtime_services(
            force=True
        )

        self._apply_scene_led()
        self._update_window_caption()

    def _init_led_runtime(
        self,
    ) -> None:
        led_data = load_led_settings()

        config = LedConnectionConfig(
            enabled=bool(
                led_data.get(
                    "enabled",
                    False,
                )
            ),
            device_id=str(
                led_data.get(
                    "device_id",
                    "",
                )
            ),
            ip_address=str(
                led_data.get(
                    "ip_address",
                    "",
                )
            ),
            local_key=str(
                led_data.get(
                    "local_key",
                    "",
                )
            ),
            version=float(
                led_data.get(
                    "version",
                    3.3,
                )
            ),
            default_mode=str(
                led_data.get(
                    "default_mode",
                    "colour",
                )
            ),
            default_brightness=int(
                led_data.get(
                    "default_brightness",
                    1000,
                )
            ),
            default_temperature=int(
                led_data.get(
                    "default_temperature",
                    500,
                )
            ),
            default_colour=RgbColor(
                *led_data.get(
                    "default_colour",
                    [
                        255,
                        255,
                        255,
                    ],
                )
            ),
        )

        try:
            led_service.configure(
                config
            )

            led_service.start()

        except Exception:
            pass

    def _apply_scene_led(
        self,
    ) -> None:
        try:
            scene_led_enabled = bool(
                getattr(
                    self.scene,
                    "scene_led_enabled",
                    False,
                )
            )

            scene_led_color = tuple(
                getattr(
                    self.scene,
                    "scene_led_color",
                    (
                        255,
                        255,
                        255,
                    ),
                )
            )

            if scene_led_enabled:
                led_service.show_color(
                    RgbColor(
                        *scene_led_color
                    )
                )

            else:
                led_service.turn_off()

        except Exception:
            # LED must never be able to crash the app.
            pass

    def quit(
        self,
    ) -> None:
        self.running = False

    def run(
        self,
    ) -> None:
        while self.running:
            dt = (
                self.clock.tick(
                    FPS
                )
                / 1000.0
            )

            camera_manager.update()
            audio_peak_detector.update()

            #
            # Read external TCP/JSON commands and convert
            # them into internal Pygame automation events.
            #
            self._post_automation_events()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    break

                # -----------------------------------------
                # External automation event
                # -----------------------------------------

                if (
                    event.type
                    == AUTOMATION_EVENT
                ):
                    self._handle_automation_event(
                        event
                    )

                    continue

                # -----------------------------------------
                # Existing scene event handling
                # -----------------------------------------

                switch = (
                    self.scene.handle_event(
                        event
                    )
                )

                if switch:
                    self._switch_to(
                        switch.new_scene
                    )

                    break

            if not self.running:
                break

            switch = self.scene.update(
                dt
            )

            if switch:
                self._switch_to(
                    switch.new_scene
                )

            hit_scanner.update(
                dt
            )

            self._update_window_caption()

            self.scene.render(
                self.screen
            )

            pygame.display.flip()

        # -------------------------------------------------
        # Shutdown
        # -------------------------------------------------

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

    def _post_automation_events(
        self,
    ) -> None:
        """
        Move commands from the TCP communication queue
        into Pygame's event queue.

        This ensures that Pygame-related work happens in
        the game's main thread.
        """

        for command in (
            self.communication_server
            .poll_commands()
        ):
            pygame.event.post(
                pygame.event.Event(
                    AUTOMATION_EVENT,
                    automation_command=command,
                )
            )

    def _handle_automation_event(
        self,
        event: pygame.event.Event,
    ) -> None:
        """
        Handle one external automation command.
        """

        command: AutomationCommand = (
            event.automation_command
        )

        if (
            command.command
            == "setWindowPos"
        ):
            self._handle_set_window_pos(
                command
            )

            return

        command.reply_error(
            "Unknown command: "
            f"{command.command}"
        )

    def _handle_set_window_pos(
        self,
        command: AutomationCommand,
    ) -> None:
        """
        Handle:

            send_command(
                "setWindowPos",
                [x, y],
            )

        The previous dictionary form is also supported:

            {
                "x": x,
                "y": y,
            }
        """

        args = command.args

        try:
            #
            # Preferred syntax:
            #
            # [x, y]
            #
            if isinstance(
                args,
                list,
            ):
                if len(args) != 2:
                    raise ValueError(
                        "Expected exactly "
                        "two arguments"
                    )

                x = int(
                    args[0]
                )

                y = int(
                    args[1]
                )

            #
            # Backwards-compatible syntax:
            #
            # {"x": x, "y": y}
            #
            elif isinstance(
                args,
                dict,
            ):
                x = int(
                    args["x"]
                )

                y = int(
                    args["y"]
                )

            else:
                raise ValueError(
                    "Unsupported args format"
                )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            command.reply_error(
                "setWindowPos requires "
                "two integer arguments: "
                "[x, y]"
            )

            return

        try:
            self._automation_window.position = (
                x,
                y,
            )

        except Exception as exc:
            command.reply_error(
                "Could not move Pygame "
                f"window: {exc}"
            )

            return

        print(
            "[Automation] "
            f"setWindowPos({x}, {y})"
        )

        command.reply_success(
            {
                "x": x,
                "y": y,
            }
        )

    # =====================================================
    # EXISTING APP FUNCTIONALITY
    # =====================================================

    def _sync_runtime_services(
        self,
        force: bool = False,
    ) -> None:
        wants_scanning = bool(
            getattr(
                self.scene,
                "wants_hit_scanning",
                False,
            )
        )

        if wants_scanning:
            if (
                force
                or not hit_scanner.enabled
            ):
                hit_scanner.enable()

        else:
            hit_scanner.disable()

    def _update_window_caption(
        self,
    ) -> None:
        if (
            hit_scanner.enabled
            and hit_scanner.state
            != hit_scanner.STATE_OFF
        ):
            caption = (
                f"{self.base_caption} "
                "(Scanning)"
            )

        else:
            caption = (
                self.base_caption
            )

        pygame.display.set_caption(
            caption
        )

    def _switch_to(
        self,
        new_scene,
    ) -> None:
        self.scene.on_exit()

        self.scene = new_scene

        self.scene.on_enter()

        self._sync_runtime_services(
            force=True
        )

        self._apply_scene_led()

        self._update_window_caption()