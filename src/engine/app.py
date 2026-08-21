from __future__ import annotations

import pygame
from pygame._sdl2.video import Window

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
)

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

from src.engine.scenes.ai_training import (
    AITrainingScene,
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

    # =====================================================
    # LED
    # =====================================================

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

    # =====================================================
    # APPLICATION
    # =====================================================

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

            # -------------------------------------------------
            # Receive external automation commands.
            # -------------------------------------------------

            self._post_automation_events()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    break

                # ---------------------------------------------
                # External automation command
                # ---------------------------------------------

                if (
                    event.type
                    == AUTOMATION_EVENT
                ):
                    self._handle_automation_event(
                        event
                    )

                    continue

                # ---------------------------------------------
                # Existing scene event handling
                # ---------------------------------------------

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
        Move commands received by the TCP thread into
        Pygame's event queue.

        All actual game/Pygame operations therefore happen
        in the application's main thread.
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
        Dispatch one automation command.
        """

        command: AutomationCommand = (
            event.automation_command
        )

        # -------------------------------------------------
        # setWindowPos
        # -------------------------------------------------

        if (
            command.command
            == "setWindowPos"
        ):
            self._handle_set_window_pos(
                command
            )

            return

        # -------------------------------------------------
        # startAITraining
        # -------------------------------------------------

        if (
            command.command
            == "startAITraining"
        ):
            self._handle_start_ai_training(
                command
            )

            return

        # -------------------------------------------------
        # getAITrainingStatus
        # -------------------------------------------------

        if (
            command.command
            == "getAITrainingStatus"
        ):
            self._handle_get_ai_training_status(
                command
            )

            return

        command.reply_error(
            "Unknown command: "
            f"{command.command}"
        )

    # =====================================================
    # AUTOMATION: WINDOW
    # =====================================================

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

        Dictionary arguments are also supported:

            {
                "x": x,
                "y": y,
            }
        """

        args = command.args

        try:
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
    # AUTOMATION: AI TRAINING
    # =====================================================

    def _resolve_ai_background(
        self,
        value,
    ) -> tuple[int, str]:
        """
        Convert an external background value into the
        AITrainingScene's zero-based bg_mode_index.

        External integer values are intentionally 1-based:

            1 = white
            2 = white_grid
            3 = gray
            4 = black
            5 = checker
            6 = checker_anim
            7 = bubbles

        Background names are also accepted.
        """

        mode_names = (
            AITrainingScene.MODE_NAMES
        )

        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                "Invalid background"
            )

        if isinstance(
            value,
            int,
        ):
            external_number = value

            if (
                external_number < 1
                or external_number
                > len(mode_names)
            ):
                raise ValueError(
                    "Background number must "
                    f"be 1-{len(mode_names)}"
                )

            index = (
                external_number - 1
            )

            return (
                index,
                mode_names[index],
            )

        if isinstance(
            value,
            str,
        ):
            value = value.strip()

            # Allow strings such as "1".
            try:
                external_number = int(
                    value
                )

            except ValueError:
                external_number = None

            if external_number is not None:
                return (
                    self._resolve_ai_background(
                        external_number
                    )
                )

            normalized = (
                value.lower()
            )

            if normalized in mode_names:
                index = (
                    mode_names.index(
                        normalized
                    )
                )

                return (
                    index,
                    mode_names[index],
                )

        raise ValueError(
            "Unknown AI training background"
        )

    def _handle_start_ai_training(
        self,
        command: AutomationCommand,
    ) -> None:
        """
        Create a completely new AITrainingScene,
        set its background and then inject a real
        Pygame F2 KEYDOWN event.

        Examples:

            send_command(
                "startAITraining",
                [1],
            )

            send_command(
                "startAITraining",
                ["checker"],
            )
        """

        args = command.args

        try:
            if isinstance(
                args,
                list,
            ):
                if len(args) != 1:
                    raise ValueError(
                        "startAITraining expects "
                        "one argument"
                    )

                background_value = (
                    args[0]
                )

            elif isinstance(
                args,
                dict,
            ):
                background_value = (
                    args["background"]
                )

            else:
                raise ValueError(
                    "Unsupported args format"
                )

            (
                background_index,
                background_name,
            ) = self._resolve_ai_background(
                background_value
            )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            command.reply_error(
                "startAITraining requires "
                "a background number 1-7 "
                "or background name. "
                f"Error: {exc}"
            )

            return

        try:
            # ---------------------------------------------
            # Always create a NEW training scene instance.
            # ---------------------------------------------

            new_scene = AITrainingScene()

            new_scene.bg_mode_index = (
                background_index
            )

            self._switch_to(
                new_scene
            )

            # ---------------------------------------------
            # Start the existing F2 code path.
            #
            # We deliberately send a real Pygame KEYDOWN
            # event instead of calling the training
            # implementation directly.
            #
            # AITrainingScene.handle_event() will therefore
            # handle this exactly like a physical F2 press.
            # ---------------------------------------------

            pygame.event.post(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    key=pygame.K_F2,
                    mod=pygame.KMOD_NONE,
                    unicode="",
                    automation=True,
                )
            )

        except Exception as exc:
            command.reply_error(
                "Could not start AI training: "
                f"{exc}"
            )

            return

        print(
            "[Automation] "
            "startAITraining("
            f"{background_name})"
        )

        print(
            "[Automation] "
            "Posted F2 KEYDOWN event"
        )

        command.reply_success(
            {
                "background_number": (
                    background_index + 1
                ),
                "background": (
                    background_name
                ),
                "f2_event_posted": True,
            }
        )

    def _handle_get_ai_training_status(
        self,
        command: AutomationCommand,
    ) -> None:
        """
        Return the current state of the active
        AITrainingScene.

        This command does not manipulate training.
        It only reports state.
        """

        scene = self.scene

        if not isinstance(
            scene,
            AITrainingScene,
        ):
            command.reply_success(
                {
                    "state": "not_running",
                    "running": False,
                    "completed": False,
                    "iteration": 0,
                    "target_iterations": 0,
                    "background": None,
                    "report": [],
                }
            )

            return

        running = bool(
            scene.auto_training_enabled
        )

        report_visible = bool(
            scene.auto_report_visible
        )

        report_lines = list(
            scene.auto_report_lines
        )

        try:
            background_name = (
                scene.MODE_NAMES[
                    scene.bg_mode_index
                ]
            )

        except (
            IndexError,
            TypeError,
        ):
            background_name = "unknown"

        # A finished F1/F2 training run builds the
        # auto report and makes it visible.
        completed = bool(
            not running
            and report_visible
            and report_lines
        )

        if completed:
            state = "completed"

        elif running:
            state = "running"

        else:
            state = "starting"

        command.reply_success(
            {
                "state": state,
                "running": running,
                "completed": completed,
                "iteration": int(
                    scene.auto_iteration
                ),
                "target_iterations": int(
                    scene.auto_target_iterations
                ),
                "background": (
                    background_name
                ),
                "headless": bool(
                    scene.auto_headless
                ),
                "phase": str(
                    scene.auto_phase
                ),
                "report_visible": (
                    report_visible
                ),
                "report": (
                    report_lines
                ),
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