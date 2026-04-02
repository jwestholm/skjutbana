import pygame

class VisualHitsSettingsScene:
    def __init__(self, engine):
        self.engine = engine
        self.settings = engine.settings

        self.index = 0

        self.options = [
            "show_hits",
            "show_all_planes",
            "fade_time",
            "persistent"
        ]

    def _get_items(self):
        return [
            ("Visa träffar", self.settings.load_visual_hits_enabled()),
            ("Visa träff i alla plan", self.settings.load_visual_hits_show_all_planes()),
            ("Fade tid", round(self.settings.load_visual_hits_fade_time(), 1)),
            ("Persistent", self.settings.load_visual_hits_persistent()),
        ]

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN:
            return

        if event.key == pygame.K_UP:
            self.index = (self.index - 1) % len(self.options)

        elif event.key == pygame.K_DOWN:
            self.index = (self.index + 1) % len(self.options)

        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN, pygame.K_SPACE):
            self._modify_current(event.key)

        elif event.key == pygame.K_ESCAPE:
            self._save()
            self.engine.pop_scene()

    def _modify_current(self, key):
        option = self.options[self.index]

        if option == "show_hits":
            val = not self.settings.load_visual_hits_enabled()
            self.settings.save_visual_hits_enabled(val)

        elif option == "show_all_planes":
            val = not self.settings.load_visual_hits_show_all_planes()
            self.settings.save_visual_hits_show_all_planes(val)

        elif option == "persistent":
            val = not self.settings.load_visual_hits_persistent()
            self.settings.save_visual_hits_persistent(val)

        elif option == "fade_time":
            val = self.settings.load_visual_hits_fade_time()
            if key == pygame.K_LEFT:
                val = max(0.1, val - 0.5)
            else:
                val = min(10.0, val + 0.5)
            self.settings.save_visual_hits_fade_time(val)

    def _save(self):
        self.settings.save()

    def update(self, dt):
        pass

    def draw(self, screen):
        screen.fill((30, 40, 45))

        font = pygame.font.SysFont(None, 36)
        small = pygame.font.SysFont(None, 24)

        # Titel
        title = font.render("Visuella träffar", True, (255, 255, 255))
        screen.blit(title, (50, 40))

        items = self._get_items()

        y = 120
        for i, (label, value) in enumerate(items):
            selected = i == self.index

            color = (255, 255, 0) if selected else (200, 200, 200)

            prefix = "> " if selected else "  "

            text = f"{prefix}{label}: {value}"
            surf = font.render(text, True, color)

            screen.blit(surf, (80, y))
            y += 50

        # Hjälptext
        help_text = [
            "UP/DOWN = navigera",
            "LEFT/RIGHT = ändra värde",
            "ENTER = toggle",
            "ESC = spara & tillbaka"
        ]

        y = screen.get_height() - 120
        for line in help_text:
            surf = small.render(line, True, (180, 180, 180))
            screen.blit(surf, (50, y))
            y += 25