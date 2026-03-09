# Skjutbana -- Architecture

Detta dokument beskriver systemets arkitektur.

------------------------------------------------------------------------

# Systemöversikt

Input Sources ├ mouse └ camera (framtida)

        ↓

HitInput

        ↓

Overlay System

        ↓

Scenes

        ↓

Game Objects

------------------------------------------------------------------------

# HitEvent

Alla träffar representeras av ett HitEvent.

Fält: - source - screen_x - screen_y - game_x - game_y - camera_x -
camera_y - timestamp

------------------------------------------------------------------------

# HitInput

`hit_input` är systemets centrala event dispatcher.

API:

push_mouse_hit() push_camera_hit() subscribe() unsubscribe() poll()

------------------------------------------------------------------------

# Overlay System

Overlay används för visuella effekter.

Nuvarande overlay: HitVisualizer

Render pipeline:

scene.render() overlay.render()

------------------------------------------------------------------------

# HitVisualizer

Visar träffmarkeringar.

Funktioner: - cirkel + crosshair - fade animation - persistent mode

------------------------------------------------------------------------

# Scene System

Scener representerar innehåll:

-   ImageScene
-   VideoScene
-   GameScene
-   TransformDebugScene
-   CameraTestScene

Scener laddas via: scene_factory.py

------------------------------------------------------------------------

# Game Architecture

Spel körs i GameScene.

GameScene kan prenumerera på träff-event:

hit_input.subscribe(self.on_hit)

GameObjects implementerar:

on_hit(event)

------------------------------------------------------------------------

# Calibration

Två kalibreringar används:

Viewport Scanport

------------------------------------------------------------------------

# Homography

Homography konverterar mellan:

kamera koordinater skärm koordinater

------------------------------------------------------------------------

# Framtida utveckling

Planerade funktioner:

-   kamerabaserad träffdetektion
-   spelobjekt
-   statistik
-   heatmaps
