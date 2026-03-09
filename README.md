# Skjutbana -- Interactive Projected Shooting Range

Skjutbana är ett modulärt Python-system för att bygga en **interaktiv
projekterad skjutbana**. Systemet projicerar bilder, video eller spel på
en fysisk måltavla och kan registrera träffar via kamera eller mus.

Systemet använder: - pygame för rendering - OpenCV för
kamerabearbetning - numpy för matematik och koordinattransformer

------------------------------------------------------------------------

# Huvudfunktioner

## Projektion

Systemet projicerar innehåll i en **viewport** som representerar den
fysiska tavlan. Viewporten kan kalibreras via:

Inställningar → Justera skjutgränser

Viewporten sparas i: content/settings.json

------------------------------------------------------------------------

## Kamerakalibrering

Kalibreringen gör:

1.  definierar ett **scanport område**
2.  identifierar hörnmarkörer
3.  beräknar en **homography-matris**

Homography används för att konvertera:

kamera-koordinater → skärm-koordinater

------------------------------------------------------------------------

# Global Hit System

Alla träffar representeras av ett **HitEvent**.

Fält: - source - screen_x - screen_y - game_x - game_y - camera_x -
camera_y - timestamp

------------------------------------------------------------------------

# HitInput

`hit_input` är systemets centrala träffhantering.

Exempel:

hit_input.push_mouse_hit(x,y)

Senare via kamera:

hit_input.push_camera_hit(camera_x,camera_y)

------------------------------------------------------------------------

# Hit Visualizer

Systemet har en global overlay som kan visa träffar.

Funktioner: - träffmarkering - fade-out - persistent mode

Inställningar finns i:

Inställningar → Visuella träffar

------------------------------------------------------------------------

# Overlay System

Alla visuella scener körs genom en **OverlayScene**.

Pipeline:

scene.render() overlay.render()

------------------------------------------------------------------------

# Scene System

Exempel på scener:

-   ImageScene
-   VideoScene
-   GameScene
-   TransformDebugScene
-   CameraTestScene

Scener laddas via: scene_factory.py

------------------------------------------------------------------------

# Menysystem

Menyn definieras i:

content/menu.json

Den beskriver kategorier, objekt och scen-typer.

------------------------------------------------------------------------

# Spelsystem

Spel körs via GameScene.

Spel kan reagera på träffar:

hit_input.subscribe(self.on_hit)

Exempel:

def on_hit(self,event): if
self.hitbox.collidepoint(event.game_x,event.game_y): self.destroy()

------------------------------------------------------------------------

# Debugverktyg

## Grid / Transform Test

Visar ett rutnät i viewporten för att verifiera:

-   kalibrering
-   homography
-   koordinater

Finns i: Inställningar → Grid / transform-test

------------------------------------------------------------------------

# Installation

pip install pygame opencv-python numpy

Starta systemet:

python main.py
