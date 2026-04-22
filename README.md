# Skjutbana

Digital skjutbana med projektor, kamera och ljuddetektion. Systemet projicerar bilder, video och spel på en vägg/tavla och detekterar träffar via kamera + mikrofon.

## Funktioner

- **Bilder och tavlor** — stillbilder projiceras, skott detekteras och markeras
- **Video** — videoklipp spelas upp, skott registreras i realtid
- **Spel** — interaktiva scenarion (Shoot/Don't Shoot, skjutbana med helfigur)
- **AI-träning** — träna AI-modellen att förbättra träffdetektering
- **Kamerakalibrering** — ArUco-baserad homografi med 24 markörer
- **LED-feedback** — Deltaco SH-LS3M styrs via Tuya-protokoll
- **Ljuddetektion** — ffmpeg-baserad peak-detektor triggar bildanalys

## Snabbstart

```bash
python main.py
```

Kräver: Python 3.10+, pygame, opencv-python, numpy, ffmpeg i PATH.

## Kalibrering (första gången)

1. **Justera skjutgränser** — Inställningar → Justera skjutgränser. Flytta/skala den gröna ramen till projektionsytan.
2. **Kalibrera kamera** — Kamera → Kalibrera viewport via kamera. ENTER startar, vänta tills markörer hittas, sparas automatiskt.
3. **Justera ljud** — Inställningar → Ljud-peak. Justera tröskeln så skott triggar men inte bakgrundsljud.

## Projektstruktur

```
main.py                  — Startpunkt
config.py                — Skärm, FPS, sökvägar
src/engine/              — Spelmotor
  app.py                 — Huvudloop
  camera/                — Kamera, hit_scanner, kalibrering
  audio/                 — Ljuddetektion
  input/                 — Träffinput och koordinattransform
  visual/                — Visualisering (träffmarkeringar, overlays)
  scenes/                — Alla scener (meny, spel, inställningar, AI)
  ai/                    — AI-modul (runtime, minne, träning)
  output/                — LED-styrning
content/                 — Menydata, inställningar, AI-minne
assets/                  — Bilder, video, spel, tavlor
```
