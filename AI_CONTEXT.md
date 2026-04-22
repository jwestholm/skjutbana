# AI-kontext för detta projekt

Detta dokument ger AI-assistenter (Kiro, ChatGPT, etc.) snabb kontext om projektet.

## Vad är det här?

En digital skjutbana. Projektor visar bilder/video/spel på en vägg. Användaren skjuter med luftgevär/CO2-vapen genom plexiglas. Kamera + mikrofon detekterar träffar.

## Teknikstack

- Python 3.10+, pygame, opencv-python, numpy
- Ubuntu (target), utveckling på Windows
- 4K USB-kamera (3840x2160, 30fps)
- ffmpeg för ljudinput
- ArUco-markörer för kamerakalibrering

## Viktiga filer

| Fil | Ansvar |
|-----|--------|
| `src/engine/app.py` | Huvudloop |
| `src/engine/camera/hit_scanner.py` | Träffdetektion (pre-shot diff, blackhat, tracking) |
| `src/engine/audio/audio_peak_detector.py` | Ljudtrigger |
| `src/engine/input/hit_input.py` | Koordinattransform (kamera → screen → viewport) |
| `src/engine/ai/runtime.py` | AI-runtime (minne, scoring, träning) |
| `src/engine/ai/bootstrap.py` | Monkey-patchar hit_scanner för AI-observation |
| `src/engine/scenes/ai_training.py` | AI-träningsscen |
| `src/engine/scenes/calibrate_camera_viewport.py` | ArUco-kalibrering (24 markörer) |
| `src/engine/scenes/calibrate.py` | Viewport-justering (skjutgränser) |
| `src/engine/visual/hit_visualizer.py` | Visuella träffmarkeringar + kandidatvisning |

## Arkitekturprinciper

1. **Viewport är master** — all rendering och alla spelkoordinater utgår från viewport-rektangeln
2. **AI är ett lager ovanpå** — AI:n observerar och rankar men påverkar inte träffar i train_only-läge
3. **AI kan aldrig krascha appen** — alla AI-anrop i try/except
4. **Minimal engine-påverkan** — AI-modulen lever i `src/engine/ai/`, kopplas in via bootstrap-patching
5. **Scener följer protokollet** — `on_enter()`, `on_exit()`, `handle_event()`, `update()`, `render()`

## Träffdetektionspipeline

1. Mikrofon hör smällen → AudioPeakEvent
2. HitScanner öppnar sökfönster
3. Pre-shot-bakgrund byggs från frames innan peak_ts
4. Diff (subtract) + blackhat → combined signal
5. Konturer → kandidater (area, circularity, radius-filter)
6. Patch-verifiering (center_change, local_contrast, pre_shot_change)
7. Known-hole penalty
8. Tracking (3 hits över 90ms → stable)
9. Emission → hit_input.push_camera_hit()
10. AI observerar (observe_scanner) och kan påverka emission (choose_for_emission)

## Kända problem och begränsningar

- Projektorflicker ger brus i pre-shot diff
- Plexiglassprickor/tejp ger falska kandidater
- Svart bakgrund i mörkt rum = inga träffar (kräver extern belysning)
- Kalibreringen kan ha zonbias i hörnen (beroende på kameravinkel)
- AI:n är i tidig fas — lär sig men kan inte ännu ersätta detektorn

## Fysisk setup

- Skytt → 6m → tavla (plexiglas + projektionsyta)
- Kamera sitter ~50cm från tavlan
- Mikrofon i kameran
- Kulan träffar innan ljudet når mikrofonen
- LED-list i tavlan (kan ge ljusa hål)
