# Arkitektur

## Lager

```
┌─────────────────────────────────────────┐
│  Scener (menu, image, video, game, AI)  │
├─────────────────────────────────────────┤
│  OverlayScene (wrappar alla scener)     │
│  - hit_visualizer                       │
│  - scanner_debug_overlay                │
│  - scanner_status_overlay               │
├─────────────────────────────────────────┤
│  Engine (app.py huvudloop)              │
│  - camera_manager.update()              │
│  - audio_peak_detector.update()         │
│  - hit_scanner.update()                 │
│  - hit_visualizer.update()              │
├─────────────────────────────────────────┤
│  Detektion                              │
│  - AudioPeakDetector → triggar sökfönster│
│  - HitScanner → pre-shot diff + blackhat│
│  - Tracking → kandidater → emission     │
├─────────────────────────────────────────┤
│  AI (lager ovanpå detektion)            │
│  - observe_scanner() varje frame        │
│  - rank_candidates() vid träning        │
│  - choose_for_emission() vid produktion │
├─────────────────────────────────────────┤
│  Transform                              │
│  - HitInput: kamera → screen → viewport │
│  - Homografi (ArUco-kalibrering)        │
│  - Scanport (fallback linjär mappning)  │
├─────────────────────────────────────────┤
│  Sensorer                               │
│  - Kamera (4K USB, 30fps)               │
│  - Mikrofon (i kameran, 50cm från tavla)│
├─────────────────────────────────────────┤
│  Feedback                               │
│  - Projektor (bilder/video/spel)        │
│  - LED (Deltaco SH-LS3M via Tuya)      │
│  - Visuella träffmarkeringar            │
└─────────────────────────────────────────┘
```

## Koordinatsystem

```
Kamera (4K pixel) → Homografi → Skärm (pygame) → Viewport → Content/Game
```

- **Viewport** = den gröna ramen, master render area
- **Content** = spelplan inom viewport
- Alla scener renderar inom viewport
- HUD/UI får ligga utanför

## AI-integration

AI:n kopplas in via monkey-patching i `bootstrap.py`:
- `HitScanner.update` → `ai_runtime.observe_scanner()`
- `HitScanner._emit_track_result` → `ai_runtime.choose_for_emission()`

AI:n kan aldrig krascha appen — alla anrop i try/except.

## Scenprotokoll

Alla scener implementerar: `on_enter()`, `on_exit()`, `handle_event()`, `update()`, `render()`.
Wrappas i `OverlayScene` som lägger till visualiseringar.
