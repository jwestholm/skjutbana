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
│  - automation command dispatch          │
├─────────────────────────────────────────┤
│  Automation / IPC                       │
│  - TCP/JSON server (localhost:8765)     │
│  - command queue -> Pygame main thread  │
│  - EventBus -> event broadcast          │
│  - external automation scripts          │
├─────────────────────────────────────────┤
│  Detektion                              │
│  - AudioPeakDetector → triggar sökfönster│
│  - HitScanner → pre-shot diff + blackhat│
│    + whitehat + absdiff                 │
│  - Tracking → kandidater → emission     │
├─────────────────────────────────────────┤
│  AI (lager ovanpå detektion)            │
│  - observe_scanner() varje frame        │
│  - rank_candidates() vid träning        │
│  - choose_for_emission() vid produktion │
│  - RoundRecord → single source of truth │
│  - FunnelTracker → per-shot diagnostik  │
├─────────────────────────────────────────┤
│  Kalibrering (återanvändbar motor)      │
│  - ArucoCalibrator (shared engine)      │
│  - Auto-kalibrering vid AI-scen start   │
│  - Manuell kalibrering via meny         │
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
- `HitScanner.update` → `ai_runtime.observe_scanner()` + `candidate_limit` propagering
- `HitScanner._emit_track_result` → `ai_runtime.choose_for_emission()`

AI:n kan aldrig krascha appen — alla anrop i try/except.

## Kalibreringsmotor (ArucoCalibrator)

Återanvändbar motor i `src/engine/camera/aruco_calibrator.py`:
- `ArucoCalibrator(viewport_rect)` — initierar ArUco-detektor
- `detect_and_calibrate(frame_bgr)` → `CalibrationResult`
- `render_markers(screen)` — ritar 24 ArUco-markörer
- `save_and_apply(result)` — sparar kalibrering + laddar om hit_input

Används av:
- `CameraViewportCalibrationScene` — manuell kalibrering via meny
- `AITrainingScene` — automatisk kalibrering vid scen-start

## Observability (RoundRecord)

All statistik i AI-träning flödar genom `RoundRecord` (dataclass i `diagnostics.py`):
- Enda sanningskälla för UI-rapport, funnel-summary och CSV-export
- Per-runda: GT-koordinater, kandidatantal, AI-gissning före facit, detektionsresultat
- Blockstatistik per 100 skott, first-100 vs last-100 jämförelse
- Round-ID state logging för felsökning av räknarmismatch

## Scenprotokoll

Alla scener implementerar: `on_enter()`, `on_exit()`, `handle_event()`, `update()`, `render()`.
Wrappas i `OverlayScene` som lägger till visualiseringar.


## Automation, IPC och EventBus

Automation är ett separat lager runt spelmotorn. Externa scripts kommunicerar med ett redan startat spel via TCP på `127.0.0.1:8765`. Protokollet är newline-delimited UTF-8 JSON.

```text
External automation script
        │ command JSON
        ▼
CommunicationServer (network thread)
        │ thread-safe command queue
        ▼
App.run() / Pygame main thread
        │
        ├─ setWindowPos / keyPress / startAITraining
        │
        ▼
Scenes / game logic
        │
        │ event_bus.emit(...)
        ▼
EventBus (in-process)
        │ broadcast queue
        ▼
CommunicationServer broadcaster
        │ event JSON
        ▼
EventListener / external clients
```

Arkitekturregler:

- Nätverkstrådar får aldrig anropa Pygame eller scenlogik direkt.
- `app.py` är gränsen där inkommande automation-kommandon går över till Pygames huvudtråd.
- `EventBus` är process-lokal och generell; den ska inte vara AI-träningsspecifik.
- Externa scripts använder `TcpNetworkHandler`/`send_command()` och `EventListener`; duplicera inte socket/JSON-kod.
- Långvariga/asynkrona flöden ska koordineras med events, inte fasta `sleep()`-tider.
- Vanlig `AITrainingScene` hålls fri från automationkoppling där det går. `AutomationAITrainingScene` är en tunn subklass som publicerar livscykel-events.

Se `NETWORK_AUTOMATION.md` för protokoll, message schemas, aktuella commands/events och extensionsregler.
