# Nuvarande status

## Fungerar

- **Kamera** — 4K USB, 30fps, rotation/spegling konfigurerbar
- **Ljud** — ffmpeg-baserad peak-detektor, PulseAudio/ALSA
- **Träffdetektion** — pre-shot diff (subtract + absdiff) + blackhat + whitehat, tracking, emission
- **Kalibrering** — ArUco-baserad homografi med 24 markörer, RANSAC. Återanvändbar motor (`ArucoCalibrator`)
- **Auto-kalibrering** — AI-träningsscenen kalibrerar automatiskt vid start (~3s: ArUco + vit/svart reference)
- **Board reference** — multi-exposure (vit + svart) ger projector response map och filtrerar bort tejp/lappningar
- **Viewport** — separat justering av skjutgränser/rityta
- **Koordinattransform** — kamera → screen → viewport → content via homografi
- **Visuella träffar** — fade/persistent, kandidatvisning med snapshot
- **LED** — Deltaco SH-LS3M via Tuya, scen-driven
- **AI-modul** — runtime, minne, träningsscen, export/import av hjärna
- **AI-observability** — RoundRecord (single source of truth), funnel-diagnostik, CSV-export, blockstatistik
- **Autoträning** — F1 (visuell) och F2 (headless), 100 skott per serie
- **Sampling modes** — center_bias, uniform, edge_bias, corners (konfigurerbar via settings.json)
- **Shot-diagnostik** — detaljerad per-skott-loggning i terminalen ([SHOT-DIAG]) + pre/post/diff bilder i content/ai/shot_diag/
- **Spel** — Shoot/Don't Shoot, skjutbana med helfigur
- **Bilder/video** — stillbilder och videoklipp med träffdetektion
- **Automation/IPC** — lokal TCP/JSON-server på `127.0.0.1:8765`, generisk command/response-kanal och persistent event-subscription
- **EventBus** — intern publish/subscribe-buss som broadcastas till externa lyssnare
- **Extern AI-träningsautomation** — `automation.autostart_ai_training` kan flytta fönstret, skapa ny AI-träningsscen, vänta på kalibrering, skicka F2 och ta emot slutrapport

## Senaste förbättringar

- **ArucoCalibrator** — återanvändbar kalibreringsmotor, används av både manuell kalibrering och AI-träning
- **Auto-kalibrering** — AI-scenen kalibrerar automatiskt vid start, slipper manuellt steg
- **Whitehat + absdiff** — hittar ljusa hål (LED-genomlysning) på mörka bakgrunder
- **Adaptiva center/ring-masker** — skalas med konturradie, hanterar stora luftgevärshål
- **Höjda area/radius-gränser** — max_area 900, max_radius 35 (stöd för luftgevär)
- **Smartare existed_before_shot** — hanterar hög-kontrast bakgrunder (checker) utan false rejections
- **RoundRecord** — enda sanningskälla för all statistik (rapport, funnel, CSV)
- **AI guess pre-facit** — rapporten visar AI:ns gissning före facit
- **Blockstatistik** — per 100 skott, visar om AI:n lär sig under körningen
- **Kandidatstatistik** — medel/min/max, noll-rundor, >50/>100/>200
- **Shot-diagnostik** — loggar per frame: konturer, rejections, patch-värden, tracking, emission/timeout
- **Pre-shot timing** — 500ms före audio peak (400-600ms fönster), extra marginal för kamerabuffring
- **Shot-diagnostik bilder** — pre/post/diff PNG + animerad GIF per skott i content/ai/shot_diag/
- **Koordinatrutnät** — nytt bakgrundsläge med A0/B1-etiketter för att verifiera pre/post-alignment
- **HUD undertrycks** under auto-kalibrering (suppress_overlays)
- **Terminal-restore** — stty sane + termios vid alla exit-vägar (atexit + try/finally)
- **Automation command path** — nätverkstråd → thread-safe queue → `app.py` → Pygame main thread
- **Event-driven AI automation** — kalibrering och träning signaleras via `aiTraining.*` events; automation behöver inte gissa med sleep/polling
- **AutomationAITrainingScene** — separat subklass för event-observability så vanlig AI-träningsscen kan förbli oförändrad

## Benchmark-resultat (F2, 100 skott per bakgrund)

| Bakgrund | Found | Top-1 | Top-3 | Medel dist |
|----------|-------|-------|-------|------------|
| white | 87% | 64% | 67% | 19.0 px |
| white_grid | 77% | 41% | 50% | 25.9 px |
| gray | 36% | 5% | 5% | 75.9 px |
| black | 37% | 2% | 3% | 76.0 px |
| checker | 20% | 9% | 10% | 141.1 px |
| checker_anim | 19% | 1% | 1% | 138.5 px |
| bubbles | inkonsistent | — | — | — |

**Mål:** white stabil som baseline, gray/black förbättra raw recall, checker förbättra filter/ranking.

## Kända begränsningar

- Automation-servern är avsiktligt localhost-only och saknar autentisering/TLS; exponera inte port 8765 på nätverket utan separat säkerhetsdesign
- Event-broadcast är best-effort i minnet; events persisteras inte och en klient som ansluter sent får inte historiska events
- Command/response har 5 s server-timeout för att själva kommandot ska bli behandlat; långvariga operationer ska startas snabbt och följas via events

- Gray/black bakgrund har fortfarande låg recall — whitehat+absdiff bör hjälpa men ej verifierat
- Checker-mönster förlorar GT i filter-steget — relaxade trösklar bör hjälpa men ej verifierat
- Bubbles-läge ger inkonsistenta resultat — troligen timing/freeze-relaterat
- Kamerabuffring ger ~100-200ms fördröjning på frame-timestamps (kompenseras med 500ms pre-shot offset)
- AI:n är i train_only-läge — påverkar inte träffar ännu
- Detektorn hittar ibland kanter/tejp/sprickor som kandidater (board reference bör minska detta)
- Luftgevärshål (stora) behöver verifieras med shot-diagnostik

---

## Automation update — repeated AI training

The event-driven automation path has been verified on the physical range:
calibration completes, `waitingForFirstShot` is emitted, automation injects F2,
progress is returned and the completed report is received. A new automation
run can then be started without restarting the game.

Added repeated-run support:

```bash
python3 -m automation.ai_training_loop 1 100
```

The loop keeps the game running and starts a fresh automation-enabled AI
training scene for each run. Results are persisted under
`content/ai/automation_runs/` as full JSON, compact JSONL, CSV and an aggregate
`summary.json`.

`aiTraining.completed` now carries structured AI-friendly diagnostics:
percentages, distance statistics, candidate statistics, funnel summary,
consistency checks, timing and all per-round `RoundRecord` rows.
