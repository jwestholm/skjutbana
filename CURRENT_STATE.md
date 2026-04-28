# Nuvarande status

## Fungerar

- **Kamera** — 4K USB, 30fps, rotation/spegling konfigurerbar
- **Ljud** — ffmpeg-baserad peak-detektor, PulseAudio/ALSA
- **Träffdetektion** — pre-shot diff (subtract + absdiff) + blackhat + whitehat, tracking, emission
- **Kalibrering** — ArUco-baserad homografi med 24 markörer, RANSAC. Återanvändbar motor (`ArucoCalibrator`)
- **Auto-kalibrering** — AI-träningsscenen kalibrerar automatiskt vid start (~1-2s)
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
- **Pre-shot timing** — 250ms före audio peak (200-350ms fönster), stöd för dubbelskott
- **Shot-diagnostik bilder** — pre/post/diff PNG + animerad GIF per skott i content/ai/shot_diag/
- **Koordinatrutnät** — nytt bakgrundsläge med A0/B1-etiketter för att verifiera pre/post-alignment
- **HUD undertrycks** under auto-kalibrering (suppress_overlays)
- **Terminal-restore** — stty sane + termios vid alla exit-vägar (atexit + try/finally)

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

- Gray/black bakgrund har fortfarande låg recall — whitehat+absdiff bör hjälpa men ej verifierat
- Checker-mönster förlorar GT i filter-steget — relaxade trösklar bör hjälpa men ej verifierat
- Bubbles-läge ger inkonsistenta resultat — troligen timing/freeze-relaterat
- AI:n är i train_only-läge — påverkar inte träffar ännu
- Detektorn hittar ibland kanter/tejp/sprickor som kandidater
- Luftgevärshål (stora) behöver verifieras med shot-diagnostik
