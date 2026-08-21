# Roadmap

## Nyligen klart

- [x] Multi-frame post-shot capture (8 frames vid ~40ms intervall)
- [x] Tvåstegs AI-pipeline: noise rejection → AI ranking
- [x] Persistence-scoring per hotspot
- [x] Funnel-diagnostik med CSV-rapporter
- [x] GT-matchning med konfigurerbar radie (5/10/15/20px)
- [x] Ökat max hotspots från 50 till 150
- [x] Kamerabuffring fixad (CAP_PROP_BUFFERSIZE=1)
- [x] Pre-shot frame timing fixad (60 frames bak)
- [x] Hole image bank — sparar skotthålsbilder automatiskt vid träning
- [x] Konservativare noise rejection (existed_before threshold 0.8)
- [x] Syntetisk autoträning med funnel-diagnostik och CSV-rapport

## Nyligen klart (observability-sprint)

- [x] RoundRecord som enda sanningskälla — ersätter auto_stats dict
- [x] AI guess pre-facit i grafiska rapporten (ai_guess_correct, avg dist)
- [x] Första 100 vs sista 100 jämförelse i rapporten
- [x] Blockstatistik per 100 skott i rapporten
- [x] Kandidatstatistik (medel/min/max, noll-rundor, >50/>100/>200)
- [x] Synkad UI-rapport, funnel och CSV (alla från round_records)
- [x] Round-ID + state logging per runda ([ROUND N] state)
- [x] Session-summary med mismatch-varning
- [x] Konfigurerbar candidate_limit via settings.json (default 200, range 1-2000)
- [x] Sampling modes förberett (center_bias, uniform, edge_bias, corners)
- [x] CSV-export med alla RoundRecord-fält + funnel-data

## Nyligen klart (detection-improvement sprint)

- [x] Whitehat (tophat) tillagd — hittar ljusa hål på mörk bakgrund
- [x] Absdiff-signal tillagd parallellt med subtract — fångar båda riktningar
- [x] existed_before_shot smartare — hanterar hög-kontrast bakgrunder (checker)
- [x] Persistence-tröskel sänkt 0.2 → 0.1 (färre false rejections)
- [x] Area-tröskel höjd 300 → 400 (checker-konturer kan vara större)
- [x] _verify_patch accepterar nu ljusa hål (bright spots on dark bg)
- [x] Sampling modes: center/full_uniform/edge/corner med alias
- [x] CSV-filnamn inkluderar bakgrundsläge och antal rundor
- [x] Rapport visar bakgrund + sampling mode i header

## Nyligen klart (moduläritet + diagnostik)

- [x] ArucoCalibrator — återanvändbar kalibreringsmotor i src/engine/camera/
- [x] Auto-kalibrering vid AI-scen start (visar markörer → detekterar → sparar)
- [x] CameraViewportCalibrationScene refaktorerad att använda ArucoCalibrator
- [x] Adaptiva center/ring-masker i _verify_patch (skalas med konturradie)
- [x] Höjda area/radius-gränser (max_area 900, max_radius 35) för luftgevär
- [x] Shot-diagnostik ([SHOT-DIAG]) — per-frame loggning av hela detektionspipelinen
- [x] Rapportfont sänkt för bättre läsbarhet
- [x] Animationsfrysning fixad mellan autoträningsrundor

## Nyligen klart (pre-shot & diagnostik)

- [x] Pre-shot timing fixad: 250ms före peak (var 2s) — stöd för dubbelskott
- [x] Shot-diagnostik bilder: pre/post/diff PNG + animerad GIF per skott
- [x] Koordinatrutnät (coord_grid) — 8:e bakgrundsläget med A0/B1-etiketter
- [x] HUD undertrycks under auto-kalibrering (suppress_overlays i OverlayScene)
- [x] Terminal-restore vid alla exit-vägar (atexit + termios + stty sane)
- [x] Död kod borttagen (_capture_pre_shot_frame)
- [x] rank_with_funnel limit synkad med candidate_limit (var hardcoded 150)

## Nyligen klart (board reference & pre-shot)

- [x] Multi-exposure board reference: vit + svart fångas vid scen-start
- [x] Projector response map (vit - svart) visar aktiv yta vs tejp/lappningar
- [x] White → scene_reference, black → surface_reference i hit_scanner
- [x] Ingen text/HUD i viewport under reference-capture (ren kamerabild)
- [x] Pre-shot offset ökat till 500ms (kamerabuffring ger ~100-200ms fördröjning)
- [x] Hit_scanner pre-shot fönster: 400-600ms, fallback 300-800ms

## Nyligen klart (automation / external control)

- [x] Lokal TCP/JSON-server på `127.0.0.1:8765`
- [x] Generisk `TcpNetworkHandler` + `send_command()` för externa scripts
- [x] `setWindowPos` som första externa command
- [x] `keyPress` som injicerar riktiga Pygame KEYDOWN/KEYUP-events i huvudtråden
- [x] `startAITraining` som skapar en ny automation-anpassad AI-träningsscen
- [x] Intern generell `EventBus`
- [x] Persistent event-subscription via `EventListener`
- [x] Event-driven AI-training lifecycle: calibration → ready → F2 → progress → completed
- [x] `automation.autostart_ai_training` som första end-to-end automation
- [x] Nätverks-/automationarkitektur dokumenterad i `NETWORK_AUTOMATION.md`

## Pågående

- [ ] Benchmark-körningar med alla bakgrundslägen (white → bubbles)
- [ ] Verifiera gray/black förbättring med whitehat+absdiff
- [ ] Verifiera checker förbättring med relaxade filter-trösklar
- [ ] Testa edge/corner sampling modes
- [ ] Analysera bubbles-konsistens efter fix
- [ ] Syntetisk bildbank-baserad träning (hole images → autotrain)
- [ ] Scen-medvetenhet i AI — feature-vektorn inkluderar scen-typ
- [ ] Readiness-indikator — visa i UI hur redo AI:n är
- [ ] Bilder och video som bakgrund i AI-träningsläge
- [ ] Förbättra AR-kodkalibrering ytterligare om zonbias kvarstår
- [ ] Broadcasta strukturerad Python logging/console-output till event subscribers
- [ ] Lägg till fler generella inputkommandon (musposition, mouse down/up/click)
- [ ] Automations-testsvit för menyer, viewport-grid, regression och längre F2-serier

## Lång sikt

- [ ] AI tar över träffdetektionen (ai_priority / ai_only)
- [ ] Weapon profiles (olika vapen, olika detektionsparametrar)
- [ ] Multi-kamera
- [ ] Replay-system
- [ ] Partikeleffekter vid träff
- [ ] Community-delade AI-hjärnor via GitHub
- [ ] Självstyrande utvecklar-agent som kan ändra kod i experimentbranch, köra automatiska benchmark och behålla/revertera ändringar baserat på metrics

## Klart

- [x] Grundläggande kamera + ljud + träffdetektion
- [x] LED-integration (Deltaco SH-LS3M)
- [x] Scen-driven LED via menu.json
- [x] Kamerarotation och spegling
- [x] Kandidatvisning med snapshot (överlever scenbyte)
- [x] Pre-shot diff (subtract-baserad)
- [x] Blackhat + pre-shot combined signal
- [x] Known-hole penalty
- [x] Zonloggning (L/M/R)
- [x] AI-modul: runtime, minne, träningsscen
- [x] AI export/import av hjärna
- [x] Homografi-bugg fixad
- [x] Kalibrering separerad (viewport vs kamera)
- [x] 24 ArUco-markörer med RANSAC
- [x] AI-träningsscen med viewport-respekt
- [x] 8 bakgrundslägen i AI-träning (vit → koordinatrutnät → bubblor)

---

## Automation / autonomous testing — next milestones

### Implemented

- [x] TCP/JSON command channel
- [x] persistent event subscription
- [x] generic EventBus broadcast
- [x] external `setWindowPos`
- [x] external generic `keyPress`
- [x] automation-enabled AI training scene
- [x] event-driven calibration → F2 → completion flow
- [x] repeated complete AI-training runs without game restart
- [x] machine-readable per-run JSON + JSONL + CSV
- [x] aggregate `summary.json` with git revision and trends

### Next useful steps

- [ ] automate all 8 backgrounds as one benchmark suite
- [ ] add mouse move/click commands for game/UI testing
- [ ] add structured log/console broadcast over the same communication layer
- [ ] add explicit benchmark configuration/seed capture for reproducible tests
- [ ] compare two git revisions/sessions automatically
- [ ] define pass/fail thresholds for regression testing
- [ ] eventually let a coding agent modify a controlled branch, run benchmarks,
      compare results and keep/reject changes automatically
