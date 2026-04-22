# AI-kontext — Onboarding för AI-assistenter

Detta dokument ger AI-assistenter (Kiro, ChatGPT, Copilot, etc.) fullständig kontext om projektet. Läs detta först.

## Vad är det här?

En digital skjutbana. En projektor visar bilder, video och spel på en vägg/tavla. Användaren skjuter med luftgevär eller CO2-vapen mot tavlan. En 4K-kamera och mikrofon (monterad nära tavlan) detekterar träffar via bildanalys och ljudtrigger.

## Teknikstack

- Python 3.10+, pygame, opencv-python, numpy
- Target: Ubuntu-maskin vid skjutbanan
- 4K USB-kamera (3840×2160, 30fps, MJPG)
- ffmpeg för ljudinput (PulseAudio/ALSA)
- ArUco-markörer (DICT_4X4_50) för kamerakalibrering

## Arkitekturregler (VIKTIGT)

1. **Viewport är master-ytan** — den gröna rektangeln ("Justera skjutgränser") definierar var allt content renderas. Spel, video, bilder, AI-träning — allt ritas INOM viewporten. Alla spelkoordinater utgår från viewporten. HUD/UI-element (statusrad, hjälptext) får ligga utanför.

2. **AI är ett lager ovanpå detektorn** — AI:n observerar och rankar men ersätter inte detektorn (i train_only-läge). Den kopplas in via monkey-patching i bootstrap.py, inte genom att ändra engine-kod.

3. **AI kan aldrig krascha appen** — alla AI-anrop i try/except.

4. **Kalibrering och viewport är separerade**:
   - `src/engine/scenes/calibrate.py` = manuell justering av viewport/skjutgränser (flytta/skala grön ram). Ingen AR, ingen homografi.
   - `src/engine/scenes/calibrate_camera_viewport.py` = kamera→viewport-kalibrering med 24 ArUco-markörer, RANSAC, per-markör error. Sparar homografi + inverse_homography.

5. **Scener följer protokollet** — `on_enter()`, `on_exit()`, `handle_event()`, `update()`, `render()`. Alla wrappas i `OverlayScene`.

## Viktiga filer

| Fil | Ansvar |
|-----|--------|
| `src/engine/app.py` | Huvudloop, scenbyte, service-sync |
| `src/engine/camera/hit_scanner.py` | Träffdetektion: pre-shot diff, blackhat, tracking, emission |
| `src/engine/audio/audio_peak_detector.py` | Ljudtrigger (ffmpeg → peak-detektion) |
| `src/engine/input/hit_input.py` | Koordinattransform: kamera → screen → viewport → content |
| `src/engine/ai/runtime.py` | AI-runtime: SimpleAIMemory, scoring, träning, choose_for_emission |
| `src/engine/ai/bootstrap.py` | Monkey-patchar HitScanner för AI-observation |
| `src/engine/ai/space_mapper.py` | Koordinatprojektion kamera ↔ screen ↔ viewport |
| `src/engine/scenes/ai_training.py` | AI-träningsscen (7 bakgrundslägen, klick-baserad inlärning) |
| `src/engine/scenes/ai_settings.py` | AI-inställningar (läge, vikt, export/import) |
| `src/engine/scenes/calibrate_camera_viewport.py` | ArUco-kalibrering (24 markörer, RANSAC) |
| `src/engine/scenes/calibrate.py` | Viewport-justering (skjutgränser) |
| `src/engine/visual/hit_visualizer.py` | Visuella träffmarkeringar + kandidatvisning |
| `content/ai/memory.json` | AI-minne (positiva/negativa exempel, feature_ranges) |
| `content/ai/settings.json` | AI-inställningar (runtime-konfiguration) |
| `content/settings.json` | Viewport, scanport, kamerakalibrering, visuella inställningar |

## Koordinatsystem

```
Kamera (4K pixel) → Homografi → Skärm (pygame) → Viewport → Content/Game
```

Homografin beräknas via ArUco-kalibrering och sparas i `content/settings.json` under `camera_calibration`. `hit_input.py` använder homografin om `prefer_homography` är satt och inverse_homography finns.

**Viktig historisk bugg (fixad):** `_prefers_homography()` i hit_input.py accepterade inte kalibreringsmetoden `"aruco_viewport_v1"` — bara `"aruco_viewport_board"`. Det gjorde att homografin beräknades men aldrig användes, och systemet föll tillbaka på enkel scanport-skalning. Resultatet var zonbias (höger bra, vänster dålig). Fixat genom att acceptera båda metodnamn + `prefer_homography`-flaggan.

## Träffdetektionspipeline

1. **Mikrofon** hör smällen → `AudioPeakEvent` (mikrofon sitter ~50cm från tavlan, kulan har redan träffat)
2. **HitScanner** öppnar sökfönster (`association_lag_s = 1.5s`)
3. **Pre-shot-bakgrund** byggs från frames 40-200ms före `peak_ts`
4. **Diff**: `cv2.subtract(pre_shot_blur, current_blur)` — detekterar mörkare förändringar
5. **Blackhat**: `cv2.morphologyEx(MORPH_BLACKHAT)` — hittar små mörka fläckar
6. **Combined**: `max(pre_shot_delta, blackhat)` — båda signalerna bidrar
7. **Konturer** → kandidater (filtreras på area 2-180, radius 0.8-12, circularity ≥ 0.02)
8. **Patch-verifiering**: center_change, local_contrast, pre_shot_change
9. **Known-hole penalty**: kandidater nära redan kända hål nedviktas (0.15-0.7×)
10. **Zonloggning**: raw blobs och kept candidates per zon (L/M/R)
11. **Tracking**: 3 hits över ≥90ms → stable → emission
12. **AI observerar** (`observe_scanner`) och kan påverka emission (`choose_for_emission`)

## AI-träning

### Träningsscen (ai_training.py)

Bakgrundslägen i progression (TAB byter):
1. **Vit** — enklast, ren bakgrund
2. **Vit + rutnät** — lägger till linjer
3. **Grå** — lägre kontrast
4. **Svart** — svårast statisk (kräver belysning)
5. **Rutmönster** — stillbild, hög kontrast
6. **Rutmönster (video)** — animerat, fryser vid skott
7. **Bubblor** — rörliga former, fryser vid skott (spel-liknande)

### Träningsflöde

1. Skjut → audio peak → hit_scanner genererar kandidater
2. AI rankar alla kandidater (upp till 50 visas)
3. Användaren klickar var kulan träffade
4. Klicket transformeras till kameraplanet via `space_mapper`
5. Närmaste kandidat inom `click_match_radius_px` (42px) → positivt exempel
6. Max 3 sämst rankade kandidater → negativa exempel
7. Om ingen kandidat nära → syntetiskt positivt exempel från klickpunkten
8. Sparas i `content/ai/memory.json`

### AI-runtime (SimpleAIMemory)

- Bounded minne: max 400 positiva, 1200 negativa exempel
- Feature-normalisering: per-feature min/max (running)
- Tidsviktning: äldre minnen penaliseras svagt
- Adaptiv AI-viktning vid scoring: 10% vid <20 minnen → 65% vid 300+
- Score = weighted(detector_norm, ai_score) baserat på datamängd

### AI-inställningar (content/ai/settings.json)

| Nyckel | Default | Beskrivning |
|--------|---------|-------------|
| `mode` | `train_only` | off / train_only / advisory / blended / ai_priority / ai_only |
| `top_k` | 10 | Max kandidater att ranka (träningsscenen använder 50) |
| `memory_limit_positive` | 400 | Max positiva minnen |
| `memory_limit_negative` | 1200 | Max negativa minnen |
| `click_match_radius_px` | 42.0 | Radie för klick-matchning |
| `trust_percent` | 0 | AI-vikt i blended-läge (0-100%) |
| `min_confidence` | 0.58 | Minsta confidence för AI-override |
| `show_overlay` | true | Visa AI-overlay |
| `auto_learn` | true | Automatisk inlärning (ej inkopplad ännu) |

### Export/import av AI-hjärna

- **E** i AI-inställningar → exporterar till `content/ai/exports/ai_brain_YYYYMMDD_HHMMSS.json`
- **I** → importerar senaste export (varnar om lokala ändringar finns)
- Exporterade hjärnor kan checkas in i GitHub som baselines

## Kalibreringsflöde (rätt ordning)

1. **Justera skjutgränser** (Inställningar → Justera skjutgränser) — flytta/skala den gröna ramen till projektionsytan
2. **Kalibrera kamera** (Kamera → Kalibrera viewport via kamera) — ENTER startar, 24 ArUco-markörer projiceras, kameran detekterar dem, homografi beräknas med RANSAC
3. **Verifiera** — kolla reprojection error (mål: <1.0px) och per-markör error (grönt <1px, gult 1-3px, rött >3px)

## Fysisk setup

```
Skytt (6m avstånd)
    │
    │  ← kulbana
    │
    ▼
┌──────────────────────────────────┐
│  Vägg                            │
│  ├─ 2mm plåt                     │
│  ├─ Ljudisolerskivor (bil) 4cm   │
│  ├─ Tomrum ~1cm med LED-list     │
│  ├─ Pappkartong (flyttlåda)      │
│  ├─ Vita A4-papper               │ ← projektionsyta, kulor träffar här
│  └─ Lappningar / markeringar     │
│                                  │
│   ┌────────────┐                 │
│   │ Projicerad  │                 │
│   │ bild/video  │                 │
│   └────────────┘                 │
└──────────────────────────────────┘
    ▲
    │
  Kamera (USB, 4K)
  Mikrofon (i kameran, ~50cm från tavla)
  Projektor
```

### Tavlans lager (utifrån → in mot vägg)

1. **Vita A4-papper + lappningar** — projektionsyta, det kameran ser
2. **Pappkartong (flyttlåda)** — bärande skikt, kulor fastnar/penetrerar
3. **Tomrum ~1cm med LED-list** — LED-bakljus, kan ge ljusa hål om påslagen
4. **Ljudisolerskivor (bil) 4cm** — dämpar ljud och fångar kulor
5. **2mm plåt** — skyddar väggen
6. **Vägg**

Hål som penetrerar pappkartongen kan bli ljusa (LED lyser igenom). Hål som bara gör en buckla/märke i pappret blir mörka. Detektorn måste hantera båda fallen.

Vapen: luftgevär (5mm BB), CO2 AR-15 (stålkulor), CO2 pistol. Max 5 skott/sekund.

## Kända problem och begränsningar

- Projektorflicker/ljusvariation ger brus i pre-shot diff
- Tejp och sprickor i tavlan kan ge falska kandidater
- Svart bakgrund i mörkt rum = svag signal (kräver extern belysning)
- Zonbias kan finnas om kalibreringen har hög reprojection error
- AI:n är i tidig fas — lär sig men kan inte ännu ersätta detektorn
- `subtract`-mode detekterar bara mörkare förändringar (inte ljusare hål med LED-bakljus)

## Vad vi redan testat och lärt oss

- `absdiff` istället för `subtract` → sämre (projektorflicker ger diff överallt)
- Vote-map (multi-frame persistens) → för tungt på 4K, opålitligt
- Feature-matching (hitta nya konturer vs gamla) → för tungt, samma problem
- Blackhat borttagen ur combined → regression (missade hål)
- Pre-shot-change med 0.80 vikt → för dominant, maskerade riktiga hål
- Homografi beräknad men inte använd → zonbias (fixat)
