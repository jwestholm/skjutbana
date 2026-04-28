# AI-kontext — Onboarding för AI-assistenter

Läs detta FÖRST. Det ger dig fullständig kontext om projektet, arkitekturen, kända problem och hur du ska felsöka.

---

## GOLDEN RULES

```
1. VIEWPORT IS THE GAME WORLD. Nothing bypasses it. All rendering,
   hit detection and input happens in viewport coordinates.

2. HOMOGRAPHY MUST BE ACTIVE. Without it, coordinate transforms
   fall back to linear scaling and everything breaks spatially.

3. AI ONLY RANKS CANDIDATES. It does not replace detection.
   If candidates are bad, AI cannot fix it.

4. ALWAYS DEBUG WITH 3×3 ZONE TEST before tuning weights or AI.

5. FIX GEOMETRY BEFORE TUNING AI. Calibration > scoring > AI training.

6. THE BIGGEST PROBLEM IS NEVER AI. It is always geometry + light +
   candidate quality. Fix those first.
```

---

## Vad är det här?

En digital skjutbana. En projektor visar bilder, video och spel på en tavla. Användaren skjuter med luftgevär eller CO2-vapen mot tavlan. En 4K-kamera och mikrofon (monterad nära tavlan) detekterar träffar via bildanalys och ljudtrigger.

## Teknikstack

- Python 3.10+, pygame, opencv-python, numpy
- Target: Ubuntu-maskin vid skjutbanan
- 4K USB-kamera (3840×2160, 30fps, MJPG)
- ffmpeg för ljudinput (PulseAudio/ALSA)
- ArUco-markörer (DICT_4X4_50) för kamerakalibrering

---

## Arkitekturregler

### 1. Viewport = MASTER (absolut regel)

ALL rendering, träffdetektion och input sker i viewport-koordinater. Inget innehåll får ritas eller tolkas utanför viewporten. Viewporten är den gröna rektangeln som ställs in via "Justera skjutgränser".

```
Fysisk skärm (pygame-fönster)
└── Viewport (grön rektangel = master render area)
    └── Content / spelplan
        ├── Spel
        ├── Video
        ├── Bilder / tavlor
        ├── AI-träning
        ├── Kandidatmarkeringar
        ├── Träffmarkeringar
        └── Klick-feedback
```

HUD/UI (statusrad, hjälptext) FÅR ligga utanför viewporten — det är applikations-UI, inte spelplan.

### 2. AI är ett lager ovanpå detektorn

AI:n observerar och rankar men ersätter INTE detektorn (i train_only-läge). Den kopplas in via monkey-patching i bootstrap.py, inte genom att ändra engine-kod. AI kan aldrig krascha appen — alla anrop i try/except.

### 3. Kalibrering och viewport är separerade

- `src/engine/scenes/calibrate.py` = manuell justering av viewport/skjutgränser. Ingen AR, ingen homografi.
- `src/engine/scenes/calibrate_camera_viewport.py` = kamera→viewport-kalibrering med 24 ArUco-markörer, RANSAC. Sparar homografi + inverse_homography.

### 4. Scener följer protokollet

Alla scener implementerar: `on_enter()`, `on_exit()`, `handle_event()`, `update()`, `render()`. Alla wrappas i `OverlayScene`.

---

## Koordinatkedjan (VIKTIGT — läs noga)

### Kamera → Skärm (vid träffdetektion)

```
Kamera (4K pixel space)
  → Homografi (cv2.perspectiveTransform)
    → Skärm (pygame pixel space)
      → Viewport (offset från skärmens övre vänstra hörn)
        → Content / Game (viewport-lokala koordinater)
```

Hanteras av `hit_input._canonical_camera_to_screen()` → `_screen_to_spaces()`.

### Skärm → Kamera (vid klick / mus)

```
Mus/klick (pygame pixel space)
  → Viewport-check (ligger klicket inom viewporten?)
    → Inverse homografi
      → Kamera (4K pixel space)
        → AI / detektor
```

Hanteras av `hit_input._canonical_screen_to_camera()`.

### Viktigt

- Om homografin INTE är aktiv faller systemet tillbaka på enkel scanport-skalning → zonbias uppstår.
- `_prefers_homography()` i hit_input.py måste returnera True. Den kollar: `prefer_homography`-flaggan ELLER metodnamn `"aruco_viewport_board"` / `"aruco_viewport_v1"`.

---

## Viktiga filer

| Fil | Ansvar |
|-----|--------|
| `src/engine/app.py` | Huvudloop, scenbyte, service-sync |
| `src/engine/camera/hit_scanner.py` | Träffdetektion: pre-shot diff, blackhat, whitehat, absdiff, tracking, emission, shot-diagnostik |
| `src/engine/camera/aruco_calibrator.py` | Återanvändbar ArUco-kalibreringsmotor (detektion, homografi, rendering) |
| `src/engine/audio/audio_peak_detector.py` | Ljudtrigger (ffmpeg → peak-detektion) |
| `src/engine/input/hit_input.py` | Koordinattransform: kamera ↔ screen ↔ viewport ↔ content |
| `src/engine/ai/runtime.py` | AI-runtime: SimpleAIMemory, scoring, träning, choose_for_emission |
| `src/engine/ai/bootstrap.py` | Monkey-patchar HitScanner för AI-observation + candidate_limit |
| `src/engine/ai/diagnostics.py` | RoundRecord (single source of truth), FunnelTracker, ShotDiagnostics, CSV-export |
| `src/engine/ai/space_mapper.py` | Koordinatprojektion kamera ↔ screen ↔ viewport |
| `src/engine/scenes/ai_training.py` | AI-träningsscen (8 bakgrundslägen, auto-kalibrering, F1/F2 autoträning, shot-diagnostik) |
| `src/engine/scenes/ai_settings.py` | AI-inställningar (läge, vikt, export/import) |
| `src/engine/scenes/calibrate_camera_viewport.py` | ArUco-kalibrering (24 markörer, RANSAC) |
| `src/engine/scenes/calibrate.py` | Viewport-justering (skjutgränser) — INGEN AR |
| `src/engine/visual/hit_visualizer.py` | Visuella träffmarkeringar + kandidatvisning |
| `content/ai/memory.json` | AI-minne (lokalt, gitignored) |
| `content/ai/settings.json` | AI-inställningar (lokalt, gitignored) |
| `content/ai/shot_diag/` | Diagnostikbilder per skott: pre/post/diff PNG + animerad GIF (gitignored) |
| `content/settings.json` | Viewport, scanport, kamerakalibrering (lokalt, gitignored) |

---

## Träffdetektionspipeline

### Hotspot-generering (hit_scanner)
1. **Mikrofon** hör smällen → `AudioPeakEvent`
2. **HitScanner** öppnar sökfönster (`association_lag_s = 1.5s`)
3. **Pre-shot-bakgrund** byggs från frames ~250ms före `peak_ts` (fönster 200-350ms, fallback 150-500ms)
4. **Diff-signaler** (parallellt):
   - `subtract` (ref - current) → hittar mörkare hål
   - `absdiff` (|ref - current|) → hittar både ljusare och mörkare
   - `blackhat` (morfologisk) → hittar mörka fläckar oavsett bakgrund
   - `whitehat` (tophat) → hittar ljusa fläckar (LED-genomlysning)
5. **Combined** = max(subtract, absdiff, blackhat, whitehat)
6. **Konturer** → upp till `candidate_limit` (default 200, konfigurerbar) raw hotspots
7. **Known-hole penalty** → nedviktar gamla hål
8. **Shot-diagnostik** loggar per frame: konturer, rejections, patch-värden, tracking

### AI-pipeline (tvåstegs)
7. **Stage 1: Noise Rejection** (`reject_noise_hotspots`)
   - Fanns hotspotten före skottet? → reject
   - Är den persistent i flera post-frames? → behåll
   - Är förändringen för stor? → reject
   - Konservativ: hellre behålla brus än döda rätt hål
8. **Stage 2: AI Ranking** (`rank_candidates`)
   - Feature-extraktion + AI-scoring + persistence-bonus
   - Adaptiv viktning: detector → AI (10% → 65% baserat på datamängd)
9. **Funnel-diagnostik** loggar var i pipelinen rätt hål försvinner

### Multi-frame post-shot
- Sparar 5 post-shot frames vid ~50ms intervall efter skott
- Persistence-scoring: hotspot som syns i alla frames = troligt hål
- Flimmer/skuggor som bara syns i 1 frame = brus

### Tracking & Emission
10. **Tracking**: 3 hits över ≥90ms → stable → emission
11. **AI kan påverka emission** (`choose_for_emission`) i blended/ai_priority/ai_only-läge

---

## Vad AI:n ÄR och INTE ÄR

### AI:n ÄR:
- En domare över hotspots — dödar brus och rankar kvarvarande
- En inlärare som minns positiva/negativa exempel
- Kapabel att använda persistence-data för bättre beslut
- Ett lager som KAN ta över emission (blended/ai_priority/ai_only)

### AI:n ÄR INTE:
- En ersättare för hotspot-generatorn (den behöver hit_scanner)
- Kapabel att kompensera för dålig kalibrering

### AI:ns pipeline:
```
Raw hotspots (150) → Noise rejection → Surviving (~50-80) → AI ranking → Top-K → Selected hit
```

### Funnel-diagnostik (per skott):
```json
{
  "raw_hotspots": 150,
  "raw_contains_gt": true,
  "filtered_count": 80,
  "gt_survived_filter": true,
  "gt_in_topk": true,
  "selected_dist": 8.3,
  "ai_selected_correct": true
}
```

**Konsekvens:** Om detektorn inte genererar en kandidat nära det riktiga hålet kan AI:n inte lära sig rätt. Fixa detektorn/kalibreringen FÖRST.

---

## AI-träning (curriculum)

Träningsscenen har 8 bakgrundslägen i progression från enklast till svårast:

| # | Läge | Typ | Syfte |
|---|------|-----|-------|
| 1 | Vit | Stillbild | Enklast — ren bakgrund, tydliga hål |
| 2 | Vit + rutnät | Stillbild | Lägger till linjer — testar kantfiltrering |
| 3 | Koordinatrutnät | Stillbild | Finmaskigt rutnät med A0/B1-etiketter — verifierar pre/post-alignment |
| 4 | Grå | Stillbild | Lägre kontrast |
| 5 | Svart | Stillbild | Svårast statisk (kräver belysning) |
| 6 | Rutmönster | Stillbild | Hög kontrast, testar kantfiltrering |
| 7 | Rutmönster (video) | Animerad | Rör sig, fryser vid skott |
| 8 | Bubblor | Spel | Rörliga former, fryser vid skott |

TAB byter läge. Börja alltid med vit och jobba uppåt.

### Träningsflöde

1. Skjut → audio peak → hit_scanner genererar kandidater
2. AI rankar alla kandidater (upp till 50 visas)
3. Användaren klickar var kulan träffade
4. Klicket transformeras till kameraplanet via `space_mapper`
5. Närmaste kandidat inom 42px → positivt exempel
6. Max 3 sämst rankade → negativa exempel
7. Om ingen kandidat nära → syntetiskt positivt från klickpunkten
8. Sparas i `content/ai/memory.json`

---

## AI-inställningar (content/ai/settings.json)

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
| `candidate_limit` | 200 | Max raw hotspots (1-2000, propageras till HitScanner) |
| `sampling_mode` | `center_bias` | Sampling vid autoträning: center_bias/uniform/edge_bias/corners |
| `save_hole_images` | true | Spara skotthålsbilder till content/ai/holes/ |

---

## Kalibreringsflöde (rätt ordning)

1. **Justera skjutgränser** — Inställningar → Justera skjutgränser. Flytta/skala den gröna ramen.
2. **Kalibrera kamera** — Kamera → Kalibrera viewport via kamera. ENTER startar, 24 markörer projiceras.
3. **Verifiera** — kolla reprojection error och per-markör error.

**Auto-kalibrering:** AI-träningsscenen kör automatisk kalibrering vid start (~1-2s). Visar ArUco-markörer, fångar kamerabild, beräknar homografi. Om det misslyckas (kameran ser inte markörerna) fortsätter den med eventuell gammal kalibrering.

Båda kalibreringsflödena använder samma motor: `ArucoCalibrator` i `src/engine/camera/aruco_calibrator.py`.

### Krav på kalibrering

- **Mål**: 18+ markörer detekterade
- **Mål**: reprojection error < 1.0px
- **Zoner ska vara jämna** — om TL/TR/BL/BR har mycket olika error är kalibreringen skev
- Per-markör error: grönt < 1px, gult 1-3px, rött > 3px

**Om kalibreringen inte uppfyller kraven:** kör om. Kontrollera att projektorn visar rätt bild, att kameran ser alla markörer, och att inget skymmer dem.

---

## Felsökning (playbook)

### När träffar inte registreras rätt:

```
1. Kör 3×3 zontest (skjut systematiskt i 9 zoner)

2. Kolla: finns kandidat nära hålet i kandidatlistan?

3. Om NEJ → DETEKTIONSPROBLEM
   - Kolla kalibrering (reprojection error, zonbias)
   - Kolla pre-shot diff (projektorflicker?)
   - Kolla ljusförhållanden (svart bakgrund + mörkt rum = ingen signal)
   - Kolla ROI-mask (täcker den hela tavlan?)

4. Om JA men fel rank → RANKINGPROBLEM
   - Kolla known-hole penalty (gamla hål tar plats?)
   - Kolla score-formel (center_change, local_contrast, pre_shot_change)
   - Kolla AI-scoring (om AI-läge ≠ train_only)

5. Om JA och rätt rank men fel position → TRANSFORMPROBLEM
   - Kolla homografi (är _prefers_homography() True?)
   - Kör om kalibrering
   - Kolla viewport-inställning
```

### Vanliga problem och orsaker:

| Symptom | Trolig orsak |
|---------|-------------|
| Zonbias (en sida sämre) | Kalibrering — kör om med fler markörer |
| Alla kandidater på kanter/tejp | Pre-shot diff fungerar inte — projektorflicker |
| Inga kandidater alls | Ljus för lågt, eller ROI-mask täcker inte tavlan |
| Gamla hål rankas högre än nya | Known-hole penalty för svag |
| Svart bakgrund = inga träffar | Whitehat + absdiff fångar nu ljusa hål. Kräver fortfarande viss belysning |
| Kandidat finns men på fel plats | Homografi inte aktiv — kolla `_prefers_homography()` |
| Luftgevärshål hittas inte | Kontrollera max_area (900) och max_radius (35) — stora hål behöver höga gränser |
| Inga kandidater trots tydligt hål | Kör shot-diagnostik (loggas automatiskt) — kolla [SHOT-DIAG] i terminalen |
| Pre/post-bilder visar fel ställe | Kolla pre_age i [REVIEW]-loggen — bör vara 200-350ms. Kolla shot_diag/ GIF:ar |
| Pre-shot innehåller redan hålet | Pre-shot timing för nära skottet — kolla [AI PRE-SHOT] offset_from_peak |

---

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

---

## Historiska buggar och lärdomar

| Vad vi testade | Resultat | Lärdom |
|----------------|----------|--------|
| `absdiff` istället för `subtract` | Sämre — projektorflicker ger diff överallt | Behåll `subtract` |
| Vote-map (multi-frame persistens) | För tungt på 4K, opålitligt | Undvik tunga per-frame-operationer |
| Feature-matching (nya vs gamla konturer) | För tungt, samma problem | Enkel diff + blackhat räcker |
| Blackhat borttagen ur combined | Regression — missade hål | Blackhat är robust, behåll den |
| Pre-shot-change med 0.80 vikt | För dominant, maskerade riktiga hål | Håll pre-shot som bonus (0.25) |
| Homografi beräknad men inte använd | Zonbias (höger bra, vänster dålig) | Kontrollera `_prefers_homography()` |
| 9 ArUco-markörer | 2.13px error, dålig i hörnen | Använd 24 markörer + RANSAC |
| `absdiff` som enda diff | Projektorflicker ger diff överallt | Använd subtract som primär + absdiff som komplement |
| Fasta center/ring-masker i patch | Stora hål (luftgevär) fick noll kontrast | Adaptiva masker baserat på konturradie |
| Duplicerad ArUco-kod i scener | Svårt att underhålla | Extrahera till ArucoCalibrator-motor |
