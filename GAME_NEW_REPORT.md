# PANG! – Ljusateljén / Ljusjakten

2026-11-22, `claude/latency`, för granskning. Inga commits/push/merge.
**Kamera/projektor orörda. Nya rummet är inte fysiskt validerat; ≥95 % är målet.**

## Spelbar version

Ett ljust, förrenderat Blender-rum med **åtta föremål åt gången, i varierade storlekar** på golv, bord,
bänk, hylla och fönsterbräda. Fem modeller: blomvas, kruka, skål, flaska och lampa;
elva funktionella deltyper. Placeringarna varierar mellan spelarturer och när ett
rum är tömt. Skador och skärvor ligger kvar mellan skotten – man plockar isär rummet.
1–2 spelare, 5/10 skott, inget tidstvång. ENTER startar, 1/2 spelare, 5/0 skott,
ESC meny, **M tystar ljuden**.

- Blomskott: blomman faller, vasen står kvar. Halsskott: hals/överdel faller,
  buken står kvar. Bukskott: kroppen krossas och överdelarna lossnar.
- Lock, skålens kant och lampskärm har egna former och fallsekvenser.
- Bekräftad del ger 100–200 poäng, svitbonus upp till +100 och +500 för tömt rum.
  Strukturellt lossnade syskon ger **inte** extra fysiska träffar eller delpoäng.
- Träffens observerade sida styr fragmentimpulsens riktning. Damm, lokal träffring,
  keramikkrasch/klonk/blomljud kommer först efter beslutet och skyddat POST-fönster.

## Mekanik, former och mätbarhet

Befintliga `GameObject.entity_id/part_id`, `make_breakable_object`, ObjectManager,
frysta former, OBJECT dense och audio `play_feedback` återanvänds.
Den enda nya generella motormekaniken är **opt-in `EffectAction.break_parts`**:
namngivna syskondelar inom samma ägare/entity kan få en uttrycklig strukturell
brottreaktion. Den skapar inga nya observerade skott eller koordinater. Andra spel
utan denna action behåller sitt beteende.

Varje del har en egen **synlig alpha-mask från samma renderkamera**, polygon/union
och projektil-touch 2,25 mm. Delmaskerna överlappar inte som raster. Flera möjliga
parts vid en fysisk delgräns ger UNKNOWN; ingen väljs med listordning eller 3D-djup.
Maskexporten rättar bara instängda ID-antialiasprickar ≤4 källpixlar, loggar antalet
och vägrar riktiga hål i polygonen. Alla former motsvarar de exporterade pixlarna;
kontur/nedskalning har fortfarande ungefär en källpixels gränsosäkerhet.

Kvarliggande skärvor ritas **bakom** aktiva delar, så de inte döljer något som
fortfarande är träffbart. Flygande skärvor får korsa rummet bara när frågorna är
stängda. Detta är en explicit renderregel, ingen z-ordning gissad från fysiskt XY.

Från första nya presentation väntar spelet **1,7 s före REDO** (1,5 s + marginal).
Hela bilden fryses vid skott. Reaktionen väntar på slutbeslut **och peak+1,2 s**.
Animationens klocka startar när den faktiskt börjar renderas, så en huvudloopspaus
inte hoppar över effekten. Därefter nytt stilla PRE-fönster. Fördröjda ljudtoppar
från en äldre spelperiod tar inte ett nytt försök. Ljudutmatning loggas via motorn;
**ingen ny mikrofonspärr eller ljudtröskel** införs. Kontrollera fysisk återkoppling
med ljud både på och av; digital volym är ingen akustisk kalibrering.

Paletten kontrolleras med Jimmys kameramodell: exporterade målytor ≥128 DN,
kompositerade spelbilder ≥121,09 DN. Detta är en prediktion; ingen ny fysisk ljusmätning.
HUD/text ligger ovanför alla mål. Detta är digital kontroll, inte ny kameramätning.
UNKNOWN/timeout ger inga poäng; CURRENT-XY och skuggmodellen ger aldrig spelpoäng.

## Start och granskning utan hårdvara

Samma meny/API-id **`light_targets`**, ny titel ”PANG! – Ljusateljén”.

```bash
python3 -m automation.play_light_targets 2>&1 | tee /tmp/light_targets_room.log
# alternativ: --menu eller --api (redan körande app)
```

API: `send_command('startGame', {'id':'light_targets'})`. Befintliga launcher-defaults
för dense, .15 s cooldown, source/window recording och avstängda fullspår är
bevarade, liksom användarens fönsterplacering. Ingen ny GAME_DX.

**Utan App, display, mixer, kamera eller projektor:**

```bash
python3 -m automation.light_room_preview --out /tmp/light_room_preview_NEW
```

Ger PNG för intakt rum samt blom-/hals-/bukträff och separata WAV-ljud att granska.
Aktuell R3-förhandsvisning: `/tmp/pang_r3_delivery/`, även MISS/UNKNOWN och fem WAV-ljud.
Förhandsprocessen mättes till cirka 159 MiB topp-RSS och 2,0 s (inte hela live-appens RAM).
Grafik/sönderfall förladdas före mätfönstret; ingen bildläsning i träffreaktionen.

## Tillgångar och Blender-pipeline

Leveransen använder **egna procedurmodeller**, inte nedladdade modeller. Nya
render-/masktillgångar levereras med **CC0-1.0**, se
`content/games/light_room_parts_assets/{SOURCES.md,LICENSE.txt,manifest.json}`.
Originala genererade ljudvågformer är också CC0-1.0, utan externa samplingar.
Befintlig projektkod omlicensieras inte. Paketet innehåller 166 PNG och manifest
(cirka 4,2 MiB); gamla tre-sprite-provet bevaras separat.

Verifierade fria alternativ, **ej importerade**, inga konton skapade:

| Källa | Tillgång/licens | Användning |
|---|---|---|
| [Kenney Furniture Kit](https://kenney.nl/assets/furniture-kit) | CC0 | Bord/hyllor/dekor |
| [Poly Haven Jug 01, Kuutti Siitonen](https://polyhaven.com/a/jug_01) | [CC0](https://polyhaven.com/license) | Kanna, manuella funktionsdelar |
| [Quaternius Ultimate Furniture](https://quaternius.com/packs/ultimatefurniture.html) | CC0 | Alternativ möblering |
| [Blender demo files](https://www.blender.org/download/demo-files/) | Impact Fracture, Simon Thommes, listad CC0 | Fracture-referens; ej hämtad |
| [Cell Fracture](https://extensions.blender.org/add-ons/cell-fracture/) | GPL-3.0-or-later, **verktyget** | Offline-fragment, inte modelllicens |

Lokal hämtning saknar DNS i sandlådan. Därför är alternativen dokumenterade och
inte felaktigt märkta som använda. Inget overifierat Sketchfab-material införs.

Pipeline: modell+källa/licens/hash → ljusa material → **manuella funktionella
parts först**, sedan Cell Fracture eller sektorskärvor inom respektive part →
intakta/delvisa/brytande sprites → synliga masker → frysta polygoner/union.
Aktuell version använder manuella meshsektorer, ingen installerad addon eller ny
realtidsrenderer. Receptet har spärr mot körning i interaktiv Blender.

```bash
/snap/blender/current/blender -noaudio -b --factory-startup \
 --python automation/assets/render_light_room_parts.py -- --out /tmp/room_raw_NEW
python3 -m automation.build_light_room_parts /tmp/room_raw_NEW /tmp/room_bundle_NEW
python3 -m automation.recolor_light_room /tmp/room_bundle_NEW /tmp/room_distinct_NEW
```

Blender 5.3.0, CPU/två rendertrådar. `--room-only` och `--only-kind vase` stöder
iteration. Alla skrivningar avslutas före explicit processavslut, vilket undviker
sandlådans PulseAudio-teardown-hängning. Ingen användaröppen Blender-scen används.

## A. Tidigare tio fysiska luftgevärsskott

Session `20261122T135436Z_7c511b31`, `live_ljusjakten2.log`: **9/10 rätt, ett osäkert**.
Approximativ `can_dense_offline`-crop-replay återger 10/10 livebeslut med sparad
sessionskalibrering. Skott 9: MF 21,28, core 0,051, toppkvot 1,142. Ingen tröskel
ändrades efter detta enda svaga positiva exempel. Negativa kontroller saknas.
Skuggmodellens ambiguitet är höga p både inne/utanför målet, inte ett upprepat id.

Reproducerbart verktyg: `automation.light_targets_review SESSION LOG TRUTH NEW.json
--model MODEL.json`. Resultat `/tmp/light_room_review/review_final.json`; ungefär
301 MiB topp-RSS och 2,81 s. Detta är **inte** live-path-ekvivalent replay och
validerar inte nya rummet. Baselinefilerna är orörda. Efter delmekaniken kontrollerades
samma 9/10 igen: `/tmp/light_room_v3_review/circle_baseline.json`.

## Fysisk acceptans och tester

Samla nya operatörsfacit: minst 60 positiva skott över delar/stödytor/positioner
plus 20 negativa kontroller, inklusive missar, delgränser och ljud utan hål.
Registrera även skott utan ljudhändelse. Facit måste komma från verkligt hål och
fryst form, **inte** träffring eller spelreaktion. Exempel:

```json
{"owner":"light_targets:…","shot_id":42,"expected":"HIT","truth_object":"vase_1_3:neck"}
{"owner":"light_targets:…","shot_id":43,"expected":"MISS"}
{"owner":"light_targets:…","shot_id":null,"attempt_id":"operator-44","expected":"HIT","truth_object":"jar_1_6:body"}
```

`python3 -m automation.light_targets_score RUN.log truth.jsonl` räknar fel del,
UNKNOWN och uteblivet resultat som fel på känt facit. Versions-/part-id och faktiska
reaktions-/ljudtider finns i LAMP-/OBJECT-loggarna. Kräv ≥95 % strikt korrekthet och
hit recall samt noll fel del/falska poäng; redovisa Wilson-intervall och nya holdouts.
9/10 är 90 %, och inget godkänt 95 %-resultat har påståtts.

Relevanta självtester: `pang_r3_selftest`, `breakable_parts_selftest`, `light_room_selftest`,
`light_targets_selftest`, `object_physical_query_selftest`, `object_hit_selftest`,
`object_readiness_selftest`, `object_dense_pair_selftest`, `object_disambiguation_selftest`,
`edge_rule_selftest`, `v250_selftest`. Körda isolerat med privata `/dev`, `/tmp`,
`/data`, skrivskyddat repo och SDL dummy. Compileall och diff-check ingår.

Alla elva moduler är gröna: bland annat 14 rums-/deltester, 5 motor-deltester och
9 tidigare spelkontrakt. Rumsdelen provar även riktiga dense-frågor på renderade
fixtures, negativa bildfall, mutning/tidsgränser, skärvocklusion, del-UNKNOWN och
rum-rensat-bonus. De syntetiska bildfallen är inte fysisk validering.

## Runda 3: PANG1-granskning och tydligare rum

Sparad fysisk session `20261122T151030Z_e3f0e267`, `live_pang1.log`:
query-matematik med sparade bildrutor/former återger **10/10 livebeslut**:
7 HIT, 3 UNKNOWN. Detta är offline query-replay, inte full live-schemaläggning.
Vit efterbild och kausal bildserie granskas separat, aldrig som detektorindata:

| Skott | Eftergranskning | Slutsats |
|---|---|---|
| 8 | Nytt hål kamera (2252,27;1598,27), spel (384,86;499,02) | Utanför alla aktiva och ursprungliga objektformer; närmaste aktiva kamera-AABB 93,23 px bort |
| 9 | Inget nytt verifierat hål identifierat | Modellens favoritpunkt har bara 3,32 DN lokal kontrast i vit efterbild; 87,47 px till närmaste granskningshål |
| 10 | Nytt hål kamera (2228,70;1433,20), spel (370,06;346,26) | Utanför alla aktiva och ursprungliga former; närmaste aktiva kamera-AABB 159,80 px bort |

Nio granskade hål har 17,15–30,49 DN lokal kontrast. Skott 9 kan exempelvis
överlappa ett befintligt hål; frånvaro av verifierat nytt hål bevisar inte uteblivet
skott. Hög skuggmodell-p är inte facit. UNKNOWN behålls; ingen regeltröskel eller
modellauktoritet ändras. Objektfrågan täcker bara objektens områden: ett hål utanför
är inte i sig ett säkert globalt MISS-besked från denna väg.

Senaste scenpresentation låg 12,97/11,93/14,73 s före skott 8–10. PRE p99-variation
var 2–3 DN; skålens medelnivå 172,1/172,2/172,0 DN. Ingen evidens för pågående
skärvanimation eller kraftig kontrastförlust i dessa PRE-fönster. Kvarliggande
skador ändrar scenen, men var stabila här.

**Digitala ändringar:** åtta objekt, höjd 78–132 px i standardlayout, mättade
turkos/grön/gul/cyan och neutral bakgrund ungefär grå 194–214. Grå 160/170 ger enligt
modellen bara 96,54/105,03 DN; därför kan grå 140–170 inte samtidigt lova ≥120 DN.
Genomsnittligt RGB-avstånd objekt–rum ökar 29,91→188,26; målfärgernas kanalspann
40,06→204,40. Detta beskriver bildkontrast, inte detektionsprecision.
Exportminimum 128,0039 DN; elva kompositerade förhandsbilder minimum 121,0947 DN
på spelytan under HUD. Alla alpha-masker och träffformer bevaras.

Delträffar färgar kvarvarande delar kontrasterande; helt/kvarstående/krossat visas
även i status. HIT/MISS/UNKNOWN har olika text och ljud; osäkert döps inte om till
miss. Tolv brytbilder vid 20 fps ger **0,6 s** animation och resultatpaus 0,95 s.
Skyddet peak+1,2 s och ny PRE-stabilisering 1,7 s kvarstår. Trettio layouter med
sparad 4K-kalibrering använder högst **642000/1000000** tillåtna ROI-läspixlar.
Detta är budgetkontroll, inte uppmätt live-latens.

Reproduktion (nya utmappar, inga baselineöverskrivningar):

```bash
python3 -m automation.pang_session_audit \
 /data/skjutbana/source_sessions/20261122T151030Z_e3f0e267 \
 /data/skjutbana/evaluation_runs/can_real_20261122/live_pang1.log \
 /data/skjutbana/evaluation_runs/can_real_20261122/board_captures_pang1/white.png \
 /tmp/pang_audit_NEW
python3 -m automation.pang_layout_budget \
 /data/skjutbana/source_sessions/20261122T151030Z_e3f0e267 \
 /tmp/pang_budget_NEW.json --seeds 30
```

Audit: `/tmp/pang_r3_audit/final/{audit.json,white_review.png}`. Digital palett:
`content/games/light_room_parts_assets/palette_report.json`; kompositkontroll
`/tmp/pang_r3_audit/composited_palette.json`. Fryst ursprungspaket finns i
`/tmp/pang_r3_audit/baseline_assets`, reproducerbart från Git
`afdaba31343d3583510a314a76ee8707b13243dd` (assetkatalogen).
Recolor-verktyget kräver v2-input, sparar hash/proveniens och vägrar befintlig utrot.
Testloggar: `/tmp/pang_r3_final_checks/`, inklusive sju nya R3-tester.

R3 behöver ny fysisk pilot, särskilt kantträffar, missar och ljudåterkoppling.
Ingen förbättrad fysisk träffprocent eller uppnådda 95 % påstås av dessa ändringar.
Appen behöver läsa den nya koden; starta inte en andra kameraägare.
