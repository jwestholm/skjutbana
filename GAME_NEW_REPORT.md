# PANG! – rummen

2026-11-23, lokal `dev`-integration. Ingen push.
**Kamera/projektor orörda. ≥95 % fysisk träffsäkerhet och visuell acceptans är inte verifierade.**

**Senaste leverans: R8 (2026-11-23).** Tre rum × tre möbleringar, naturliga
rumstexturer och valfri `object_highlight`-presentation. Se R8-avsnittet först;
R3–R7 nedan beskriver tidigare leveranser.
Förhandsbild: [/tmp/pang-dev-r8-pastel-living/show_targets.png](/tmp/pang-dev-r8-pastel-living/show_targets.png).

## R8 — naturliga rum och diskret målupplysning (2026-11-23)

- Nio naturliga rumsversioner (`living|kitchen|study` × tre möbleringar) med
  318 paketerade PNG-filer. Möbler och objekt använder lokala CC0 Poly Haven-
  källor; vägg- och golvytor använder dokumenterade CC0-atlasutsnitt där en
  separat arkitekturtextur inte fanns tillgänglig. Manifestet låser fil- och
  recept-hashar. R7-paketet lämnades separat.
- Motorflaggan `object_highlight` är av som standard. PANG-spelet kan slå på den
  via `SKJUTBANA_PANG_OBJECT_HIGHLIGHT=1`, `--object-highlight` eller G i spelet.
  `--object-glow` och den äldre `SKJUTBANA_PANG_OBJECT_GLOW` fungerar som alias.
  En uttrycklig `precision_flash=off`/miljö-off stänger av presentationen.
- PANG-launchern kör nu `automation.bench calibration-check --save` före
  appstart när `--object-highlight` används. Den visar befintliga ArUco-markörer,
  tar en ny kamerakalibrering och använder befintlig backup-/linsmodellspolicy.
  Misslyckad kalibrering stoppar appstarten. Kamera och ljud spelas sedan in i
  en ny rå session; efteråt kan `automation.bench replay --export-candidates`
  spela samma indata genom den inspelade live-pipeline-konfigurationen.
- Endast objektens frysta synliga silhuetter ljusas upp, i en mildare skala av
  samma ursprungliga färg. Varje objekt får en tunn ljusgul ring. De större
  cirklarna används bara för mätning och visas inte som fyllda fält. Markeringen
  försvinner direkt efter den korta mätvisningen; återstående analys kan fortsätta
  i bakgrunden. Skador får bara minska den frysta ytan.
- `automation.pang_glow_preview` renderar mjukvaruförhandsbilder och ett
  simulerat klick. Tre rumstyper har granskats visuellt; alla nio naturliga
  rumsbilder exporterades. Detta är inte kameradata eller fysisk träffvalidering.
- Självtesterna för `object_highlight` (11), `object_glow` (19) och
  `precision_flash` (19) passerar; hela sviten klarade 169/169 moduler.
  Digital pixel-/tidsprobe använde 12 frön × fyra bakgrunder × sju scenarier
  (336 fall), syntetisk 720×480-kamerabild och Gaussian-darkening, inte verkliga
  hålstämplar. Tydlig träff, närmiss, gammalt+nytt, inget hål, utanför mask och
  ljusning matchade förväntan i 48/48 fall vardera; den avsiktligt svaga
  signalen blev UNKNOWN i 48/48 fall. Inga oväntade träffutdelningar. Detta är
  digital pixel-/tidsverifiering, inte fysisk noggrannhet eller full
  source-session replay.

Källor: `content/games/light_room_natural_assets/`,
`automation/assets/pang_rooms_r8.py`, `automation/object_glow_bench.py` och
`automation/object_glow_replay.py`. Operatörsprov med kamera/projektor behövs
fortfarande för faktisk synlighet, timing och träffsäkerhet.


## Spelbar version

Ett ljust, perspektiviskt Cycles-vardagsrum byggt av lokala Poly Haven-modeller
med PBR-texturer: soffa, bokhylla, soffbord, sidobord, dagsljusfönster och tavelram. **Åtta stora föremål åt gången** ur tio typer:
bladvas, keramikvas, gryta, slank vas, armlampa, retro-TV, böcker, krukväxt,
väckarklocka och mjölkkanna. Interna familje-id:n behålls för loggkompatibilitet. Totalt
21 funktionella deltyper. TV och böcker finns i varje rum; övriga nya typer varierar.
Placeringar och storlekar varierar mellan spelarturer och när rummet töms.
Skador och skärvor ligger kvar mellan skotten. Fönster, soffa och väggtavla är
bakgrundsdekor i denna version; de åtta lösa föremålen är poänggivande mål.
1–2 spelare, 5/10 skott, inget tidstvång. ENTER startar, 1/2 spelare, 5/0 skott,
ESC meny, **M tystar ljuden**.

- Bladträff: bladverket faller, vasen står kvar. Halsskott: hals/överdel faller,
  buken står kvar. Bukskott: kroppen krossas och överdelarna lossnar.
- Lock, skålens kant och lampskärm har egna former och fallsekvenser.
- Bekräftad del ger 100–200 poäng, svitbonus upp till +100 och +500 för tömt rum.
  Strukturellt lossnade syskon ger **inte** extra fysiska träffar eller delpoäng.
- Träffens observerade sida styr fragmentimpulsens riktning. Damm, lokal träffring,
  keramikkrasch/klonk/blomljud kommer först efter beslutet och skyddat POST-fönster.

## Mekanik, former och mätbarhet

Befintliga `GameObject.entity_id/part_id`, `make_breakable_object`, ObjectManager,
frysta former, OBJECT dense och audio `play_feedback` återanvänds.
Den tidigare införda mekaniken **opt-in `EffectAction.break_parts`** återanvänds:
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
kompositerade spelbilder ≥124,28 DN. Detta är en prediktion; ingen ny fysisk ljusmätning.
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
python3 -m automation.light_room_preview --out /tmp/light_room_preview_NEW --all-parts
```

Ger PNG för intakt rum samt blom-/hals-/bukträff och separata WAV-ljud att granska.
Aktuell R5-förhandsvisning: `/tmp/pang_r5/delivery/`, även nya delreaktioner, MISS/UNKNOWN och fem WAV-ljud.
Den äldre R3-förhandsprocessen mättes till cirka 159 MiB topp-RSS och 2,0 s;
detta är inte en RAM-/tidsmätning av R5 eller hela live-appen.
Grafik/sönderfall förladdas före mätfönstret; ingen bildläsning i träffreaktionen.

## Tidigare tillgångar och Blender-pipeline (R3–R5)

Leveransen använder **egna procedurmodeller**, inte nedladdade modeller. Nya
render-/masktillgångar levereras med **CC0-1.0**, se
`content/games/light_room_parts_assets/{SOURCES.md,LICENSE.txt,manifest.json}`.
Originala genererade ljudvågformer är också CC0-1.0, utan externa samplingar.
Befintlig projektkod omlicensieras inte. Paketet innehåller 316 PNG och manifest
(cirka 8,1 MiB); gamla tre-sprite-provet bevaras separat.

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
 --python automation/assets/render_light_room_parts.py -- --out /tmp/room_raw_NEW --living-room
python3 -m automation.build_light_room_parts /tmp/room_raw_NEW /tmp/room_bundle_NEW --realistic
python3 -m automation.recolor_light_room /tmp/room_bundle_NEW /tmp/room_distinct_NEW --realistic
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

## Runda 4: PANG2, två verkliga men svaga hål

Session `20261122T161620Z_4b470454`, byggd från `858486d`: operatören anger
10/10 verkliga objektträffar. Live gav 8 HIT och 2 UNKNOWN, inga felaktiga poäng.
Samma sparade query-matematik återger 10/10 beslut. Separat vit efterbild bekräftar
båda bortsorterade punkterna; den används aldrig som detektorindata.

| Skott | Objekt | MF S | core | bipolar | S/S2 | Orsak |
|---|---|---:|---:|---:|---:|---|
| 8 | bowl_1_3:rim | 24,207 | 0,1098 | 0,1797 | 1,337 | core <0,15; isoleringskvot <1,5 |
| 9 | vase_1_7:body | 23,823 | 0,0976 | 0,1297 | 1,335 | core <0,15; isoleringskvot <1,5 |

Högsta peak ligger kamera (2518,1394) respektive (2132,1261), mot operatörens
(2519,1394)/(2132,1261). Det är inte recall-/rankingfel. Konkurrenterna har core
ungefär 0,012 men räknas ändå i isoleringskvoten som avsiktligt skydd mot fel.
Lokalt runt hålen är PRE 152/160 DN, patch-standardavvikelse 1,34/1,45 DN;
ringkorrigerad förändring 14,68/13,13 DN. Tre separata POST-bilder ger 14,38–15,71
respektive 12,49–15,72 DN: ihållande men svag mörkning. Ingen tydlig fintextur eller
kantstörning vid dessa punkter. Aktiva regioners PRE p99-variation 3–5 DN. Skott 8
kom 10,76 s efter loggad presentation; skott 9 ändrar inga delar och har därför ingen
egen presentationsmarkering. PRE-data ger inget stöd för pågående skärvanimation.

**Säkerhetsbeslut:** ingen ändring i detektor, trösklar, modell eller poängauktoritet.
Att sänka core till under dessa två mätvärden vore efteranpassning.
Dessutom hade PANG1:s skott 9 S=21,7667 och isoleringskvot **1,334117**,
nästan samma som PANG2:s svaga hål: en kvotgräns 1,33 skulle också släppa igenom
den gamla obestyrkta punkten (white-spot bara 3,32 DN). Detta är ett konkret
motexempel till en ren kvotsänkning, inte en ny acceptansregel. Ett möjligt
separat experiment är en bruskalibrerad verifiering av grunda, persistenta toppar,
innan svaga konkurrenter kan undantas från kvoten. Det kräver frysta negativa
kontroller och samtliga tidigare fysiska sessioner; dessa två positiva exempel
bevisar inte noll nya fel. Ingen förbättrad detektionsprocent påstås här.

R4 byter grafikpipeline: Cycles-material med glasyrlager, IOR 1,46 och breda
reflexer, mjuk key/fill-belysning samt bevarad rumsskuggning. Opaka keramiska/
porslinsliknande material prioriteras framför verkligt transparent glas, som
skulle göra hålsynligheten innehållsberoende. Inga mörka texturmönster läggs till.
Renderfärgerna behålls; bara pixlar under kameramodellens golv lyfts minimalt mot
vitt. R3:s relativa gråskaleomfärgning används inte för normalmaterialen.
Modellerna är fortsatt egna CC0-procedurmodeller; inga externa modeller påstås
vara importerade. Delindelning, maskexport och manuellt delade sönderfall återanvänds.
Profilerna interpoleras mjukt mellan ursprungliga knutar, med begränsad radie
så att inga extra smala kanter skapas; synliga masker och träffpolygoner exporteras
på nytt från samma kamera. Nya former är alltså inte gamla masker över ny grafik.

Reproduktion:

```bash
# Separat headless Blender, aldrig användarens interaktiva Blender:
/snap/blender/current/blender -noaudio -b --factory-startup \
 --python automation/assets/render_light_room_parts.py -- --out /tmp/pang4_raw_NEW
python3 -m automation.build_light_room_parts /tmp/pang4_raw_NEW /tmp/pang4_pack_NEW --realistic
python3 -m automation.recolor_light_room /tmp/pang4_pack_NEW /tmp/pang4_final_NEW --realistic
```

Analysfiler: `/tmp/pang_r4/audit/{audit.json,white_review.png}` och
`/tmp/pang_r4/hole_details.json`, patchbilder `shot8_patches.png`/`shot9_patches.png`.
R3-baslinjetillgångar sparade i `/tmp/pang_r4/baseline_assets` före ändring.
Nya material måste fortfarande godkännas visuellt av Jimmy och mätas fysiskt.

### Leverans och slutkontroll R4

- **166 PNG, 5 178 633 byte** inklusive paketmetadata. Vanliga målytor min
  128,0008 DN, delträffstillstånd 128,0005 DN, rum 128,0004 DN enligt proxyn.
- **Elva sammansatta förhandsbilder:** minimum **124,3115 DN** under HUD, både
  intakt, sönderfall, efterläge och MISS/UNKNOWN. Dessa bilder är mjukvarurenderade,
  inte fysiska kameramätningar. Förhandsbild: `/tmp/pang_r4/preview/room.png`.
- **30 layouter:** högst 642000/1000000 ROI-läspixlar; sparad PANG2-kalibrering.
  `/tmp/pang_r4/budget.json`. Metadata binder resultat till aktuell asset-manifesthash.
- **Tolv självtestmoduler gröna**, inklusive tre nya R4-materialtester, befintliga
  R3-färg-/timingtester, 14 rums-/deltester och OBJECT-regressioner. Loggar:
  `/tmp/pang_r4/final_checks/`. Compileall och `git diff --check` godkända.
- Meny/API-id är fortsatt `light_targets`; loggversion `ceramic-room-parts-4` och
  assethash skiljer nya bygget från de fysiskt testade R3-bilderna. Inga commits.

Råpaketet `/tmp/pang_r4/raw_assembly` använder de färdiga objektrenderingarna från
`raw_complete` och den färdiga rumsrenderingen från `raw_v6` (samma rumsgeometri,
material, ljus, kamera; exponering −0,35, 32 samples). Tidigare mörkare rumsförsök
bevaras. `source_provenance.json` och `raw_inputs_sha256` anger ursprung och hash.
Stillbilder använder 32 samples och rörelsebilder 8; samtliga renderas CPU/två trådar.
Arbetet delades per `--only-kind` efter en avbruten lång export; `--resume` återanvänder
färdiga råbilder i just samma oförändrade export. Använd nya utmappar för nya recept.

Nya paket kan granskas före installation:

```bash
python3 -m automation.light_room_preview --assets /tmp/pang4_final_NEW --out /tmp/pang4_preview_NEW
```

R4 förbättrar material-/formåtergivningen men är inte den större fotorealistiska
vardagsrumsleveransen. Visuell acceptans och ett nytt fysiskt försök återstår.
Det är inte visat att de nya materialen ökar träffdetekteringen; inga nya fysiska
noll-fel/noll-falska eller 95 %-resultat påstås.

## Runda 5 – vardagsrummet

**Levererat i worktree, utan commits eller kamera/projektoråtkomst.** Samma
meny/API-id `light_targets`, titel ”PANG! – Vardagsrummet”, loggversion
`living-room-parts-5`. Startskriptet och dess opt-in OBJECT/source-recording-miljö
är bevarade. Inga globala detektorinställningar, trösklar eller modeller ändras.

Blender-rummet har perspektivkamera (50 mm), kontinuerligt golv, mineralfärgade
väggar, ljus ek, breda stoppade soffkuddar, bokhylla, matta och fönsterljus.
Föremålen är fortfarande friställda sprites från samma exportkedja och har
silhuettpolygoner från synliga partmasker. Ljusa, stora och förenklade ytor
prioriteras framför mörk träådring, text och finmönster. Det är en ljus egenbyggd
Cycles-tolkning av vardagsrummet, inte importerade fotogrammetrimodeller eller en
påstått verifierad fotorealistisk slutkvalitet. Jimmy behöver bedöma det faktiska
utseendet i förhandsbilden.

| Typ | Träffdel och efterförlopp |
|---|---|
| TV | Skärmen får ljusa sprickor och ligger kvar; foten är separat och kan fälla skärmen |
| Böcker | Vänster/mitten/höger bok är separata; bara den träffade volymen faller |
| Krukväxt | Bladverket faller separat; krukan krossas med tre stora jordklumpar och lossat bladverk |
| Mugg | Handtaget kan lossna ensamt; kroppsträff krossar muggen och lossar handtaget |
| Klocka | Glastavlan får sprickor |
| Tidigare keramik | Hals/buk/blomma/lock/kant/lampskärm enligt tidigare delregler |

TV:n visar en egen ljus, helt stilla procedurbild. Klockvisarna är också stilla;
inga omgivningsanimationer eller subtilt fladdrande videobilder körs under mätning.
Skyddad POST till peak+1,2 s, 1,7 s ny PRE-stillhet, 0,6 s reaktion och 0,95 s
resultatpaus är bevarade. Kvarvarande delar behåller sina material efter en träff;
saknade delar, sprickor, skärvor och HUD visar skadan. Inga extra poäng ges för
delar som bara lossnar strukturellt. Fragmentens nya fallbanor stoppas vid deras
lokala stödyta. Detta är förrenderade, manuellt författade reaktioner, inte en ny
live-3D-/rigid-body-motor.

Placeraren använder begränsad backtracking och placerar stödkrävande föremål
först: TV på mediabänk/bord, böcker i bokhyllan, övriga på passande stöd. Det
åtgärdar en verklig regression där den gamla giriga algoritmen kunde returnera
ett tomt rum vid 640×480 trots att en giltig layout fanns. Målens storlek och
icke-överlappande former behålls; slitaget sprids genom varierade ankare.

### Digitala mätningar och regression

- **10 typer, 21 delar, 316 PNG; 8 506 794 byte** inklusive paketmetadata.
- Exporterade vanliga målytor: minimum **128,0004 DN** enligt den givna proxyn.
- **31 kompositerade spelbilder**, inklusive samtliga nya delreaktioner:
  minimum **124,2830 DN** under HUD. Källa `/tmp/pang_r5/composited_delivery.json`.
- **30 layouter** på sparad PANG2-kalibrering: max **591840/1000000** läspixlar,
  inga budgetfel. `/tmp/pang_r5/budget_final.json`. Detta är geometri/resurskontroll,
  inte uppmätt live-latens eller fysisk precision.
- **13 självtestmoduler gröna**, inklusive sju nya R5-tester. Kontrollerna omfattar
  TV/stand-ägarskap, enskilda böcker, kruka/blad, muggens strukturella delar,
  alpha-kanter, hash/proveniens, 640×480-layout, frysta former och skyddad timing.
  `/tmp/pang_r5/final_checks/results.json`; relevanta OBJECT-regressioner ingår.
- Kontaktskuggornas första version gav mörka resamplingsfransar (minimum 113,12 DN).
  Orsak: svarta RGB-värden i transparenta pixlar. De färglades före skalning;
  ett regressionstest skyddar nu mot felet. De tidigare bilderna är bevarade.
- Färgskyddet använder nu en exakt RGB-uppslagstabell för upprepade färger.
  Ingen sampling: det är byte-identiskt mot den frysta pre-optimeringsfixturen
  (`4ebd2785…f3f624`), och de befintliga materialtesterna är gröna.

Dessa är **mjukvaru-/renderkontroller**. Nya fysiska träffar, missar, delgränser,
redan spruckna ytor och ljudåterkoppling måste fortfarande testas med operatörsfacit.
Ingen ny fysisk noll-fel/noll-falska eller ≥95%-nivå påstås. R4:s detektoranalys
förblir giltig; ingen poängauktoritet har öppnats för de två grunda hålen.

### Källor, licenser och reproduktion

Poly Havens [licens](https://polyhaven.com/license) är CC0, men den lokala
filhämtningen från `api.polyhaven.com` misslyckas med `Temporary failure in name
resolution`. **Inga Poly Haven-/Kenney-/Sketchfab-modeller ingår.** Samtliga nya
modeller, renderbilder, TV-motiv och befintliga syntesljud är egna och levereras
som CC0-1.0; se paketets SOURCES.md/LICENSE.txt. Befintlig projektkod omlicensieras inte.

```bash
# Separat headless CPU-Blender, aldrig den interaktiva Blender-sessionen:
/snap/blender/current/blender -noaudio -b --factory-startup \
 --python automation/assets/render_light_room_parts.py -- --out /tmp/living_raw_NEW --living-room
python3 -m automation.build_light_room_parts /tmp/living_raw_NEW /tmp/living_pack_NEW --realistic
python3 -m automation.recolor_light_room /tmp/living_pack_NEW /tmp/living_final_NEW --realistic
python3 -m automation.light_room_preview --assets /tmp/living_final_NEW \
 --out /tmp/living_preview_NEW --all-parts
```

Den färdiga exporten återanvänder fem oförändrade R4-modeller; nytt rum och fem
nya typer renderades separat. `raw_provenance`/`raw_inputs_sha256` binder paketet
till sparade indata; originalen finns kvar i `/tmp/pang_r4` och `/tmp/pang_r5`.
R4-installationens tillgångar är dessutom frysta i `/tmp/pang_r5/baseline_assets`.
`--only-kind`/`--resume` kan återuppta en avbruten, oförändrad råexport; använd alltid
nya utmappar vid nytt recept/färghantering. Det separata AgX-stillbildsprovet används
inte i spelets levererade sprites.

**Förhandsbild:** `/tmp/pang_r5/delivery/room.png`. Där finns också
`tv_screen_after.png`, `books_middle_after.png`, `plant_body_breaking.png` och
`mug_handle_after.png`, övriga delsteg och WAV-filer. Bildmappen är en simulering;
träffpunkterna väljs inne i respektive form för att testa spelreaktioner och får
inte användas som mätning av detektorns träffsäkerhet.

## Runda 6 – Poly Haven / PBR, 2026-11-23

**Nuvarande grafik är ersatt med importerade Poly Haven-modeller.** Inga commits,
push/merge, kamera-, projektor- eller live-Blender-åtkomst. Alla exporter kördes i
en separat headless Cycles-process med privat `/dev`. API/meny-id `light_targets`
är bevarat. Användarens befintliga ändringar i menu.json/AI-settings lämnas kvar.
Loggversionen är `polyhaven-room-parts-6` med ny asset-manifesthash.

### Modellurval, äkta PBR och licenser

27 glTF-paket inventerades read-only. **15 använda modellpaket, 84 originalfiler**
kopierades till `content/games/pang_polyhaven_sources/`. Samtliga kopior verifierades
byte för byte mot `/data/skjutbana/assets/polyhaven/`; inga original ändrades.
Modeller/buffertar och refererade 1k-texturer tar **19 954 336 byte** inklusive
källmanifest/licensfiler. Endast renderade PNG används av spelet; glTF-texturerna
laddas inte i live-appen.

| Användning | Poly Haven-id |
|---|---|
| Rum | sofa_02, modern_coffee_table_01, Shelf_01, side_table_01, fancy_picture_frame_01 |
| Vas/blad | ceramic_vase_02, potted_plant_02 |
| Keramikvas / slank vas | ceramic_vase_01, ceramic_vase_03 |
| Gryta/lock | ceramic_pot |
| TV | Television_01 |
| Tre individuella böcker | book_encyclopedia_set_01 (tre valda volymer) |
| Krukväxt | potted_plant_02 |
| Väckarklocka | alarm_clock_01 |
| Armlampa | desk_lamp_arm_01 |
| Kanna/handtag med fäste | metal_jug |

Alla dessa inputs är **CC0-1.0** enligt den tillhandahållna källförteckningen och
[Poly Havens licens](https://polyhaven.com/license). Exakta tillgångslänkar och
SHA256 per glTF/bin/textur finns i källmappens `SOURCES.md`/`manifest.json` och
spritepaketets `asset_provenance`. Befintlig projektkod/Blender omlicensieras inte.
Golvet, väggskalet, fönstret, ljus TV-bild och effektgeometri är egna tillägg;
de dekorativa möblerna kommer från de listade importerade modellerna.

Importerade roughness-/metall-/normalkartor och UV-koordinater behålls. Diffusfärgen
får en ljus variant (gamma 0,55, varm ljus inblandning), normalstyrkan mjukas något.
Kameran är perspektivisk för rummet, Cycles kör CPU/två trådar, AgX, exponering +0,5.
De separata föremålsrenderingarna får dessutom **1,35 display-linjär förstärkning**
före kameragolvet. Detta är grafikbehandling, **inte kamera- eller mikrofonförstärkning**.
Det minskar andelen synliga modellpixlar som klipps till ljusgolvet från 63,1 %
till 27,3 % och behåller mer av skuggningen/texturen (`exposure_check.json`).
Originalkartorna ändras inte. TV-bilden och klockvisarna är helt stilla under skottet.

### Delar, silhuetter och reaktioner

Samma ObjectManager, frysta OBJECT-former, projektil-touch och strukturella
`break_parts` används. **Tio typer / 21 delar**, åtta stora föremål åt gången.
Rumsmöbler/fönster/tavla är dekor; de lösa föremålen är poänggivande mål.

Meshdelar kapas vid hals, mynning, lock, lampdel eller handtag/fäste. Grytans lock
har en egen riktig del; den historiska familjen `bowl/rim` används fortfarande i
loggar. `mug/handle` avser nu kannans övre högra handtag/fäste, inte en påhittad
mugg. Bladverket beskärs i 3D till tre kompletta importerade blad, vars texturer/UVs
behålls. Ingen öppning i en färdig träffmask fylls igen för att få en enklare form.

Hål, handtag och öppningar exporteras som raster-exakta konturvandringar eller
högst åtta delpolygoner. Nollängdskanter förbjuds; raster och motorns befintliga
ray-test kontrolleras, inklusive en uttömmande tvåhålsfixtur utan pixelavvikelser.
Ingen motorgeometri behöver ändras. Antialias-/kantosäkerheten är fortfarande
ungefär en källpixel, utöver fysisk kalibreringsosäkerhet.

Skärvorna är manuella sektorer med **överförda original-UVs och materialindex**,
inte en platt ersättningsfärg. Fallbanor stannar vid den lokala stödytan. Grytan
får egen spriteinramning så att handtagen inte kapas. En individuell kamerastorlek
har egen pivot i manifestet; äldre råpaket använder den befintliga globala fallbacken.
Importerade texturer spegelvänds inte vid vänsterträff: en liten translation ger
impulsskillnaden utan att flytta en vänsterbok till höger eller spegla tryck/text.
TV-/klocksprickor ligger kvar på sin ursprungliga yta.

0,6 s reaktion, skyddad POST till peak+1,2 s och 1,7 s stillhet före nästa REDO är
bevarade. Armlampan står på bord/bänk/hylla. Ingen detektortröskel, modellauktoritet,
kamerakonfiguration eller motor-loop ändras. Reaktionerna är författade sprites,
inte en ny live-rigid-body-motor.

### Reproduktion och förhandsbild

```bash
# Behövs bara för ett nytt urval; befintlig repokopia kan användas direkt.
python3 -m automation.stage_pang_polyhaven \
 /data/skjutbana/assets/polyhaven /tmp/pang_sources_NEW
/snap/blender/current/blender -noaudio -b --factory-startup \
 --python automation/assets/render_light_room_parts.py -- \
 --out /tmp/pang_raw_NEW --polyhaven-root /tmp/pang_sources_NEW
python3 -m automation.build_light_room_parts /tmp/pang_raw_NEW /tmp/pang_pack_NEW --realistic
python3 -m automation.recolor_light_room /tmp/pang_pack_NEW /tmp/pang_final_NEW --realistic
python3 -m automation.light_room_preview --assets /tmp/pang_final_NEW \
 --out /tmp/pang_preview_NEW --all-parts
```

`--only-kind` och `--only-part` delar exporten i korta jobb. Atomiska checkpoints
skrivs efter varje färdig del; `--resume` används endast med samma recept och
oförändrade råindata. I denna leverans finns färdiga inputbilder i
`/tmp/pang_r6/raw_delivery/` och ursprungliga/alternativa försök i separata mappar.
R5:s installerade sprites bevaras i `/tmp/pang_r6/baseline_assets/`.

Huvudbild: **`/tmp/pang_r6/preview.png`**. Alla 21 positiva delreaktioner samt
MISS/UNKNOWN finns i slutmappen som anges i slutkontrollen nedan. Det är simulerade
spelreaktioner: inga simulerade klick används som evidens för fysisk träffsäkerhet.
Nästa fysiska pilot behöver nya träffar, missar och delgränser med operatörsfacit.
≥95 % och fysisk noll-fel/noll-falska är inte verifierade av denna grafikleverans.


### R6 slutkontroll — 2026-11-23

- Installerat: 316 PNG, 10 föremålsfamiljer, 21 brytbara delar. Spritepaket
  8 996 164 byte; använda originalmodeller med källor/licenser 19 954 336 byte.
  Totalt 28 950 500 byte (cirka 29 MB); inga oanvända originalmodeller kopieras.
- Slutlig förhandsbild: `/tmp/pang_r6/preview.png`. Alla 47 bilder (21 positiva
  delreaktioner, MISS/UNKNOWN och rum) finns i `/tmp/pang_r6/final_delivery/`.
  Minsta predikterade kameranivå i spelytan under HUD är **125,04 DN**.
  Detta är färgguidens beräkningsmodell, inte en ny kameramätning.
  Mätvärden: `/tmp/pang_r6/final_measurements.json`.
- 30 placeringsfrön klarar den sparade kalibreringens ROI-budget:
  högst **598 408 / 1 000 000 bildpunkter** (`/tmp/pang_r6/budget_final.json`).
- **14 självtestmoduler gröna**, inklusive åtta R6-testfall för originalkällor,
  alfa/exponering, maskhål, polygonkanter och att texturer inte spegelvänds.
  Modulernas exitkoder finns i `/tmp/pang_r6/release_checks/results.json`.
- API/meny-id `light_targets`, OBJECT-loggning, breakable parts och skyddade
  mätfönster är kvar. Ingen detektorändring, ingen commit, ingen fysisk utrustning
  använd. Fysisk ≥95 % träffsäkerhet är fortfarande ett separat acceptanstest.


## R7 — naturlig kontrast, rättad skala och tre rum (2026-11-23)

### Vad som ändrades

- Den globala bildljusningen/kameragolvet är borttaget från bakgrunderna. Sex
  **oförändrade Cycles-PNG** används: vardagsrum, kök och arbetsrum med två
  möbleringar vardera. Rumsljuset använder AgX, exponering 0 och material/ljus i
  3D-receptet. Original-PBR används på möbler; soffan har en varm beige
  materialvariant och hyllan en ljus invändig rygg. Matta, böcker/prydnader,
  sidobord/växt och extra möbler i förgrunden gör rummen mindre tomma.
- TV:n använder en egen TV-bänk, aldrig soffbordet. Målskalan beräknas från
  ankarets projicerade meterhöjd. Grytan är cirka 0,24 m mot TV:ns 0,85 m i
  placeringsreceptet; golvvaser cirka 0,40–0,46 m. Grytan placeras inte på golvet.
  Visuella minimihöjder 40/60 px och max 125 px behåller spelbarhet; det är alltså
  ingen strikt fysisk arkitekturvisualisering. Smala mål/delgränser behöver
  fortfarande fysisk provning.
- **Tio** aktiva målfamiljer per rum (R6 hade åtta): vas, kruka, gryta, smal vas,
  lampa, TV, böcker, krukväxt, klocka och mjölkkanna. Den sistnämnda behåller
  historiskt internt id `mug`; det är ingen ny muggmodell. Ramar/fasta möbler är
  fortsatt dekor, inte poängmål. Glasyr/emaljfärger gör målen mer distinkta;
  käll-PBR-skuggning, UV-silhuetter, alfamasker och delanimationer återanvänds.
- Enbart mål och deras omedelbara omgivning skyddas. Bakgrunden skyddas inom
  **12 px från den synliga silhuetten**, med 32 px mjuk övergång, till 124
  predikterade DN. Inga rektangulära helt ljusa AABB-paneler. Naturligt ljusa
  placeringar prioriteras för att minska behovet av lokal ljusning.
  Bakgrundspixlar utanför skyddet är verifierat byte-identiska med renderingen.
  Målens PNG-färggolv är 128 DN; skalning/alfablandning verifieras separat.
- Sparad historik undviker de två senaste rumstyperna och nyligen använda
  möbleringar. De senaste 96 **presenterade målcentrumen**, inte uppmätta
  kulhål, används som placeringskostnad för bättre spridning. Extra säkra
  golvplatser till vänster/höger breddar ytan. Historiken ligger som burkspelets
  historik under `~/.local/state/skjutbana/pang_room_recent.json`; skrivning sker
  atomiskt och ett skrivfel stoppar inte spelet. Test-/analysverktyg använder
  `SKJUTBANA_PANG_ROOM_HISTORY` till en privat fil i `/tmp`.

### Mätning och verifiering

- 30 layouter med tio mål vardera använde samtliga sex rum. Lägsta
  kompositerade kamerafärgproxy i målens skyddade närområden: **121.87 DN**.
  Högst **22.4 %** av bakgrundspixlarna ändrades lokalt;
  resten behölls. Detta är en skärm-/färgmodell, inte en ny kameramätning.
- De 300 målcentrumen ligger över cirka 9–91 % av bildbredden och 30–86 % av
  höjden; HUD och dåligt kalibrerade ytterkanter undviks. Det är spridning över
  den säkra spelytan, inte garanti för jämn verklig hålfördelning.
- Samma 30-frösvit mot den sparade PANG2-kalibreringen klarar OBJECT-budgeten:
  högst **438,462 / 1 000 000** kamerabildpunkter.
- **15 självtestmoduler gröna.** Senaste omkörningen av sex berörda spelmoduler
  har **50 testfall**; där ingår R7:s fem nya testfall för lokal färgskyddsgräns,
  sex distinkta bakgrunder, historik över omstart, alla rum i 640×480/760×580/
  1280×720 samt fryst bakgrund under skottrendering. En separat placeringsprobe
  klarade 54 kombinationer (tre frön × sex rum × tre upplösningar) med tio mål.
- Gamla tester som krävde ljusgolv över hela rummet uppdaterades till den nya
  lokala kontraktsgränsen. Ett test som alltid förutsatte en kruka använder nu
  den garanterade vasen. Spårtestets tillfälliga `/data` är en privat bindning
  till `/tmp/pang_r7/object_test_data`; den riktiga `/data` har inte skrivits.
- Paketet innehåller **322 PNG** (inklusive sex rum); cirka 17,25 MB renderade
  tillgångar + 19,95 MB oförändrade, använda CC0-modellberoenden. Samma 15
  Poly Haven-källor som R6, med käll-URL/licens och filhashar. Ingen ny nedladdning.

Underlag: `/tmp/pang_r7/review_final.json` (färg/spridning),
`budget_final.json` (geometribudget), `checks/release_results.json` (testmoduler),
`checks/wide_games.log` (50 testfall) och `delivery_results.json` (sex exporter).
Förhandsbilder per möblering finns i `/tmp/pang_r7/delivery/<rum>_<0|1>/`;
`living_0` innehåller samtliga 21 simulerade delreaktioner samt MISS/UNKNOWN.
R6 och tidigare R7-försök är bevarade separat under `/tmp/pang_r7/`.

### Var och hur

`automation/assets/pang_room_variants.py` bygger rum/möbler/ankare i Blender.
`render_light_room_parts.py --room-template living|kitchen|study --furnishing 0|1
--room-only --polyhaven-root content/games/pang_polyhaven_sources --out NEW_RAW`
körs i separat headless CPU-Blender, aldrig mot användarens Blender eller display.
`automation/build_pang_rooms.py BASELINE RAW_ROOMS NEW_BUNDLE` paketerar sex
råkataloger. `content/games/light_room_art.py` väljer bakgrund och bereder maskerna;
`light_room_safety.py` gör endast lokalt färgskydd. Spelregler/historik ligger i
`light_targets_game.py`. Den installerade artefaktens `room_recipes_sha256` skiljs
från ärvda R6-recept för målsprites. `palette_report.json` redovisar den lokala
R7-policyn och naturliga rumsvärden, inte det gamla globala ljusgolvet.

Reproducera kontroll utan utrustning (välj nya outputnamn):

```bash
python3 -m automation.light_room_preview --room living_0 --all-parts --out /tmp/pang_review_NEW
python3 -m automation.pang_room_review /tmp/pang_colors_NEW.json --layouts 30
SKJUTBANA_PANG_ROOM_HISTORY=/tmp/pang_budget_history_NEW.json python3 -m automation.pang_layout_budget /data/skjutbana/source_sessions/20261122T161620Z_4b470454 /tmp/pang_budget_NEW.json --seeds 30
```

API-/meny-id `light_targets` och startverktyget `automation/play_light_targets.py`
är kvar. Loggarna har version `polyhaven-rooms-7` och explicit `room_id`; OBJECT-
semantik, exakta delpolygoner, kantradie, poäng och skyddat PRE/POST-fönster är
oförändrade. Rummet byts vid ny omgång/spelare eller tömt rum, aldrig mitt i ett
mätfönster. Detektor, modell, ljudtrösklar och cooldown är orörda.

Ingen commit/push/merge. Användarens befintliga meny-/AI-inställningsändringar
lämnas orörda. **Fysisk ≥95 % träffsäkerhet och visuell acceptans är inte
verifierade här**; nästa pilot behöver träffar, missar och delgränser med facit
per rum. De simulerade klickbilderna är spelreaktionstest, inte detektorevidens.
