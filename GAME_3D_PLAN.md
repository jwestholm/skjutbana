# Ljusjakten: fasplan 2026-11-22

1. **Uppföljning före grafik.** Återspela de tio luftgevärsskotten med befintliga
   can_dense_offline-verktyg; undersök skott 9 och modellens ambiguity utan att
   ändra en tröskel efter ett enda positivt exempel. Behåll regelauktoriteten om
   inget säkert, reproducerbart förbättringsförslag stöds av negativa kontroller.
2. **Första spelbara rumsversionen.** Återanvänd Ljusjaktens meny/API, frysta
   skottkontext, OBJECT dense och ObjectManager. Tre ljusa, förrenderade föremål
   (vas, kruka, keramik) med mjuk 3D-volym på en ljus rumsyta. Tre separerade, aktiva föremål
   per omgång, ett skott mot valfri keramik, utan överlapp/fiktiv z-ordning. Sprite-
   silhuetter ger träffpolygoner; projektilradien är fortsatt 2,25 mm.
   Damm/skärvor först efter slutbeslut. Nya ytan står still minst 1,5 s före REDO;
   inga animationer innan skottets mätfönster och fråga avslutats.
3. **Blender-förfining efter första granskning.** Exportera ortografiskt rum,
   friställda föremål (RGBA), silhuettmasker och numrerad break-sekvens med samma
   kamera, skala/pivot och färghantering. Blender körs endast headless i separat
   process/egen .blend, aldrig användarens öppna scen. Första versionen får använda
   enklare offline-renderade sprites om Blender saknas. Exportskript och manifest
   ska bevara geometri, version och färgkrav; inga handritade kollisionsboxar.
4. **Verifiering.** Självtester: silhuett/shape, ljus palett, ägarskap, dubletter,
   frysta bildrutor, minst 1,5 s PRE-stabilitet och inga tidiga effekter. Därefter
   operatörsägd fysisk pilot med nya bilder: positiva, missar och kanter, spridda
   positioner. >=95 % är ett mål, inte ett resultat av sprite- eller replaytester.

Risker: dekor/blad/skuggor kan skapa falska prickar; håll detaljering borta från
målens omgivning och G>=190 på träffytor. Silhuettmaskens små detaljer ska undvikas
redan i modellen. Effekter får inte läcka till nästa PRE. Sparade tio skott har inga
missar och kan inte bevisa noll falsklarm eller generalisering till nya föremål.

Status: fas 1 (A + första spelbara rumsversion) implementerad och programvarutestad.
Tre föremål kan väljas per omgång, och sprites/sönderfall är exporterade i headless
Blender 5.3.0. Ny fysisk verifiering och eventuell grafikförfining återstår; se
GAME_NEW_REPORT.md. Inga detektortrösklar ändrades på de tio positiva skotten.

## Jimmys tillägg – rum, delar och variation

1. **Inventering och fria källor:** motorns `GameObject.entity_id/part_id`,
   `make_breakable_object`, damage/reactions och `audio_peak_detector.play_feedback`
   återanvänds. Cell Fracture är ett offlineverktyg, inte en live-dependency.
2. **Rum med stödytor:** headless Blender renderar bord, hyllor, fönsterbräda och
   golv med namngivna, projicerade stödankare. Placering randomiseras mellan rundor;
   ingen möbelkant eller mörk textur läggs genom en träffyta.
3. **Parts i befintliga GameObjects:** separata exakta synliga masker per kroppsdel,
   samma entity_id, unika part_id. En explicit opt-in strukturell reaktion kan
   bryta loss namngivna syskondelar inom samma ägare/entity. Kropp krossas, hals
   faller, blomma faller separat. Ingen ny XY-auktoritet eller fiktiv z-sortering.
4. **Roligare loop:** flera föremål står kvar genom en runda, skador bevaras,
   delbonus/sviter och rum-rensat-bonus. Effekter/ljud efter beslut + POST-skydd;
   ny PRE-stabilisering efter varje visuell förändring. M kan tysta ljud.
5. **Mätbarhet:** test av delgränser/UNKNOWN, ägarskap, frysta gamla generationer,
   kausala presentations-/ljudtider och dense-resultat på renderade fixtures.
   Ny fysisk pilot måste innehålla missar och delgränser; 9/10-serien bevisar inte 95 %.

### Verifierade fria källor (2026-11-22)

| Tillgång/verktyg | Källa | Licens | Plan |
|---|---|---|---|
| Furniture Kit, Kenney | https://kenney.nl/assets/furniture-kit | CC0 | Bord/hyllor/dekor; ommaterialisera ljust |
| Jug 01, Kuutti Siitonen / Poly Haven | https://polyhaven.com/a/jug_01 | CC0 | Möjlig kanna; ta bort dekor/fin textur, dela manuellt |
| Ultimate Furniture, Quaternius | https://quaternius.com/packs/ultimatefurniture.html | CC0 | Alternativ möblering |
| Cell Fracture | https://extensions.blender.org/add-ons/cell-fracture/ | GPL-3.0-or-later (verktyget) | Bake av fragment, inte runtimekod eller modelllicens |

Poly Havens assetlicens: https://polyhaven.com/license . Samtliga ovan är
primärkällor; inga konton skapades. Kenney har en direkt nedladdning utan donation.
Nätverk från den lokala renderprocessen saknar DNS i denna sandlåda, så de externa
modellerna är **undersökta alternativ**, inte redan importerade tillgångar. Aktuell
spelversion använder våra reproducerbara procedurmodeller; rapporten skiljer detta
från externa CC0-tillgångar. Inget obekräftat Sketchfab-material införs.

Pipeline: verifierad modell + sparad källa/licens/hash → ljusa material → manuella
funktionella delar (hals/buk/blomma) → Cell Fracture eller enkla sektorskärvor per
del → headless render av intakta/delvisa/brytande tillstånd → synliga alpha-masker
→ normaliserade polygoner → GameObject per del. Delgränser med flera fysiskt möjliga
parts avstår; de avgörs aldrig med listordning eller 3D-djup från fysiskt XY.

## Levererad fortsättning

PANG! / Ljusateljén är nu spelbar med sex kvarstående föremål, fem modeller och
elva funktionella deltyper. Bord, bänk, hylla, fönsterbräda och golv har projicerade
stödankare. Buk/hals/blomma och lock/kant/skärm har separata synliga former och
reaktioner. Structural `break_parts` är opt-in, scoped till ägare/entity; inga
extra fysiska träffar skapas. Skärvor bakom aktiva mål skyddar den synliga formen.
Originala ljusa modeller, render-/maskfiler och syntetiserade vågformer levereras
som CC0-1.0. Källkodens licens ändras inte. Externa CC0-modeller är forsknings-/
produktionsalternativ, inte redan importerade tillgångar.

Ytterligare fracture-referens: Blenders officiella
[demofillista](https://www.blender.org/download/demo-files/) listar Impact Fracture
av Simon Thommes/Blender Studio som CC0. Filen är inte hämtad eller verifierad
i vår pipeline. Nästa faktiska steg är fysisk pilot av delgränser, stödytor och
ljudåterkoppling; inte fler detektortrösklar anpassade till de tio positiva skotten.

## Runda 3 – genomförd, fysisk ommätning återstår

- PANG1: samma query-matematik återger 7 HIT + 3 UNKNOWN. Eftergranskning av
  foton/tidsserie ger hål utanför samtliga objekt vid 8 och 10, inget nytt verifierat
  hål vid 9. Inga detektortrösklar ändras efter dessa data.
- Neutral bakgrund och tydliga mättade objekt, oförändrade alpha-masker/partformer.
  Grå 160 ger bara 96,5 predikterade DN; därför används cirka grå 194–214 i stället
  för 140–170 för att behålla kameramarginalen. Exportgolv 128 DN, kontrollerat
  kompositgolv >120 DN. Detta är en proxyberäkning, inte en ny ljusmätning på riggen.
- Åtta föremål med olika storlekar och varierade stödankare. Partialträffar färgar
  kvarvarande delar kontrasterande; krossade delar blir skärvor. Distinkt träff,
  miss och osäkert – UNKNOWN döps inte om till MISS utan belägg.
- Tolv brytbilder vid 20 fps ger 0,6 s animation. Efterbesked 0,95 s, oförändrat
  POST-skydd peak+1,2 s och ny PRE-stabilisering 1,7 s.
- Reproducerbar palettstatistik, sparad-query-audit och ROI-budget finns i
  automation.{recolor_light_room,pang_session_audit,pang_layout_budget}.
  Nästa acceptans kräver nya fysiska skott med oberoende facit, inklusive missar
  och delgränser; ingen fysisk förbättring för R3 påstås.

## Runda 4 – material och former levererade

- PANG2:s två underkända hål är starkaste MF-kandidat men för grunda för befintlig
  acceptans. Ingen detektortröskel ändrad; PANG1 ger ett konkret motexempel till
  sänkt isoleringskvot. Separat experiment med oberoende negativa kontroller krävs.
- Mjukare profiler, glaserad keramik/porlinsliknande material och breda reflexer;
  rumsljus och färger bevaras från Cycles. Ytor under proxygolvet lyfts mot vitt.
  Silhuettmasker/partpolygoner exporteras om från samma kamera; ingen gammal mask
  läggs över ändrad geometri. Egna CC0-tillgångar, inga nya externa modeller.
- 166 exporterade PNG, proxygolv >=128 DN; elva kompositerade bilder >=124,31 DN.
  30 layouter klarar ROI-budgeten. Förhandsbild `/tmp/pang_r4/preview/room.png`.
  Detta är ett materialsteg, inte ett fotorealistiskt vardagsrum eller ny fysisk
  validering. Den större vardagsrumsvisionen kräver ytterligare scen-/tillgångsarbete.

## Runda 5 – vardagsrum, pågående implementation

Återanvänd samma sprite-/maskexport, frysta OBJECT-former och delreaktioner.
Bygg ett perspektiviskt Cycles-rum med soffa, bokhylla, soffbord/mediaförvaring,
stort dagsljusfönster och varma material. Aktiva nya föremål: stilla ljus TV
(skärm spricker, fot kan falla), böcker (enskilda volymer faller), krukväxt
(kruka/blad separata), klocka (spräckt urtavla). Behåll stor ljus keramik och
spridda placeringar. Inga nya detektortrösklar. Ingen live hårdvara startas.

Poly Haven CC0/licens kontrollerad 2026-11-22: https://polyhaven.com/license .
Nedladdning från api.polyhaven.com misslyckas med DNS i denna miljö. Levererade
modeller förblir därför egna CC0-procedurmodeller; inga externa tillgångar får
påstås vara införda. Visuell kvalitet granskas på faktiska renderbilder, medan
kameranivå/alpha/partmasker/timing kontrolleras separat. ≥95% kan bara avgöras med
nytt fysiskt facit, inte av snygga renderingar eller syntetiska selftests.


## Runda 5 – levererad spelversion och nästa acceptans

Perspektiviskt vardagsrum, tio typer/21 funktionella delar, TV och böcker i varje
rum samt varierande växt/klocka/mugg. Åtta stora föremål samtidigt. Sprickor,
enskilda böcker, separata växtdelar och muggdelar använder befintlig delmotor.
Möbler/fönster/väggtavla är ännu dekor, inte poänggivande mål. Begränsad
backtracking säkrar placeringen utan mindre mål. 13 testmoduler är gröna;
31 kompositbilder >=124,28 DN och 30 layouter inom ROI-budget. Inga nya fysiska
noggrannhetssiffror påstås. Externa CC0-hämtningar blockerades av DNS; egna
procedurmodeller levereras med korrekt käll-/licensredovisning. Visuell
fotorealistisk slutkvalitet måste bedömas på renderingen; det här är en ljus,
förenklad och mätbar tolkning. Förhandsbild `/tmp/pang_r5/delivery/room.png`.
Nästa acceptans: Jimmys bildgranskning och nya fysiska skott med negativkontroller,
kanter och facit. Ingen automatisk modellpromotion eller sänkt acceptansgräns.


## R6 genomförd — 2026-11-23

Lokalt tillhandahållna CC0 Poly Haven-modeller används nu: 15 tillgångar,
84 hashverifierade glTF-beroenden. Cycles-rummet använder verklig soffa, hylla,
soffbord, sidobord och ram; tio målfamiljer har 21 separata brytbara delar.
Original-UV, PBR-material och skärvornas UV bevaras. Manuella delsnitt återanvänder
motorns breakable parts. Silhuettpolygoner bevarar verkliga öppningar; inga hål
fylls för att få enklare träffformer. Skador/animationer börjar efter mätfönstret.

Ljusa materialvarianter och fast linjär exponering ger 47 förhandsbilder med
minst 125,04 predikterade DN i spelytan; 30 layouter klarar ROI-budgeten.
14 självtestmoduler är gröna. Källmodeller cirka 20 MB, spritepaket cirka 9 MB.
Käll-/licenslistor medföljer. Preview `/tmp/pang_r6/preview.png` och reproduktion i
`GAME_NEW_REPORT.md`. Fysisk pilot, negativkontroller och Jimmys visuella
acceptans återstår; färgproxy och simulerade träffar bevisar inte fysisk 95 %.
