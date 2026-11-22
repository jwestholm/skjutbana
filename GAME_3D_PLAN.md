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
