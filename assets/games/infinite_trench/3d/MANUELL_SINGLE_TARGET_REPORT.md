# MANUELL: ett mål och inga köade dubbelskott

2026-10-16. Endast `codex/graphics-realism` i grafikworktreet. Ingen commit, push, merge, reset, städning eller ändring av ordinarie dev. Tidigare lokalt arbete bevarat.

## Rotorsak och patch

Tidigare MANUELL använde samma `EnemyDirector.active` som NORMAL. Denna samling försörjer AI, renderaren och de frysta träffregionerna. Flera giltiga mål gav därför korrekt avvisad, men demonstrationsmässigt oanvändbar, HIT. Musloopen spärrade redan extraklick i samma eventbatch. Direkta `submit_game_hit`/adapteranrop saknade motsvarande spärr och kunde registrera skott under markeringspausen.

- `enemy_system.py`: liten presentationsgrind i befintlig Director. I MANUELL finns högst en **levande** aktiv fiende. Övriga närliggande aktörer väntar i `manual_waiting`, utan AI-tick, renderinstans eller träffregion. Encountergenerator, antal, kompetenskurva och aktörernas AI är oförändrade. Befintligt synligt mål prioriteras deterministiskt vid B; därefter behålls målet tills det besegras eller passeras. Huvudgångens kommande mål prioriteras framför passerade/sidoplacerade patruller. Ingen extra väntetimer införs.
- `trench_3d_test.py`: synligt namn MANUELL; låst mål-ID **och aktörsreferens** från presenterad bild. Gemensam skottingång serialiserar callbacks och läges-/världsbyte med RLock och generationsräknare. Anrop under MARKERA avvisas innan de väntar på låset; konkurrerande anrop från samma generation kan inte skapa nya markeringar efter resume. SDL blockerar och tömmer mouse-down-event under pausen; befintlig batchspärr behålls för redan hämtade event. Inga nya skottköer.
- Efter bedömning/lägesbyte/omstart öppnas skottingången först efter en ny presenterad bild. Första uppdateringen efter bedömning hoppar över väntetiden. Avslut/död rensar pending utan att skapa en bedömning. NORMAL:s ursprungliga XY/raycast-kropp är AST-identisk bakom dispatchgrinden; hitboxar, render-/skuggkod och seedgeneratorer ändras inte.

`active` innehåller även lik under dödsanimationen: dessa kan fortfarande synas men har inga träffregioner och räknas inte som levande mål. En döende och en ny levande fiende kan därför synas samtidigt. Väntande aktörer behåller identitet/tillstånd vid lägesväxling. Befintlig spatial streaminglivslängd gäller fortfarande: aktörer långt utanför fönstret kan avlastas, deras seedplan raderas inte. Besegrade ID:n återkommer inte under samma run.

## Reglage

- **B:** NORMAL ↔ MANUELL. Till NORMAL återförs väntande aktörer utan dubbelspawn. Lägesbyte är spärrat under MARKERA.
- **SPACE:** en HIT på låst mål. Noll mål, försvunnet mål eller ersatt aktör avvisas med text. **M:** en MISS, även utan mål. Upprepade tangenttryckningar ger inte flera resultat.
- **R:** samma seed/initiala plan. **S:** nytt seed/värld. Fungerar i spel och på dödsskärmen; båda spärras under MARKERA tills SPACE/M besvarat skottet. **F9/F10** behålls.
- Extraskott under MARKERA ändrar inte skott-/poängräknare eller simulation. Separat ammunitionstillstånd finns inte i dagens ArcadeRun. Programvaran stoppar **inte fysisk avfyrning**. Ingen ljud-/Skjutbanan-integration har tillkommit; endast befintliga ingångar provas.

## Mätt före/efter

Kontrollerat tvåfiendemöte, seed 1944, samma kamera. [NORMAL: två mål](manual_validation/normal_two_targets.png), [MANUELL: ett mål](manual_validation/manual_one_target.png), [nästa mål och döende föregångare](manual_validation/manual_next_target.png). Bilderna är granskade. NORMAL-bilden visar flerfiendepresentationen som MANUELL tidigare använde; ingen historisk före-rendering påstås.

| Mätning | Före patch (sparad produktionskod) | Efter |
|---|---:|---:|
| Planerade fixturefiender | 2 | 2 |
| Levande aktiva / presenterade / träffbara | 2 / 2 / 2 | 1 / 1 / 1 |
| Väntande | 0 | 1 |
| HIT vid detta möte | avvisad, tvetydig | ett låst mål |
| Extra resultat från 5 direkta pausskott | 5 | 0 |
| Extra SDL-klick under paus | tidigare batchspärr fanns | 5 av 5 förkastade vid ingress |
| Köade skott efter resume | ej historiskt mätt | 0 |

Maskinläsbart: [före/efter](manual_validation/before_after.json), [renderprov](manual_validation/render_result.json), [fem seeds](manual_validation/seeds.json). Sex samtidiga callbacks gav en pending och ett resultat efter bedömning. Fem mus-event plus fem direkta anrop i riktiga spelprovet gav inga extra resultat.

## Verifiering

- **PASS:** 41 riktade tester, `manual_validation/tests.log`: manual admission/callbacks, befintliga läges-/seedtester, fem seeds, topologi och encounterprogression.
- **PASS:** 26 kompletterande tester, `manual_validation/lifecycle_tests.log`: R/S från spel/död/pending, död, befintlig arcade och atomisk streaming. Loggens injicerade CHUNK ERROR är ett avsiktligt felhanteringstest.
- **PASS:** 25 tester efter sista dispatchlåset, `manual_validation/final_trigger_tests.log`; separat reproduktion av gamla callbackfelet, `baseline_test.log`. Antalen överlappar; inte 93 unika tester.
- **PASS:** fem seeds (1944, 98765, 44123, 44223, 20261016), 0–1000 m i 5-meterssteg: NORMAL:s aktörsspecifikationer, positioner, AI-state och klockor exakt lika före-versionen. MANUELL högst en levande aktiv och högst en aktör med träffregioner, inklusive chunkgränser. Döda återkommer inte vid reload; reset återställer initialplanen.
- **PASS:** riktigt SDL/OpenGL-prov, 640×480 Mesa llvmpipe: två mål, NORMAL:s koordinatbaserade bom/träff, B, MANUELL HIT, väntande mål, paus, R/S/B-spärr och återgång NORMAL. Nio kontrollerade draws gav GL_NO_ERROR. Den sista ändringen serialiserade skottdispatch och kontrollerades därefter med riktade tester; renderkoden ändrades inte. Första exportförsöket misslyckades på äldre Pillows `Image.Transpose`; verktyget anpassades och provet kördes klart.
- **PASS:** `py_compile` på samtliga ändrade Pythonfiler och `git diff --check`. Inga GLSL-ändringar; spelprovet kompilerar/använder befintliga shaders. Ingen full regressionssvit körd.

## Filer, kostnad och kvarstående kontroll

Produktionsfiler i detta pass: `enemy_system.py`, `trench_3d_test.py`. Test-/verktygsfiler: `tests/test_demo_mode.py`, `tests/test_demo_polish.py`, nya `tests/test_manual_single_target.py`, `smoke_demo.py`, nya `verify_manual_single_target.py`. Historiska rapporten `DEMO_SPRINT_REPORT.md` länkar till dessa aktuella instruktioner. `manual_validation/` innehåller före-snapshots, loggar, JSON och PNG; inga nya spelassets.

Inga nya material, texturer eller draw calls. Väntande aktörer är begränsade till det befintliga spatiala fönstret. MANUELL sorterar detta lilla fönster och använder ett gemensamt lås; callbacks kan vänta på pågående bild/tick när spelet kör, men aldrig köas under markeringspausen. CPU/GPU-tid och GTX 1070-FPS är **inte mätta**. Långvarigt projektorprov, verklig extern eventkälla och spelkänslan vid täta naturliga encounters återstår; headless-statistik bevisar inte balans. Ingen fysisk skottkälla är ansluten i detta repo.

## Två minuters operatörsprov

1. Starta `python3 trench_3d_test.py`. Kontrollera NORMAL och B → MANUELL; ett levande mål åt gången. F9 ska fortfarande flytta projektorfönstret.
2. Skjut ett simulerat musklick. Under MARKERA: klicka fem gånger, prova B/R/S, vänta några sekunder. Bild/simulation ska stå still och inget nytt resultat registreras.
3. SPACE ger en träff. Nästa mål får presenteras; inga gamla skott ska dyka upp. Upprepad SPACE ger inget extra. Prova M som bom och SPACE utan giltigt mål (ska avvisas; M avslutar).
4. B tillbaka till NORMAL återger flerfiendesemantik. R återger samma bana, S annan bana; prova även från dödsskärmen. Paus i programmet är aldrig ett fysiskt skjutstopp.

Fullständig slutstatus och diffstat, inklusive tidigare arbete: [manual_validation/git_status.txt](manual_validation/git_status.txt).
