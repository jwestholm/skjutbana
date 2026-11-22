# Ljusjakten — luftgevär 20 m

2026-11-22. Implementerat för granskning, inga commits. **95 % fysisk korrekthet är
ett acceptansmål, inte ett uppmätt resultat.** Ingen kamera eller projektor har använts.

## Spelet

En eller två spelare, fem eller tio skott var. En stor ljus cirkel på ljusgul botten
byter plats efter varje bedömt skott. Träff ger 100 poäng och svitbonus upp till 200;
miss eller osäkert ger inga poäng. Osäkert visas uttryckligen och räknas som förbrukat
försök. Inga tidspoäng, inga flashar. Skjut bara när REDO visas, ett skott per lampa.
ENTER startar; 1/2 väljer spelare, 5/0 väljer fem/tio skott, ESC återgår till menyn.
Mus är märkt SIMULATED och ingår inte i fysisk mätning.

Tolv spridda positioner används före återanvändning, med marginal från dukens kanter.
Vid 760×580 är målet 120 px i diameter, cirka 14 cm på en 90 cm bred projektion.
20 m avser den avsedda verkliga skjutdistansen; målet krymps inte virtuellt.
Paletten har hög grönkanal, utan text, mörk kontur eller smådetaljer nära målet.
Ren presenterad yta får stabiliseras 1,1 s före REDO. Hela bilden fryses under frågan;
feedback börjar först när frågan stängts. Renderingen cachas mellan förändringar.

## Filer och återanvändning

- `content/games/light_targets_game.py`: spel, poäng, låst skottägarskap, regioner.
- `content/menu.json`: id `light_targets`, ”Ljusjakten – luftgevär 20 m”.
- `src/engine/input/hit_regions.py`: explicit `evidence_method='dense'` i fryst region.
- `src/engine/camera/object_physical_query.py` och `input/object_hit_v2223.py`:
  befintlig dense-väg/deadline/parhantering kan väljas per fråga utan global env-ändring.
  Frågor utan denna opt-in behåller befintligt val. Inga detektortrösklar har justerats.
- `src/engine/scene_factory.py`, `app.py`: `startGame` väljer registrerat meny-id via
  befintlig fabrik/kö och utför scenbytet i huvudtråden.
- `automation/play_light_targets.py`: start och API-klient.
- `automation/light_targets_score.py`: oberoende facit jämfört med spelloggen.
- `automation/light_targets_selftest.py`: nio kontraktstester.

OBJECT-träffar går via befintlig ObjectManager/resolve_object_query med frysta
ellipsformer och projektilradie 2,25 mm (4,5 mm luftgevär). Vanliga CURRENT-XY ger
aldrig poäng. Fel skott-id, gammal målkontext, followers och dubbletter avvisas.
Fyra sekunder utan slutresultat ger UNKNOWN; inget påhittat missbeslut.
Modellen är fortsatt SHADOW. Ingen standardpolicy ändras för andra spel.

## Start

Från worktree-roten, när operatören vill starta riggen:

```bash
python3 -m automation.play_light_targets 2>&1 | tee /tmp/light_targets.log
```

Starta med vanlig meny och samma mätmiljö:

```bash
python3 -m automation.play_light_targets --menu
```

I redan körande app:

```bash
python3 -m automation.play_light_targets --api
# motsvarar send_command('startGame', {'id': 'light_targets'})
```

API öppnar intro; ENTER startar omgången. Vanlig meny fungerar även utan global
OBJECT_DENSE eftersom just detta spels regioner väljer dense explicit.
API kan inte aktivera inspelning i en redan körande app.

Launchern sätter defaults (befintliga env-värden respekteras):
`SKJUTBANA_OBJECT_DENSE=1`, `SKJUTBANA_AUDIO_COOLDOWN_S=0.15`,
`SKJUTBANA_SOURCE_RECORD=1`, `SKJUTBANA_SOURCE_RECORD_WINDOW=1`,
`SKJUTBANA_PHYSICAL_TRACE_DISABLE=1`. Befintlig v6aug-modell används enbart i SHADOW
om filen finns. Ingen GAME_DX/korrigering införs. Källsessioner skrivs av befintlig
inspelare under `/data/skjutbana/source_sessions/`; kontrollera dess start-/diskstatus.
En redan satt env-variabel kan avsiktligt stänga av inspelningen.

## Fysisk verifiering före påstående om 95 %

Lås kod, kalibrering, mikrofon och inställningar. Använd färskt papper och verifierad
kalibrering. Samla först en pilot med minst 60 verkliga målträffar över samtliga
positioner och minst 20 negativa kontroller (verkliga missar, ljud utan nytt hål).
Inkludera dokumenterade kantträffar; facit ska använda projektilens touch-regel,
inte enbart hålcentrum. Undvik snabba serier: spelet begär ett skott per REDO.

Operatören registrerar varje avlossat skott, också när motorn inte reagerar. Facit
kommer från faktiskt hål och fryst målposition, vid behov oberoende granskning av
sparade bilder. **Använd inte motorns vinnare som facit.** Oklart facit särredovisas
som UNKNOWN och hindrar ett rent godkännande. Bevara källsession och konsollogg.
Befintliga `[OBJECT-DENSE]`/`[OBJECT-HIT]` kompletteras med `[LAMP-TARGET]`,
`[LAMP-PRESENTED]`, `[LAMP-SHOT]`, `[LAMP-RESULT]` (owner, token, shot_id, peak, poäng).
Ingen parallell detektor eller automatisk etikettering har införts.

Facit JSONL, koppla id från loggen men bestäm expected oberoende:

```json
{"owner":"light_targets:…","shot_id":42,"expected":"HIT"}
{"owner":"light_targets:…","shot_id":43,"expected":"MISS"}
{"owner":"light_targets:…","shot_id":null,"attempt_id":"operator-shot-44","expected":"HIT"}
```

Den sista raden fångar utebliven ljudhändelse/resultat. Owner kan utelämnas endast
om loggen har exakt en owner; ange den alltid vid flera omgångar.

```bash
python3 -m automation.light_targets_score /tmp/light_targets.log truth.jsonl
```

Rapporten visar strikt korrekthet, Wilson 95 %-intervall, träffåterkallning, falska
poäng, avståenden, oetiketterade resultat och pilotstorlek. UNKNOWN/uteblivet resultat
på känt facit räknas som fel. Kräv minst 95 % strikt korrekthet **och** minst 95 %
träffåterkallning samt noll falska poäng i piloten. Pilotens punktestimat bevisar inte
95 % generaliserbar korrekthet: använd därefter ny oberoende inspelning och redovisa
konfidensintervallet. Exempelvis räcker inte ens 60/60 till 95 % nedre Wilson-gräns.
Ingen fysisk pilot eller live-path replay av nya spelet har körts här.

## Programvarutester

Körda utan hårdvara, SDL dummy och privata `/dev`, `/tmp`, `/data` i bubblewrap:
`light_targets_selftest`, `object_physical_query_selftest`, `object_hit_selftest`,
`object_readiness_selftest`, `object_dense_pair_selftest`, `v250_selftest`.
Alla sex moduler är gröna: 9 nya speltester, 43 OBJECT-querytester, 14 OBJECT-hit,
43 readiness samt dense-pair- och V2.25-kontrakten.
Detaljloggar: `/tmp/light_targets_checks/*_verified.log`.
Ordinarie runnerns nätverksisolering nekas av denna miljö (NETLINK_ROUTE); den
ersattes enbart i testkommandot med samma isolering utan `--unshare-net`.
Ingen App/device-start görs i testerna; API-handlern testas med stubbat scenbyte.
`compileall`, launcher `--help` och `git diff --check` ingår också.

Befintliga användarändringar i `content/ai/settings.json` och lokala data är orörda.
