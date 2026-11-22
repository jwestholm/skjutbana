# Can Shootout – dense OBJECT-beslut, AI-skugga och mätningar (2026-11-21)

Status: allt bakom `SKJUTBANA_OBJECT_DENSE` (av som standard) på `claude/latency`. Inga standardbeteenden
ändrade, inget mergat. Startskript för live: `python3 automation/play_can_dense.py` (spelar in källsession).

## Vad som var fel live 2026-11-20 (0 % träff)
1. Varje riktigt skott gav en andra ljudtopp 65–97 ms senare; den avbröt det första skottets fråga innan
   hålet hunnit synas (rättat: parfönster 0,15 s + följaren publiceras aldrig före ägaren; ljud-cooldown 0,15 s
   i startskriptet).
2. Beslutet togs innan hålet syntes i kameran (rättat: POST-bilder från 0,35 s, 3 bilder, deadline 1,2 s).

## Mätverktyg
- `automation/can_dense_offline.py`: spelar upp inspelade sessioner genom samma beslut, ~12 s per session,
  överensstämmer med live 298/299 och 20/20 (riktiga skott).
- Testprogram (scratchpad `can_harness.py`): projicerade hål 75–100 ms efter ljudet, följartopp, lappning var
  5:e skott, kantmissar 2–8 px utanför formen, realistiska grunda/ljusa hål (`CAN_HOLE_FACTOR`).
- `automation/can_peak_dataset.py` / `can_peak_model.py` / `can_peak_ensemble.py`: kandidatdata, numpy-modeller
  (logistisk, MLP, boostade stubbar), utvärdering per skott med exakt form, tau via leave-one-session-out.

## Resultat (projicerade hål om inget annat anges)
| Material | Regel | Modell (skugga) | Fel objekt | Falsk poäng |
|---|---|---|---|---|
| Färskt test, 6 sessioner (206 träffar, 154 missar) | 176 | 173 (regel∪modell 178) | 0 | 0 |
| Validering v4, 12 sessioner (529/251) | 438 | 451 (regel∪modell 465) | 0 | 0 |
| Grunda/ljusa hål, 4 sessioner (157 träffar) | 49 | 78 | 0 | 0 |
| Riktigt luftgevär, 20 skott (17 träffar, operatörsfacit) | 9 live / 10 med x-korr. | 10–14 | 0 | 0 |

## Förkastade hypoteser (mätta)
Lösare isolerad regel (spökträffar ×4), konsistensspärr (−15 träffar), L*-gråskala (verkligt +1, projicerat −1..−3),
lägre kameraskärpa (ingen nettoeffekt), längre exponering 333 (sämre, mättnad), kvadratisk x-korrektion (förlorar
verklig kantträff), MLP/stubbar/ensembler (inte bättre än logistisk modell på validering).

## Kända risker
- Spökträffar: kant-/textskimmer i spelgrafiken ger ibland starka "nya" prickar; aldrig inne i en form i regeln
  på ~1 900 projicerade skott, men skuggmodell v2 gav 2 fel objekt på kantmiss-sessioner (därför v4, tau 0,994).
- Långa stopp i huvudloopen (> 1 s) gav en falsk poäng i testprogrammet (hål ritades sent). Skuggan körs därför i
  bakgrundstråd.
- Riktiga hål är grunda (core 0,05–0,15) och ibland ljusa; projicerade testhål överskattar detekteringen.
- Codex: modellen ska vara SKUGGA tills ~300 oberoende riktiga skott utan fel.

## Natten 21–22 nov: realistiska hål, v5 och universellt spår
- Testprogrammet kan nu rita grunda (30–55 % mörkare) och ljusa hål samt kantmissar.
- v5 (mörka+ljusa kandidater, tränad med grunda hål): validering grunda hål regel 46/220, v5 127/220,
  regel∪v5 131/220; vanliga 502→538/603; orört test 176→181/206; 0 fel, 0 falska på 591 missar.
  Live, färska sessioner utan inspelning: grunda 29→51/74, vanliga 59→62/73, 0 fel/0 falska på 93 missar.
- Universellt (hela ytan, 12 bakgrundstyper, ~800 skott): H15-regeln 26 % rätt, universell modell 19 %
  med fler fel; riktiga hålstämplar på dart/bågskytte/skidskytte/5-tavla/rörlig/mörk: regel 13/44 (1 fel),
  modell 0/44. Modellen generaliserar INTE mellan domäner; det stora hindret är rörlig och mörk bakgrund.

## Förslag: motorfunktion för universell detektering (ej byggd)
"Projektionsmedveten differens": motorn vet exakt vilken bild som projicerades. Med en fotometrisk modell
projektor→kamera (per pixel, kalibreras med några testbilder) kan motorn förutsäga kamerabilden och jämföra
med den verkliga; residualen är fysiska förändringar (hål) även när spelet animerar. API: spelets renderade
bilder loggas med presentationstid (finns delvis: render_history 'presented'), motorn bygger förväntad
kamerabild för PRE/POST-tidpunkterna. Alternativ/komplement: spelet kan begära "stillbild" efter skott
(hålla senast visade bild ~0,5 s) – finns som idé i Skyttegrav/bench --freeze-on-shot-ms.
Mörka scener: hål syns inte i mörkt projicerat ljus; kräver ljusare golvnivå eller kort ljuspuls (precision_flash).
