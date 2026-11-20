# H15-REF-1 och upplägg för fysisk verifiering (2026-11-20)

Status: referensvariant låst; fysisk verifiering förberedd men **inte genomförd**. Inga standardbeteenden ändrade.

## 1. Låst referensvariant H15-REF-1

| Del | Värde |
|---|---|
| Kod | `src/engine/shot_dense_h15.py` vid commit 19c87a0, sha256 `bc111be745dee4db…` |
| Aktivering | `SKJUTBANA_H15_DENSE=1` (+ `SKJUTBANA_H15_OWN_THREAD=1` för egen tråd), allt annat standard |
| Fönster (kausala) | PRE [toppen−0,4 s, toppen−0,02 s], POST [toppen+0,15 s, inlämningsruta], fördröjning 0,6 s, deadline 1,3 s, ≤9 rutor per median |
| Beslutsregel | förstaval accepteras om s ≥ 16, s1/s2 ≥ 1,15, core ≤ 0,4, bipolar ≤ 1,0; ingen reserv till gamla kedjan |
| Avstängda tillägg | `SKJUTBANA_H15_MIN_PRE_DN` (mörkermask, 0 = av), `SKJUTBANA_H15_LOG_EXTRA` (endast mätning) |
| Utvärdering | `python3 automation/h15_reference_eval.py RUN_DIR [UTMANARE_RUN_DIR …]` |

Referenssiffror (replay av inspelade projicerade sessioner, tolerans 8 px; återskapade med skriptet):

| Material | Skott | H15 rätt | fel | falsklarm | avstår | missade | precision | Gamla kedjan rätt / omatchade emissioner |
|---|---|---|---|---|---|---|---|---|
| DEV projicerad (4 sess.) | 340 | 122 | 2 | 0 | 216 | 218 | 98,4 % | 110 / 210 |
| DEV fysisk (2 sess.) | 22 | 14 | 1 | 1 | 7 | 8 | 87,5 % | 23 / 14 |
| TODAY projicerad (10 sess.) | 827 | 277 | 10 | 0 | 540 | 550 | 96,5 % | 200 / 581 |
| TODAY + mörkermask 16 (utmanare) | 827 | 306 | 24 | 0 | 497 | 521 | 92,7 % | – |

Regler för framtida ändringar:
- Varje ändring körs som utmanare mot H15-REF-1 och gamla kedjan på **samma** sessioner med samma skript.
- TODAY har använts flera gånger och är inte längre helt orörd; beslut om standardläge kräver **nya** sessioner
  (split `NEW` i skriptet) och fysiska hål.
- Fysiska och projicerade hål redovisas alltid separat. Projicerade testhål är mörka prickar i bilden; fysiska hål
  ändrar papper/fiber, kan ha flisor och skuggor och påverkas av ljusnivå. Underlaget för fysiska hål är i dag 22 skott.

## 2. Fysisk verifiering – testupplägg

### Princip
- **Parade jämförelser där det går.** Gamla kedjan, H15 (egen tråd) och H15 + mörkermask kan alla utvärderas på
  *exakt samma skott*: kör live med H15 i egen tråd och skuggloggning, spela samtidigt in källsessionen, och kör
  sedan kedjan, H15-REF-1 och masken som replay på samma rådata. Ingen separat skjutserie behövs för dessa tre.
- **Separata serier bara när stimulus ändras:** svartnivå och ljuspuls ändrar det kameran ser, och fördröjning med
  och utan H15-tråd ändrar processlasten.
- **Samma förutsättningar:** samma kamerainställningar (exponering 167, gain 0, WB 4600, gamma 100), samma avstånd,
  vapen och ammunition, samma kalibrering, ingen ändring av spelens ordinarie färger.
- Blocken körs i ABBA-ordning (A B B A …) så att drift i ljus och tavla inte gynnar någon variant.

### Före passet
1. `python3 automation/disk_guard.py status`: minst 20 GB reserv plus cirka 2 GB per 100 skott för källinspelning.
2. Spårskrivning: `physical_trace_capture_enabled` endast via förberedd session
   (`physical_test start`, som skriver `physical_trace_session_mode=prepared`), aldrig mot gamla sökvägar.
3. Rena tavelytor eller nytt papper per serie, så att varje nytt hål går att identifiera. Gamla hål är kontext, inget veto.
4. Facit: punkt för punkt efter varje skott med GT-granskningen (`automation/physical_gt_capture.py`),
   eller fotografi av tavlan efter varje block med numrerade hål.

### Serier

| Serie | Syfte | Upplägg | Skott |
|---|---|---|---|
| A. Huvudserie (parad) | Gamla kedjan vs H15-REF-1 vs H15 + mask, fysiska hål | Vit och en mörk bakgrund, standard svartnivå; live H15 egen tråd + skugglogg + källinspelning | ≥ 2 × 30 (60 ger ungefär ±7 procentenheter i träffandel) |
| B. Svartnivå | Hur mörka ytor påverkar fysiska hål | Mörk scen med projicerad svartnivå 64 / 72 / 80; ordning 64-72-80-80-72-64 | 6 block × 8 |
| C. Ljuspuls (valfri) | Exakt XY med kort ljuspuls | Först efter att de 4 Codex-punkterna i `precision_flash.py` är åtgärdade; puls av/på växelvis per skottpar; en puls per skur | 2 × 15 |
| D. Fördröjning live | H15 egen tråd på/av | ABBA-block om 10 skott; `SKJUTBANA_LOOP_TIMING=1` och stoppsond på | 4 × 10 |
| E. Negativa kontroller | Falsklarm | Handklapp/slag nära mikrofonen utan träff, ekosituationer; 10 st spridda mellan blocken | 10 |

### Mätvärden per serie och variant
Rätt (≤ 8 px), fel, falsklarm, avstår, missade, precision, fördröjning till träff (p50/p90), huvudloopens p99,
samt H15:s s/kvot/core-fördelning för fysiska hål jämfört med projicerade testhål.

### Beslutskriterier (förslag, att godkännas av dig)
- Mörkermasken går till standard endast om den på nya fysiska och projicerade sessioner ger fler rätt **utan** lägre
  precision än H15-REF-1.
- H15 som huvudmetod kräver att fysisk precision ≥ referensens projicerade (cirka 96 %) och att den inte missar fler
  än gamla kedjan på samma skott.

### Det som kräver din närvaro
Allt i serierna A–E (riktiga hål). Det jag kan göra i förväg: åtgärda `precision_flash.py`, förbereda körskript och
utvärdering, samt köra hålfria interferenstester.

## 3. Hur oberoende är H15 av den gamla vägen? (Codex-granskning 2026-11-20, ej åtgärdat)
H15 är självständig i **förslagsgenereringen** (egen täta sökning på PRE/POST-medianer), men inte som körbar detektor.
Den ärver från gamla vägen: ROI/kalibrering (`_viewport_roi`, cachas och följer inte kalibreringsbyte),
scannerhändelser + ljudtid + livscykel (vilka jobb och fönster som finns), kamerahistorik och mottagningstider,
CO₂-parantagande och tidigare beslut (`_decided_xy` kan läsa kedjans spår), H11-kärnan inne i `dense_top`
(därför tillför "topp5 → H11" delvis samma signal igen) samt selektor/readiness/fallback (accepterade resultat blir scanner-spår).

**Risk att åtgärda innan H15 får mer ansvar:** toppar nära föregående beslut tas bort, och närliggande spår markeras
som emitterade utan kontroll av vem som äger dem; det kan i sällsynta fall konsumera ett annat verkligt skott.

**Föreslaget oberoendetest:** mata samma tidsstämplade grårutor och frysta ljudhändelser till H15 med kandidatgenerering
och rankare avstängda; jämför jobbidentitet, PRE/POST-ruta, toppar och acceptans mot befintlig skugglogg. Därefter
separat: rå ljud → händelser. Live-köer, reserv och deadline kräver egen kontroll.
