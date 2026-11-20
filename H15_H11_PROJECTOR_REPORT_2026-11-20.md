# H15, H11, mörka områden och animerade bakgrunder – rapport 2026-11-20 (kväll)

Självständigt arbete medan du var borta. Inga fysiska skott. Projektorn användes bara för korta, automatiska bildtester utan helskärmsblixtar. Allt nytt ligger bakom flaggor som är avstängda som standard. Inget är mergat.

**Märkning av resultat:**
- **[verifierat]**: uppmätt på materialet som anges.
- **[hypotes]**: rimligt men inte bevisat.
- **[kräver fysiskt]**: går bara att avgöra med riktiga skott.

Materialet är detsamma i alla tabeller:
- **DEV:** 6 sessioner (4 projicerade från 18–19 nov och 2 fysiska). Alla trösklar och anpassningar är valda här.
- **TODAY:** 10 projicerade AI-testsessioner från 20 nov med 827 skott. Codex påpekar att TODAY har samma rigg, bakgrunder och spelsekvens. Ju fler varianter vi jämför på TODAY, desto mer blir det utvecklingsmaterial. Slutsatserna behöver därför bekräftas på **nya, oöppnade sessioner**.

## 1. Diskutrymme [verifierat]

| Åtgärd | Frigjort |
|---|---|
| Radering du godkänt: 3 poster i förmiddags, plus den trasiga inspelningen `20261120T133146Z_5d51f5d4` (6,2 GB) | 9,3 GB |
| Packning till `.tar.zst`: 11 `physical_traces`-sessioner och `fusion_week_01/trials` (174,7 GB → 35,0 GB) | **139,7 GB** |
| **Ledigt på `/data`** | **1,9 GB → 143 GB** (69 % använt) |

- **Kontroll före borttag:** varje arkiv lästes tillbaka. Varje fil kontrollerades med sha256 mot originalet, och antal och storlek måste stämma. Först då togs originalet bort.
- **Utfall:** 0 avvikelser och 0 avbrott. `fusion_week` hade 1 840 432 filer.
- **Spårbarhet:** manifest (`*.tar.zst.sha256.json`) ligger bredvid varje arkiv. Loggen finns i `/data/skjutbana/archive_log_20261120.jsonl`.
- **Diskvakten** (`automation/disk_guard.py`, incheckad som 61bd6cf):
  - Varje experiment städar sina egna tillfälliga filer efter lyckad körning.
  - Originaldata, arkiv och modeller rörs aldrig, och inte heller filer som en process har öppna.
  - Den vägrar starta körningar som skulle gå under 20 GB ledigt.
  - Allt loggas i `/data/skjutbana/disk_guard_log.jsonl`.
- Inga andra äldre originaldata har raderats.

### ⚠ Incident efter packningen – kräver ditt beslut

**Vad som hände:**
- Både din huvudkatalog och min arbetskatalog har `physical_trace_capture_enabled: true` i `content/ai/settings.json`, med roten `content/ai/physical_traces/session_20260909_194947_D01_60806ff2`.
- Före packningen vägrade spårskrivaren att skriva där, eftersom katalogen redan innehöll skott ("CAPTURE REFUSED").
- När originalet hade packats och tagits bort skapade min första prestandakörning (`base_a`, 17:27–17:31) katalogen på nytt och skrev **25 GB på 4 minuter**.
- Det gav också 40 % iowait, så den körningen är ogiltig.

**Åtgärd, utan radering:**
1. Motorn stoppades.
2. Den nyskrivna katalogen flyttades till `evaluation_runs/live_perf_20261120/QUARANTINE_physical_trace_spill_base_a`. Den innehåller bara filer från i dag 17:27–17:31, inga originaldata.
3. En platshållare skapades: `physical_traces/session_20260909_194947_D01_60806ff2/shots/_PACKED_SEE_ARCHIVE.txt`. Den återställer exakt beteendet före packningen, så att skrivaren vägrar igen. Det är verifierat i nästa körning.

**Ditt beslut behövs:**
- **Karantänkatalogen (25 GB)** får inte raderas utan ditt OK.
- **Spårskrivningen:** slå av `physical_trace_capture_enabled` eller koppla den till en ny session när du vill spela in fysiska spår igen. Platshållaren kan tas bort först efter det.

## 2. H15 och H11 tillsammans

Underlaget är kausal skuggloggning från live-koden i replay. Regler och anpassningar är valda på DEV, och siffrorna gäller TODAY. Inget alternativ faller tillbaka på gamla kedjan.

| Metod (TODAY, 827 skott) | Rätt | Fel | Avstår | Precision |
|---|---|---|---|---|
| Gamla kedjan (replay-emissioner) | 200 (24 %) | 581 omatchade | – | 26 % |
| H15 förstaval, alltid | 358 | 387 | 82 | 48 % |
| **H15 live-regel** (s≥16, kvot topp1/topp2 ≥1,15, form) | **277 (33,5 %)** | **10** | 540 | **96,5 %** |
| H15 topp5 + H11-form, s≥18 | 309 | 52 | 466 | 86 % |
| H15 + gamla kedjan (s≥18, eller s≥12 och kedjekandidat med H11≥10) | 323 | 54 | 450 | 86 % |
| H15 topp5 + lärd domare (logit ≥0,5) | 272 | 14 | 541 | 95 % |
| H15 topp5 + lärd domare med mörker- och stabilitetsdrag (logit ≥1,0) | 239 | 9 | 579 | 96 % |
| **Mörkermask + live-regel + ljusgrind k=5** (ny, se 3.3) | **305 (36,9 %)** | **16** | 506 | **95,0 %** |

Om metoderna:
- **"AI"** är här en lärd logistisk domare över H15:s topp 5. Den ordinarie AI-modellen har inte körts på H15-punkter, eftersom den kräver bildutsnitt per kandidat. Det är inte gjort.
- **Gamla kedjan har inga facitfria falsklarm här**, eftersom AI-testet alltid har ett hål. Falsklarm från ljud utan träff (klapp, eko) går inte att mäta på detta material. **[kräver fysiskt]**

**Slutsatser:**
1. **H15 hittar fler kandidater än gamla kedjan** [verifierat]. Rätt punkt finns bland topp 5 i 400 skott, mot 261 för kedjan.
2. **Bästa slutbeslut är den enkla live-regeln**, med 96,5 % precision vid 33,5 % träff. Det är 77 fler rätta och 570 färre fel än gamla kedjan på samma material.
   - Inga varianter med H11, gamla kedjan eller lärd domare slår regeln vid samma precision.
   - **Kvottestet** (topp1 tydligt starkare än topp2) är den viktigaste delen. Det stoppar animerad bakgrund. På `scroll_landscape` ger s≥18 utan kvot 21 rätt och 28 fel, medan regeln ger 13 rätt och 1 fel.
3. **H11 tillför lite ovanpå H15.** H15-värdet s är redan H11-kärnan applicerad tätt. Codex påpekar att "H15 → H11" delvis är samma evidens två gånger.
4. **Gamla kedjan återinför fel.** Att kombinera med kedjan (323 rätt / 54 fel) ger fler rätta men femdubbelt så många fel som regeln. Därför rekommenderas **ingen** automatisk kombination.
5. **Kostnad:** 60 ms CPU per H15-jobb (p95 71 ms) och cirka 17 MB indata.

### Rättat efter Codex granskning [verifierat]

- **Matchning:** ett H15-jobb kan nu bara matchas mot ett facitskott, och skottindex följer hela facitlistan. Resultaten blev identiska, så det fanns inga dubbelmatchningar.
- **Läckage:** den gemensamma regeln får bara använda kedjekandidater som fanns när H15-jobbet skars av (kandidatens tidsstämpel ≤ jobbets sista bildruta).
- **Etikett:** domarens tröskel är en logit, inte en sannolikhet. Det är rättat.

## 3. Projektorn – svartnivåer och mörka områden

Mätningarna gjordes med live-inställningarna: exponering 167, gain 0, gamma 100. Det fanns inga öppna, verifierbara hål på tavlan. Den automatiska hålsökningen i vit bild hittade bara platta ytor, se `hole_candidates_white.png`. **Därför mäts ingen direkt hålkontrast.** Det som redovisas är kamerans ljus och brus, och vilken hålsignal som skulle synas.

### 3.1 Kamerans ljus per svartnivå [verifierat] (`projector_levels_20261120`)

| Projicerad grå | 0 | 40 | 48 | 52 | 56 | 60 | 64 | 72 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Kamerans medel-DN | 0,2 | 0,9 | 5,7 | 9,2 | 14,1 | 18,0 | 23,2 | 32,1 | 41,3 | 66,2 |
| Andel pixlar under 2 DN | 100 % | 77 % | 25 % | 8 % | 2 % | 0,9 % | 0,2 % | 0 | 0 | 0 |

- **Färgtonade nivåer:**
  - Blåsvart (40,48,80) och neutral (56,58,60) ger samma ljus som grå 60 (16 DN).
  - Grönsvart (40,72,48) ger 23 DN, alltså som grå 64. Kameran ser ungefär luminans, och färgtonen ger ingen extra fördel.
- **Svag statisk textur** (±8 eller ±16) och **gradienter** ändrar inte bruset över tid. Skillnadsmetoderna (H15) påverkas därför inte av statisk textur.
- **Stabilitet:** bildmedelvärdet drev mindre än 0,35 DN under 2 s på alla nivåer.

### 3.2 Brus och minsta synliga hålsignal [verifierat] (`projector_null_20261120`)

Metoden är direkt nollmätning: samma median- och core-operator som H15, på disjunkta 9-rutorsfönster utan hål.

| Grå | 48 | 52 | 56 | 60 | 64 | 72 | 80 | 100 | 255 |
|---|---|---|---|---|---|---|---|---|---|
| Brus i core (std) | 0,156 | 0,111 | 0,054 | 0,037 | 0,020 | 0,0135 | 0,0098 | 0,0060 | 0,0017 |
| Minsta core vid 5σ | 0,78 | 0,56 | 0,27 | 0,19 | **0,10** | **0,068** | **0,049** | 0,030 | 0,009 |

Upptäckta fysiska hål hade core med median 0,093 och p25 0,045. De var dock 18 st och ett selekterat urval. På alla AI-testbakgrunder låg core mellan 0,04 och 0,15.

**Lägsta användbara nivå:**
- **64** räcker för ungefär hälften av hålen.
- **72–80** behövs för de flesta. **[hypotes: kräver fysiskt]**
- Under 56 ger kameran praktiskt taget ingen information.

Detta stämmer med förmiddagens fynd (golv 60–64), men visar att 64 är ett minimum, inte ett bra värde. Spelens ordinarie färger är inte ändrade.

**Viktigt för H15:** på en statisk scen utan hål ger brus ett toppvärde s på 14–20 vid ljus bakgrund och 24–26 vid grå 40–48. Tröskeln s≥16 skiljer alltså inte signal från brus. Det är kvottestet som gör jobbet.

### 3.3 Mörkermask i H15 (ny flagga, avstängd: `SKJUTBANA_H15_MIN_PRE_DN`)

**Mekanism [verifierat]:**
- På `dark_scene`, `images` och `black` hamnar H15:s förstaval på oupplysta pixlar. Ljuset där är 1–3 DN, och kvoten 1−post/pre blir rent kvantiseringsbrus.
- Då tränger sådana toppar undan riktiga kandidater. För `dark_scene` fanns rätt punkt bland topp 50 i 24 av 73 skott, men bara i 7 bland topp 5.

**Masken:** pixlar där PRE-ljuset (9×9-medel) är under N DN tas bort före toppsökningen. N=16 är valt från nollkalibreringen, alltså gränsen där bruset blir för stort, utan att titta på facit.

**Replay med masken** (`h18_mask16_20261120`, samma 16 sessioner):

| TODAY | Rätt punkt bland topp 1 | Rätt punkt bland topp 5 | Live-regel, rätt/fel | Regel + ljusgrind k=5, rätt/fel |
|---|---|---|---|---|
| Utan mask | 358 | 400 | 277/10 | 277/8 |
| Med mask N=16 | 376 | 410 | 306/24 | **305/16** |

**Ljusgrind:** core ≥ 5 × brus(DN), där bruset kommer från tabell 3.2. Faktorn 5 är bestämd i förväg. Utan mask tar grinden bort 2 fel och inga rätta.

**Slutsats:**
- Masken ger fler rätta: +29 rätt, varav `coord_grid` står för +11 och `images` för +5. Den ger också fler fel.
- Med ljusgrinden blir det +28 rätt och +6 fel jämfört med dagens regel.
- De kvarvarande felen ligger främst i `images` (10 rätt, 8 fel).
- **Detta är en lovande men inte färdig förbättring.** Den behöver nya sessioner, och felen i `images` behöver förstås innan den slås på.

## 4. Animerade bakgrunder

### 4.1 Kontrollerade projektorsekvenser utan hål [verifierat] (`projector_anim_20261120`)

Upplägg:
- 9 sekvenser på 9 s vardera.
- 7 påhittade skottider per sekvens, med exakt live-fönstren (PRE −0,4…−0,02 s, POST +0,15…+0,6 s).
- Allt som accepteras räknas som falsklarm.

| Sekvens | Bakgrundsförändring PRE→POST (medel/p99 DN) | Topp-5 s, median/max | Accepterade (falska) av 7 |
|---|---|---|---|
| Statisk | 0,3 / 2 | 13,7 / 16,3 | 0 |
| Långsam mörk skiva | 1,5 / 66 | 13,2 / 18,5 | **2** |
| Snabb mörk skiva | 7,2 / 122 | 13,4 / 16,5 | 0 |
| Ljusa objekt på mörkt | 6,7 / 142 | 15,2 / 19,0 | 0 |
| Mörka objekt på ljust | 5,8 / 129 | 12,9 / 15,1 | 0 |
| Blinkande objekt (2 Hz) | 2,7 / 60 | 14,8 / 17,2 | 0 |
| Långsam ljusramp | 8,4 / 11 | 13,9 / 16,4 | 0 |
| Rullande ränder | 25,8 / 102 | 11,4 / 13,9 | 0 |
| Svart scen med en grå skiva | 0,1 / 1 | **23,7** / 25,7 | 0 (men alla topp-5 på svart) |

**Falsklarm:**
- Live-regeln gav 2 falsklarm på 63 påhittade skott. Båda kom från en långsamt glidande mörk kant.
- Rörelse ger sällan toppar som klarar kvottestet. Däremot ger svarta ytor starka bruspunkter som tar platser i topp-K, vilket masken i 3.3 åtgärdar.

### 4.2 Verkliga projicerade skott per bakgrund [verifierat] (TODAY)

| Bakgrund | n | Kedjan har kandidat | H15 topp5 | H15 topp50 | Regel rätt/fel | Ljus vid H15 topp1 (DN) | Bakgrundsförändring vid topp1 |
|---|---|---|---|---|---|---|---|
| white | 67 | 46 | 57 | 63 | 54/0 | 190 | 0,34 |
| coord_grid | 68 | 41 | 59 | 61 | 44/0 | 186 | 0,48 |
| bubbles | 71 | 11 | 47 | 49 | 38/2 | 185 | 0,27 |
| moving_targets | 73 | 17 | 52 | 60 | 32/2 | 148 | 0,36 |
| scroll_landscape | 69 | 14 | 25 | 45 | 13/1 | 142 | 1,01 |
| checker_anim | 69 | 5 | 9 | 11 | 2/0 | 48 | **2,70** |
| dark_scene | 73 | 12 | 7 | 24 | 1/1 | **1,0** | 0,09 |
| images | 72 | 17 | 13 | 18 | 5/1 | **3,0** | 0,75 |
| black | 74 | 3 | 0 | 1 | 0/0 | 1,0 | – |

**Svar på frågan om medianerna påverkas [verifierat]:** ja, på två olika sätt.
1. **Snabb eller stor animation**, som `checker_anim`:
   - Bakgrundsförändringen mellan PRE- och POST-medianen (2,7) är många gånger större än hålets signal (cirka 0,15 i relativ mörkning).
   - Rätt punkt finns då inte ens bland topp 50 (11 av 69), och ingen metod kan hitta den.
   - Bara en ljusblixt kan hjälpa här, se avsnitt 7.
2. **Glidande innehåll**, som `scroll_landscape`:
   - Rätt punkt finns ofta bland topp 50 (45 av 69) men trängs undan av rörelsekanter som fastnar i medianen.
   - Kvottestet stoppar de flesta felen.

**Mörka scener** är inte ett rörelseproblem utan ett ljusproblem, se avsnitt 3. H15 är där sämre än kedjan utan mask, eftersom svarta pixlar dominerar topp-K. Masken lyfter `dark_scene` från 7 till 13 i topp 5.

## 5. Stabilitet över flera bildrutor

**Hypotesen:** ett riktigt hål ligger kvar på samma plats i varje POST-ruta, medan projektorinnehåll ändras.

**Skillnad mot H14** (falsifierat 19/11): H14 omrankade kedjans spår med ytterligare en median-SNR. Här mäts i stället tidsförloppet vid fast position, ruta för ruta:
- andelen POST-rutor med core ≥ 0,5 × median-core,
- teckenkonsistens och 20-percentilen,
- PRE-rutornas variation,
- lokal bakgrundsförändring.

Detta loggas nu med `SKJUTBANA_H15_LOG_EXTRA=1` (endast mätning).

**Resultat [verifierat]**, H15 topp 5 med s≥12 där riktiga hål jämförs med övriga toppar:

| Drag | Median, riktiga hål | Median, övriga | AUC DEV | AUC TODAY |
|---|---|---|---|---|
| Andel stabila POST-rutor | 1,00 | 0,73 | 0,80 | 0,74 |
| 20-percentil / core | 0,93 | 0,41 | 0,75 | 0,72 |
| PRE-variation (lägre är bättre) | 0,003 | 0,017 | 0,79* | 0,79* |
| Ljus under punkten (DN) | 185 | 102 | 0,73 | 0,74 |

\* Riktningen är omvänd (AUC 0,21), så lägre värde betyder hål.

**Men som beslutsregel ger stabilitet ingen vinst.** Bästa varianten, vald på DEV, var regel + stabilitet ≥0,7. Den gav 252 rätt och 7 fel på TODAY, mot regelns 277 rätt och 10 fel. I den lärda domaren gav stabilitetsdragen +29 rätt och +2 fel vid logit ≥1, men det slår ändå inte den enkla regeln.

**Förklaring** (Codex påpekade detta i förväg): POST-medianen och kvottestet väljer redan ut ihållande signaler. Stabilitet hjälper alltså att skilja kandidater åt, men tillför inget utöver regeln. Den kan bli användbar som skydd mot animation med lägre s-tröskel. **[hypotes]**

## 6. Prestanda live

**Upplägg:**
- AI-testet F2 med 48 omgångar och vit bakgrund, utan inspelning.
- Armarna körs omväxlande: utan H15 → H15 i egen tråd → utan H15 → H15 i egen tråd.
- Inga andra jobb körde: packning och replay var klara, och systemlasten låg på 2–3.
- **Mätningen är obelastad** (`live_perf_20261120`).
- Den första körningen utan H15 störs av spårskrivningen (se incidenten i avsnitt 1) och räknas inte.

| | utan H15 a | **H15 egen tråd a** | utan H15 b | **H15 egen tråd b** |
|---|---|---|---|---|
| Rätt | 20/83 (24 %) | **31/74 (42 %)** | 19/86 (22 %) | **32/83 (39 %)** |
| Falska/extra | 40 | **34** | 41 | **37** |
| Träfflatens median / p90 (ms) | 980 / 1756 | 910 / **1192** | 967 / 1830 | 934 / **1316** |
| Huvudloop, arbete p50 / p95 / p99 (ms) | 28 / 113 / 883 | 28 / 86 / 449 | 28 / 122 / 878 | 28 / 124 / 872 |
| Minne (RSS, median) | 11,4 GB | 11,4 GB | 11,6 GB | 11,6 GB |

**Svar [verifierat på live, obelastat]:**
- **H15 sänker inte huvudloopen.** Toppfördröjningarna på cirka 0,9 s (p99) finns lika mycket utan H15. De är ett befintligt problem i grundmotorn, inte orsakade av H15. **Det är nästa stora prestandafråga.**
- **CPU:**
  - H15-tråden använder 61 ms CPU per jobb (trådtid), vilket blir cirka 2 % av en kärna vid AI-testets skottakt.
  - Väntetiden i H15:s egen kö är som mest 5–16 ms (p95).
  - Väggtiden per jobb är 270 ms (p95), eftersom tråden delar CPU med motorn.
  - Processens totala CPU varierade mellan körningarna (104–191 % av en kärna) utan koppling till H15. Mätningen per tråd är därför osäker. Jobbens egen trådtid är det tillförlitliga måttet.
- **Minne:** ingen mätbar ökning. Indata är cirka 17 MB per jobb.
- **Försenade och gamla resultat:**
  - 0 sena H15-resultat och 4 `no_frames` på 153 jobb.
  - Resultaten är kopplade till skott-id, ljudtopp och generation, och gamla generationer kastas. Det är testat i självtestet. Inga resultat hamnade på fel skott.
- **Kamerans bildfrekvens och tappade bildrutor:** detta mättes inte i dessa körningar, eftersom inspelningen var avstängd. I dagens 13 inspelade körningar, inklusive 3 med H15 i egen tråd, var `dropped_camera` 0 överallt. Bildfrekvensen låg på cirka 20–23 bilder/s i alla armar.
- **Effekt:**
  - Fler rätt: 23 % → 40 %.
  - Något färre falska: 40,5 → 35,5.
  - Kortare latens i svansen (p90): 1,8 s → 1,25 s.
- **Osäkerhet:** det är bara 2+2 körningar med vit bakgrund, så ingen statistisk säkerhet utöver att båda paren pekar åt samma håll. Fler bakgrunder behövs.

## 7. Ljusblixtar (analys, inga nya blinkserier)

**Vad som redan är uppmätt [verifierat]:**
- En vit puls på 3 projektorbilder (cirka 35 ms) ger alltid exakt en fullt upplyst kamerabild.
- 1–2 bilder ger opålitliga eller delvis upplysta bilder.

**Vad dagens mätningar lägger till [verifierat]:**
- Vid helvitt är bruset i core 0,0017. Ett hål med core 0,05–0,09 får då **SNR 30–50 i en enda bild**, utan median.
- Vid grå 72 är SNR för samma hål bara 4–7.
- **Därför löser en blixt både mörka scener och animation:** referensblixten och blixten efter skottet visar samma vita fält, så spelinnehållet tas bort helt. Det gäller precis de bakgrunder där inget annat fungerar: `black`, `dark_scene`, `checker_anim` och `images`.

**Begränsningar:**
- Synligt för skytten.
- Kräver en referensblixt.
- Ger cirka 0,7–0,8 s latens med 0,6 s väntetid för dubbelskott.
- Codex gav i november fyra blockerande anmärkningar på `precision_flash.py`, och de är inte åtgärdade i koden (senaste ändringen är 765580a):
  - topparna registreras efter skottvägen,
  - hållna träffar släpps vid för få hål,
  - blixtrutor går in i HitScanner,
  - burst-tillståndet är föränderligt.

**Förslag:** i ett gemensamt försök med dig (inte obevakat):
1. Åtgärda de fyra punkterna.
2. Testa blixt bara i precisionslägen (EXACT), med 35 ms och en blixt per skottserie.
3. Mät hålkontrast vid blixt mot grå 72 på riktiga hål. **[kräver fysiskt]**

## 8. Kan kedjan förenklas?

**Arkitekturen "kamera → H15 → topp 5 → H11 → acceptans → träff"** testades på samma material:

| Kriterium | Gamla kedjan | Förenklad (H15-regeln) |
|---|---|---|
| Rätt / fel (TODAY) | 200 / 581 | 277 / 10 |
| Rätt punkt bland kandidaterna | 261 | 400 (topp 5) |

**Vad kedjan har som H15 saknar [verifierat]:** 62–65 TODAY-skott där bara kedjan har en kandidat.
- **22 av dem saknar H15-jobb helt** (`no_frames` eller `no_job`). Det är fönster- och timingfall, till exempel tätt följande skott, där H15 inte har ett giltigt PRE-fönster.
- **Resten fördelar sig så:**
  - `images` 12, `moving_targets` 10, `gray` 8, `dark_scene` 8 och `scroll_landscape` 6;
  - övriga bakgrunder 1–5 skott vardera.
- **Gamla kedjan ger dessutom tidigare beslut**, ibland redan efter 0,2–0,3 s. Det har fortfarande betydelse för snabba spel.

**Evidensfälten** (förmiddagens rapport):
- Bara H11-familjen kostar något när den tas bort (−15).
- Cirka 20 fält är konstanta, och 159 fältpar har |ρ| ≥ 0,9.

**Rekommendation:** ta inte bort något nu, men flytta tyngdpunkten.
1. **H15-regeln som huvudbeslut** för skott där H15 har ett giltigt fönster.
2. **Gamla kedjan bara där H15 saknar fönster** (cirka 7 % av skotten), och bara med samma höga precisionskrav. Inte som allmän reserv.
3. **De ~20 konstanta evidensfälten** kan tas bort ur loggning och modell efter ett eget A/B-test. De påverkar inte besluten i dag.

Förenklingen kräver fortfarande verifiering av:
- dubbelskott och CO2-par,
- objekt- och regionträffar,
- negativa ljudfall (klapp och eko),
- nya sessioner som inte har använts i utvecklingen. **[kräver fysiskt för klapp/eko och riktiga hål]**

## 9. Codex granskning (två rundor)

**Runda 1**, före resultaten (`codex_a7.md`). Åtgärdat:
- unik jobbmatchning,
- kausal gräns för kedjeevidens,
- logit-etiketten,
- direkt nollmätning i stället för härledd SNR.

Varningar som står kvar:
- TODAY har blivit utvecklingsmaterial.
- 12 px-uteslutningen kring föregående skott är ett spatialt veto mot nära träffar.
- `CO2_PAIR_S` kan slå ihop riktiga snabba dubbelskott.

**Runda 2**, efter resultaten (`codex_a8.md`):
- **Masken med ljusgrind är en avvägning**, inte en entydig vinst. Den ska vara avstängd tills den frysta kombinationen har testats på nya sessioner.
- **5σ per punkt** motsvarar inte falsklarmsrisken efter toppsökning över hela bilden. Det behöver mätas med hela detektorn på nya hålfria sekvenser.
- **Felen i `images`** ska klassificeras innan någon ny tröskel införs (rörelse, registrering, kvantisering eller kvarstående bilddetalj). Inga bakgrundsspecifika regler.
- **Stabilitet som hård spärr** passar möjligen en försiktighetsprofil, men är ingen generell förbättring.
- **Live med 2+2 körningar** räcker inte som statistiskt bevis, och vinsten kan inte tillskrivas just den egna tråden.
  - Svar på den sista punkten: förmiddagens jämförelse med samma H15-regel gav egen tråd 67/163 mot delad 39/164. Även det är bara en körning per arm.

## 10. Svar på dina frågor

| # | Fråga | Svar | Status |
|---|---|---|---|
| 1 | Diskutrymme frigjort? | 9,3 GB raderat (godkänt) och 139,7 GB genom förlustfri packning. Ledigt gick från 1,9 till 143 GB. | verifierat |
| 2 | H15 + H11? | H15 hittar fler kandidater (400 mot kedjans 261 i topp 5). Bästa beslutet är den enkla H15-regeln: 277 rätt och 10 fel mot kedjans 200 rätt och 581 fel. H11 ovanpå H15 tillför lite, eftersom det är samma kärna. Kombination med gamla kedjan ger fler rätta men femdubbla fel. | offline/replay verifierat; live 23 % → 40 % rätt |
| 3 | Bästa svartnivå? | Under 56 får kameran ingen användbar information. 64 räcker för ungefär hälften av hålen, och **72–80 behövs för de flesta**. Färgton (blå eller grön) och statisk textur hjälper inte. | ljus och brus verifierat; hålkontrast **kräver fysiskt** |
| 4 | Animerade bakgrunder? | **Två mekanismer.** (a) Stor animation (`checker_anim`): bakgrundsändringen överröstar hålet i medianerna, och rätt punkt saknas även bland topp 50. Bara en blixt kan hjälpa. (b) Glidande innehåll: rörelsekanter tränger undan hålet, och kvottestet stoppar de flesta felen. **Mörka scener är ett ljusproblem**, inte ett rörelseproblem. | verifierat |
| 5 | Stabilitet över bildrutor? | Den skiljer kandidater åt (AUC 0,74–0,80), men som beslutsspärr tar den bort fler rätta än fel (252/7 mot 277/10). Skillnad mot H14 och skäl: POST-medianen och kvottestet väljer redan ihållande signaler. | verifierat; ingen vinst |
| 6 | Påverkar H15 huvudloopen? | Nej. p99 är cirka 0,9 s med och utan H15, och de topparna kommer från grundmotorn. H15 kostar 61 ms CPU per jobb, kön är högst 16 ms, och inga resultat kom sent eller hamnade på fel skott. | live verifierat (2+2 körningar) |
| 7 | Förenkling av kedjan? | H15-regeln kan bli huvudbeslut. Kedjan behövs i dag där H15 saknar fönster (cirka 7 %), för tidiga beslut och i `images`/`dark_scene`. Cirka 20 konstanta evidensfält kan tas bort efter ett eget A/B-test. | delvis verifierat |
| 8 | Faktisk förbättring på samma material? | **Replay TODAY:** 200 → 277 rätt, 581 → 10 felaktiga emissioner (H15-regeln utan reserv). **Live, vit bakgrund:** 23 % → 40 % rätt och något färre falska. **Mörkermask med ljusgrind:** +28 rätt och +6 fel, inte frysta än. | se respektive avsnitt |
| 9 | Nästa steg | Se nedan. | – |

**Gemensam rekommendation med Codex, i prioritetsordning:**
1. **Frys H15-regeln**, med mörkermask och ljusgrind som separat arm, och mät på **nya, oöppnade sessioner**:
   - flera bakgrunder,
   - hålfria sekvenser för falsklarm,
   - ett fysiskt pass med riktiga hål, inklusive klapp och eko. **[kräver fysiskt]**
2. **Profilera huvudloopens ~0,9 s-stopp** (p99) steg för steg. Det är ett separat problem i grundmotorn som påverkar alla spel.
3. **Hålkontrast med riktiga hål** vid grå 64/72/80 och vid en 35 ms-blixt i ett gemensamt försök. Åtgärda först Codex fyra punkter i `precision_flash.py`. **[kräver fysiskt]**
4. **Klassificera felen i `images` och `dark_scene`** innan någon ny tröskel införs.
5. **Spårskrivningen i `settings.json`** behöver ditt beslut (se incidenten i avsnitt 1).

## 11. Ändringar, data och flaggor

**Incheckat på `claude/latency`** (inte pushat, inte mergat):
- 61bd6cf: diskvakt.
- 7c5b1ba: lagringsrapport.
- 19c87a0: H15 mörkermask och extra loggning. Båda är avstängda som standard, och självtestet visar bit-identiskt resultat när de är av.

**Nya flaggor:**
- `SKJUTBANA_H15_MIN_PRE_DN` (0 = av).
- `SKJUTBANA_H15_LOG_EXTRA` (endast mätning).

**Data** (`/data/skjutbana/evaluation_runs/`):

| Katalog | Innehåll |
|---|---|
| `projector_levels_20261120` | svartnivåer, ljus och brus |
| `projector_null_20261120` | nollmätning av brus |
| `projector_anim_20261120` | animerade sekvenser utan hål |
| `h17_extra_20261120` | replay med extra loggning |
| `h18_mask16_20261120` | replay med mörkermasken |
| `live_perf_20261120` | prestandamätning live |

Råa bildstaplar sparades inte.

**Spelens ordinarie färger och inställningar är oförändrade.** Kamerainställningarna återställdes efter varje försök, och projektorn lämnades svart.
