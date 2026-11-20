# Kandidatkedja, enkel prickdetektor, projektor och H15 live – 2026-11-20 (förmiddag)

Detta svarar på arbetsordern "Kandidatgenerering, enkel håldetektering, H11/H15 och projektorexperiment".

Allt är mätning; inget standardbeteende är ändrat och inget är incheckat. Resultaten ligger under `/data/skjutbana/evaluation_runs/` med en egen katalog per experiment.

## Kort svar på huvudfrågan

> Varför genererar vi inte rätt kandidat oftare, och kan enklare bildanalys lösa det?

Kedjan letar på fel sätt, inte med för få kandidater:

- Den föreslår punkter från **enskilda bilder** tidigt efter skottet, med en **osignerad** förändringskarta (absdiff/zscore/saliency).
- Mörkning väger bara 0,32 i saliency-formeln.
- I de skott där rätt kandidat saknas ligger kedjans närmaste kandidat i median **59 px** bort. Det är alltså inte nära missar som en större Top-N skulle rädda.

En **enkel prickdetektor** fungerar mycket bättre:

- Den använder signerad relativ mörkning `1 − post/pre` på **medianer** av PRE- och POST-bilder.
- Den filtreras med en hålformad kärna (samma matchade filter som H15 använder).
- Kandidat-recall blir **233/349 inom topp 50**, mot kedjans **150/349**.
- Tillsammans med kedjan blir det **249/349 (71 %)**, mot 43 %.

## 1. Kandidatkedjan (kodkartläggning, fil:rad i worktree `skjutbana-claude`)

**Aktiv väg live och i replay:**
- Ljudtopp → worker-jobb (1 per skott, ≥ 55 ms mellan bilder).
- Legacy V1 + V2 FAST-extraktor.
- Hybridsammanslagning → V2222-städning → lokal bekräftelse → spår (12 px) → val.

**Gränser (inga styrs av miljövariabler; de kommer från AI-settings eller `content/ai/detector_v2*.json`):**

| Steg | Gräns | Typ | Var |
|---|---|---|---|
| FAST primära lokala maxima | 520 | per bild | shot_fast_v2225.py:95,363 |
| FAST temporal räddning (bara om primära < 90) | 300 | per bild | :96,374,402 |
| FAST efter NMS 3,5 px och rutkvot 8×6×7 + 100 | **180** | per bild | :454-466 |
| Legacy V1 | 200 | per bild | hit_scanner.py:1159 |
| Hybridsammanslagning | 200 (50 V2 + 150 legacy) | per bild | candidate_generator_v2.py:2722-2730 |
| Kandidatbank | 240 / bär 40 | per skott | :2567 |
| V24-ackumulator | 320 | per skott | detector_v24_extension.py |
| V26-valv | 680 | per skott | detector_v26_extension.py |
| V2222 slutlig gräns | **200** | per worker-resultat | hit_scanner_v2222.py:291 |
| Lokal bekräftelse | 200, 2 rundor | per skott | shot_fast_v2225.py:650 |
| AI-rankning (advisory) | top 10 | – | ai/runtime.py |

**Hårda borttag före val:**
- ROI-erosion 5 px.
- Legacy: area, cirkularitet och radie; hela diffen förkastas vid > 5 % förändring.
- FAST: evidensgrindar och saliency-tröskel.
- Känt hål inom 18 px med låg PRE-förändring.
- Horisontella rippor.
- Ägarskap: resultat efter nästa ljudtopp kastas.
- Grind i lokal bekräftelse.
- Redo-krav: 3 träffar och 0,09 s, eller 2 träffar och 0,3 s.
- Timeout 2 s.

**Vilken lista bedöms var:**
- **H11** mäter FAST- och FULL-förslagen samt den sammanslagna listan. Det är samma lista som exporteras till `candidates.jsonl`.
- **AI-modellen** ser workerresultatet (≤ 200) men är bara rådgivande.
- **H15** söker hela viewporten oberoende av förslagen.

**Live och replay skiljer sig så här:**
- Replay har en fast workerkostnad på 0,15 s; live använder verklig CPU-tid.
- Replay har ingen AI-skuggning och inga regioner som standard.
- `detector_v2*.json` läses alltid från repot.

**Latent fel (från kodläsning):**
- I FAST-läge återinjiceras V24:s senaste tile-probes, upp till 100 st, utan kontroll av bild eller skott.
- De förnyas bara av FULL-extraktion.
- I de 6 replay-exporterna finns inga V24-fält alls, så felet har inte påverkat dagens siffror. Det bör ändå åtgärdas.

**Kandidatgränserna 25–500:**
- Eftersom kedjans missar ligger långt från hålet (§2.2) hjälper det inte att höja gränserna 180/200.
- Därför har jag i stället mätt rangordning för rangordning hos detektorerna nedan; det motsvarar "max N kandidater".

## 2. Enkel prickdetektor (`dot_recall_20261120`)

**Material:** samma 6 utvecklingssessioner som i natt (AI 1336/1343/1227/1623, fysisk 04f1be81, tejpad da06f87a), 349 skott med facit.

**Metod per skott:**
- PRE = median [topp−0,4, topp−0,02] s; POST = median [topp+0,15, min(topp+0,6, nästa skott)] s. Det är samma fönster som H15.
- Varje detektor rangordnar punkter i tavlans ROI.
- Recall@K räknas inom 8 px.

### 2.1 Resultat

| Detektor | @1 | @5 | @20 | @50 | @100 | @300 | Räddar av kedjans 199 missar (@50) | Kedja ∪ detektor @50 | ms/skott |
|---|---|---|---|---|---|---|---|---|---|
| **Nuvarande kedja** | – | – | – | – | – | – | – | **150** | – |
| **MF** (hålkärna på mörkning, = H15) | **147** | 196 | 217 | **233** | 241 | 261 | **99** | **249** | ca 50 |
| Ring (centrum minus ring, 3 skalor) | 112 | 158 | 190 | 209 | 223 | 243 | 79 | 229 | ca 50 |
| Blob (OpenCV SimpleBlobDetector) | 78 | 140 | 195 | 211 | 221 | 232 | 83 | 233 | ca 60 |
| CC (tröskel z>3, sammanhängande områden) | 98 | 161 | 185 | 195 | 205 | 212 | 67 | 217 | ca 140–270 |
| LoG (3 skalor) | 68 | 94 | 121 | 151 | 174 | 204 | 41 | 191 | ca 45 |
| Statisk (bara POST-bilden, ingen PRE) | 17 | 49 | 106 | 153 | 161 | 178 | 57 | 207 | ca 50 |

### 2.2 Tolkning

- **Har vi redan en sådan metod?** Ja och nej. Matchningsfiltret finns i `hole_likeness.dense_peaks` och H15, men **bara H15 använder det, och bara toppkandidaten**.
  - H12, matchade filter-förslag i kedjan, arbetar på enskilda bilders kartor, och det var just det som inte hjälpte.
  - Kedjans egna generatorer är inga prickdetektorer.
- **Varför missar kedjan?** I 199 missade skott är medianavståndet till närmaste kedjekandidat 59 px; bara 7 ligger inom 8–16 px. 15 skott har inga kandidater alls.
- **Vad betyder topp 1?** MF tar rätt hål som *topp 1* i 147 skott. I 44 av dem saknar kedjan rätt kandidat; kedjans närmaste ligger då i median 118 px bort.
- **Formfilter jämfört med cirkelform:** det enkla filtret på hålform slår blob, LoG och CC. Medianer jämfört med enskilda bilder, och signerad mörkning jämfört med osignerad förändring, är troligen huvudskälen. Det är inte verifierat var för sig.
- **Hålliknande strukturer direkt i kamerabilden (statisk)** fungerar dåligt, eftersom gamla hål, tejp och spelinnehåll stör. Jämförelsen mellan före och efter är nödvändig.
- **71 skott hittas inte av någon detektor**, inte ens i topp 300. Det är främst bakgrunderna *black* (1–2 av 26 hittas), *checker_anim* och *images*. Där syns hålet inte fysiskt i kameran (se §3).

### 2.3 Per bakgrund

Kolumnerna är kedjan, MF@50 och ring@50.

| Bakgrund | Kedja | MF@50 | Ring@50 | n |
|---|---|---|---|---|
| black | 1 | 2 | 1 | 26 |
| bubbles | 3 | **21** | 21 | 28 |
| moving_targets | 6 | **27** | 25 | 28 |
| dark_scene | 7 | **19** | 8 | 31 |
| scroll_landscape | 5 | **15** | 10 | 26 |
| coord_grid | 18 | 30 | 30 | 32 |
| checker | 9 | 12 | 12 | 26 |
| checker_anim | 6 | 3 | 4 | 24 |
| images | 12 | 12 | 11 | 26 |
| gray | 18 | 21 | 17 | 26 |
| white | 23 | 23 | 21 | 26 |
| white_grid | 23 | 28 | 28 | 28 |
| co2_ar15 (fysiskt) | 19 | 20 | 21 | 22 |

De största vinsterna finns i rörligt och mörkt innehåll. Där drunknar kedjans osignerade förändringskarta i rörelse, medan medianen och den signerade mörkningen filtrerar bort den.

## 3. Projektorn (`projector_floor_20261120_*`, `projector_pulse_20261120_a`)

**Kamerainställningar:** exponering 167, gain 0, vitbalans 4600 K fast, autofokus av.

Mätvärdet är tavlans medelljus i kameran (0–255) och tidsbrus per pixel.

| Projicerat | Tavla i kameran | p5 | Brus |
|---|---|---|---|
| svart 0 | 0,3 | 0 | 0,20 |
| golv 6 / 12 / 24 | 0,4 / 0,6 / 0,9 | 0 | 0,2–0,3 |
| golv 40 | 1,1 | 0,1 | 0,43 |
| blåsvart 12,16,28 / grönsvart 12,24,16 | 0,6 | 0 | 0,3 |
| golv 24 + textur 8 / 16 | 0,9 | 0 | 0,3 |
| golv 44 | 2,6 | 0,3 | 0,5 |
| golv 48 | 5,4 | 0,8 | 0,8 |
| golv 52 | 9,0 | 2,0 | 1,0 |
| golv 56 | 14,1 | 4,8 | 0,9 |
| golv 60 | 17,9 | 9,6 | 0,9 |
| golv 64 | 23,6 | 16,5 | 0,8 |
| golv 72 | 32,3 | 23,5 | 0,9 |
| golv 80 / 96 | 41,8 / 61,0 | 33 / 51 | 0,8 |
| grå 128 | 102 | 90 | 0,7 |
| vit 255 | 193 | 185 | 0,6 |

**Slutsatser:**
- **Svartgolv 12–40 är osynliga för kameran.** Projektorns ljus är så lågt (gamma: 24 → ca 0,5 % av vitt) att det försvinner under kamerans svartnivå. Det gäller också färgtonerna och texturen. **Ett hål på sådan bakgrund kan inte synas**, och det stämmer med att bakgrunden *black* ger 1/26.
- **Golvet börjar verka vid ca 52–56 och är användbart från ca 60–64.** Det motsvarar en tydligt mörkgrå bild, ca 4–5 % av vitt. Det är ett märkbart ingrepp i spelbilden.
- **Kamerans gamma (separat hypotes, inte ändrad i motorn):**
  - Gamma 200 ger golv 60 → 42 (i stället för 18) med brus 1,2.
  - Gamma 300 ger golv 40 → 21, men med brus 2,0.
  - Vitt klipps inte (205–215).
  - `brightness` lägger bara på en offset och hjälper inte.
- **Hålkontrasten kunde inte mätas:** tavlan har för få öppna hål (nästan allt är tejpat). Det kräver några nya hål.

**Ljuspuls** (svart bakgrund, vit puls, 4 repetitioner per längd, kamera ca 24 fps):

| Pulslängd | Kamerabilder upplysta |
|---|---|
| 1 projektorbild (≤ 4 ms) | 1 av 4 pulser syntes – opålitligt |
| 2 bilder (ca 17 ms) | alltid 1, men delvis upplyst (43–150 av 185) |
| **3 bilder (ca 35 ms)** | **alltid exakt 1 fullt upplyst** |
| 6 bilder (ca 83 ms) | 2 fullt + 1 delvis |

Med en puls på ca 35 ms får man en fullt upplyst referensbild. Om den ger bättre hålbilder kräver mätning med riktiga hål. Pulsen har inte använts under skjutning.

## 4. H11 i live-rankningen (nattens holdout, 285 skott)

| Rangordning | Rätt topp 1 |
|---|---|
| Nuvarande live-poäng | 154 |
| Bara H11 | 161 |
| Live-poäng + w·H11 (robust z, w valt på train = 5) | **162** |
| Samma, w = 1–1,5 (bäst på holdout, inte valt på train) | 166 |
| Live-poäng, H11 avgör vid jämnt läge (inom 5 %) | 155 |
| H11 primär, live-poäng avgör jämnt läge | 161 |

Att lägga till H11 ger +8 (+12 i bästa fall), men det är litet jämfört med recall-vinsten i §2. Som avgörare vid jämnt läge hjälper H11 knappt.

## 5. H15 live med snabbmedian (AI-testet, 48 projicerade skott)

Alla arms kördes utan annan belastning på datorn.

| Arm | Rätt | Falska | Sena H15-resultat | Träfflatens median |
|---|---|---|---|---|
| Bas | 11/75 | 48 | – | 0,93 s |
| H15, tidsgräns 1,3 s | 19/77 | 37 | 30 av 48 | 1,23 s |
| **H15, tidsgräns 1,8 s** | **25/80** | 39 | 11 av 48 | 1,31 s |
| (H15 1,3 s med offline-jobb samtidigt: störd, räknas inte) | 11/80 | 28 | 39 | – |

**Observationer:**
- **H15 fungerar live**: 11 → 25 rätt, och färre falska än basen.
- Begränsningen är att workern är upptagen: ordinarie jobb tar ca 0,5 s i median. H15-jobbet köar 0–380 ms och tar 100–230 ms live (ca 90 ms offline).
- Med en tidsgräns på 1,8 s ryms det inom precisionsspelens budget på ca 2 s, men inte inom snabba spels 1 s.
- Snabbmedianen är bitidentisk med `np.median`; det är verifierat offline för n = 1–12.

## 6. Rekommendation (nästa enskilda hypotes)

**H16 – prickförslag från mediandiff in i kedjan.** Det ska vara bakom en flagga och avstängt som standard.

- När POST-fönstret har ≥ 3 bilder (ca topp+0,3 s) körs MF-prickdetektorn på medianerna, en gång per skott.
- Topp K (t.ex. 20) läggs in som kandidater i den ordinarie kedjan. Rangordningen görs av H11 och H15:s acceptansregel, som utvidgas från bara topp 1 till topp K.
- **Förväntad takhöjd:**
  - Kandidat-recall 150 → ca 250/349.
  - H15 är redan topp 1 av samma karta, och därför ger det redan +10 procentenheter.
  - Vinsten ligger i att H11 och kedjans spår kan välja bland topp 20 i stället för att bara ta topp 1 eller avstå.

**Plan:**
1. Mät offline H11-rankningen på MF:s topp-20 med samma facit (räknas som en hypotes).
2. Testa i replay på 485-skottsmaterialet med falska positiva.
3. Testa live.

**Prestanda:** ca 50 ms per skott för kartan. Det kostar mindre än ett ordinarie workerjobb, men det måste få egen tid. Tidsgränsen är den verkliga flaskhalsen; se §5.

**Övrigt:**
- **Projektor:** svartgolv bara om spelet tål ca 60–64 (mörkgrått). Under det gör det ingen nytta.
- **Hålkontrast:** golv, puls och vitt jämförs med riktiga hål i ett separat test.
- **Förenkling:** när H16 är på plats kan man mäta vilka av FAST/legacy/V24/V26-vägarna som fortfarande ger unika rätt kandidater. Inget tas bort innan dess.

---

# Del 2 – Prickdetektorn undersökt (eftermiddag 2026-11-20)

**Underlag:** samma 6 utvecklingssessioner, 349 skott. Katalog: `dot_merge_20261120`.

Alla kandidater, både kedjans och prickdetektorns, mäts med samma täta hålmått på PRE/POST-medianerna. Därför jämförs rangordningarna på lika villkor.

## P1. Hur många prickkandidater? (topp K)

| K | Skott med rätt kandidat (kedja + prick) | Nya jämfört med kedjan |
|---|---|---|
| 0 (bara kedjan) | 150 | – |
| 5 | 222 | +72 |
| 10 | 231 | +81 |
| 20 | 237 | +87 |
| 30 | 246 | +96 |
| 50 | 249 | +99 |

**Beräkningstid:** själva kartan tar ca 46 ms per skott, oberoende av K. Live tar hela jobbet (medianer + karta + mått) 100–230 ms. Varje extra kandidat kostar bara en liten patch-mätning.

## P2. Slutligt val – kedja, prick + H11 och inlärd bedömning

Siffrorna är rätt/fel. "Fel" betyder att vald punkt ligger mer än 8 px från rätt hål.

| Rangordning | Kandidater | Rätt | Fel | Avstår |
|---|---|---|---|---|
| Live-poäng (dagens) | kedjan | 104 | 230 | – |
| H11 `ev_hole_morph_snr` | kedjan | 119 | 215 | – |
| Tät H11 (MF-värde) | kedja + prick, alla K | 147 | 202 | – |
| Inlärd bedömning (lämna en session utanför) | kedja + prick, K ≥ 5 | 159 | 190 | – |
| Tät H11 + acceptansregel (s ≥ 16, core ≤ 0,4, bipolar ≤ 1) | kedja + prick | **144** | **87** | 118 |
| Samma, faller tillbaka på kedjans H11 vid avstående | | 153 | 188 | – |
| Inlärd bedömning, avstår lägsta 30 % | kedja + prick | 140 | 104 | 105 |
| Inlärd bedömning på bara prickarna, K = 3, avstår lägsta 30 % | prick | 152 | 92 | 105 |
| Bara prickarnas topp 1, avstår lägsta 30 % | prick | 146 | 98 | 105 |

**Slutsatser:**
- **Vinsten kommer från kartan, inte från K.** Prickdetektorns förstaval är redan bättre än allt kedjan väljer: 147 rätt mot 119.
- Med acceptansregeln blir det **+25 rätt och 128 färre fel** jämfört med kedjans H11, men 118 skott avstår.
- Ett större K höjer kandidat-recall (+72 → +99), men vid ett givet antal falska ger K > 3 nästan ingenting (152/92 mot 146/98).
- **Rekommenderat K = 5:** det ger 73 % av recall-ökningen till försumbar kostnad. Uppåt 10 om man vill ha marginal.
- **AI jämfört med H11:** den inlärda bedömningen slår H11 när man alltid väljer en punkt (159 mot 147). Vid samma andel avstående blir skillnaden liten.
- Enligt AGENTS ska en inlärd bedömning bara köras i skuggläge. Den frysta vikten finns i `mf_judge_frozen.json` och är bara för utvärdering.
- **Fallback till kedjan återinför felen** (153/188). Fail-open skyddar alltså inte mot falska träffar, och det påpekade även Codex.

## P3. Prestanda och kö

- **Live:** ordinarie workerjobb tar ca 0,5–0,6 s i median. H15-jobbet köade 0–380 ms bakom dem, och 30 av 48 resultat blev sena.
- **Infört, av som standard:** `SKJUTBANA_H15_OWN_THREAD=1` i `shot_dense_h15.py`. Det ger en egen tråd (`shot-dense-h15`) för det täta jobbet. numpy och cv2 släpper GIL, så jobbet kan köras parallellt. Kamerainsamlingen berörs inte.
- **Ännu inte mätt.** Enligt Codex ska det jämföras på identisk inspelning:
  - delad eller egen tråd,
  - kötid, CPU och beslutslatens,
  - kamerans bildluckor och FPS,
  - huvudloopens p95/p99,
  - fast antal OpenCV-trådar.
- Risken är konkurrens om fyra kärnor och OpenCV:s trådpool.
- **Lärdom:** inga offline-jobb samtidigt med livetester. Det förstörde en körning i förmiddags.

## P4. Varför missar kedjan? Isolerade hypoteser

Måttet är rätt hål inom topp 50 för varje variant, av 349 skott.

| Variant | Topp 1 | @50 | Av kedjans 199 missar |
|---|---|---|---|
| MF på medianer (prickdetektorn) | 147 | 233 | 99 |
| Samma men osignerad \|D\| | 138 | 220 | 89 |
| Tidig bild: en POST-bild direkt efter +0,15 s | 113 | 169 | 46 |
| En PRE- och en POST-bild (som en enskild bild) | 118 | 161 | 39 |
| Kedjeliknande saliency (5,8·z + 0,72·abs + 1,75·DoG + 0,32·mörkning), medianer | 47 | 99 | 16 |
| Samma, en tidig bild | 13 | 70 | 4 |

**Tolkning:**
1. **Hur bevisen viktas och formas är den största faktorn** (233 → 99). En hålformad kärna på relativ mörkning hittar hål. En summa av förändringsmått lockas av rörelse och kanter.
2. **Tidiga eller enskilda bilder** är nästa faktor (233 → 161–169). Medianen av flera bilder tar bort brus och rörelse.
3. Signerad jämfört med osignerad är liten (233 → 220).

**Förbehåll (Codex):**
- Saliency-varianten är en approximation, inte kedjans riktiga karta, som har registrering, egen brusskattning och egen förslagslogik. Den förklarar alltså inte kedjans fel exakt.
- Att kedjans närmaste kandidat ligger långt bort utesluter inte helt att gränser kastat bort närmare förslag. Det kräver en replay med höjda gränser, som inte är gjord.
- Fönstren här bygger på facit-tid (+0,025 s), inte på detektorns egna ljudtoppar.
- "Fel" mäter lokalisering på riktiga skott, inte falsklarm från klappar och ekon.

## P5. Det latenta V24-felet

- **Regressionstest:** `automation/v24_stale_tile_probe_selftest.py`.
  - Testet visar att en tile-probe från en tidigare FULL-extraktion (skott 1) slås in i en senare FAST-bild (skott 2).
  - Med `SKJUTBANA_V24_FRESH_TILE_PROBES=1` slås den inte in, medan en probe från samma bild behålls.
- **Korrigering (av som standard):** i `detector_v24_extension.py` sparas probens bildtid, och proben slås bara in om bildtiden är densamma.
- Befintliga V24-självtester går igenom.
- I de 6 replay-exporterna finns inga V24-fält, så felet har inte påverkat dagens mätningar. Om det slår till live (efter FULL-räddning) är ännu inte mätt.

## P6. Gamma 200 live (AI-testet, standard gamma 100 återställd efteråt)

| Körning | Rätt | Falska |
|---|---|---|
| gamma 100 a | 17/86 | 45 |
| gamma 200 a | 14/91 | 30 |
| gamma 100 b | 8/83 | 41 |
| gamma 200 b | 13/85 | 41 |
| **Summa gamma 100** | **25/169 (15 %)** | **86** |
| **Summa gamma 200** | **27/176 (15 %)** | **71** |

**Slutsats:**
- Ingen säkerställd skillnad. Variationen mellan körningar med samma inställning (17 mot 8) är större än skillnaden mellan inställningarna.
- Gamma 200 räddar inte svart eller mörk scen i den här kedjan: 0 rätt i båda.
- Prickdetektorn på dessa inspelningar per gamma körs efter replay, se P9.
- Små hål och brus med gamma kräver de riktiga hålen.

## P8. 249 respektive 71 – vad skiljer?

Kombinationen kedja + prickar topp 50 hittar rätt kandidat i 249 av 349 skott, så **100 skott saknar korrekt kandidat**. Dessa 100 består av:

| Grupp | Antal |
|---|---|
| Ingen metod alls hittar hålet ens i topp 300 (MF, ring, blob, LoG, CC, statisk) | 71 |
| MF hittar det, men på plats 51–300 | 19 |
| Bara en annan detektor hittar det (statisk 4, ring 3, blob 3) | 10 |

- **De 71 är fysiskt eller optiskt osynliga hål:** black 23, checker_anim 14, checker 8, dark_scene 7, images 6, bubbles 4, scroll_landscape 4 med flera.
- **De 29 övriga är synliga men rangordnas för lågt.** De går i princip att rädda med större K eller fler detektorer, till priset av fler kandidater.

## P9. H15 på jämförbart underlag (`live_paired_20261120`)

AI-testet slumpar bakgrunder och hål per körning, så live-körningarna har olika nämnare. Därför har varje inspelad live-session spelats om i replay med varje variant, så att jämförelsen görs på **identiska bilder och facit**.

**A. Replay, 7 nya sessioner från i dag, 577 skott (samma för alla varianter):**

| Variant | Rätt | Andel | Falska |
|---|---|---|---|
| Bas | 137/577 | 23,7 % | 410 |
| H15 (tidsgräns 1,3 s) | **217/577** | **37,6 %** | **330** |
| H15 (tidsgräns 1,8 s) | 217/577 | 37,6 % | 330 |

- Det blir +80 rätt och −80 falska. H15 är bättre i alla 7 sessionerna, med +9 till +14 rätt per session.
- Tidsgränsen påverkar inte replay, eftersom den virtuella workern inte modellerar kön. **Förlusten live är alltså ren timing.**
- **Gamma-strata i replay:**

| Gamma | Bas | H15 |
|---|---|---|
| 100 (2 sessioner) | 45/169 | 67/169 |
| 200 (2 sessioner) | 41/176 | 60/176 |

  Gamma ger ingen vinst.

**B. Live jämfört med replay av samma session:**

| Live-körning | Live | Replay av samma session, samma variant | Live / replay |
|---|---|---|---|
| Bas | 11/75 | 17/75 | 65 % |
| H15 delad worker, 1,3 s | 19/77 | 27/77 | 70 % |
| H15 delad worker, 1,8 s | 25/80 | 31/80 | 81 % |
| **H15 egen tråd (`SKJUTBANA_H15_OWN_THREAD=1`), 1,3 s** | **38/81 (47 %), 31 falska** | 43/81, 31 falska | **88 %** |

Replay av bas på samma session (egen tråd-körningen) gav 21/81 och 52 falska.

**Den egna tråden löser köproblemet:**
- 0 av 69 H15-svar var sena, mot 30 av 48 med delad worker.
- Kötiden är 0–19 ms och beräkningen 64–301 ms.
- Träfflatensen live är 0,96 s i median och 1,20 s vid p90, mot 1,23 s / 1,72 s tidigare.
- Kameran stördes inte: 22,0 fps mot 20,9–21,2 i de andra körningarna, och lika många luckor över 80 ms (235 mot 232–249).

**Förbehåll:**
- Det är en live-körning. Den behöver upprepas med omväxlande delad och egen tråd för att visa att skillnaden inte är slump.
- Enligt Codex bör huvudloopens p95/p99 och CPU per tråd också mätas.
- Bakgrunderna är AI-testets projicerade hål, inte fysiska skott.

**Upprepning, omväxlande delad worker och egen tråd (11:13–11:21):**

| Live-körning | Live | Replay H15 samma session | Replay bas samma session | Live / replay |
|---|---|---|---|---|
| H15 delad worker, 1,3 s (2:a körningen) | 20/87, 28 falska, 37 sena | 41/87, 41 falska | 22/87 | 49 % |
| H15 egen tråd, 1,3 s (2:a körningen) | 29/82, 35 falska, 1 sen | 33/82, 46 falska | 19/82 | 88 % |

**Summa live H15, tidsgräns 1,3 s:**

| Variant | Rätt | Falska | Sena H15-svar |
|---|---|---|---|
| Delad worker | 39/164 (24 %) | 65 | 67 av ca 100 |
| **Egen tråd** | **67/163 (41 %)** | 66 | 1 av ca 100 |

- Med egen tråd når live ca 88 % av replay i båda körningarna. Med delad worker når live 49–70 %.
- Antalet falska är oförändrat.
- **Rekommendation:** gör `SKJUTBANA_H15_OWN_THREAD=1` till en del av H15-paketet inför fysisk verifiering. Det kräver ditt godkännande; standard är fortfarande av.
