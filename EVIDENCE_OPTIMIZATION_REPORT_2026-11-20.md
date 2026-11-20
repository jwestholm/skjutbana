# Evidensoptimering – rapport 2026-11-20

Det här är en svar på arbetsordern från 2026-11-19 om att systematiskt förbättra träffdetekteringen. Allt är mätt offline med projektorn avstängd, och ingen detektorlogik är ändrad som standard. Alla nya beteenden ligger bakom flaggor som är avstängda som standard. Varje körning har en egen katalog under `/data/skjutbana/evaluation_runs/`, och ingen baslinje är överskriven. Inget är incheckat.

**Ärlighetsnotering:** efter 00:09 stod arbetet still till ca 09:45. Ett väntejobb matchade sig självt med `pgrep` och blev aldrig klart. Brus-experimentet i §4.3 startades därför först i morse.

## 1. Sammanfattning

| Fråga | Svar |
|---|---|
| Bästa enskilda evidens | **H11 `ev_hole_morph_snr`** (matchat filter för hålform på V2-förändringskartor): AUC 0,962. Ensam ger den top-1 161/285, lika mycket som hela den inlärda modellen (161). |
| Takhöjd för omrankning | 189/285 skott har *någon* korrekt kandidat. Bästa rankning når 161 (85 %). Resterande 96 skott saknar kandidat helt, alltså är **recall** den största förlusten. |
| Live-poängen `score` | 154/285, AUC 0,921. **Formeln innehåller inte H11.** |
| AI jämfört med regelbaserat val | Båda rätt 149. Bara modellen rätt 12. Bara live-poängen rätt 5. |
| H11-varianter i natt (fler skalor, större provyta) | Högre AUC (0,945 → 0,953 / 0,962) men **0 fler rätta skott**. Behålls inte. |
| H15, tät sen andra åsikt (incheckad, avstängd som standard) | Replay på 8 sessioner: **49,7 % rätt (241/485), 228 falska**, mot baslinjen 39,6 % (192/485), 275 falska. |
| H15 live | Gav ingen effekt live: 35/46 resultat kom efter tidsgränsen på 1,3 s. Orsaken är `np.median` (285 ms). Den är nu ersatt med en exakt snabbmedian (15 ms). **Inte verifierat live.** |

## 2. Inventering av evidensmetoder (bänk: `evidence_bench_20261120a`)

Underlaget är nattens fryst train/holdout-split med hela sessioner, 285 holdout-skott och 42 fält. Tecknet för varje fält väljs på train och mäts på holdout.

### 2.1 Bästa enskilda fält

| Fält | AUC holdout | Top-1 ensam |
|---|---|---|
| ev_hole_morph_snr (H11) | 0,962 | **161** |
| blackhat_value (egentligen DoG, se 2.3) | 0,944 | 149 |
| ev_hole_morph_patch_snr | 0,927 | 136 |
| v2_darkening | 0,924 | 146 |
| score (live) | 0,921 | 154 |
| v2_dog | 0,916 | 148 |
| v2_zscore | 0,911 | 150 |
| pre_shot_change | 0,909 | 143 |
| v2_saliency | 0,909 | 151 |
| v2_absdiff | 0,897 | 144 |
| local_contrast_gain | 0,788 | 101 |
| center_darkening | 0,744 | 105 |

### 2.2 Familjeablation

Modellen tränas om utan respektive bara med varje familj.

- **Utan ev_hole (H11): 161 → 146 (−15).** Det är den enda familjen vars borttagning kostar något.
- Bara ev_hole: 162. Alla andra familjer tillför 0 eller ±1, alltså brus.
- Det finns 159 par med |Spearman| ≥ 0,9. Många fält mäter samma sak.
- Ungefär 20 fält är konstanta eller nästan konstanta (AUC 0,578, top-1 150 = "välj första kandidaten"). Exempel är `shot_accumulator_*`, `v26_vault_*`, `candidate_bank_hits/streak`, `ev_hole_morph_valid` och `v2_rescue_blob`.

### 2.3 Fynd som inte var dokumenterade

1. **`blackhat_value` är inte blackhat.** I V2/FAST/V24/V251 sätts det från `dog_value`. En äkta blackhat finns bara i den gamla `hit_scanner`-vägen.
2. **Live-poängen saknar H11:** `0.16*v2_saliency + 0.23*center_darkening + 0.15*local_contrast_gain + 0.11*blackhat_value + 0.06*min(v2_zscore,25)`. Den tar dessutom med `center_darkening` och `local_contrast_gain`, som hör till de svagaste fälten.
3. `ev_hole_morph_snr` och rå H11-SNR är inte samma storhet, och ingen av dem är en kalibrerad sannolikhet.

### 2.4 AI jämfört med regelbaserat val

| | Antal skott |
|---|---|
| Båda rätt | 149 |
| Bara modellen rätt | 12 (6 av dem i AI-sessionen 20261117T2228) |
| Bara live-poängen rätt | 5 |
| Ingen rätt | 119 (varav 96 saknar korrekt kandidat) |

När modellen vinner beror det på att den har **högre H11** (+2,1 sd) och **lägre zscore/saliency/absdiff** (−1,3 sd). Live-poängen lockas av starka förändringar som inte har hålform.

**Rekommendation, inte implementerad:** som nästa enskilda hypotes, ta in H11 i live-poängen (eller låt H11 avgöra lika lägen). Mätt offline borde det ge ungefär +7 skott av 285.

## 3. H15 – tät sen andra åsikt (commit 014737a, avstängd som standard)

| Session | H15 (rätt/falska) |
|---|---|
| AI 1336 | 46/40 |
| AI 1343 | 43/40 |
| AI 1227 | 37/42 |
| AI 1223 | 46/39 |
| AI 1623 | 36/39 |
| Rörliga mål | 8/16 |
| Fysiskt, nya hål | 16/5 |
| Fysiskt, tejpat | 9/7 |
| **Totalt H15** | **241/485 (49,7 %), 228 falska** |
| **Totalt bas** | 192/485 (39,6 %), 275 falska |

De fysiska sessionerna ligger i paritet med baslinjen, så inga nya falsklarm uppstår där.

**Live 2026-11-19:**
- Replay av samma live-session: bas 30/82, H15 40/82.
- Live gav ingen vinst, eftersom resultaten kom för sent (median 285 ms och upptagen worker).

**Åtgärd, oincheckad i `shot_dense_h15.py`:**
- Exakt Batcher-median på uint8, högst 9 jämnt fördelade bilder (`SKJUTBANA_H15_MAX_FRAMES`).
- Verifierad identisk med `np.median` för n = 1–12.
- 12 bilder tar nu 15 ms, och hela jobbet ca 90 ms (`dense_top` ca 61 ms dominerar).

## 4. Optimering av H11 per metod

Metoden är att ändra en sak i taget och jämföra på identiska kandidater. Underlaget är 6 sessioner (AI 1343/1336/1227/1623, fysiskt 04f1be81, tejpat da06f87a), 362 skott, varav 153 går att rädda.

### 4.1 Mätmått

- **AUC** per kandidat.
- **top-1** över alla skott.
- **cond:** top-1 bland skott som går att rädda.
- **+/−:** parvisa vinster och förluster per skott jämfört med baslinjen.

Codex påpekade att AUC ensam överviktar skott med många negativa kandidater.

### 4.2 Resultat (`h11_variants_20261120`)

| Variant | Flagga | AUC | cond top-1 | mot bas |
|---|---|---|---|---|
| bas | – | 0,945 | 122/153 | – |
| skalor | `SKJUTBANA_H11_SCALES=0.75,1,1.35` | 0,953 | 122/153 | +1/−1 |
| större provyta | `SKJUTBANA_H11_SAMPLE_R=4` | 0,962 | 122/153 | ±0 |

Som jämförelse på samma material: `score` 107/153 och `blackhat_value` 85/153. **Slutsats:** bättre separation men samma vinnare. Inget av detta behålls som förbättring. Flaggorna finns kvar i koden, avstängda som standard och utan påverkan.

### 4.3 Pågår: brusnormalisering (`h11_noise_20261120`)

Två varianter körs:
- `SKJUTBANA_H11_NOISE_RING=81,21`: större ring för brusskattningen.
- `SKJUTBANA_H11_NOISE_FLOOR=1.0`: högre brusgolv.

Frågan är om ljusa, texturerade störningar får uppblåst SNR för att ringen runt dem är ovanligt tyst. **Resultat:**

| Variant | cond top-1 | Jämfört med bas |
|---|---|---|
| ring81 | 119/153 | +1/−4 |
| floor1 | 123/153 | +3/−2 |

Inget av det är en förbättring; skillnaderna är brus. H11:s brusskattning är inte flaskhalsen.

### 4.4 Inte gjort ännu (Codex ordning)

1. Beständighet på fast XY: förändring i kärnan minus ringen över det kausala POST-fönstret, utan rumslig sökning. Detta skiljer sig från H14, som är falsifierad.
2. Fel stratifierade efter registreringskvalitet (respons och förskjutning).
3. Fältaggregat på den registrerade V2-`darkening`-kartan.

## 5. Recall – den större möjligheten

- 96 av 285 holdout-skott har ingen korrekt kandidat, och ingen omrankning kan rädda dem.
- I first-loss-analysen (2026-11-19) saknade ungefär 50 % av skotten en kedjetrack vid rätt position.
- H15 är just ett recall-grepp, och därför ger den +10 procentenheter där omrankning ger ±0.
- Analysen av mörka områden (aldrig kandidat jämfört med bortsorterad) är **inte klar**. Underlaget finns i `firstloss_20261119c/e` och visibility-proben.
- Visibility-proben visade att hål på svart eller mörkt innehåll är osynliga, med 0 rätt på "black" i alla varianter. Se §6.

## 6. Akvisition: projektor, kamera och IR (dokumentation, ej prioriterat)

| Område | Resultat |
|---|---|
| Kamera | 25 fps uppmätt, inte ca 20. MJPG 4K. |
| Exponering | Manuell 167 (16,7 ms = en 60 Hz-period). Brus 0,56 %, bandning 0,25 %. 333 var marginellt bättre (0,37/0,18). 250/120/83 var mycket sämre (upp till 9,7 % bandning). Live H15 med 167: 20/82, med 333: 17/82, ingen mätbar skillnad. **Behåll 167.** Ny override: `SKJUTBANA_CAMERA_EXPOSURE`. |
| IR | Kameran ser 850–940 nm (fjärrkontrollens LED syns som en magenta punkt). NIR-belysning med IR-pass-filter är möjlig utan ny kamera. Se `EXTERNAL_RESEARCH_2026-11-19.md`. Broadlink RM mini 3 är för svag och har fel strålgeometri för att fungera som belysning. |
| Svart golv | `src/engine/display_floor.py`, avstängt som standard. Se 6.1. |

### 6.1 Svart golv

Flaggor:
- `SKJUTBANA_DISPLAY_BLACK_FLOOR=<0-255>`
- `_RGB=r,g,b`
- `_TEXTURE=<amp>`: statisk textur som aldrig ändras mellan bilder.

Golvet läggs på sist före varje flip, i alla spel, utan att ändra spellogik. Kalibrering är undantagen. Kostnaden är 0,4 ms. **Ska testas med projektorn** som separata inspelningar: golv 0/12/24/40, en blågrå RGB-variant och textur. Det rapporteras skilt från offline-resultaten.

## 7. Rättelser av premisser i arbetsordern

| Påstående | Uppmätt |
|---|---|
| Kameran ca 20 fps | 25 fps |
| Max offline ca 65 % | ca 49 % (MAX-utvärderare, 1,0 s), ca 55 % med regionstöd |
| Ca 70 stämplar | 67 (59 luftgevär + 8 AR15) |

## 8. Viktigt om valideringsdata

Sessionerna 20261118T1701–1704 ligger i nattens holdout och syns därför i evidensbänken i §2. De har bara mätts, aldrig använts för att välja varianter. H11-varianterna i §4 valdes på andra sessioner. En sista frusen kontroll kräver ändå nya, orörda sessioner, helst inspelade efter dagens projektortest.

## 9. Förslag till nästa steg, i prioritetsordning

1. **Med projektorn:** H15 live med snabbmedianen. Målet är ≥ 45 % live, så att replay-vinsten blir verklig.
2. **Med projektorn:** svarta golvnivåer, som separat akvisitionshypotes.
3. **Offline:** H11 in i live-poängen eller som avgörare vid lika läge. Det är en enda hypotes, mätt på samma kandidater.
4. **Offline:** recall i mörka områden. Klassa de 96 skott som aldrig blir kandidat.
5. Beständighet på fast XY och registreringsstratifiering (§4.4).
6. Live-verifiering av `SKJUTBANA_BOUNDED_SHUTDOWN=1`: motorn avslutas fortfarande inte på SIGINT.

## 10. Oincheckade filer (`claude/latency`, kräver ditt OK)

| Fil | Innehåll |
|---|---|
| `AUDIT_FINDINGS_2026-11-18.md` | §31-tillägg med slutsiffror för H15 |
| `EXTERNAL_RESEARCH_2026-11-19.md` | Webbresearch |
| `EVIDENCE_OPTIMIZATION_REPORT_2026-11-20.md` | Den här rapporten |
| `automation/bench/evidence_bench.py` + `__main__.py` | Evidensbänken |
| `src/engine/shot_dense_h15.py` | Snabbmedian |
| `src/engine/winner_selection_flags.py` | H11-experimentflaggor, standard oförändrad |
| `src/engine/camera/camera_manager.py` | `SKJUTBANA_CAMERA_EXPOSURE` |
| `src/engine/display_floor.py` + krokar i `app.py`, `shot_critical_v2223.py`, `shot_async_v2224.py` | Svart golv |

`content/ai/*` är lokala artefakter och ska aldrig checkas in.
