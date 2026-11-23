# Färgguide för spel – hålsynlighet (mätt 2026-11-22)

Mätt på riktiga CO2-hål i duken med projektor + kamera (exponering 167). 125 enfärgade bakgrunder (5 nivåer per
kanal). Mått: signal/brus för det svagaste av 5 hål. Data och färgkarta: `/data/skjutbana/evaluation_runs/hole_lab_20261122/matrix/`
(`color_chart.png`). Gäller ytor **bakom och runt mål** (där hål ska upptäckas); HUD och dekor utanför målområden är fria.

## Huvudregel: kameran ser nästan bara GRÖNT

Kamerans ljusnivå ≈ 16·R^2.2 + 115·G^2.2 + 35·B^2.2 + 37 (projektorvärden 0–1 upphöjt till 2,2). Rött bidrar knappt.

| Grön kanal (G) i färgen | Svagaste hålets S/N (median över R,B) | Klass |
|---|---|---|
| 0 | 2.7 | UNDVIK |
| 64 | 3.3 | UNDVIK |
| 128 | 5.6 | OK |
| 192 | 6.3 | OK |
| 255 | 7.2 | OK |

## Använd
- Gula, gröna, cyan, ljusa pastell- och gråtoner, vitt (G ≥ 190).
- Röda/rosa föremål: håll G ≥ 120 (t.ex. läskburken (240,120,110)); ren röd (255,0,0) syns inte.
- Bakgrund nära mål: ljus (gråskala ≥ 128) – mörk bakgrund ger brus som döljer hål (gamma 0,5 på burkspelet).

## Undvik bakom mål
- Svart, mörkt, och färger med G ≤ 64: ren röd, blå, magenta, lila, mörkbrun (trä), mörkgrön.
- Fina mönster, text och skarpa kanter precis där hål kan hamna (ger falska prickar även i stillbild).

## Bäst / sämst uppmätt
Bäst: (0,255,128) 13,8 · (0,255,64) 11,7 · (0,192,64) 10,9 · (0,255,0) 9,7 · (64,255,255) 9,3 · (255,255,192) 9,2.
Sämst: (255,0,64) · (0,0,0) · (0,0,64) · (64,0,0) · (0,0,255) · (192,0,64) · (255,64,0) – alla ≈ 0–0,4.

Obs: absoluta S/N beror på kamerans exponering och projektorns ljus; använd rangordningen och regeln, inte decimalerna.
