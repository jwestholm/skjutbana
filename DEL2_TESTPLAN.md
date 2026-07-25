# Del 2 – kandidatens egen bilddata

## Bas

Ändringen är gjord ovanpå commit:

```text
6c3819c9d4c091bfa70138cccd1d6d9e72a0c9f3
Help from ChatGPT part 1
```

## Fil som ska ersättas

```text
src/engine/ai/runtime.py
```

Det är den enda produktionsfilen som ändras i del 2.

## Vad som är gjort

- Varje AI-kandidat får ett internt `evidence_id`.
- Kandidatens kameratidsstämpel används för att hitta den bildruta som faktiskt skapade kandidaten.
- En liten patch sparas från kandidatens egen källbild.
- Motsvarande patch från pre-shot-bilden sparas när bildstorleken stämmer.
- En diff-patch mellan pre-shot och kandidatens källbild sparas.
- Patcharna lagras i skottets `AIShotContext`, inte som stora bildmatriser i kandidatens publika dictionary.
- `patch_mean`, `patch_std` och `edge_strength` beräknas från kandidatens egen källpatch.
- `existed_before_shot()` använder kandidatens pre-shot- och källpatch.
- Persistens jämför mot rätt pre-shot-patch och räknar bara kompatibla post-shot-bilder.
- En riktig skottkandidat utan bildbevis får neutrala bildfeatures. Den får aldrig analyseras mot en senare global kamerabild.
- Avslutningsloggen visar nu både kandidatantal och antal kandidatbevis:

```text
[AI SHOT] finished shot_id=... state=... candidates=N evidence=N post_frames=N
```

## Installation

Kopiera filen till:

```text
<repository>\src\engine\ai\runtime.py
```

Kontrollera därefter:

```powershell
python -m py_compile src/engine/ai/runtime.py
git diff --check
git status
```

## Testläge

Behåll AI-läget i `train_only` under verifieringen. Del 2 ska inte börja flytta verkliga träffar.

## Test 1 – start och grundfunktion

1. Starta programmet normalt.
2. Öppna kamera- och AI-vyerna du brukar använda.
3. Kontrollera att inga exceptions kommer från `runtime.py`.
4. Kontrollera att kandidat-overlay och automatisk testning fortfarande går att starta.

## Test 2 – ett vanligt skott

1. Skjut eller simulera ett tydligt skott.
2. Vänta tills skottet avslutas.
3. Leta efter avslutningsraden.

Förväntat exempel:

```text
[AI SHOT] finished shot_id=12 state=matched candidates=10 evidence=10 post_frames=8
```

`evidence` bör normalt vara samma som `candidates`. Ett enstaka lägre värde är inte automatiskt ett fel, men återkommande `evidence=0` när det finns kandidater ska rapporteras.

## Test 3 – rörlig bild

1. Kör video, animation eller ett spel med tydlig rörelse.
2. Skapa ett skott eller en testtrigger.
3. Låt bilden fortsätta röra sig medan kandidaterna rankas.
4. Kontrollera att programmet inte kraschar och att avslutningsraden innehåller kandidatbevis.

Syftet är att kontrollera att AI:n inte längre läser kandidatens visuella egenskaper från den senare videobild som råkar visas när rankningen sker.

## Test 4 – automatisk AI-testning

1. Starta den automatiska testning som du använde efter del 1.
2. Kör minst 10 syntetiska försök.
3. Kontrollera att testningen slutförs utan exception.
4. Kontrollera att topplistan fortfarande visas och att kandidater kan läras in.

Resultatnivåerna behöver inte förbättras i denna del. Kandidatgenerator och rankingmodell är fortfarande oförändrade.

## Test 5 – två skott

1. Gör två separata skott.
2. Kontrollera att de får olika `shot_id`.
3. Kontrollera att båda får egna `candidates` och `evidence`.
4. Kontrollera att skott 2 inte använder hålet från skott 1 på grund av gammalt AI-state.

## Godkänt resultat

Del 2 är godkänd när:

- programmet startar utan nya fel,
- vanliga skott fortfarande fungerar,
- automatisk testning fungerar,
- `evidence` normalt följer kandidatantalet,
- rörlig video inte orsakar exceptions,
- två skott hålls separerade,
- inga riktiga skottkandidater analyseras mot en senare global bild.

## Det som inte är ändrat

Del 2 ändrar inte:

- kandidatgeneratorns recall,
- detectortrösklar,
- blackhat/whitehat,
- known-hole-avdrag,
- track-association,
- AI-minnets modell eller vikter,
- vilket hål som blir top 1.

Att ett gammalt hål fortfarande kan bli top 1 är därför möjligt. Det angrips i kommande delar när kandidatgenerering, known-hole-logik och ranking görs om.
