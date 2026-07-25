# Skjutbana – del 1: isolering av skott och tidsdata

## Bas

Filerna är baserade på `dev` vid commit:

`7a3e87df31097980de8fa86ac3aa8ea9f3661b04`

Kontrollera före installation:

```powershell
git switch dev
git pull
git rev-parse HEAD
```

Om HEAD inte är committen ovan, granska `git diff` extra noggrant innan incheckning.

## Filer som ersätts

Kopiera från paketet till motsvarande sökvägar i repositoryt:

```text
src/engine/ai/runtime.py
src/engine/ai/bootstrap.py
```

Paketet har redan samma katalogstruktur och kan därför packas upp i repositoryts rot.

## Vad del 1 gör

1. Varje ljudhändelse får en separat `AIShotContext`, identifierad av `shot_id`.
2. AI:n får endast använda kandidater märkta med samma `shot_id` som träffen som emitteras.
3. AI-runtime synkroniseras med det aktuella skottet innan emission sker inuti `HitScanner.update()`.
4. En ny kandidatbild ersätter föregående kandidatlista även när den nya listan är tom. Gamla kandidater får alltså inte ligga kvar.
5. En kamerabild kopplas till högst ett skott, även när tidsfönster överlappar.
6. Post-shot-persistens räknas bara från unika kameratidsstämplar. Samma kamerabild kan inte räknas flera gånger.
7. Skott som timeoutar speglas som `missed` i AI-runtime.
8. AI-lagret är fortsatt fail-open: om AI-koden får fel används den ordinarie detektorns träff och programmet ska fortsätta.
9. Gamla avslutade shot-contexts rensas för att undvika att flera uppsättningar 4K-bilder blir kvar i minnet.

Del 1 ändrar **inte** tröskelvärden, morfologi, kandidatdetektion eller AI-modellens scoring. Det är avsiktligt så att testet endast mäter state- och tidskopplingen.

## Installation

Från repositoryts rot:

```powershell
git switch dev
git status
```

Se till att dina nuvarande ändringar är sparade. Packa därefter upp paketet i repositoryts rot och tillåt överskrivning av de två filerna.

Kontrollera:

```powershell
python -m py_compile src/engine/ai/runtime.py src/engine/ai/bootstrap.py
git diff --check
git diff -- src/engine/ai/runtime.py src/engine/ai/bootstrap.py
```

## Test 1 – start och normal funktion

1. Starta programmet på samma sätt som vanligt.
2. Öppna AI-inställningar och den syntetiska träningsscenen.
3. Verifiera att inga exceptions visas vid start, scenbyte eller avstängning.
4. Verifiera att kandidat-overlay och vanlig träffdetektering fortfarande fungerar som före ändringen.

Förväntat resultat: ingen regression. Standardläget `train_only` ska fortfarande inte ändra detektorns valda träff.

## Test 2 – gammal kandidat får inte återanvändas

Detta kan först göras med syntetisk scen, inspelat material eller annan säker testsekvens.

1. Skapa skott-/ljudhändelse A där det finns tydliga kandidater.
2. Skapa därefter skott-/ljudhändelse B utan ett nytt synligt hål eller med en bild där detektorn ger noll kandidater.
3. Titta på konsolloggen.

Förväntad logik:

```text
[AI SHOT] created shot_id=A ...
[AI SHOT] finished shot_id=A ... candidates=N ...
[AI SHOT] created shot_id=B ...
[AI SHOT] finished shot_id=B ... candidates=0 ...
```

Skott B får inte använda eller markera kandidatens koordinat från skott A. I ett AI-läge som tillåter override ska orsaken internt bli `no_candidates_for_shot`, och originaldetektorn får i stället bestämma.

## Test 3 – två snabba ljudhändelser

1. Generera två separata testtriggers relativt tätt.
2. Kontrollera att två olika `shot_id` skapas.
3. Kontrollera att kandidater för den senare händelsen inte ändrar kandidatantalet som loggas för den tidigare avslutade händelsen.

Förväntat resultat: varje skott skapas och avslutas separat. Ingen kandidatlista delas mellan dem.

## Test 4 – unik post-shot-bild

Kör programmet med en huvudloop som är snabbare än kamerans bildfrekvens, vilket normalt redan är fallet.

Förväntat resultat: `post_frames` i avslutningsloggen ska motsvara nya kamerabilder, inte antalet varv i huvudloopen. Det ska aldrig bli åtta persistensbilder bara genom att samma kamerabild läses åtta gånger.

## Test 5 – timeout/miss

1. Skapa en ljudtrigger utan att detektorn hittar en stabil träff.
2. Vänta tills HitScanner loggar `MISS`.

Förväntat resultat: AI-runtime ska också logga att samma `shot_id` avslutades med `state=missed`. Nästa skott ska få en ren kontext.

## Vad du bör skicka tillbaka efter test

Spara gärna konsolloggen från ungefär fem sekunder före första testtriggern till fem sekunder efter sista. Det mest användbara är rader med:

```text
[AI SHOT]
[SHOT #]
[PRE-SHOT]
HIT
MISS
```

Notera också:

- bakgrundstyp: vit, grå, svart, checker, film eller spel,
- om kandidaten syntes i overlay,
- om träffen blev rätt/fel,
- om fel träff låg på ett äldre hål,
- ungefärlig tid mellan två triggers.

## Incheckning efter godkänt test

```powershell
git add src/engine/ai/runtime.py src/engine/ai/bootstrap.py
git commit -m "Isolate AI candidates and frames per shot"
```
