# CURRENT_STATE.md
# Skjutbana – Current State Snapshot

> Detta dokument beskriver **nuvarande läge i projektet** ur ett praktiskt perspektiv.
> Syftet är att en ny chattinstans, en ny utvecklare, eller framtida vi snabbt ska förstå:
>
> - vad som redan fungerar
> - vad som fungerar delvis
> - vad som just nu är största problemen
> - vad vi tror att nästa steg bör vara
>
> Det här dokumentet är **inte** samma sak som `ARCHITECTURE.md`.
> `ARCHITECTURE.md` beskriver strukturen.
> `CURRENT_STATE.md` beskriver **hur läget faktiskt är just nu**.

---

# 1. Projektets nuvarande mål

Projektet försöker bygga ett visuellt skjutbanesystem där:

- en projektor visar innehåll på en fysisk tavla/yta
- en kamera ser tavlan
- en mikrofon hör skott
- systemet försöker avgöra **var träffen sitter**
- träffen omvandlas till koordinater i:
  - appfönstret
  - viewporten
  - det visade innehållet
  - framtida spel/scenarion

På längre sikt ska projektet också bli en spelmotor/träningsplattform med:
- vapenprofiler
- partikelsystem
- material/penetration
- ljud ut/in
- replay/träning
- scenario-/gameplaylogik

---

# 2. Vad som fungerar just nu

## 2.1 Appen startar och scenflödet fungerar
Den grundläggande appstrukturen fungerar:

- Pygame startar
- scener kan laddas
- meny fungerar
- image/video/game/debug/settings-scener kan nås

Det här är inte huvudproblemet i projektet just nu.

---

## 2.2 Kamera-input fungerar
Kameran fungerar i praktiken som inputkälla:

- kameran kan startas
- frame hämtas löpande
- scanport kan definieras
- kamerabilden används i scanner/debug

Det innebär att den visuella sensordelen finns på plats.

---

## 2.3 Audio-peak fungerar som skotttrigger
Mikrofon/ljudpeak fungerar nu tillräckligt bra för att:

- upptäcka när ett skott sannolikt har skett
- skapa en timing-signal
- låta den visuella analysen undersöka rätt tidsfönster

Det här är en stor vinst jämfört med tidigare läge.

Vi betraktar ljuddelen just nu som:
- en **trigger**
- inte ännu som ett klassificeringssystem för vapentyper

---

## 2.4 Hit-eventkedjan fungerar
Systemet kan redan generera och skicka `HitEvent` genom den globala pipelinen.

Det gäller både:
- mus-hit
- camera-hit

Det betyder att vi redan har:
- en central hitmodell
- subscribers/listeners
- visualisering via visualizer
- debugvy som kan visa senaste hit

Det här är viktigt, för det betyder att eventarkitekturen inte är vårt stora problem.

---

## 2.5 Visuella träffar fungerar globalt
Global hit-visualisering fungerar i praktiken:

- musträffar kan ritas
- kameraträffar kan ritas
- overlay-systemet fungerar
- visuella träffar kan vara en global funktion i motorn

Det betyder att det finns en fungerande väg från:
- detektion / input
- till visualisering i appen

---

## 2.6 Grundläggande transformation fungerar nu mycket bättre än tidigare
Tidigare fick vi extrema och orimliga koordinater (t.ex. långt utanför skärmen), vilket visade att transformen var fel.

Nu har vi kommit till ett bättre läge där:
- träffar i många fall hamnar **inom viewporten**
- ringar ritas i projektionsområdet
- mappingen från kamera till viewport inte längre är totalt trasig

Det här betyder att vi inte längre är på nivån “allt är fel”.
Vi är nu på nivån “grovmappingen fungerar, precisionen gör det inte”.

---

# 3. Vad som fungerar delvis

## 3.1 Hörn-/viewgrid-kalibrering finns som idé och delvis system
Det finns redan en kalibreringstanke i projektet:

- en synk-/kalibreringsbild
- hörn
- viewgrid / viewport

Det finns alltså ett begrepp om att kameran måste synkas mot den projicerade ytan.

Men detta är ännu inte tillräckligt robust för att garantera exakt träffposition över hela ytan.

Så status är:
- **finns**
- **delvis användbart**
- **inte tillräckligt för exakt träffkalibrering**

---

## 3.2 Scanport, viewport och content finns konceptuellt men inte fullt synkat
Vi har nu tydliga begrepp för:

- full kamera
- scanport
- viewport
- content rect

Det är mycket bra.

Men vi har fortfarande inte full garanti att:
- det som faktiskt ritas
- exakt matchar det som transformkedjan tror är content

Det här är en viktig nyckelorsak till att träffar fortfarande känns “slumpiga”.

Så status är:
- **begreppen finns**
- **debug kan visa dem**
- **sanningen i runtime behöver göras säkrare**

---

## 3.3 Scanner-debug fungerar delvis
Debuginformation kan visas, och den har hjälpt oss att förstå flera fel:

- fel koordinatsystem
- träff utanför viewport
- felaktig transform
- kandidater som uppstår på konstiga ställen

Men debugen är ännu inte tillräcklig för att fullt förstå:
- alla kandidater
- varför en viss kandidat valdes
- hur samma kandidat ser ut i alla koordinatsystem

Vi saknar framför allt:
- full kandidatplotting
- överlagring av kandidater i flera coordinate spaces

---

# 4. Det största problemet just nu

## 4.1 Systemet hittar inte träffen robust ens på en enkel vit tavla
Det här är kärnproblemet.

Även när:
- tavlan är enkel
- debug finns
- audiopeak fungerar

så blir träffarna fortfarande ofta:
- utspridda
- feltolkade
- slumpmässiga
- långt från det faktiska hålet

Det betyder:

> Systemet är inte robust nog i själva **kandidaturvalet / håldetektionen**.

Det är alltså inte bara en fråga om “lite tuning”.
Det är en djupare fråga om:
- hur vi observerar kandidater
- hur vi validerar dem
- hur vi väljer rätt kandidat

---

## 4.2 Vi vet ännu inte tillräckligt väl *vad systemet ser*
Det här är kanske det viktigaste diagnosproblemet.

Vi vet ofta:
- vilken träff som valdes
- var ringen ritades

Men vi vet inte tillräckligt tydligt:
- vilka kandidater som fanns samtidigt
- om det riktiga hålet fanns bland kandidaterna
- om rätt kandidat fanns men förlorade
- eller om hålet inte hittades alls

Detta gör att fortsatt algorithm-tuning lätt blir gissning.

---

## 4.3 Projektionen och tavlans verkliga yta är stökiga
Den fysiska verkligheten innehåller redan nu:

- gamla hål
- tejp
- ljusvariationer
- projektorbrus
- skuggor
- struktur i tavlan

Det betyder att systemet inte jobbar på en ren yta.
Det här gör detection svår.

Det är också därför idéen om en **vit session reference-bild** är så viktig.

---

# 5. Vad vi tror om problemet

Här är vår bästa nuvarande tolkning.

## 5.1 Audio-triggern är inte huvudproblemet längre
Ljudet känns inte längre som den stora boven.

Det kan fortfarande förbättras senare, men just nu är det inte där huvudarbetet bör ligga.

---

## 5.2 Mappingen är inte perfekt, men inte heller huvudorsaken till slumpen
Mappingen har haft stora problem tidigare.
De största katastrofelen där verkar nu vara reducerade.

Det betyder att precision fortfarande kan behöva förbättras, men slumpen vi ser just nu känns främst som ett problem i:
- detection
- candidate ranking
- geometri/content-sync
- avsaknad av bra observationsverktyg

---

## 5.3 Scannern väljer sannolikt fel kandidat väldigt ofta
Det här är vår starkaste nuvarande hypotes.

Den riktiga träffen kan:
- finnas bland kandidaterna men förlora
- eller inte komma fram tydligt nog alls

Båda fallen kräver bättre visualisering/debug av kandidater.

---

## 5.4 Vi saknar ett stabilt “före-skott” och “så här såg tavlan ut från början”
Detta är ett stort gap.

Om systemet inte vet hur tavlan såg ut innan sessionen:
- gamla hål
- tejp
- permanenta märken

så är allt detta bara “möjliga kandidater”.

Det är därför session reference (vit bild direkt efter sync) känns som ett mycket viktigt nästa steg.

---

# 6. Vad vi kommit fram till metodmässigt

Det här är centralt.

Vi vill **inte** fortsätta med:
- fler slumpmässiga thresholdändringar
- fler halvblinda tweaks i scannerlogiken

Istället vill vi gå över till en tydligare metod:

## 6.1 Först: instrumentering
Systemet måste visa:
- vad det ser
- vilka kandidater som finns
- hur kandidater transformeras

## 6.2 Sedan: stabil geometri
Vi måste säkerställa:
- scanport
- viewport
- content
- verklig fysisk tavla

## 6.3 Sedan: session reference
Systemet måste veta:
- hur tavlan såg ut innan skott i just denna session

## 6.4 Först därefter: förbättrad detection
När vi har ovanstående kan vi förbättra detektion på ett intelligent sätt, istället för att gissa.

---

# 7. Nästa konkreta prioritering

Det här är den viktigaste delen av dokumentet.

## TOP 1 – Kandidatplotting
Vi måste bygga ett debugläge där systemet visar:

- alla kandidater
- filtrerade kandidater
- top candidates
- vald kandidat

Gärna med olika färger, exempel:

- blå = rå blob
- gul = klarar threshold
- grön = bra kandidat
- röd = vald träff

Detta behöver ritas i minst:
- scanport
- gärna också viewport/content

Utan detta famlar vi fortfarande lite i mörkret.

---

## TOP 2 – Verifierad transformkedja
Vi måste kunna se att samma punkt överlagras i:

- verklig kamera / scanport
- viewport
- content
- game/app

Målet är att samma fysiska punkt ska bli samma logiska punkt genom hela kedjan.

Det här bör testas med:
- gridbild
- klickdebug
- manuella kontrollpunkter

---

## TOP 3 – Session-start med syncbild + vit referens
När vi startar nytt:
- spel
- tavla
- bild
- video

bör vi göra:

1. syncbild
2. hörnsynk / viewgrid
3. vit bild
4. capture av 3–5 frames
5. skapa session reference

Den referensen bör användas som baseline för:
- gamla hål
- tejp
- bakgrundsstruktur
- projektorgradienter

---

# 8. Större roadmap-spår som redan är definierade

Det här är sådant vi redan vill ha med på längre sikt.

## 8.1 Weapon profiles
Framtida stöd för olika vapen:

- luftgevär
- CO2-pistol
- CO2-gevär
- senare andra typer

Varje vapenprofil ska på sikt kunna bära:
- audio-signatur
- förväntad hålstorlek
- damage-profile
- spread-model
- particle-profile
- penetration power

Viktigt:
spelmotorn ska kunna veta **vilket vapen som sköt**.

---

## 8.2 Audio output i motorn
Motorn ska senare kunna spela:
- TTS
- countdown
- berättarröst
- gameplayljud
- videoljud
- musik

Detta ska integreras så att:
- mic/skottanalys vet vad motorn själv spelar upp
- outputljud behandlas som brus / ignoreras så långt möjligt
- speech input kommer senare

---

## 8.3 Partikelsystem
Vi vill senare ha träffeffekter som:
- damm
- sand
- blod
- splitter
- granris
- träflisor
- metallgnistor

Detta ska bero på:
- material
- vapen
- gameplaytolkning

---

## 8.4 Material, penetration och armor
Framtida spelmotorlogik ska stödja:

- material per objekt
- penetration genom flera objekt
- restenergi
- skyddsvästar / armor
- olika skydd mot pistol / AK / hagel etc

Detta är gameplaylagret ovanpå den fysiska träffpunkten.

---

## 8.5 Multi-camera / replay
Det finns en framtida idé om:
- en extra kamera som filmar skytten
- replay
- slow motion
- analys av agerande efteråt

Detta är inte prioritet nu, men en tydlig framtidsgren.

---

# 9. Praktiska beslut vi redan tagit

## 9.1 Vi ska inte försöka lösa allt med hotspots just nu
Hotspots/target zones är en bra framtidsidé, men inte nästa steg.

Anledning:
- systemet hittar inte ens stabilt på en vit tavla ännu
- då hjälper det inte att vikta kandidater mot målzoner förrän basdetection fungerar bättre

---

## 9.2 Vi ska inte fortsätta göra många fler scanner-tweaks blint
Vi behöver först bättre observability.

---

## 9.3 Vi ska montera kameran fast
Takmontering med stabil fysisk setup ses som ett viktigt nästa steg.
Stabil hårdvara först, bättre software sedan.

---

# 10. Vad vi misstänker kommer ge störst lyft

Om vi ska vara brutalt ärliga:
de tre största sakerna som sannolikt kommer ge mest effekt per arbetstimme är:

## 10.1 Kandidatplotting
Det här kommer sannolikt omedelbart visa om:
- hålet hittas
- hålet väljs bort
- hålet inte hittas alls

## 10.2 Vit session reference
Det här kommer sannolikt minska:
- gamla hål
- tejp
- permanenta strukturer
som falska kandidater

## 10.3 Bättre content/geometry truth
Vi behöver kunna lita på att:
- det vi ritar
- är samma content_rect som systemet mappar till

---

# 11. Risker / reality check

Det här är också viktigt att skriva ner.

## 11.1 Det finns en chans att vanlig frame-difference inte räcker bra nog
Det är möjligt att vi längre fram kommer fram till att:
- enkel differens mellan pre/post inte är tillräckligt robust
- projektionen stör för mycket
- hålen är för subtila i vissa lägen

Om det händer behöver vi kanske gå mot:
- bättre edge/kantanalys
- annan hole detection-metod
- eller mer explicit referensbaserad modell

Men vi är inte där än.
Vi ska först göra systemet observerbart.

---

## 11.2 Vi får inte bygga för smart för tidigt
Det vore lätt att kasta in:
- hotspots
- ML
- target prediction
- vapenklassificering
- content priors

Men om grunddetection fortfarande är blind så hjälper det inte.

Så just nu gäller:
**förstå först, smartness sen**.

---

# 12. Current summary in one paragraph

Nuvarande läge är att systemet har en fungerande appstruktur, fungerande kamera- och audioinput, fungerande hit-eventkedja, global hitvisualisering och en grundläggande mapping från kamera till viewport/content. Det som fortfarande inte fungerar tillräckligt bra är den visuella träffdetektionen: systemet väljer ofta fel kandidat och känns därför slumpmässigt även på enkel tavla. Vår nuvarande arbetslinje är att sluta gissa i scannerlogiken och istället bygga bättre insyn: kandidatplotting, full transformdebug och session-start med syncbild + vit referensbild. Det är den tydligaste och mest sannolikt framgångsrika vägen framåt.

---

# 13. Immediate next-session plan

När vi fortsätter nästa gång bör vi göra detta i ordning:

1. Bygga kandidatplotting
2. Visa kandidater i flera coordinate spaces
3. Verifiera transformkedjan med rutnät / klickdebug
4. Bygga session-start:
   - syncbild
   - vit referens
5. Först därefter återvända till själva detektionen

