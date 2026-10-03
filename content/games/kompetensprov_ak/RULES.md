# Kompetensprov AK – regler och källor

## BAS (officiell Försvarsmakten)

Implementationen följer **H Skjutb AK 2021, Delmoment 12 – Kompetensprov Bas**:

- avstånd: **50 m**
- mål: **1/1-figur / Helfigur 2020**
- start: stående grundställning
- första 3 skott: valfri liggande ställning
- nästa 3 skott: valfri knästående/sittande ställning
- därefter omladdning och stående
- eventuella bättringsskott genomförs efter omladdning och stående
- **de nio bästa träffarna räknas**
- krav: **9 träffar och Pk minst 1,00**
- tiden mäts från startsignal till **sista skottet**

Poängkvot: `poäng / tid`, avrundad till två decimaler.

### Helfigur 2020

Officiell poängtabell:

| Zon | Poäng |
| --- | ---: |
| H | 1 |
| A | 5 |
| X | 2 |
| B | 4 |
| C | 3 |
| D | 3 |

Om en högre poänglinje tydligt bryts ska den högre poängen räknas. Simulatorn approximerar detta med 7,62 mm kuldiameter mot zonmasken.

## BAS Hv

Spelet har också en valbar **BAS Hv**-profil: samma grundprincip, men **30 m** och **Pk minst 1,30**. Detta är beskrivet som den av Rikshemvärnschefen fastställda Hemvärnsvarianten i publicerad facktext.

## 45 sekunder i spelet

Det fasta **45-sekunders skjutfönstret är en simulatorregel för detta spel**, inte en ersättning för BAS-regelns tidtagning. Spelet låter 45 s gå till `ELD UPPHÖR`, men resultatets tid är fortfarande tiden från `ELD` till **sista registrerade skottet**.

## AK4B/C – kulbana

V1 använder Försvarsmaktens ungefärliga banhöjd för **Ak4B / Ptr 10 / rödpunktsikte** vid de två profilavstånden:

- 30 m: **+2 cm**
- 50 m: **+5 cm**

Den fysiska kameraträffen sparas oförändrad. Banhöjden appliceras endast som en separat virtuell scoring-transform.

## Källor

- Försvarsmakten: *Handbok Skjututbildning Automatkarbin 2021 (H Skjutb AK 2021)*
  https://www.forsvarsmakten.se/siteassets/2-om-forsvarsmakten/dokument/handbocker/h-skjutb-ak-2021.pdf
- Försvarsmakten: *Reglemente Skjutning Automatkarbin 2021*
  https://www.forsvarsmakten.se/siteassets/2-om-forsvarsmakten/dokument/reglementen/skjutr-ak-2021.pdf
- Försvarsmakten: *Skjutbok AK* (resultatblad visar BAS 50 m, 9 träff, kvot 1,00 och HF2020-poängtabellen)
  https://www.forsvarsmakten.se/siteassets/2-om-forsvarsmakten/dokument/publikationer/skjutbok-ak-2022.pdf
- Taktisk.se: *45 sekunder* – beskrivning av Kompetensprov BAS Hv, 30 m / Pk 1,3.
  https://www.taktisk.se/2024/06/21/45-sekunder/

## Grafik

`hf2020.png` och zonmasken är en **simulatorreproduktion** byggd efter den officiella HF2020-zonmodellen och poängtabellen. Den är inte en kopia av Försvarsmaktens tryckoriginal. Detta gör zonerna deterministiska och maskbaserade för spelets träffbedömning.


## V2 – pre-start och manuell verifiering

- Spelet startar alltid i ett förberedelseläge; inga skott eller tider räknas före `ELD`.
- `←/→` flyttar hela HF2020-figuren 5 cm per tryck på den fysiska projektionsytan.
  HitRegion och poängzoner använder samma flyttade målrektangel.
- Efter `ELD UPPHÖR` fångas om möjligt en färsk kamerabild och transformeras med
  befintlig homography/scanport till viewporten.
- Manuell efterkontroll sparas separat från detectorns automatiska `ShotRecord`.
  Den manuella kontrollen kan användas för det visade verifierade provresultatet,
  men skriver aldrig över eller maskerar detectorns emitted truth.
- Reviewfiler sparas i första hand under
  `/data/skjutbana/kompetensprov_manual_reviews/`.

Den visuella simulatorfiguren är en egen återgivning baserad på HF2020:s
officiella siluett/zonprincip. Försvarsmaktens egna fotografier/illustrationer
kopieras inte in som spelasset.


## V3 – projektion först i manuell kontroll

Manuell kontroll använder nu som standard **den rena spelprojektionen i exakt samma
viewport-position och storlek som under skjutningen**. Syftet är praktiskt: den
projicerade muspekaren kan placeras direkt över det verkliga kulhålet på den
fysiska tavlan.

- Vänsterklick: lägg till manuellt verifierat kulhål.
- Högerklick: ta bort närmaste manuell markering.
- `R`: rensa manuella markeringar.
- `A`: kopiera AUTO-träffarna till det manuella lagret som utgångspunkt.
- `C`: växla till kamerabild/homography som diagnostik.
- `ENTER`: godkänn manuell verifiering.
- Antalet manuella markeringar är **inte** längre begränsat av hur många skott
  AUTO-detektorn lyckades registrera.
- Om manuell kontroll godkänns får den styra visad träff-/zonpoäng, men
  automatiska ShotRecords sparas fortfarande separat.
- Tiden till sista skott kommer fortsatt från det automatiskt registrerade
  skott-eventet; V3 hittar inte på en ny tidsstämpel för manuellt tillagda hål.

Mus-simulering kräver ingen separat spelkod: GameScene/OverlayScene skickar
vänsterklick under skjutläge genom samma `HitInput`-väg som andra simulerade
träffar.
