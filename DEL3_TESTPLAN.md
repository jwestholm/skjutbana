# Del 3 – rörelse- och video-fokuserad kandidatförbättring

## Filer i paketet
- `src/engine/ai/runtime.py`

## Vad som ändrats
Den här del 3-versionen ändrar **inte** `hit_scanner.py` direkt. I stället lägger den ett nytt AI-sidelager i `AIRuntime.observe_scanner()` som **kompletterar** (`supplement`) de kandidater som `HitScanner` redan producerar.

Det är avsiktligt: vi ville höja träffsäkerheten på rörliga bakgrunder utan att samtidigt riskera att slå sönder den befintliga detektorn på vita/statiska tavlor.

### Nya huvudidéer
1. Om `HitScanner` redan levererar många kandidater (t.ex. vit tavla med ~150–200 kandidater) görs **ingen** komplettering.
2. Om kandidaten är gles/svag (t.ex. rörlig checker/video) genererar AI-lagret extra hotspots direkt från kandidatbilden.
3. De extra hotspotsen tas fram med:
   - multiskalig **blackhat** och **tophat**
   - lokal mörk/ljus-analys mot medianfilter
   - lokal **temporal change** som försöker bevara små permanenta förändringar men tona ned bred scenrörelse
4. Alla extra kandidater slås ihop med ordinarie kandidater, dedupliceras och skickas sedan vidare i samma AI-pipeline som tidigare.

## Nya runtime-inställningar
Dessa ligger i `DEFAULT_SETTINGS` och kräver ingen manuell ändring för första testet:

- `supplement_candidates_enabled = True`
- `supplement_min_candidates = 120`
- `supplement_peak_percentile = 99.6`

## Förväntat resultat
### Vit tavla
Bör ligga ungefär kvar där ni redan ligger. Del 3 ska helst **inte** försämra den.

### Rörlig bakgrund / checker_anim
Målet är att höja:
- `Rätt hål bland raw hotspots`
- `Top-1 rätt`
- `AI valde rätt`

Den viktigaste förbättringen att titta efter är att `Rätt hål bland raw hotspots` går upp tydligt från ~19%.

## Installera
Skriv över:

```text
<skjutbana>\src\engine\ai\runtime.py
```

Kontrollera gärna efteråt:

```powershell
python -m py_compile src/engine/ai/runtime.py
```

## Rekommenderat testupplägg
1. Starta om programmet helt.
2. Kör `F1` på **vit tavla** (100 rundor).
3. Notera slutresultatet.
4. Kör `F1` på **checker_anim / rörlig ruta** (100 rundor).
5. Notera slutresultatet.
6. Jämför mot tidigare siffror.

## Titta särskilt på
- `Hittade hålet`
- `Top-1 rätt`
- `Top-3 rätt`
- `Missade`
- `Rätt hål bland raw hotspots`
- `Överlevde filter`
- `AI valde rätt`
- `Medel / Min / Max` kandidater

## Godkänt utfall för del 3
Minst ett av följande bör förbättras tydligt på rörlig bakgrund:
- raw hotspot recall
- top-1
- AI guess correct

Det viktigaste är fortfarande **raw hotspot recall**. Om den går upp markant, då har del 3 gjort rätt sak och nästa steg kan fokusera mer på ranking och gamla hål.
