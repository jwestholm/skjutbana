# Nuvarande status

## Fungerar

- **Kamera** — 4K USB, 30fps, rotation/spegling konfigurerbar
- **Ljud** — ffmpeg-baserad peak-detektor, PulseAudio/ALSA
- **Träffdetektion** — pre-shot diff + blackhat, tracking, emission
- **Kalibrering** — ArUco-baserad homografi med 24 markörer, RANSAC
- **Viewport** — separat justering av skjutgränser/rityta
- **Koordinattransform** — kamera → screen → viewport → content via homografi
- **Visuella träffar** — fade/persistent, kandidatvisning med snapshot
- **LED** — Deltaco SH-LS3M via Tuya, scen-driven
- **AI-modul** — runtime, minne, träningsscen, export/import av hjärna
- **Spel** — Shoot/Don't Shoot, skjutbana med helfigur
- **Bilder/video** — stillbilder och videoklipp med träffdetektion

## Senaste förbättringar

- Homografi-bugg fixad (`_prefers_homography` accepterar nu rätt metodnamn)
- `inverse_homography` sparas explicit vid kalibrering
- Pre-shot diff med `subtract` (inte `absdiff`) — hittar hål bättre
- Blackhat återinförd i combined signal
- Known-hole penalty — nya hål prioriteras över gamla
- Zonloggning — raw blobs och kept candidates per zon (L/M/R)
- AI-träningsscen med viewport-respekt och 7 bakgrundslägen
- Kalibrering separerad: viewport-justering vs kamerakalibrering
- 24 ArUco-markörer med RANSAC och per-markör error

## Kända begränsningar

- Svart bakgrund i mörkt rum ger inga träffar (kräver extern belysning)
- Kalibreringen kan fortfarande ha zonbias i hörnen
- AI:n är i train_only-läge — påverkar inte träffar ännu
- Detektorn hittar ibland kanter/sprickor i plexiglaset som kandidater
