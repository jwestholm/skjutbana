# Roadmap

## Nyligen klart

- [x] Multi-frame post-shot capture (8 frames vid ~40ms intervall)
- [x] Tvåstegs AI-pipeline: noise rejection → AI ranking
- [x] Persistence-scoring per hotspot
- [x] Funnel-diagnostik med CSV-rapporter
- [x] GT-matchning med konfigurerbar radie (5/10/15/20px)
- [x] Ökat max hotspots från 50 till 150
- [x] Kamerabuffring fixad (CAP_PROP_BUFFERSIZE=1)
- [x] Pre-shot frame timing fixad (60 frames bak)
- [x] Hole image bank — sparar skotthålsbilder automatiskt vid träning
- [x] Konservativare noise rejection (existed_before threshold 0.8)
- [x] Syntetisk autoträning med funnel-diagnostik och CSV-rapport

## Pågående

- [ ] Verifiera funnel-diagnostik med autoträning
- [ ] Analysera var i pipelinen hål försvinner
- [ ] Syntetisk bildbank-baserad träning (hole images → autotrain)
- [ ] Scen-medvetenhet i AI — feature-vektorn inkluderar scen-typ
- [ ] Readiness-indikator — visa i UI hur redo AI:n är
- [ ] Bilder och video som bakgrund i AI-träningsläge
- [ ] Förbättra AR-kodkalibrering ytterligare om zonbias kvarstår

## Lång sikt

- [ ] AI tar över träffdetektionen (ai_priority / ai_only)
- [ ] Weapon profiles (olika vapen, olika detektionsparametrar)
- [ ] Multi-kamera
- [ ] Replay-system
- [ ] Partikeleffekter vid träff
- [ ] Community-delade AI-hjärnor via GitHub

## Klart

- [x] Grundläggande kamera + ljud + träffdetektion
- [x] LED-integration (Deltaco SH-LS3M)
- [x] Scen-driven LED via menu.json
- [x] Kamerarotation och spegling
- [x] Kandidatvisning med snapshot (överlever scenbyte)
- [x] Pre-shot diff (subtract-baserad)
- [x] Blackhat + pre-shot combined signal
- [x] Known-hole penalty
- [x] Zonloggning (L/M/R)
- [x] AI-modul: runtime, minne, träningsscen
- [x] AI export/import av hjärna
- [x] Homografi-bugg fixad
- [x] Kalibrering separerad (viewport vs kamera)
- [x] 24 ArUco-markörer med RANSAC
- [x] AI-träningsscen med viewport-respekt
- [x] 7 bakgrundslägen i AI-träning (vit → bubblor)
