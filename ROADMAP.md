# Roadmap

## Pågående

- [ ] Verifiera kalibrering med 24 markörer — mål: reprojection error < 1.0px
- [ ] Testa AI-träning med viewport-respekt
- [ ] Verifiera zonbias efter ny kalibrering

## Kort sikt

- [ ] Förbättra hit_scanner pre-shot diff — lösa projektorflicker/skakning
- [ ] AI: lokal diff vid klick — bättre träningsdata när kandidat saknas
- [ ] Rensa oanvända settings-nycklar i content/ai/settings.json
- [ ] Verifiera AI-logik med riktig data (normalisering, viktning, neg-sampling)
- [ ] Prestanda-verifiering av AI-scoring på target-hårdvara

## Medel sikt

- [ ] AI-overlay i spelscener — visa AI:ns förslag under vanligt spel
- [ ] Auto-learn i spelläge — AI lär av egna beslut (kräver feedback-loop-skydd)
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
