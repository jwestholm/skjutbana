# Hårdvara

## Setup

- **Dator** — Ubuntu-maskin (hyfsat prestanda)
- **Kamera** — USB, 4K (3840x2160), 30fps, MJPG
- **Mikrofon** — inbyggd i kameran, ca 50cm från tavlan
- **Projektor** — projicerar på vägg/tavla
- **LED** — Deltaco SH-LS3M (Tuya-protokoll, version 3.3)
- **Tavla** — flerskiktad konstruktion med LED-bakljus (se lager nedan)

## Fysisk layout

```
Skytt (6m avstånd)
    │
    │  ← kulbana
    │
    ▼
┌──────────────────────────────────┐
│  Vägg                            │
│  ├─ 2mm plåt                     │
│  ├─ Ljudisolerskivor (bil) 4cm   │
│  ├─ Tomrum ~1cm med LED-list     │
│  ├─ Pappkartong (flyttlåda)      │
│  ├─ Vita A4-papper               │ ← projektionsyta
│  └─ Lappningar / markeringar     │
└──────────────────────────────────┘
    ▲
    │
  Kamera (USB, 4K)
  Mikrofon (i kameran)
  Projektor (bakom/ovanför)
```

### Tavlans lager (utifrån → in mot vägg)

1. Vita A4-papper + lappningar — projektionsyta, det kameran ser
2. Pappkartong (flyttlåda) — bärande skikt, kulor fastnar/penetrerar
3. Tomrum ~1cm med LED-list — bakljus, kan ge ljusa hål om påslagen
4. Ljudisolerskivor (bil) 4cm — dämpar ljud, fångar kulor
5. 2mm plåt — skyddar väggen
6. Vägg

## Tidslinje vid skott (6m avstånd)

1. t=0: Avfyrning
2. t=30-60ms: Kulan träffar tavlan (beroende på vapen)
3. t=~1.5ms efter träff: Mikrofon hör smällen (50cm / 340 m/s)
4. t=33-66ms efter träff: Kameran fångar frame med hålet (30fps)

Audio peak ≈ kulan har redan träffat. Pre-shot-bakgrund måste vara från innan peak_ts.

## Vapen

- Luftgevär (5mm BB): ~100-120 m/s
- CO2 AR-15 (stålkulor): ~150-200 m/s
- CO2 pistol: ~100-130 m/s
- Max 5 skott/sekund (200ms mellanrum)

## LED

- Styrs via lokal nätverksanslutning (Tuya)
- Version låst till 3.3 (SH-LS3M)
- Används för feedback (träffblink) och belysning
- Kan ge ljusa hål om påslagen (LED bakom tavlan)
