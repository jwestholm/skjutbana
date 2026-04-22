# Hårdvara

## Setup

- **Dator** — Ubuntu-maskin (hyfsat prestanda)
- **Kamera** — USB, 4K (3840x2160), 30fps, MJPG
- **Mikrofon** — inbyggd i kameran, ca 50cm från tavlan
- **Projektor** — projicerar på vägg/tavla genom plexiglas
- **LED** — Deltaco SH-LS3M (Tuya-protokoll, version 3.3)
- **Tavla** — plexiglas framför projektionsyta, LED-list monterad i tavlan

## Fysisk layout

```
Skytt (6m avstånd)
    │
    │  ← kulbana
    │
    ▼
┌──────────────────┐
│   Plexiglas       │ ← kulor träffar här
│   ┌────────────┐  │
│   │ Projektions-│  │
│   │ yta         │  │
│   └────────────┘  │
│   LED-list        │
└──────────────────┘
    ▲
    │
  Kamera (USB, 4K)
  Mikrofon (i kameran)
  Projektor (bakom/ovanför)
```

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
- Kan ge ljusa hål genom plexiglaset om påslagen
