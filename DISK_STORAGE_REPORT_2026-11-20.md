# Lagring av skjutbanedata – inventering och förslag (2026-11-20)

Inventeringen är skrivskyddad: `disk_inventory.py`. Rådata ligger i `/data/skjutbana/evaluation_runs/disk_inventory_20261120.json`.

## Radering i dag (godkänd)

| Post | Storlek | Kontroll före radering |
|---|---|---|
| `evaluation_runs/h16_shadow_20261120` (felaktig skuggkörning, ersatt av `…b`) | 457 MB | Ingen process eller rapport hänvisade till den |
| 38 st `*_gstack.npy` i `projector_floor_20261120_*` (råa bildstaplar) | 1 206 MB | PNG, meta.json och mätvärden är kvar |
| `source_sessions/20261120T092249Z_7184c2c5` (störd live-körning) | 1 431 MB | Den ingick inte i någon jämförelse |
| **Summa** | **3 092 MB** | |

Ledigt på `/data` efter raderingen är 9,4 GB (98 % använt). En live-session tar ca 1,5 GB, och den största källsessionen tar 7 GB.

Inspelningen vägrar starta under 4 GB ledigt. Det är det som stoppade dagens livetest.

## Inventering (`/data/skjutbana`, ca 268 GB)

| Katalog | Storlek | Filer | Ålder (nyast–äldst) | Används av | Innehåll |
|---|---|---|---|---|---|
| `physical_traces` (11 sessioner) | 132,5 GB | 27 920 | 55–74 dagar | `physical_trace.py`, `accuracy_*`, `physical_label_*`, `*_research.py` | `.npy`-bildrutor och `trace.json` |
| ↳ `session_20260909_194947_D01` | 79,8 GB | | 70–72 dagar | | En enda session står för 30 % av all data |
| `source_sessions` (68 st) | 66,1 GB | 751 | 0–3 dagar | replay, nightly (cron 02:30), alla dagens mätningar | `camera.bin` (MJPEG), `.jsonl`, `.pcm` |
| `physical_ai_lab` | 47,2 GB | 1,9 miljoner | 44–48 dagar | `physical_ai_*`-skript | `fusion_week_01/trials`: 45 GB hyperparametersök (14 600 `model.npz` + 29 GB JSON) |
| ↳ `fusion_week_01/leaderboard.json.tmp` | 0,57 GB | 1 | 44 dagar | ingen (`.tmp`, avbruten skrivning) | |
| `evaluation_runs` (151 st) | 16,4 GB | 8 953 | 0–85 dagar | rapporter och jämförelser | Under 2 dagar: 10,2 GB; över 30 dagar: 4,5 GB |
| `nightly` | 3,5 GB | 265 | 0–3 dagar | nattligt jobb | |
| Övrigt (`backups`, `nightly_evidence`, `ir_test`, `demo_backups` …) | ca 2 GB | | | | |

**Filtyper:**

| Typ | Storlek |
|---|---|
| `.npy` | 133 GB |
| `.bin` | 65 GB |
| `.json` | 43 GB |
| `.npz` | 15 GB |
| `.jsonl` | 10 GB |

## Hur mycket går att komprimera?

Mätt med zstd -3 på stickprov av 50 MB.

| Filtyp | Exempel | Faktor | Kommentar |
|---|---|---|---|
| Rå bildrutor `.npy` (physical_traces, physical_ar15_b) | `frames/post_000036.npy` | **5,7–6×** | Förlustfritt |
| `trace.json` | physical_traces | **24×** | |
| JSON i fusion_week (journal, trials) | `journal.json` | **8,9×** | |
| `candidates.jsonl` (replay-export) | live_paired | **17×** | |
| `camera.bin` (MJPEG) | source_sessions | 1,3× | Redan komprimerat, ingen idé |
| `model.npz` | fusion trials | – | Redan deflate-komprimerat |

## Rekommenderade åtgärder

Inget av detta är gjort. Allt kräver ditt godkännande.

| Prioritet | Åtgärd | Frigör (uppskattat) | Risk | Förutsättning |
|---|---|---|---|---|
| 1 | Packa `physical_traces`-sessioner äldre än 30 dagar till `.tar.zst` (förlustfritt). Packa upp vid behov. | **ca 105 GB** (132 → ca 25 GB) | Låg; läsning kräver uppackning | Verifiera checksumma (sha256) före borttag av originalet; behåll inventeringsrapporten |
| 2 | Packa `physical_ai_lab/fusion_week_01/trials` till `.tar.zst`. `leaderboard.json` ligger kvar okomprimerad som sammanfattning. | **ca 30–35 GB** | Låg | Samma checksumma-kontroll |
| 3 | Ta bort `fusion_week_01/leaderboard.json.tmp` (avbruten skrivning) | 0,57 GB | Mycket låg | Kontrollera att `leaderboard.json` är komplett |
| 4 | Komprimera `candidates.jsonl` och stora `.npy` i `evaluation_runs` äldre än 7 dagar (zstd per fil) | ca 3–5 GB | Låg; analysskript behöver läsa `.zst` | Liten hjälpfunktion i bench-läsarna |
| 5 | Policy för `source_sessions`: behåll alla, men flytta sessioner äldre än 14 dagar som inte ingår i nightly/holdout till arkiv (extern disk eller NAS). `camera.bin` går inte att komprimera. | ca 4–5 GB per vecka med dagens takt | Medel; de är facitdata | Lista över vilka sessioner nightly och holdout använder |
| 6 | Automatisk vakt: varna (och vägra starta nightly/replay-batchar) under 15 GB ledigt | – | – | Liten kontroll i `nightly` och i livetestskriptet |

**Prioritet 1 och 2 räcker ensamma** för att nå målet 10–20 GB ledigt, med stor marginal (> 100 GB). Ingen data förloras, bara packning.

**Förslag till automatisk regel:**
- **Aldrig radera automatiskt.** Packa bara automatiskt, och bara:
  - `.npy`/`.json` i `physical_traces` och `evaluation_runs` äldre än 30 dagar,
  - med verifierad checksumma innan originalet tas bort.
- `source_sessions` och träningsdata (`content/ai`, `nightly_evidence`) rörs aldrig utan beslut.
