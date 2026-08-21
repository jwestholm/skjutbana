# Skjutbana

Digital skjutbana med projektor, kamera och ljuddetektion. Systemet projicerar bilder, video och spel på en vägg/tavla och detekterar träffar via kamera + mikrofon.

## Funktioner

- **Bilder och tavlor** — stillbilder projiceras, skott detekteras och markeras
- **Video** — videoklipp spelas upp, skott registreras i realtid
- **Spel** — interaktiva scenarion (Shoot/Don't Shoot, skjutbana med helfigur)
- **AI-träning** — träna AI-modellen att förbättra träffdetektering
- **Kamerakalibrering** — ArUco-baserad homografi med 24 markörer
- **LED-feedback** — Deltaco SH-LS3M styrs via Tuya-protokoll
- **Ljuddetektion** — ffmpeg-baserad peak-detektor triggar bildanalys
- **Automation/API** — lokal TCP/JSON-styrning och event-stream för externa test- och AI-skript

## Snabbstart

```bash
python main.py
```

Kräver: Python 3.10+, pygame, opencv-python, numpy, ffmpeg i PATH.

## Kalibrering (första gången)

1. **Justera skjutgränser** — Inställningar → Justera skjutgränser. Flytta/skala den gröna ramen till projektionsytan.
2. **Kalibrera kamera** — Kamera → Kalibrera viewport via kamera. ENTER startar, vänta tills markörer hittas, sparas automatiskt.
3. **Justera ljud** — Inställningar → Ljud-peak. Justera tröskeln så skott triggar men inte bakgrundsljud.

## Projektstruktur

```
main.py                  — Startpunkt
config.py                — Skärm, FPS, sökvägar
src/engine/              — Spelmotor
  app.py                 — Huvudloop
  camera/                — Kamera, hit_scanner, kalibrering
  audio/                 — Ljuddetektion
  input/                 — Träffinput och koordinattransform
  visual/                — Visualisering (träffmarkeringar, overlays)
  scenes/                — Alla scener (meny, spel, inställningar, AI)
  ai/                    — AI-modul (runtime, minne, träning)
  communication/         — TCP/JSON server + klientmotor
  events/                — Intern EventBus
  output/                — LED-styrning
automation/              — Fristående automations- och testscripts
content/                 — Menydata, inställningar, AI-minne
assets/                  — Bilder, video, spel, tavlor
```


## Automation

Starta spelet normalt och kör externa scripts från repositoryts rot. Exempel:

```bash
python3 main.py
# annan terminal:
python3 -m automation.set_window_pos
python3 -m automation.autostart_ai_training 1
```

Automation använder den gemensamma TCP/JSON-motorn i `src/engine/communication/`. För protokoll, threading-regler, commands och events: se **`NETWORK_AUTOMATION.md`**.

---

## Repeated automated AI training

With Skjutbana already running, complete F2 AI-training sessions can be run
repeatedly from another terminal:

```bash
python3 -m automation.ai_training_loop 1 100
```

The first argument is the background (1-8 or name), the second is the number
of complete runs. Results are saved under `content/ai/automation_runs/` in
machine-readable JSON/JSONL plus CSV. See `NETWORK_AUTOMATION.md` for protocol,
events and result schema.
