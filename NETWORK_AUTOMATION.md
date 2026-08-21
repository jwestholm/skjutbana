# Nätverk, automation och EventBus

> **Canonical reference för framtida AI-assistenter.** Läs denna fil innan du ändrar extern styrning, TCP/JSON-protokoll, EventBus, automation-scripts eller automationkopplingen i `app.py`.

## Syfte

Skjutbana har ett litet generellt automationlager som gör att fristående Python-program kan styra ett redan startat spel och ta emot strukturerade status-events. Målet är att samma grundmotor ska kunna användas för testautomation, AI-träning, mus/tangentbordssimulering och senare en självstyrande utvecklar-AI.

Nuvarande implementation är medvetet liten och lokal. Den ska utökas stegvis utan att koppla nätverkskod direkt till spel-/AI-logik.

## Canonical filer

| Fil | Ansvar |
|---|---|
| `src/engine/communication/communication_server.py` | TCP-server, JSON parsing, command queue, event subscribers, event broadcast |
| `src/engine/communication/tcp_network_handler.py` | Extern klientmotor: `TcpNetworkHandler`, `send_command`, `EventListener` |
| `src/engine/events/event_bus.py` | Process-lokal, thread-safe-ish publish/subscribe för strukturerade events |
| `src/engine/app.py` | Tar commands från kön och utför Pygame/game-operationer i huvudtråden |
| `src/engine/scenes/automation_ai_training.py` | Tunn `AITrainingScene`-subklass som emitterar automation-events |
| `automation/set_window_pos.py` | Enkelt command-test för `setWindowPos` |
| `automation/autostart_ai_training.py` | Event-driven end-to-end automation för F2 AI-träning |

## Absoluta arkitekturregler

1. **Nätverkstrådar får aldrig anropa Pygame eller scenlogik direkt.**
2. Inkommande commands går: socket -> JSON -> `AutomationCommand` -> thread-safe queue -> `App.run()` -> Pygame main thread.
3. Externa scripts ska använda `send_command()` / `EventListener`; skriv inte egen socket/JSON-kod i varje script.
4. Asynkrona operationer ska signalera lifecycle-events. Använd inte fasta `sleep()` som synkroniseringsmekanism.
5. `EventBus` är generell engine-infrastruktur. Undvik AI-specifik speciallogik i själva bussen.
6. Automation får inte i onödan ändra normal spelkod. Använd tunna wrappers/subklasser där det går.
7. Nya commands/events ska dokumenteras här i samma commit som implementationen.
8. Porten är localhost-only och saknar auth/TLS. Ändra inte bind-address till `0.0.0.0` utan en separat säkerhetsdesign.

## Transport

- TCP
- Default host: `127.0.0.1`
- Default port: `8765`
- Encoding: UTF-8
- Framing: **newline-delimited JSON / JSONL** — ett komplett JSON-objekt per rad
- Max inkommande meddelande: 64 KiB (`MAX_MESSAGE_BYTES`)

En rå transportström kan alltså se ut så här:

```text
{"type":"command","command":"setWindowPos","args":[2130,50]}\n
{"type":"response","command":"setWindowPos","success":true,"data":{"x":2130,"y":50}}\n
```

## Två kommunikationsmönster

### 1. Command / response

Normala automation-kommandon skickas med:

```python
from src.engine.communication.tcp_network_handler import send_command

response = send_command("setWindowPos", [2130, 50])
```

Klienten öppnar en TCP-anslutning, skickar ett command och väntar på ett response. Serversidan väntar högst 5 sekunder på att spelets huvudtråd ska behandla själva commandet.

**Långvariga jobb ska därför inte hålla commandet öppet tills de är klara.** Commandet ska starta jobbet och svara snabbt; fortsatt progress/resultat går som events.

### 2. Event subscription

För asynkrona händelser används en persistent anslutning:

```python
from src.engine.communication.tcp_network_handler import EventListener

with EventListener() as listener:
    event = listener.wait_for_event("aiTraining.completed")
```

Vid connect skickar klienten:

```json
{"type":"subscribe"}
```

Servern bekräftar:

```json
{
  "type": "response",
  "success": true,
  "subscription": "events"
}
```

Därefter skickas EventBus-events löpande på samma anslutning.

## Message schemas

### Command

```json
{
  "type": "command",
  "command": "keyPress",
  "args": ["F2"]
}
```

`args` får för närvarande vara JSON-array eller JSON-object.

### Success response

```json
{
  "type": "response",
  "command": "keyPress",
  "success": true,
  "data": {
    "key": "F2",
    "key_code": 1073741883
  }
}
```

### Error response

```json
{
  "type": "response",
  "command": "keyPress",
  "success": false,
  "error": "..."
}
```

### Event

`EventBus.emit()` skapar ett strukturerat meddelande:

```json
{
  "type": "event",
  "event": "aiTraining.calibrationDone",
  "sequence": 12,
  "timestamp": 1787300000.123,
  "source": "AutomationAITrainingScene",
  "data": {
    "background": "white",
    "attempts": 1
  }
}
```

`sequence` är process-lokal och monotont ökande under den aktuella körningen. Events persisteras inte.

## Threading och dataflöde

### Inkommande command

```text
External script
    -> TCP client thread
    -> CommunicationServer JSON parser
    -> queue.Queue[AutomationCommand]
    -> App._post_automation_events()
    -> custom Pygame AUTOMATION_EVENT
    -> App._handle_automation_event()
    -> command handler in Pygame main thread
    -> AutomationCommand.reply_success/error()
    -> TCP response
```

Detta är en central invariant. Flytta inte Pygame-operationer till sockettråden även om det verkar enklare.

### Utgående event

```text
Scene / engine code
    -> event_bus.emit(name, data, source=...)
    -> CommunicationServer EventBus subscriber
    -> broadcast queue
    -> broadcaster thread
    -> all connected EventListener clients
```

En subscriber som ansluter efter ett event får inte eventet retroaktivt.

## Aktuella commands

### `setWindowPos`

Flyttar det befintliga Pygame-fönstret.

```python
send_command("setWindowPos", [2130, 50])
```

Backwards-compatible dict-form kan också accepteras av handlern:

```python
send_command("setWindowPos", {"x": 2130, "y": 50})
```

Fönster-wrappern från `pygame._sdl2.video.Window.from_display_module()` ska skapas **en gång** och återanvändas. Att skapa nya wrappers vid varje command har tidigare gett instabilitet/segfault.

### `startAITraining`

Skapar en ny `AutomationAITrainingScene`, väljer bakgrund och går in i scenen. Den startar scenens normala auto-kalibrering.

```python
send_command("startAITraining", [1])
send_command("startAITraining", ["checker"])
```

**Detta command skickar inte F2.** Automation ska vänta på `aiTraining.waitingForFirstShot` och först därefter skicka `keyPress(F2)`.

Bakgrunder enligt `AITrainingScene.MODE_NAMES`:

| # | namn |
|---:|---|
| 1 | `white` |
| 2 | `white_grid` |
| 3 | `coord_grid` |
| 4 | `gray` |
| 5 | `black` |
| 6 | `checker` |
| 7 | `checker_anim` |
| 8 | `bubbles` |

**Framtida AI ska läsa `MODE_NAMES` i koden om listan kan ha ändrats; duplicerade nummerlistor i automation scripts måste hållas synkade.**

### `keyPress`

Injicerar ett normalt Pygame `KEYDOWN` följt av `KEYUP` i Pygames event queue. Detta låter automation testa samma eventväg som fysisk tangentinput.

```python
send_command("keyPress", ["F2"])
send_command("keyPress", ["ESC"])
send_command("keyPress", ["TAB"])
```

Grundmappningen stöder F1-F12, ESC/ESCAPE, ENTER/RETURN, SPACE, TAB, piltangenter samt enskilda bokstavstangenter där motsvarande `pygame.K_*` finns.

## AI-training events

`AutomationAITrainingScene` emitterar för närvarande:

| Event | När | Viktig data |
|---|---|---|
| `aiTraining.started` | automation-scenen har gått in | background, background_number |
| `aiTraining.calibrationStarted` | auto-kalibrering pågår | phase, background |
| `aiTraining.calibrationDone` | kalibrering/reference capture klar | result, attempts, background |
| `aiTraining.calibrationFailed` | kalibrering misslyckades/avbröts | result, attempts, background |
| `aiTraining.waitingForFirstShot` | scenen är redo för skott/F2 | background, message |
| `aiTraining.trainingStarted` | F2 startade headless-auto | mode, background, target_iterations |
| `aiTraining.iterationCompleted` | en auto-runda blev klar | iteration, target_iterations, background |
| `aiTraining.trainingStopped` | F1/F2 stoppade en aktiv körning före rapport | background, iteration |
| `aiTraining.completed` | slutrapport skapad | iterations, found, top1, top3, ai_guess_correct, report |
| `aiTraining.exited` | automation-scenen lämnas | background, training_running |

## Exempel: korrekt AI-training automation

Kör från repositoryts rot:

```bash
python3 -m automation.autostart_ai_training 1
```

Logiskt flöde:

```text
1. EventListener ansluter FÖRE start-commandet
2. setWindowPos(2130, 50)
3. startAITraining(1)
4. vänta: aiTraining.started
5. vänta: aiTraining.calibrationStarted
6. vänta: aiTraining.calibrationDone
7. vänta: aiTraining.waitingForFirstShot
8. keyPress(F2)
9. ta emot aiTraining.trainingStarted
10. ta emot iterationCompleted 1..N
11. ta emot aiTraining.completed
12. läs strukturerad final data/report
```

**Varför subscriba först?** Event-streamen har ingen replay. Om lyssnaren ansluter efter `startAITraining` kan tidiga lifecycle-events redan vara förlorade.

## Hur ett nytt command ska läggas till

Exempel framtida `mouseClick`:

1. Definiera extern syntax, t.ex. `send_command("mouseClick", [x, y, button])`.
2. Lägg command-dispatch i `App._handle_automation_event()`.
3. Validera args i en separat handler.
4. Utför Pygame-operationen i huvudtråden.
5. Svara med `reply_success()` eller `reply_error()`.
6. Lägg ett litet automation-script/test som bevisar end-to-end-flödet.
7. Uppdatera command-tabellen i denna fil.

Gör **inte** Pygame-anrop i `CommunicationServer._handle_client()`.

## Hur ett nytt event ska läggas till

1. Emittera nära den faktiska state-transitionen som känner sanningen.
2. Använd stabilt namn, gärna `<subsystem>.<event>`, t.ex. `camera.calibrationDone`.
3. Payload ska vara JSON-serialiserbar: dict/list/string/number/bool/null.
4. Skicka meningsfull strukturerad data, inte bara en formatterad console-sträng.
5. Lägg eventet i dokumentationen här.
6. Automation ska reagera på eventet, inte försöka inferera samma state från timing.

## Logging / console output

**Status: planerat, inte fullt implementerat som generell log-stream ännu.**

Målet är att Python `logging` ska kunna ha både console-handler och en communication/event-handler så att samma debugdata går till terminalen och externa lyssnare. Implementera inte detta genom att ersätta `sys.stdout` globalt utan tydlig anledning. Föredra strukturerad logging där `level`, `source`, `message` och valfri `data` skickas som ett separat `type: "log"` eller ett väldefinierat engine-event.

Tills detta är implementerat ska framtida AI **inte anta att all vanlig console-output automatiskt broadcastas**. Endast explicit emitterade `EventBus`-events är garanterade på eventanslutningen.

## Nuvarande begränsningar

- Ingen autentisering eller TLS.
- Endast avsett för localhost.
- Inga event lagras/replayas.
- Ingen backpressure/persistence för långsamma subscribers utöver OS/socket-buffert och intern queue.
- Command protocol har ingen request-ID/correlation-ID ännu; varje `send_command()` använder egen anslutning så svaret är implicit kopplat till just det commandet.
- Ingen generell mouse input ännu.
- Ingen generell console/log broadcast ännu.

## Avsedd framtida riktning

Detta lager ska kunna bära:

- automatiska meny-/inputtester
- mouse click / mouse movement och syntetiska skott
- kamera- och kalibreringstester
- F2 benchmark över flera bakgrunder
- strukturerade metrics/resultat
- debug/log streaming
- inspelade/replaybara testdataset
- en extern AI-agent som får ändra kod i isolerad git-branch, köra tester och jämföra resultat

Utöka protokollet konservativt. Kommunikationsmotorn ska förbli enkel; domänlogik ska ligga i engine/scenes/tester, inte i socketservern.

---

# Repeated AI training / benchmark loops

## Current automated loop

A complete F2 training session can be repeated without restarting the game.
The loop runner reuses one persistent event subscription and starts a fresh
`AutomationAITrainingScene` for every run.

Run from the project root:

```bash
python3 -m automation.ai_training_loop 1 100
```

Arguments:

- first argument = background number/name
- second argument = number of complete training runs
- each F2 run currently contains 100 synthetic rounds

Example: `1 100` therefore produces approximately 10,000 synthetic rounds.

Background numbers:

1. `white`
2. `white_grid`
3. `coord_grid`
4. `gray`
5. `black`
6. `checker`
7. `checker_anim`
8. `bubbles`

## Result storage — canonical machine-readable output

Repeated automation sessions are stored below:

```text
content/ai/automation_runs/
└── YYYYMMDD_HHMMSS_<background>_<N>runs/
    ├── session.json
    ├── run_001.json
    ├── run_002.json
    ├── ...
    ├── runs.jsonl
    ├── runs.csv
    └── summary.json
```

### `run_NNN.json`

This is the most complete artifact for one run. It contains:

- schema version
- session/run identity
- git commit when available
- wall-clock duration
- the complete `aiTraining.completed` event
- structured aggregate metrics
- funnel diagnostics
- consistency checks
- every `RoundRecord` as structured JSON
- the human-readable report as secondary information

**AI rule:** use structured fields (`metrics`, `funnel`, `consistency`,
`round_records`) instead of parsing the human `report` strings.

### `runs.jsonl`

One compact JSON object per completed run. This is intended for streaming,
quick comparisons, scripts and AI analysis without loading all per-shot rows.

### `runs.csv`

Tabular compact overview of runs. Useful for spreadsheets/plots and quick
manual inspection.

### `summary.json`

Canonical session-level summary. Contains:

- requested/completed runs
- total number of synthetic shots
- aggregate found/top1/top3/AI-correct counts and percentages
- average per-run metrics
- funnel averages
- first/last/best run points
- first-to-last deltas
- consistency status
- git commit

A future coding AI should normally read `summary.json` first, then
`runs.jsonl`, and only open individual `run_NNN.json` files when detailed
per-shot diagnosis is required.

## `aiTraining.completed` schema

The event currently includes compatibility fields (`found`, `top1`, etc.) and
preferred structured fields:

```json
{
  "schema_version": "1.0",
  "background": "white",
  "sampling_mode": "center_bias",
  "match_radius_px": 42.0,
  "training_duration_seconds": 12.3,
  "metrics": {
    "found": 80,
    "found_pct": 80.0,
    "top1": 70,
    "top1_pct": 70.0,
    "top3": 75,
    "top3_pct": 75.0,
    "ai_guess_correct": 68,
    "ai_guess_correct_pct": 68.0,
    "nearest_distance_px": {},
    "ai_guess_distance_px": {},
    "candidates_raw": {},
    "candidates_ranked": {}
  },
  "funnel": {},
  "consistency": {
    "counts_match": true
  },
  "round_records": []
}
```

Exact values vary by run. The schema is versioned so future automation can
change safely without silently changing field meaning.

## Files used by repeated training

```text
automation/autostart_ai_training.py
automation/ai_training_loop.py
automation/ai_training_results.py
src/engine/scenes/automation_ai_training.py
```

`ai_training_loop.py` orchestrates multiple complete runs.
`ai_training_results.py` owns persistent machine-readable result storage.
The normal `src/engine/scenes/ai_training.py` remains the actual training
implementation and is deliberately not duplicated by automation.
