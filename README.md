# 🎯 Skjutbana

Ett visuellt skjutbanesystem där vi kombinerar:
- kamera (träffdetektion)
- mikrofon (skotttrigger)
- projektion (tavla / spel / video)
- spelmotor (logik, feedback, effekter)

---

# 🚀 Vision

Systemet ska gå från:

👉 enkel projekterad tavla  
till  
👉 komplett tränings- och spelplattform

med:
- realistisk träffdetektion
- vapenprofiler
- penetration & material
- partikelsystem
- ljud (in/ut)
- replay & analys

---

# 🧠 Hur det funkar (kort)

1. 🎤 Mikrofon registrerar skott (audio peak)
2. 📷 Kamera analyserar tavlan
3. 🎯 Systemet hittar träff (x,y)
4. 🔄 Koordinater transformeras:
   - kamera → scanport → viewport → content
5. 🎮 Spelmotor tolkar träffen
6. 💥 Feedback:
   - visuella markörer
   - partiklar (framtid)
   - ljud (framtid)

---

# 🏗️ Arkitektur (översikt)

Systemet är uppdelat i lager:

## 1. Sensor layer
- kamera (bild)
- mikrofon (audio peak)

## 2. Detection layer
- blob/candidate detection
- hit selection

## 3. Transform layer
- mapping mellan:
  - camera
  - scanport
  - viewport
  - content

## 4. Engine layer
- scene system
- game logic
- content rendering

## 5. Feedback layer
- hit visualizer
- debug overlays
- (framtid: partiklar, ljud)

---

# 📁 Struktur (kort)

```
main.py
config.py
content/
assets/
src/engine/
    app.py
    scene.py
    input/
    camera/
    audio/
    scenes/
    visual/
```

---

# ⚙️ Viktiga begrepp

## Viewport
Området där vi ritar/projicerar.

## Scanport
Området i kamerabilden vi analyserar.

## Content rect
Området där själva bilden/tavlan finns.

## HitEvent
Globalt event med:
- screen_x/y
- viewport_x/y
- content_x/y
- camera_x/y

---

# 🔥 Status just nu

✔ App + scenes fungerar  
✔ Kamera fungerar  
✔ Audio peak fungerar  
✔ Hit-event pipeline fungerar  
✔ Visualisering fungerar  

⚠ Detection är fortfarande instabil  
⚠ Kandidatval känns slumpmässigt  
⚠ Mapping behöver finjusteras  

---

# 🎯 Nästa steg (kritiska)

1. Kandidatplotting (se vad systemet ser)
2. Transform-debug (verifiera mapping)
3. Session-start:
   - syncbild
   - vit referensbild

---

# 🛣️ Roadmap highlights

## Core
- stabil träffdetektion
- kalibrering

## Engine
- spelstöd
- target zones

## Audio
- TTS
- gameplay-ljud
- suppression i mic

## Weapon system
- vapenprofiler
- audio-signaturer
- damage & spread

## Partiklar
- damm, sand, blod, splitter

## Material & penetration
- trä, glas, metall
- armor / skyddsvästar

## Multi-camera
- skyttkamera
- replay

---

# 💡 Designprinciper

- Bygg enkelt först
- Debugga visuellt
- Separera ansvar
- Undvik “gissning tuning”

---

# 🧪 Dev tips

- Testa alltid på vit tavla först
- Verifiera koordinater visuellt
- Lita inte på logik du inte kan se

---

# 🤝 Målet

Det här ska bli:

👉 en riktigt bra träningsplattform  
👉 en flexibel spelmotor  
👉 ett system som faktiskt funkar i verkligheten  

---

Built with curiosity, iteration och ganska mycket trial & error 😄
