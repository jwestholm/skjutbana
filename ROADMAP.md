# 🎯 Skjutbana System – Full Roadmap

## 📌 Översikt
Detta projekt syftar till att bygga ett komplett visuellt skjutbanesystem med:
- kamerabaserad träffdetektion
- projektion av innehåll (tavlor, spel, video)
- spelmotor
- träningssystem
- avancerad fysik (penetration, material)
- audio-system (in/ut)
- framtida AI & analys

Systemet består av flera lager:
1. Sensor (kamera + ljud)
2. Detection (träffidentifiering)
3. Transformation (koordinatmapping)
4. Game Engine (logik & rendering)
5. Feedback (ljud, partiklar, UI)

---

# 🧱 FAS 1 – Grund & Stabilitet

## Kamera & Setup
- Montera kamera i tak (fast position)
- Rotera bild 180°
- Säkerställ konstant exposure/fokus

## Geometri
- Definiera:
  - Camera space
  - Scanport
  - Viewport
  - Content rect
- Rita overlays för alla

---

# 👁️ FAS 2 – Debug & Insyn

## Kandidatplotting
Visa:
- Alla blobbar
- Filtrerade blobbar
- Toppkandidater
- Vald träff

## Multi-space visualization
Rita varje kandidat i:
- kamera
- scanport
- viewport
- content

## Transform-debug
- Klick-debug tool
- Visa hela kedjan:
  camera → scanport → viewport → content → game

---

# 📸 FAS 3 – Session System

## Session start
1. Visa syncbild
2. Kalibrera hörn
3. Visa vit bild
4. Ta 3–5 frames
5. Bygg median → session reference

## Användning
- ignorera gamla hål
- filtrera tejp
- stabilare detection

---

# 🔍 FAS 4 – Detection (Baseline)

- Enkel blob detection
- filter:
  - storlek
  - circularity
- verifiera på vit tavla

---

# 🎯 FAS 5 – Träffkalibrering

- Grid (9–16 punkter)
- Skjut per punkt
- Spara:
  - expected vs measured
- Skapa korrigeringsmodell

---

# 🎮 FAS 6 – Game Engine

- Rendering pipeline
- Content layers
- Z-order
- Object system

---

# 💥 FAS 6.5 – Partikelsystem

- materialbaserade effekter
- vapenbaserade effekter
- entry/exit effects
- pooling

---

# 🔫 FAS 7 – Weapon System

## Weapon profiles
- name
- type
- audio signature
- hole size
- penetration power
- spread model
- damage
- particle profile

## Event
Shot event innehåller:
- position
- weapon_id

---

# 🔊 FAS 7.5 – Audio System

## Output
- TTS
- musik
- gameplay
- video

## Input
- mic
- shot detection

## Separation
- ignorera engine audio
- noise suppression

---

# 🎯 FAS 8 – Target Awareness

- target zones (0..n)
- center + radius + weight

---

# 🎥 FAS 9 – Multi Camera

- extra kamera (skytt)
- recording
- replay
- slow motion

---

# 📊 FAS 10 – Training System

- logga skott
- statistik
- heatmaps
- progression

---

# 🧱 FAS 11.5 – Material & Penetration

## Material
- type
- thickness
- resistance

## Penetration
- energi
- multiple hits
- layering

## Armor
- skyddsnivå
- durability

---

# 🚀 FAS 12 – Advanced

- motion detection
- ML (optional)
- speech input

---

# 🎯 Designprinciper

- Separation of concerns
- Debug everything visually
- Build simple first
- Add complexity later

---

# 🔥 Prioritet NU

1. Kandidatplotting
2. Transform-debug
3. Session reference

---

# 🧠 Vision

Systemet ska kunna gå från:
- enkel tavla

till:

- full spelmotor
- träningsplattform
- realistisk simulering
