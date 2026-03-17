# HARDWARE.md
# Skjutbana – Hardware Setup

> Beskriver den fysiska uppställningen för systemet.
> Syftet är att en ny utvecklare snabbt ska förstå hur den verkliga miljön ser ut
> och vilka antaganden mjukvaran gör.

---

# 🧱 Tavla / Skjutyta

Tavlan är uppbyggd i lager (framifrån → bakåt):

1. **Vita A4-papper**
   - utbytbara
   - ger tydlig kontrast för hål

2. **Dubbla lager pappkartong (flyttlådor)**
   - fångar projektiler
   - ger struktur för håldetektion

3. **4 cm ljudisoleringsmatta (bilmatta)**
   - dämpar ljud
   - stoppar energi

4. **2 mm plåt**
   - stoppar projektiler
   - säkerhetslager

5. **Tunt skummgummi (bakre lager)**
   - extra dämpning
   - minskar resonans

---

# 🎯 Markeringar

Tavlan är tydligt markerad:

## 🟢 Grön zon (skjutområde)
- området där man får skjuta
- motsvarar projekterad bild (viewport/content)

## 🔴 Röd zon (förbjudet område)
- markerar kanter
- får inte träffas

---

# 📽️ Projektion

- **Projektor monterad i taket**
- projicerar bild/video/spel på tavlan
- definierar det visuella “content”-området

Viktigt:
- projektionsytan måste matcha viewport/content i mjukvaran

---

# 📷 Kamera

- monterad i taket (eller nära tak)
- riktad mot tavlan
- fångar hela skjutytan (full frame)

Används för:
- visuell träffdetektion
- debug (scanport)

---

# 🎤 Mikrofon

- kopplad till systemet (samma enhet eller extern)
- används för att detektera skott via ljud (audio peak)

Viktigt:
- måste kunna skilja skott från bakgrundsljud
- senare: ignorera ljud från systemets egna högtalare

---

# 🔊 Ljud (framtid)

Systemet kommer även använda ljud ut:
- TTS (”skjut”, ”3-2-1”)
- gameplayljud
- video/musik

Mikrofonen måste:
- ignorera dessa så gott det går
- behandla dem som brus

---

# ⚠️ Viktiga antaganden

Mjukvaran antar:

- tavlan är **plan**
- kameran är **fast monterad**
- projektorn är **fast monterad**
- skjutområdet är **tydligt avgränsat**
- ljusförhållanden är relativt stabila

---

# 🧪 Praktiska tips

- Byt A4-papper regelbundet (gamla hål stör detection)
- Håll tavlan så jämn som möjligt
- Undvik starka skuggor
- Se till att kameran inte rör sig

---

# 🧠 Sammanfattning

Hardware setupen är byggd för att:

- stoppa projektiler säkert
- ge bra visuell kontrast
- minska ljud/reflektioner
- möjliggöra stabil kameradetektion

Detta är grunden som hela mjukvarusystemet bygger på.
