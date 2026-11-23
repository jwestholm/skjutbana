# Webbresearch: mer avancerade sätt att hitta nya hål på en projicerad skjuttavla

Datum: 2026-11-19. Metod: webbsökning plus läsning av patent, artiklar och produktsidor. Inga filer i repot har ändrats.
Markeringar: **[belagt]** = står i en primärkälla som jag har läst (patent, artikel, datablad). **[sekundär]** = från sammanfattning eller abstract. **[spekulation]** = min egen bedömning, ej verifierad.

Det här är redan provat och nämns bara där en idé bygger vidare på det: PRE/POST-ändringskartor, matchat filter, median/dense, GB-scorer, CIE L*, objektregionstips, precision flash, syntetiska stämplar, ljudtoppar, region/novelty-steg. IR och DISAG nämns i dokumentationen men verkar inte vara byggda.

---

## Rangordning efter förväntat värde för att nå 95–100 %

| # | Idé | Förväntat värde | Kostnad | CPU-ok <1 s? |
|---|---|---|---|---|
| 1 | NIR-kanal: 850/940 nm IR-belysning + kamera med IR-passfilter (projektionen försvinner) | Mycket högt | Låg–medel (~300–1500 kr) | Ja, lättare än idag |
| 2 | IR-bakgrundsbelysning genom tavlan (hål = ljusa punkter) | Högt, beror på mekanik | Medel | Ja |
| 3 | Radiometrisk framåtmodell (förutsäg kamerabilden ur framebuffern) + albedokvot | Högt för rörligt innehåll | Medel (mjukvara) | Ja, på ROI |
| 4 | Akustisk/piezo TDOA-lokalisering av träffen som ROI-prior | Medel–högt (minskar falsklarm) | Medel (~1–2 tkr) | Ja |
| 5 | Exponeringen låst till projektorns uppdateringsfrekvens, manuell kamerastyrning | Medel (billig grundhygien) | Mycket låg | Ja |
| 6 | Logaritmisk/homomorf ändringsdetektion + texturdeskriptorer (LBSP) som features | Medel | Låg | Ja |
| 7 | Kodat/kort känt mönster i ROI (vidareutveckling av flash): färgväxling, fiduciella mönster | Medel | Låg | Ja |
| 8 | Ljusridå/IR-ram (Meyton-princip) eller linjesensorer: mäter projektilen, inte hålet | Potentiellt mycket högt men dyrt/svårt | Hög | Ja (separat elektronik) |
| 9 | YOLO-liknande småobjektsdetektor + inpainting-augmentering | Låg–medel givet befintlig scorer | Medel | Gränsfall |
| 10 | Polarisationsfilter | Låg (fungerar troligen inte mot diffus papperreflex) | Låg | – |
| 11 | Imperceptible structured light, DLP-subframe-synk | Låg med 30 fps-kamera och konsumentprojektor | Hög | – |

---

## 1. NIR-kanal: IR-belysning + IR-passfilter (högst värde)

**Vad.** Belys tavlan med 850 nm (eller 940 nm) IR-LED och filma med en kamera som har IR-passfilter (långpass ~830 nm eller bandpass 850 nm). Projektorns synliga bild syns då nästan inte. Kameran ser en jämnt belyst, oföränderlig tavla. Därmed försvinner animationer, svart innehåll, rullande landskap och projektorflimmer som störkällor.

**Belägg.**
- **[belagt]** US10789729B2 (Ti Training Corp), "System and method(s) for determining projectile impact location": kamera ≥30 fps (gärna monokrom), IR-LED-lampa som lyser upp tavlan och ett 850–880 nm-filter i drift. "The camera sees only the IR-lit target, so the projected animation does not appear". Filtret tas bort vid kalibrering, så att projicerade linjemönster kan mappa kamera→projektor. Detektionen är enkel: jämför med en bild 1–2 s tidigare, kantdetektion eller ljusstyrka, uteslut pixlar som redan är markerade som hål, bounding boxes, och förkasta för små boxar samt boxar med bildförhållande >4:3 ("flared edges"). https://patents.google.com/patent/US10789729
- **[sekundär]** Interaktiva projektorer (Epson BrightLink, Vivitek D-Point) kör 850 nm IR-kamerasystem precis bredvid projektorns egen bild. Det tyder på att projektorljuset ger lite störning vid 850 nm. Tillverkarna varnar däremot för solljus, halogen och värmekällor. https://epson.com/faq/SPT_V11H921520~faq-0000868-shared , https://manualedistruzioni.it/vivitek/interactive-whiteboard/d-point-2/user-manual/26jjh
- **[spekulation, måste mätas]** Lampprojektorer kan läcka mer NIR än LED- och laserprojektorer. Testa med den egna projektorn: filma svart, vitt och rörligt innehåll med IR-filter och IR-lampan släckt. Det ska nästan inte synas någon skillnad.

**Varför det hjälper här.** Grundproblemet idag är att PRE/POST-skillnaden blandar ihop fysisk förändring med projicerad förändring, och att svart innehåll gör hål osynliga. I NIR försvinner båda problemen, och ändringsdetektionen blir det enkla fallet "statisk scen, ett nytt mörkt hål". Det dokumenterade matchade filtret och de riktiga hålstämplarna kan återanvändas rakt av, men de behöver tränas om för NIR-utseende.

**Risker och öppna frågor.**
- Upplösning: med en 1 MP global-shutter-kamera (OV9281, 1280×800) över en ~1,2 m bred tavla blir det ~1 mm/px, så ett hål på 2–4 mm täcker 2–4 px. Det är på gränsen. Bättre alternativ: en högupplöst kamera utan IR-spärrfilter ("NoIR", t.ex. moduler med IMX477/IMX519/IMX708 eller en 4K-USB-kamera där IR-cut-filtret plockas ut) plus ett skruvat IR-passfilter. Det går också att behålla den nuvarande 4K-kameran för synligt ljus och lägga till en NIR-kamera. Två vyer kräver dock homografi och synk.
- Hål som inte går igenom: framifrån syns en buckla/nagg bara via skuggor. En sned IR-belysning ("raking light") från sidan förstärker reliefen **[spekulation]**.
- Papperets NIR-reflektans är hög och jämn, vilket är bra. Tejp och tryckfärg kan se annorlunda ut i NIR än i synligt ljus. Många svarta tryckfärger (kolsvart) är mörka även i NIR, medan färgbaserade tryck ofta blir "osynliga" **[spekulation, allmän erfarenhet]**.
- IR-LED-strålkastare för övervakning (850 nm, 5–20 W) kostar ~150–400 kr. Bandpassfilter M12/M27/M37 kostar ~100–300 kr. OV9281 global shutter USB kostar ~25–50 USD med extern trigger (Arducam B0332 49,99 USD; Waveshare ~25 USD). IMX296 global shutter finns som CSI-modul för ~43–63 USD, inte som USB. https://www.arducam.com/product/arducam-100fps-global-shutter-usb-camera-board-1mp-720p-ov9281-uvc-webcam-module-with-low-distortion-m12-lens-without-microphones-for-computer-laptop-android-device-and-raspberry-pi/ , https://www.waveshare.com/product/raspberry-pi/cameras/global-shutter-cameras.htm
- Kalibrering: precis som i Ti-patentet behövs en synlig-ljus-kalibrering (projicera mönster) för att mappa NIR-kamera → projektorkoordinater. Det går också att låta IR-filtret sitta kvar och kalibrera med ett IR-märkt mönster. Med två kameror räcker en fast homografi per kamera mot tavlan.

**CPU.** Klart lättare än idag: en statisk bakgrund gör att en ren differens plus det befintliga matchade filtret räcker.

**Mätbarhet.** Fullt testbar offline. Spela in synkade NIR- och synligt-ljus-sessioner med samma skott och jämför recall och XY-fel per innehållstyp (svart, rörligt, rutmönster).

---

## 2. IR-bakgrundsbelysning (transillumination) genom tavlan

**Vad.** En IR-ljuskälla bakom tavlan, i kulfångets hålrum. Ett genomgående hål blir en ljus punkt i NIR, med mycket hög kontrast och oberoende av projicerat innehåll.

**Belägg.**
- **[belagt]** US10180310B2 (Meggitt Training Systems, nu Inveris), "Mobile target carrier for gun range with coupled mobile projector": "an infrared light source behind the target shines through the bullet holes to aid detection". Samma patent beskriver också att projektorn kort växlar till en känd referensbild (vit eller färg) med fiduciella markeringar. Bilden triggas av en sensor (mikrofon, mikrovågssensor eller optisk sensor), och flera bilder med olika projicerade färger förbättrar positionsnoggrannheten. Det är i praktiken er "precision flash" plus fiducialer och färgcykling. https://patents.google.com/patent/US10180310

**Varför det hjälper.** Hålet blir en positiv ljuspunkt (högt SNR) i stället för en svag mörk förändring, och det gäller även över svart innehåll. Gamla hål syns också, men de är statiska och kan subtraheras.

**Risker.**
- Kulfånget bakom tavlan måste ha plats för IR-LED:er utan att de skjuts sönder. Belysning från sidan eller indirekt via en vit bakvägg är ett alternativ.
- Kartong släpper igenom lite NIR, så en jämn "glöd" uppstår. Tunna partier, som en BB som nästan gått igenom, kan synas som en svagare ljuspunkt **[spekulation, värd ett enkelt test]**.
- Tejplagningar blockerar hål. Det stämmer med regeln att reparationer är kontext, inte veto.
- Kostnad: IR-LED-list eller strålkastare för ~200–500 kr, plus en NIR-kamera enligt punkt 1.

**CPU.** Trivialt: tröskel eller toppdetektion på en differens.

---

## 3. Radiometrisk framåtmodell + albedokvot (procams-litteraturen)

**Vad.** Systemet vet exakt vilken bild det projicerar. Kalibrera hur projektorpixelvärden blir kamerapixelvärden, så kan den förväntade kamerabilden Ĉ(t) räknas fram för vilken spelbild som helst. Därefter beräknas ett albedo-/kvotmått a = C_observed / Ĉ_predicted. Ett hål sänker albedot (a < 1) oavsett vad spelet visar. Det gör det möjligt att jämföra PRE- och POST-bilder även när innehållet har ändrats däremellan: jämför a_pre och a_post i stället för råpixlar.

**Belägg.**
- **[sekundär]** Den klassiska modellen (Nayar, Peri, Grossberg, Belhumeur): kamerasvar = per-pixel 3×3-färgblandningsmatris × (projektorrespons-linjäriserat P) + ambient term, alltså C = V·P + K. Grossberg et al. kalibrerar den med minst ~6 bilder. https://ics.uci.edu/~majumder/vispercep/nayar_procam.pdf , https://www.cs.columbia.edu/~belhumeur/conference/projector-cvpr04.pdf , https://openjournals.uwaterloo.ca/index.php/vsl/article/download/164/154
- **[sekundär]** "Visual echo cancellation", där Microsoft Research löser exakt det här problemet för interaktiva projektorytor. Liao, Yang, Zhang, "Robust and Accurate Visual Echo Cancelation in a Full-duplex Projector-camera System", IEEE TPAMI 30(10), 2008 (CVPRW 2006): geometrisk och fotometrisk transfer lagras i en LUT, och projicerat innehåll undertrycks i realtid. Zhou, Zhang, Huang (projektor-kamera-whiteboard) definierar albedokvoten som uppmätt intensitet delat med förutsagt "eko". Skrift på tavlan absorberar ljus, så kvoten blir ≤ 1, vilket är samma fysik som för ett hål. https://www.microsoft.com/en-us/research/?p=328061 , https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/VISUAL-ECHO-CANCELLATION-IN-A-PROJECTOR-CAMERA-WHITEBOARD-SYSTEM.pdf
- **[sekundär]** Uhm et al., "User region extraction from dynamic projected background": kalibrera geometri och radiometri, subtrahera den förutsagda bakgrunden och få kvar förgrunden. https://www.aminer.cn/pub/56d86472dabfae2eee9593dc/user-region-extraction-from-dynamic-projected-background-for-a-virtual-environment-system
- **[belagt, abstract]** Neurala varianter (CompenNet++, DeProCams, Huang & Ling, TVCG 2021) lär sig hela relighting-funktionen, men kräver GPU-träning och hundratals bildpar. Kod: https://github.com/BingyaoHuang/DeProCams . För det här projektet är en klassisk per-pixel-LUT tillräcklig och CPU-vänlig **[spekulation]**.
- **[sekundär]** CompenNet-artikeln konstaterar att projektorn läcker ljus även vid svart ingång ("backlight"). Det är relevant eftersom svart innehåll alltså inte är exakt noll ljus. https://ar5iv.labs.arxiv.org/html/1904.04335

**Varför det hjälper här.** Den hanterar rörligt innehåll (bubblor, scrollande landskap, rutmönster) där PRE/POST-differensen idag ger falska ändringar. Kvoten är dessutom en naturlig "observerbarhetskarta": där Ĉ är nära projektorns svartnivå är kvoten odefinierad. Systemet kan då *veta* att hålet är oobserverbart där och bara flasha just den regionen (koppling till precision flash).

**Kritiska förutsättningar.**
- Exakt tidssynk mellan den visade framebuffern och kamerabilden. Projektorns latens (typiskt 1–3 bildrutor) måste mätas och bildrutor med övergång mellan två spelbilder kastas eller blandas. Det kräver att spelmotorn loggar vilken bild som visades när.
- Rullande slutare och projektorns PWM ger radvisa avvikelser (se punkt 5).
- Kameran måste köras med fast exponering, förstärkning och vitbalans, annars driver modellen.

**CPU.** Ja. Förutsägelsen är en LUT plus en 3×3-matris per pixel, och den kan begränsas till ROI eller en nedskalad bild. En warp av framebuffern till kamerageometri är en `cv2.remap` per bildruta, några ms på ROI.

---

## 4. Akustisk/piezo-lokalisering av träffen (TDOA) som ROI-prior

**Vad.** 3–4 kontaktmikrofoner (piezo) på tavlans ram eller backstop, eller luftmikrofoner i tavlans hörn. Tidsskillnaden för nedslagsljudet (TDOA, GCC-PHAT) ger en grov position. Det ger inte mm-noggrannhet, men det kan begränsa sökområdet till t.ex. en cirkel på 3–10 cm. Det tar bort de flesta falsklarm från animationer, och det kan avgöra vilket av flera kandidathål som är det nya.

**Belägg.**
- **[sekundär]** TAI-CHI-projektet (EU, "Tangible Acoustic Interfaces") lokaliserade knackningar på MDF, stålwhiteboard och trä med fyra kontaktsensorer i hörnen. Metoder: TDOA, mallmatchning med korskorrelation och time reversal. Noggrannheten anges som "reasonable", i praktiken cm-nivå enligt mitt minne **[ej verifierat siffermässigt]**. https://new.eurasip.org/Proceedings/Eusipco/Eusipco2006/papers/1568982376.pdf , https://re.public.polimi.it/retrieve/e0c31c0a-1d1e-4599-e053-1705fe0aef77/DAFx05_TAI.pdf , https://orca.cardiff.ac.uk/54576/1/U584936.pdf
- **[belagt, patentöversikt]** Patentlitteraturen bedömer piezo-TDOA på stålmål som "prone to very low accuracy" och problematisk vid snabbeld, eftersom vibrationer måste klinga av. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10190854 . Pappersmål med tre piezo på målarket: US5095433. https://patents.google.com/patent/US5095433
- **[sekundär]** Öppen kod: GCC-PHAT TDOA i Python https://github.com/xiongyihui/tdoa . Billigt open-source SSL-system med 4 USB-ljudkort (IJEEI) https://beei.org/index.php/EEI/article/view/10523 . kdwaterman/raspi_audio_geolocation har testats med en CO2-luftpistol.
- **[spekulation]** Luftljud: 343 m/s och 48 kHz ger ~7 mm per sampel, och med subsampelinterpolation är 1–3 cm möjligt i teorin. Kartong är dispersiv och dämpande för böjvågor, så luftmikrofoner nära tavlan kan vara mer förutsägbara än kontaktsensorer. Mynningsknallen kommer före nedslaget och måste hållas isär, men den sitter redan i den befintliga ljudpipelinen (de två CO2-topparna).

**Varför det hjälper.** Det är en *oberoende* sensor, och den fungerar även när kameran inte ser något (svart innehåll). Den behöver inte vara exakt för att höja precision och recall, eftersom den gör sökningen mindre.

**Kostnad.** Flerkanaligt USB-ljudinterface (4 in, synkade kanaler; billiga USB-ljudkort är *inte* synkade sinsemellan) ~1–2 tkr, plus piezo/elektretmikrofoner ~100 kr. **CPU:** GCC-PHAT på 4 kanaler × några tiotals ms ljud tar under 1 ms.

**Projektregler.** Det här är en ny sensor och därmed en ny hypotes. Den bör först utvärderas som en offline-prior utan att ändra detektorbeteendet.

---

## 5. Exponering låst till projektorns frekvens + manuell kamerastyrning

**Vad.** Sätt kamerans exponeringstid till en heltalsmultipel av projektorns uppdateringsperiod, t.ex. 1/60 s eller 2/60 s vid 60 Hz, och helst av färghjulets eller PWM-periodens subframes. Lås förstärkning, vitbalans och fokus.

**Belägg.**
- **[sekundär]** Flimmerpatent och leverantörsguider: band uppstår när exponeringen inte är ett heltal antal ljuscykler, och en rullande slutare gör att raderna fångar olika faser. Med exponering = n·period försvinner banden. https://www.imatest.com/imaging/flicker/ , https://support.enttec.com/pixel/pixel-general-knowledge/why-do-leds-flicker-on-camera , https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8711245
- **[belagt]** TI DLPC900: synk av kamera mot DLP-mönster sker via TRIG_OUT, och "dark time" kan läggas in mellan mönster. Det finns bara på utvecklingskort och "pattern mode"-projektorer, inte på konsumentprojektorer. https://www.ti.com/document-viewer/DLPC900/datasheet/GUID-0C998F07-A7A0-44DD-AF13-84A3D80FF94B
- **[sekundär]** Färghjulets fas kan läsas med färgfiltrerade fotodioder och användas som trigger (stereoskopiskt patent US8167431). Med en global-shutter-kamera med extern trigger (OV9281 har det) går det i princip att fånga exakt samma subframe varje gång. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8167431

**Varför.** Projektorflimmer och rolling-shutter-band syns idag som brus i differenskartorna. Om de tas bort sjunker brusgolvet för det matchade filtret. Det är billigt och hör till grundhygienen innan nya hypoteser mäts. Värdet ska mätas som skillnad i brusgolv per innehållstyp.

**Att "fånga i mörkerintervallet"** för att se tavlan utan projektion går inte med en vanlig konsumentprojektor och en 30 fps-kamera. DLP-mörkertider är bara µs–ms långa. **[spekulation]**

---

## 6. Logaritmisk/homomorf och texturbaserad ändringsdetektion

**Vad.** (a) Räkna differensen i log-domän: log(post) − log(pre), eller log-högpass. Det tar bort *multiplikativa* belysningsändringar, t.ex. projektorns ljusstyrkevariation och gradvisa färgskiften, medan albedoändringar (hål) finns kvar. (b) Lokala binära texturdeskriptorer (LBSP, från SuBSENSE/PAWCS) är robusta mot belysningsändring men känsliga för strukturändring. De kan bli extra features till den befintliga GB-scorern.

**Belägg.**
- **[sekundär]** Toth, Aach, Metzler: logaritm + linjärt högpass tar ungefär bort belysningen, och ändringsdetektionen görs på reflektansdelen. https://www.isip.uni-luebeck.de/fileadmin/files/publications/ssiai00_01.pdf
- **[belagt, abstract]** St-Charles et al., SuBSENSE (CVPRW 2014): färg + LBSP med lokal adaptiv känslighet, i topp på CDnet 2014 "dynamic background". https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf
- **Not:** CIE L* (kubrot) är redan provat. Log är en närliggande men starkare kompression av mörka värden, så vinsten kan vara liten. Det är billigt att testa som en ren feature-ablation.

**Begränsning.** Log-trick hjälper inte när det projicerade *mönstret* ändras (rörliga objekt). Det fallet kräver punkt 3 eller punkt 1.

---

## 7. Utveckla precision flash: kort, kodad, regional

**Vad.** I stället för en vit blixt över hela tavlan:
- Flasha bara i den lokaliserade ROI:n (från ljud-TDOA eller kandidater), med ett *känt* mönster som innehåller fiducialer.
- Använd två korta bildrutor med olika färger eller komplementära mönster (P och 1−P). Ett hål är mörkt i båda, medan projicerat innehåll växlar.

**Belägg.** **[belagt]** Meggitt/Inveris-patentet ovan: referensbild med fiducialer och flera bilder med olika projicerade färger "improves shot-location fidelity". **[sekundär]** Imperceptible structured light med inverspar kräver höghastighetskamera och projektor i tät synk (Cole, Ziauddin, Greenspan, VISAPP 2020). https://www.scitepress.org/Papers/2020/89559/89559.pdf . Med 30 fps blir mönstret alltså *synligt* för spelaren.

**Spelregelaspekt.** Enligt minnesreglerna styr spelet om flash får användas. En regional flash är mindre störande än en flash över hela tavlan **[spekulation]**.

---

## 8. Mäta projektilen i stället för hålet: ljusridå/IR-ram

**Vad.** En ram med korsade IR-ljusbarriärer framför tavlan. Projektilen bryter strålar i X och Y. Det är så professionella elektroniska luftvapentavlor fungerar.

**Belägg.**
- **[belagt]** Meyton MF5R1 "Black Magic": 192 korsade ljusbarriärer, 182×182 mm genomskjutningsyta, noggrannhet "upp till 1/10 mm". Patent US9823051. https://meyton.info/en/product/mf5r1-black-magic/ , https://patents.google.com/patent/US9823051
- **[sekundär]** Meggitt FATS 100LE live-fire: gummiduken används bara för projektionen, och träffdetektionen mäter kulan i flykten. https://www.officer.com/training-careers/training-simulators/product/20982159/inveris-training-solutions-virtual-and-live-fire-weapons-training-systems-live-fire-screen

**Bedömning.** Det här är det enda spåret som är helt oberoende av projektion, slitage och svart innehåll. Ytan är dock stor (hela projektionen, ~1 m+), och en BB passerar en stråle på ~50 µs. Det kräver snabb dedikerad elektronik (mikrokontroller, fotodiodmatriser). Kommersiella IR-pekramar (touch-overlays) skannar troligen för långsamt för att fånga en BB **[spekulation]**. Stort DIY-projekt med hög risk. Det nämns som det långsiktiga "100 %-alternativet", inte som nästa steg.

---

## 9. Öppna projekt och artiklar om hålfinding: vad de faktiskt gör

- **[belagt]** Belcher et al., arXiv:2601.17062 (jan 2026), "A Computer Vision Pipeline for Iterative Bullet Hole Tracking in Rifle Zeroing": YOLOv8 med 97,0 % mAP50 men bara 63,3 % mAP50-95. Gamla och nya hål skiljs åt med IoU>0,5 mot föregående bild, efter ORB-homografi. Augmentering: riktiga hål tas *bort* med inpainting för att skapa "före"-bilder, i stället för att generativa hål läggs till (de syntetiska hålen såg orealistiska ut). End-to-end-noggrannheten var bara 40,3 %. Slutsats för projektet: en objektdetektor löser *inte* problemet med "nytt hål exakt" av sig själv. Inpainting-tricket (ta bort riktiga hål ur riktiga bilder) kan däremot ge realistiska PRE/POST-par från befintliga inspelningar **[spekulation om nyttan här]**. https://arxiv.org/html/2601.17062v1
- **[belagt]** US12435958B2 (Eley Ltd, 2025): bilder med fast intervall, föregående bild subtraheras, Hough-cirklar ger hålcentrum. Konfidenstest: många Hough-cirklar per träff ska ligga inom ±4 mm och blobben ska vara ≥70 % cirkulär. Annars får användaren mata in poäng manuellt. Det är en idé för en "abstain"-väg: en spridningsbaserad konfidens för XY. https://patents.google.com/patent/US12435958
- **[sekundär]** MDPI AI 5(1):5 (2024): YOLOv8 och Detectron2 på skjutkort. https://www.mdpi.com/2673-2688/5/1/5 . Universidade do Porto-avhandling: morfologi + perspektiv, ~1 s latens, "98,3 % efficiency". https://repositorio-aberto.up.pt/bitstream/10216/160791/2/681549.pdf . Watershed för överlappande hål (IJRASET 2020).
- **[sekundär]** ShootOFF (öppen, Java): projektorarena med autokalibrering via rutmönster, men den detekterar *laserprickar*, inte hål. https://github.com/phrack/ShootOFF/wiki/Using-the-Projector-Arena/12510611b47c6c7aff73f2f8a38ef59a3400fbb7
- **[sekundär]** AccuShoot (kommersiellt, kamera + projektor i stålbox): recensenter noterar att kameran "eventually starts having difficulty seeing your impacts as the screen wears". Det bekräftar att slitage är ett känt problem även kommersiellt. https://shootingnewsweekly.com/shot-show/accushoot-live-fire-training-experience-hands-on-at-shot-show-2025
- Roboflow har flera dataset för "bullet hole detection" (främst skarp ammunition på tryckta mål, inte BB på projicerad kartong). https://universe.roboflow.com/perbakinsam/bullet-hole-object-detection-2cat5

**Sammanfattning:** inga öppna projekt hittades som löser just "nya BB-hål under rörlig projektion". De kommersiella live-fire-systemen som löser projektionsproblemet gör det med **IR (front eller bak)**, **referensbild/flash med fiducialer** eller **mätning i flykten**, inte med smartare synlig-ljus-differens.

---

## 10. Polarisering (låg förväntan)

- **[sekundär]** DLP-projektorer avger opolariserat ljus. LCD- och LCoS-projektorer avger polariserat ljus, ofta med grönt vinkelrätt mot rött och blått, och vissa modeller depolariserar internt. https://www.projectorreviews.com/home-theater-and-projectors-the-technical-side/passive-3d-projection-part-4/
- **[spekulation, men fysikaliskt välgrundad]** Matt papper och kartong ger en *diffus* reflex som till stor del depolariserar ljuset. Ett korsat polarisationsfilter framför kameran tar därför bort spegelglans men inte den projicerade bilden på pappret. Låg nytta för att skilja hål från innehåll, och ~50 % ljusförlust. Kan möjligen dämpa blänk från tejp.

---

## Förslag till mätordning (respekterar repots regler: en hypotes i taget, egen run-katalog, ingen tuning under mätning)

1. **Billig grundhygien (punkt 5):** lås exponeringen till n/refresh. Mät brusgolvet i PRE/POST-kartor per innehållstyp före och efter. Ingen detektoränd­ring.
2. **NIR-förstudie (punkt 1):** ett snabbt hårdvarutest med IR-pass + IR-lampa. Spela in svart, vitt och animerat innehåll *utan* skott: syns projektionen? Därefter en skottsession med parallell synlig-ljus-kamera, och jämförelse av recall och XY-fel offline.
3. **Bakbelysning (punkt 2):** samma testprotokoll om mekaniken tillåter. Testa även halvgenomskjutna hål.
4. **Framåtmodell + albedokvot (punkt 3):** ren mjukvara, offline-utvärdering på befintliga inspelningar *om* framebuffer-tidsstämplar finns loggade. Annars måste loggningen läggas till först.
5. **Akustisk prior (punkt 4):** spela in 4-kanaligt parallellt med de befintliga sessionerna och mät TDOA-felet mot facit-XY innan det kopplas in.

## Viktigaste källor
- Ti Training, US10789729B2 (IR-filtrerad kamera, projicerade scenarier): https://patents.google.com/patent/US10789729
- Meggitt/Inveris, US10180310B2 (IR-bakbelysning, referensbild med fiducialer, flerfärgsbilder, ljudtrigger): https://patents.google.com/patent/US10180310
- Eley, US12435958B2 (realtidsdifferens, Hough-konfidens): https://patents.google.com/patent/US12435958
- Liao/Yang/Zhang, Visual echo cancelation, TPAMI 2008: https://www.microsoft.com/en-us/research/?p=328061
- Nayar et al., procam-radiometri: https://ics.uci.edu/~majumder/vispercep/nayar_procam.pdf
- Huang & Ling, DeProCams: https://github.com/BingyaoHuang/DeProCams
- Belcher et al., arXiv:2601.17062: https://arxiv.org/html/2601.17062v1
- Meyton MF5R1 / US9823051: https://meyton.info/en/product/mf5r1-black-magic/
- TAI-CHI akustisk lokalisering: https://new.eurasip.org/Proceedings/Eusipco/Eusipco2006/papers/1568982376.pdf
- SuBSENSE: https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf
- Toth/Aach belysningsinvariant ändringsdetektion: https://www.isip.uni-luebeck.de/fileadmin/files/publications/ssiai00_01.pdf
- TI DLPC900 trigger/dark time: https://www.ti.com/document-viewer/DLPC900/datasheet/GUID-0C998F07-A7A0-44DD-AF13-84A3D80FF94B
