# Ny seedvärld, encounters, lampskugga och dödsvapen

Branch **codex/graphics-realism**, bas `33512ee6ba2eb8b6225daaf1e5e749fb2d25f54f`.
Allt lokalt och ocommittat. Tidigare arbete, assets och backup bevarade; ingen
Git-flytt, städning, integration, commit, push eller branchändring.

## Rotorsaker och ändringar

**Seed/topologi: åtgärdat.** `turn_for_meter` använde samma sex kurvor,
34-metersperiod och stora svängar vid 155/274/412 m för alla seeds. Seed ändrade
bara amplitud/längd lite; tidigare meshhash-test bevisade inte ny banstruktur.
Alternativseeds styr nu segmentavstånd, första sväng, målriktningar och
svänglängd genom separat stabil heltalshash. Begränsade absoluta riktningar ger
positiv framåtrörelse och undviker loopar. **1944 behåller exakt gamla rutten.**
Ingen ny generator, korridorbredd, anslutning eller collider.

**Fiender: utökat enligt senaste användarregeln.** Tidigare huvudfiender vid
32/164/258/384 m och sedan 72-metersnät gav långa, likadana luckor.
Spelets befintliga `EnemyDirector` får nu explicit world seed. Befintliga
encounterfunktioner varierar position, flank, profil, initial blickfas,
reservrendererns utrustningsvariant och partnerantal med en separat hashström.
Tidigt 52 m basintervall med ±7 m jitter, sedan 44 respektive 34 m; säker
startzon minst 27 m. Femte basankaret är 238 m. Partnerchansen ökar mjukt till
högst 55 %. Aktiveringsfönster och eliminerade ID:n återanvänds; ingen växande
plancache/dubbelspawn. Äldre anrop utan seed bevarar befintliga dekor-/coverankare;
levande spelet använder alltid seed. **Defaultens encounterplan ändras avsiktligt
för att uppfylla den nya stridsregeln, medan dess geometri bevaras.**

Kompetenskurvan **fanns redan** och behålls: `difficulty_at` höjer accuracy_scale
.60 → .79 → 1.04 → 1.20 och sänker reaction_grace 2.25 → 1.95 → 1.65 → 1.40 s
vid 180/450/850 m. Både EnemyAgent och ArcadeRun använder den. Modern GLB har
bara en utrustningsmodell; inga nya GLB-varianter påstås.

**Lampskugga: reproducerad och åtgärdad.** `closest_shadow_lamp` slog av kartan
bortom **7,5 m kameraavstånd**, trots redan ritad geometri/belysning. Verklig
renderer, lampan vid ruttmeter 16, seed 98765: index −1 vid 7,80 m lampavstånd,
index 0 vid 7,40 m. Alla fyra chunkar/caster-VAO:er fanns hela tiden. Default
visar motsvarande övergång mellan ruttproven 7 och 8,4 m. Detta är skuggaktivering,
inte sen chunkuppladdning. Kartan förbereds nu inom 18 m och får mjuk vikt till
full styrka vid 12 m. Byte mellan närliggande lampor tonas ned vid lika avstånd.
Samma enda 768²-lampkarta, samma sol/måne, castermesh och globala renderavstånd.
Begränsning: en lokal karta ger inte full samtidiga skuggor från alla lampor.

**Dödsvapen: markpose förbättrad, rapporterad lodrät slutpipa ej reproducerad
med aktuell Tactical-renderer.** Sparad före-kod/bilder visar redan horisontell
pipa men gevär på magasin/grepp. Nu rullas det ned på sidan med liten
seedstabil variation per fiende. Kroppens settle-vinkel kompenseras helt före
markanpassning; utökat lutningstest hittade och fångar det felet. Kroppens ben,
levande grepp, skala och hitboxar är oförändrade. Ingen säker rotorsak till
användarens exakt 90°-observation påstås; kontrollera samma worktree/renderer.

## Seedkedja och reproducerbarhet

Kallstart = **1944**, alternativt `--seed`. S byter seed, R behåller den; båda
spärras vid obesvarat DEMO-skott. Befintlig reset invaliderar jobbgeneration,
rutt, CPU-/skottmeshcache, GPU-chunks, skugg-VAO:er och aktörscacher.
Fiendedirectorns seed sätts före reset. Processbygge får explicit seed.
BattlefieldEventSystem och ArcadeRun har redan separata seedade Random-instancer;
miljö/rekvisita använder befintliga saltade noise-/Random-strömmar och globala
ankare. Ingen global RNG eller processberoende `hash()` införs.

Fasta designregler behålls: chunklängd, tunnel-/rampcykel, grundläggande biomeordning,
dugoutmarkörer och demorekvistaregler, inklusive torktumlaren. Befintliga slumpval
för rum, väggar, material, dekor och slagfält följer seed. Detta gäller initial
värld; identisk stridssimulation kräver också samma actions/tidssteg.

[Fem seeds: full rutt, encounters, props, signaturer och intervallstatistik](seed_shadow_validation/routes.json).
Endast **4,0–4,8 % identiska centerlineprov** (gemensam startzon), medelseparation
13,3–51,3 m. Separata processer och omkastad rutt-/propordning ger samma resultat.
[Faktiskt meshbygge 80 → 0 m](seed_shadow_validation/mesh_order.json) ger också
exakt samma hash som 0 → 80 för seed 98765. Defaultstartens mesh är fortfarande
`f1770a5620b3adb48ed5fa14039a4d694dc2ccf53db4b07096b8ae95e6ecf4b0`.

| Seed | Fiender 0–250 / 250–500 / 500–1000 m | Max planlucka tidigt | Max provad siktlucka |
|---|---|---|---|
| 1944 | 5 / 6 / 24 | 55,9 m | 65,9 m |
| 98765 | 5 / 7 / 25 | 55,9 m | 55,9 m |
| 44123 | 5 / 9 / 27 | 52,0 m | 56,0 m |
| 44223 | 6 / 6 / 24 | 56,8 m | 56,8 m |
| 20261016 | 5 / 7 / 27 | 63,5 m | 59,5 m |

Senare täthet 4,8–5,4/100 m mot 2,0–2,4 tidigt. Medel-accuracy .68–.70 →
1.08–1.09, reaction_grace cirka 2,10–2,13 → 1,57–1,59 s. Befintliga sidovakter
räknas med; huvudencounters särredovisas i JSON.
[Mesh-LOS i alla fem seeds](seed_shadow_validation/clearance.json): alla tidiga
huvudfiender/partners har fri skjutlinje inom 20 m och ±38° från gångriktningen.
Femte encounter flyttades 12 m tidigare efter ett konservativt 82,6 m siktgap;
nya placeringarna omprovade. Golvets maxsteg vid 0,5 m sampling: 2,14 cm.

## Faktisk verifiering och kostnad

- **PASS:** 72 riktade tester, därefter 9 fokuserade reset-/encounter-/inputtester.
  [Logg](seed_shadow_validation/final_tests.log), [slutlig resetregression](seed_shadow_validation/final_reset_tests.log).
  Första lutningstestet hittade settle-felet; korrigerat. Loggens `CHUNK ERROR`
  är avsiktlig retry-felinjektion. Ingen full suite körd.
- **PASS:** [verklig S/R och DEMO-spärr](seed_shadow_validation/play/result.json):
  S i spel → 98765, S efter död → 44123, R samma rutt/mesh/encounters/props,
  repeat spärrad. Provet hittade först seedtilldelning i fel resetväg; flyttad
  till `restart_world`, regression tillagd, verkliga provet omkört godkänt.
- **PASS:** aktuell seed 98765, 251 route-sampel till 250 m; genererade fiender
  0–4 träffade genom NORMAL:s frysta XY-raycast vid **14/69/125/174/218 m**.
  Default provades också genom 250 m före sista femte-encounterjusteringen;
  dess nya placering verifierades sedan med mesh-LOS. Detta är ett samplat
  spelprov, inte normalhastighetsvideo eller balanserad AI-överlevnad.
- **PASS:** [riktiga NORMAL/DEMO-smoketestet](seed_shadow_validation/demo/result.json),
  inklusive B/SPACE/M, miss/träff och poäng. Alla draw-anrop `GL_NO_ERROR`.
- **PASS:** `py_compile`, `git diff --check`, verklig EGL/GLSL-kompilering/länkning.
  [Oförändrade skyddade kodvägar](seed_shadow_validation/source_contracts.json).
- [Granskade före/efterbilder](seed_shadow_validation/index.html): default har
  identisk kamera/seed/FOV/ljus; alternativseedens topologi ändras avsiktligt och
  är inte en isolerad shaderpixeljämförelse. Lampbildernas gång-/springtider är
  nominella 2/4 m/s, inte en uppmätt realtidsvideo.

**Renderbegränsning:** Mesa-bildläsning gav långa stopp; stackdump pekar på
`Framebuffer.read`, inte chunkarbetaren. Avläsning ger även `GL_INVALID_OPERATION`
efter returnerade bilder, medan draw är felfri. Begränsad avläsning användes i
spelprovet. Verktygsfel med NumPy-JSON och syntetisk key-up rättades också.
**GTX 1070/FPS och verklig spelbalans är ej verifierade.**

[Skuggkostnad](seed_shadow_validation/render_cost.json): 7 identiska defaultvyer,
72 → 84 world shadow-draws (+16,7 %), max fortfarande tre kartpass. Median
draw+finish 708 → 589 ms på llvmpipe är störningskänslig och bevisar ingen
prestandavinst. Inga nya texturer/modeller/skuggkartor. Fler soldater kostar fler
aktörsdraws; täthet/gruppstorlek är begränsade. [CPU/meshdata](seed_shadow_validation/clearance.json):
startchunk 1944 = 260004 vertices/6,50 s, 98765 = 285990/6,64 s; alla fyra chunkar
per seed redovisas där. Olika världar, inte före/efter-prestandapåstående.

## Filer, Git och demoprov

Ändrat i detta pass: `trench_3d_test.py` (rutt/seedkoppling/shader),
`enemy_system.py` (auktoriserad spawnplanering/initial blick), `rig_renderer.py`
(befintligt variantval), `gltf_actor.py` (dödsvapen), `shadow_lighting.py` (lampvikt).
Nytt: `tests/test_seed_shadow_revision.py`, fyra `verify_*`-script för route_seeds,
seed_shadow_render, seed_play och encounter_clearance, samt `seed_shadow_validation/`.
Inga nya spelassets/beroenden. [Full Git-status](seed_shadow_validation/git_status.txt)
och [diffstat](seed_shadow_validation/git_diff_stat.txt) inkluderar tidigare lokalt arbete;
ospårade verifieringsfiler finns i statuslistan.

Start: `python3 trench_3d_test.py`; alternativ: `python3 trench_3d_test.py --seed 98765`.
Prova första lampan runt 7–12 m, S/R från spel/dödssida, DEMO-spärren och
stridstäthet på projektorn. Kontrollera `LIVE: taktisk soldat aktiv` i terminalen.
Nästa tre kontroller: GTX/projector-prestanda; spelbalans; exakt renderer/state/vinkel
om det rapporterade lodräta vapnet fortfarande förekommer.
