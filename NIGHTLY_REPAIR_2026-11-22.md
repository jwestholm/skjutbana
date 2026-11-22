# Nattjobb: minne, datasetval och rapportering — 2026-11-22

Implementerat på `codex/astra-2`. Ingen spel-/detektorkod ändrad, ingen modell installerad.
Cron har inte ändrats eller återaktiverats. Befintliga data/baslinjer i `/data` har bara lästs.
Körartefakter finns i `/tmp/nightly_repair_20261122/` eftersom denna arbetsmiljö endast medger
skrivning i repot och `/tmp`. `claude/latency` har inte ändrats.

## Vad som ändrats

- `candidate_cache.py`: strömmar exporten i två pass till en innehållsadresserad float32-NPY-cache
  per export. SHA256 över originalbytes, kolumnschema, label-/impact-data och gruppnycklar sparas.
  NaN bevarar missingness; `shots.jsonl` bevarar även skott utan kandidater. Atomisk publicering.
- `train.py`: mmap-cache, TRAIN-only kolumnval/median/skalning, float64 arbetsaritmetik och samma
  logistiska optimerare. Tillfälliga arbetsmatriser ligger i `/tmp` och städas efter körningen.
  Holdout innehåller alla kandidater/skott; inga utvärderingsrader samplas bort.
- Fryst urval: seed 20261122; alla kandidater i positiva impact-grupper, högst 64 slumpade kandidater
  i övriga impact-grupper. Index sparas per export tillsammans med ursprungshash. Ingen ny slump per natt.
- `research/nightly_candidate_dataset_20261121.json`: exakt de 64 tränings- och 4 holdout-exporter
  som referensmodellen från 21 november använde. Även regressionernas baslinjer är hashade.
  Ingen nattlig katalogupptäckt, nya syntetiska träningsfrön eller automatisk sessionsexport.
- `nightly.py`: sekventiella övervakade barnprocesser; träningsprocessen har 6 GB adressrymdsgräns.
  Max 6 GB JSON-indata, 100 exporter, 2 GB arbetsmatriser, konservativ 16 GB diskbudget plus 20 GB reserv.
  Budgetar kan anges explicit med `--config`; inget befintligt dataset raderas för att frigöra plats.
- RUNNING publiceras före inläsning/träning, FAILED vid fel inklusive barnprocessens SIGKILL/OOM,
  och COMPLETE först efter lyckad jämförelse/träning. En dödad supervisor lämnar synligt RUNNING.
  NOT_COMPARABLE och WORSE är fel med icke-noll exitkod. Ingen gammal framgång visas som aktuell körning.

## Viktigt tidigare mätfel: grupp-ID kolliderade mellan exporter

Den gamla `_top1` slog ihop exempelvis `double:900000:7` från standard/stress/stamps innan vinnaren
valdes. Ett skott kunde därför få sin vinnare från en annan export. Primär top-1 använder nu
`(export, group)`. Historisk matchning finns bara som märkt `legacy_global_group_top1` för paritetskontroll.
Detta ändrar mätningen, inte detektorn. 130/150 mot 127/150 får inte längre anges som korrekt totalmätning.

## Reproducerbara mätningar

Alla tester här är offline kandidatvärdering på tidigare live-path-exporter, inte ny fysisk validering.
Den redan inspekterade holdouten är en regressionsmängd.

| Samma indata | Gammal laddare | Ny cache, utan sampling |
|---|---:|---:|
| 3 TRAIN-exporter + alla 4 holdout-exporter | 1 507,7 MiB topp-RSS | 162,5 MiB |
| Historisk top-1 på denna delmängd, modell / live | 132/150 / 127/150 | identiskt |
| Modell-AUC på denna delmängd | 0,9676903055705787 | identiskt |

Full osamplad träning: 64 TRAIN-exporter, 977 259 kandidater, 3 116 positiva, 3 412 skott.
128 features i samma ordning; största viktavvikelse från sparad referens 1,05e-7 efter float32-lagring.
Historisk matchning återskapar **130/150 / 127/150**; korrekt exportägd matchning är **128/150 / 126/150**.
Modell-AUC identisk med sparad referens: 0,9927275683123439. Topp-RSS **1 671,1 MiB**.
Tid 479,7 s inklusive kall cachekonvertering; inte jämförbar som ren träningstid mot varm cache.

Fryst 64-urval på SAMMA 64 exporter: **453 375 kandidater**, alla **3 116 positiva** kvar.
Primär top-1 identisk: **128/150 / 126/150**, topp-RSS **1 149,4 MiB**, tid 63,7 s med varm cache.
Alla 64 indexlistor återskapades exakt (`sampling_repeat_check.json`).
Ett separat försök att även minska datasetet till 51 exporter gav 119/150; den datasetminskningen används inte.

Disk: originalexporter **4 755 767 465 byte**; kompakt cache **645 321 301 byte** (cirka 7,4 gånger mindre).
Originalen är bevarade, så cachen är en engångskostnad, inte redan frigjort utrymme.
Beständiga modell-/urvalsartefakter för samplad körning: **3 715 545 byte**.
Arbetsmatriser: cirka 516 MB tillfälligt för samplad träning, inte nya beständiga dataset varje natt.
Jämförelsen med rapporterad OOM på 21,6 GB den 22 november gäller olika datasetstorlekar; den parade
minnesmätningen ovan är den direkta före/efter-kontrollen. Ingen ny full gammal OOM-körning startades.

Återskapa delmängdsjämförelsen, separata processer för separata RSS-mätningar:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m automation.bench.verify_training \
  --manifest research/nightly_candidate_dataset_20261121.json --mode legacy --subset 3 \
  --cache-root /tmp/new-nightly-check/cache --output /tmp/new-nightly-check/legacy.json
OPENBLAS_NUM_THREADS=1 python3 -m automation.bench.verify_training \
  --manifest research/nightly_candidate_dataset_20261121.json --mode cached --subset 3 \
  --cache-root /tmp/new-nightly-check/cache --output /tmp/new-nightly-check/cached
```

`verify_training --mode reference` läser den frysta modellens vikter och jämför historisk/korrigerad
holdout-matchning. Varje utdata måste vara en ny sökväg. Fulla osamplade/samplade körningars
`model.json`, `metrics.json`, `sampling.json` och index ligger i artefaktroten under `full` / `sampled_full`.

## Verifiering och kvarvarande driftvillkor

- `automation.bench_training_selftest`: 7 testfall gröna, inklusive tom kandidatgrupp, missingness,
  exakt indexurval, gruppkollision, hashändring, splitöverlapp, dry-run utan skrivning,
  NOT_COMPARABLE och träningsbarn som avslutas med -9.
- `automation.bench_selftest`: grönt.
- `automation.bench_fidelity_selftest`: grönt, 92 installerade metoder matchar startup.
- Verklig train-CLI med 6 GB adressrymdsgräns på paritetsdelmängden: exit 0, 162,5 MiB RSS.
- `git diff --check` samt Python-kompilering: gröna.
- Verklig dry-run på hela frysta manifestet: exit **2**, avsiktligt synligt **NOT_COMPARABLE** för båda armarna.
  Dagens `content/settings.json` har hash `84957dd0…`, baslinjen från 21 november `cc966bdf…`.
  De äldre regressionsfilerna saknar också hash för AI-settings; nya synth-manifest registrerar denna
  och compare/preflight kontrollerar den. Gamla settings/baslinjer har inte ersatts. Innan schemalagd körning krävs en uttryckligt fryst
  kompatibel regressionsbaslinje eller återfunnen exakt settings-snapshot. Ingen dold ombasering görs.
  Ingen full ny syntetisk regressionskörning startades eftersom preflight redan visar mismatch.
- Cron förblir pausad. Denna miljö kan inte läsa användarens crontab; pausstatus kommer från Jimmy/Claude.

## Föreslaget OBJECT-schema att stämma av med Claude (INTE implementerat)

Behåll en exportägare; Claude äger OBJECT-export, burkmodellen och packad kameraläsning.
Föreslagen adapter till samma cache, först efter avstämning:

- Skottmanifest: schema/version, session/stream, stabil shot_uid och audio-event-/burst-ID med
  grupperingsversion; begärd result_semantics; frysta regioners/mappingens hash; beslutets tidsgräns.
- Facit separat: VERIFIED_IMPACT / VERIFIED_NO_IMPACT / UNKNOWN, verifierade objekt-ID,
  eventuell exakt rå kamerakoordinat, annoteringskälla/hash. UNKNOWN blir aldrig negativ etikett.
- Kandidat: stabil candidate_uid/coordinate_version, shot_uid, producent/version, rå kamera-XY,
  alla exakta objektmedlemskap, separat kvalificeringsstatus; inga spelprioriteter som fysisk sanning.
- Evidens: numeriskt värde + giltighet/missingness, familj/version/enhet, PRE/POST-frame-ID,
  kausalt cutoff, faktisk tillgänglighet/leveranstid och uppmätt kostnad. Facit/position/identitet
  får inte hamna i featurematrisen genom automatisk numerisk upptäckt.
- Split före modell-/tröskelval, per fysisk källsession (alla bearbetningar av samma skott tillsammans).
  Negativa ljudkontroller och skott utan kandidater måste ha egna manifestrader.

Inga nya OBJECT-fält, trösklar eller packade-session-läsare har införts i denna ändring.

## Commitstatus

Ingen ny commit skapades. En första version kunde stageas, men senare `git add` nekades med
`Unable to create .git/index.lock: Read-only file system`. Senaste ändringarna är därför delvis
stageade och delvis ostageade; använd `git diff HEAD` för hela leveransen. Ingen push/merge gjordes.
