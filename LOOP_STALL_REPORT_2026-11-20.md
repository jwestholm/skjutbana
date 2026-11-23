# Huvudloopens stopp (~0,9 s): orsak identifierad (2026-11-20)

Mätning: `SKJUTBANA_STALL_PROBE=<fil>` (ny, avstängd som standard, `src/engine/loop_stall_probe.py`). Den registrerar
varje loopvarv ≥ 150 ms med tid per fas, stickprov av alla trådars stackar var 20 ms, CPU-tid per tråd och GC-pauser.
Live: F2 48 automatiska projicerade skott, vit bakgrund, utan H15, utan spårskrivning, ostörd maskin.
Data: `/data/skjutbana/evaluation_runs/loop_stall_20261120/` (base_a: 19:14–19:18, kontroll nosave_a: 19:19–19:23).
OBS: under base_a kördes självtester cirka 19:15–19:16 (nice); slutsatsen bygger på stackprover och bekräftas av kontrollen.

## Resultat

| Körning | huvudloop work p99 | varv ≥ 150 ms | varv ≥ 500 ms | max |
|---|---|---|---|---|
| base_a (som tidigare) | 870–917 ms | 110 | 80 | 3070 ms (start) |
| nosave_a (`SKJUTBANA_TRAINING_SKIP_DIAG_SAVES=1`) | 249–291 ms | 74 | 3 (alla vid start) | 3139 ms (start) |

**Orsak till 0,9 s-stoppen (verifierad):** AI-träningsscenen (den scen F2-mätningarna körs i) sparar
diagnostik synkront på huvudtråden efter varje skott:
1. `automation_ai_training._save_v216_shot` → `candidate_pack_v216.capture_shot` → `np.savez_compressed` av hela
   bildrutor (`save_full_frames`): cirka 52 % av alla stoppstickprov.
2. `ai_training._save_shot_diagnostic` / `_save_hole_image`: `glob` över `content/ai/shot_diag` (60 230 filer) för att
   hitta nästa löpnummer, varje skott, plus PNG/GIF-skrivning.
Det är alltså **inte** grundmotorns bildhantering, låsningar eller GC (GC-pauser ≤ 35 ms) som ger 0,9 s, utan
blockerande diskskrivning i testscenen. Vanliga spel anropar inte dessa funktioner (ej mätt live i spel).

## Kvarvarande stopp ~250–300 ms (kontrollkörningen)
| Andel stickprov | Plats | Typ |
|---|---|---|
| 15 % | `hit_scanner._maybe_capture_reference`: `np.median` över staplade hela bildrutor | grundmotor, CPU |
| 15 % | `ai_training._do_clean_capture` → `camera_manager.capture_fresh_frame` | träningsscen, väntar på ny kamerabild |
| 16 % | `synthetic_hole_overlay._alpha_blend_bgra_on_bgr` i render | träningsscen, CPU |
| 4 % | `ranking_dataset_v29.write_shot` | rankningsdataset, diskskrivning |
| 3 % | `runtime.save` (study_click_area) | JSON-skrivning |
| 21 % | utan projektstack (C-kod/GC) | – |

Därtill ett engångsstopp på ~3 s vid första skottet (`detector_luma.detector_gray`, uppvärmning).

## Förbehåll (Codex-granskning)
- Slutsatsen vilar på två oberoende mått: det billiga `SKJUTBANA_LOOP_TIMING` (p99 900 → 270 ms) och stickproven.
- Stickproven tas av en Pythontråd som måste få GIL; andelar av stickprov är inte exakt andelar av stopptid, och
  CPU-mätningen börjar först efter tröskeln. Den utesluter inte GIL-konkurrens eller andra motorstopp generellt.
- Varvet mäts från efter `clock.tick()` till efter flip; `gt_review.observe()` och sondens egen skrivning ligger utanför.
- Kontrollflaggan stoppar även insamling av kandidatpaket/hålbilder och ska anges i mätmanifest.
- Codex rekommenderar ordningen 3 → 1 → 2 nedan.

## Förslag (ej genomförda; kräver godkännande)
1. Flytta v2.16-sparning och skottdiagnostik till en bakgrundsskrivare (kopierade arrayer), eller stäng av
   `save_full_frames` i mätningar. Ersätt `glob`-numreringen med en räknare som läses en gång.
2. `_maybe_capture_reference`: samma exakta Batcher-median som H15 eller beräkning i arbetstråd.
3. Gör om F2-latensmätningarna med `SKJUTBANA_TRAINING_SKIP_DIAG_SAVES=1`, eftersom tidigare live-latenser
   (både med och utan H15) innehöll dessa stopp.
