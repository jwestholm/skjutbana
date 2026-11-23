# PANG! Poly Haven rendering provenance

Imported models/textures: CC0-1.0, https://polyhaven.com/license .
Light diffuse variants preserve source roughness, metallic, normal and alpha maps.
Source depot remains unmodified. Manual mesh cuts, UV-preserving sectors, bright TV
screen and original room shell/effects are recorded by the recipe hashes.
Runtime loads sprites, not glTF. Exact per-file hashes are in asset_provenance.

- sofa_02: https://polyhaven.com/a/sofa_02 — CC0-1.0
- modern_coffee_table_01: https://polyhaven.com/a/modern_coffee_table_01 — CC0-1.0
- Shelf_01: https://polyhaven.com/a/Shelf_01 — CC0-1.0
- side_table_01: https://polyhaven.com/a/side_table_01 — CC0-1.0
- fancy_picture_frame_01: https://polyhaven.com/a/fancy_picture_frame_01 — CC0-1.0
- Television_01: https://polyhaven.com/a/Television_01 — CC0-1.0
- book_encyclopedia_set_01: https://polyhaven.com/a/book_encyclopedia_set_01 — CC0-1.0
- ceramic_vase_01: https://polyhaven.com/a/ceramic_vase_01 — CC0-1.0
- ceramic_vase_02: https://polyhaven.com/a/ceramic_vase_02 — CC0-1.0
- ceramic_vase_03: https://polyhaven.com/a/ceramic_vase_03 — CC0-1.0
- ceramic_pot: https://polyhaven.com/a/ceramic_pot — CC0-1.0
- potted_plant_02: https://polyhaven.com/a/potted_plant_02 — CC0-1.0
- alarm_clock_01: https://polyhaven.com/a/alarm_clock_01 — CC0-1.0
- desk_lamp_arm_01: https://polyhaven.com/a/desk_lamp_arm_01 — CC0-1.0
- metal_jug: https://polyhaven.com/a/metal_jug — CC0-1.0

Palette packaging: automation.recolor_light_room; render_preserving=True.
Baseline alpha/masks/geometry are preserved. Generated sprites retain CC0-1.0.
See palette_report.json; DN values are predictions, not physical camera validation.

R7: six natural PBR room plates, three templates/two furnishings each.
No global camera-floor postprocessing or display gain on room plates.
Original R6 target sprites and UV silhouettes retained; local safety is applied
only to active-target surroundings by content/games/light_room_safety.py.
New room recipe hashes are separate from inherited target recipe hashes.
