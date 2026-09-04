# V2.25.1 delta — balanced object-region physical proposal / confirmation

This cumulative delta is intended to overlay the current V2.25.0/dev checkout.
It retains V2.24.1-V2.25.0 files and adds V2.25.1.

## Main runtime change

`src/engine/shot_region_proposal_v251.py` installs after V2.25.0 and partitions
normal live object-context proposals by frozen physical HitRegion search area.
Each region gets a local robust threshold and bounded candidate quota. Merged
V1/V2 output and V2.22.5 local confirmation are re-balanced per physical region.
Confirmed track choice uses physical confirmation/evidence rather than game roles.

The patch does not move candidate XY and does not change GameObject collision,
damage, penetration or effect semantics.

V2.22.5 FULL-RESCUE remains global and bypasses the V2.25.1 restriction.

## Install

```bash
unzip -o skjutbana_v2.25.1_object_region_physical_proposal_delta.zip -d .
python3 -m automation.v251_prepare
python3 -m automation.v251_selftest
python3 -m automation.v251_verify_install
python3 -m automation.v251_status
python3 main.py
```

Open **Spel -> Game Objects Test (V2.25.1)** and repeat the five-object physical
series used for V2.25.0.
