# V2.22.2 r2 installer-fix DELTA

Apply this over the already installed V2.22 + V2.22.1 + V2.22.2 tree.

Changed files:

- `src/engine/camera/hit_scanner_v2221.py`
- `src/engine/camera/hit_scanner_v2222.py`
- `automation/v2222_selftest.py`
- `automation/v2222_verify_install.py`
- `V2222_R2_INSTALL_FIX.md`

This r2 only fixes the real startup installation path and adds regression checks.
It does not retune CV or AI behaviour.

Run:

```bash
python3 -m automation.v2222_selftest
python3 -m automation.v2222_verify_install
python3 main.py
```

Do not fire until startup contains BOTH successful V2.22.1 and V2.22.2 install
messages and no `unavailable` message.
