from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def check(label,cond):
    if not cond: raise AssertionError(label)
    print(f"[PASS] {label}")
def _find(node):
    if isinstance(node,dict):
        if node.get('id')=='game_objects_test_v250': return node
        for v in node.values():
            r=_find(v)
            if r is not None: return r
    if isinstance(node,list):
        for v in node:
            r=_find(v)
            if r is not None: return r
    return None
def main():
    print('V2.25.2 INSTALL VERIFICATION\n=============================')
    req=[
      'src/engine/shot_region_freshness_v252.py','src/engine/shot_region_proposal_v251.py',
      'src/engine/shot_context_v250.py','content/games/game_objects_test_v250.py',
      'menu_games_entry_v252.json','automation/v252_prepare.py','automation/v252_apply_docs.py',
      'automation/v252_apply_menu.py','automation/v252_selftest.py','automation/v252_verify_install.py',
      'automation/v252_status.py','V252_REGISTERED_FRESHNESS_AUTHORITY.md','AI_REGISTERED_FRESHNESS.md',
      'V252_TEST_PLAN.md','V252_DOC_PATCH.md','DELTA_README_V252.md','main.py']
    check('required V2.25.2 files exist',all((ROOT/p).exists() for p in req))
    main_src=(ROOT/'main.py').read_text()
    check('runtime wired in main','install_v252_runtime(App)' in main_src)
    check('runtime installs after V2.25.1',main_src.rfind('install_v252_runtime(App)')>main_src.rfind('install_v251_runtime(App)'))
    src=(ROOT/'src/engine/shot_region_freshness_v252.py').read_text()
    check('registered PRE->POST maps used','absdiff' in src and 'zscore' in src and 'temporal_noise' in src)
    check('early waiting-post authority gated','await_registered_frame' in src)
    check('legacy can only gain local authority by revalidation','legacy_revalidated' in src and 'v252_fresh_physical' in src)
    check('V2.22.5 persistence remains second-frame proof','previous_confirm' in src and 'REGISTERED-CONFIRM' in src)
    check('global FULL rescue preserved','rescue_router_v2225.was_consumed(sid)' in src)
    check('candidate XY is never rewritten','camera_x"] =' not in src and "camera_y\"] =" not in src)
    menu=ROOT/'content/menu.json'
    check('content/menu.json exists',menu.exists())
    entry=_find(json.loads(menu.read_text()))
    check('Game Objects Test entry exists',entry is not None)
    check('Game Objects Test labelled V2.25.2',entry is not None and entry.get('title')=='Game Objects Test (V2.25.2)')
    scene=(ROOT/'content/games/game_objects_test_v250.py').read_text()
    check('scene logs V2.25.2 ObjectHit','[V2.25.2 OBJECT-HIT]' in scene)
    print('\nV2.25.2 installation verification passed.')
if __name__=='__main__': main()
