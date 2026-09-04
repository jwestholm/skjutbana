from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def check(label,cond):
    if not cond: raise AssertionError(label)
    print(f'[PASS] {label}')
def _find(node):
    if isinstance(node,dict):
        if node.get('id')=='game_objects_test_v250': return node
        for v in node.values():
            r=_find(v)
            if r is not None:return r
    elif isinstance(node,list):
        for v in node:
            r=_find(v)
            if r is not None:return r
    return None
def main():
    print('V2.25.3 INSTALL VERIFICATION\n=============================')
    req=['src/engine/shot_cross_thread_novelty_v253.py','src/engine/shot_region_freshness_v252.py',
      'src/engine/shot_region_proposal_v251.py','content/games/game_objects_test_v250.py','menu_games_entry_v253.json',
      'automation/v253_prepare.py','automation/v253_apply_docs.py','automation/v253_apply_menu.py','automation/v253_selftest.py',
      'automation/v253_verify_install.py','automation/v253_status.py','V253_CROSS_THREAD_NOVELTY_AUTHORITY.md',
      'AI_CROSS_SHOT_NOVELTY.md','V253_TEST_PLAN.md','V253_DOC_PATCH.md','DELTA_README_V253.md','main.py']
    check('required V2.25.3 files exist',all((ROOT/p).exists() for p in req))
    main=(ROOT/'main.py').read_text(); check('V2.25.3 wired after V2.25.2',main.rfind('install_v253_runtime(App)')>main.rfind('install_v252_runtime(App)'))
    src=(ROOT/'src/engine/shot_cross_thread_novelty_v253.py').read_text()
    check('lock-protected shared bridge exists','CrossThreadShotBridgeV253' in src and 'threading.RLock' in src)
    check('readiness is peak-validated','peak_ts' in src and 'is_ready' in src)
    check('history is canonical full-camera','v253_full_camera_x' in src and '_candidate_full_camera_xy' in src)
    check('rehit recovery is soft','signature_gain' in src and 'never a hard exclusion' in src)
    check('base V2.22.5 confirmation used before old balancing','_v251_previous_local_confirm' in src)
    check('FULL rescue stays global','_v251_previous_best_track' in src and 'was_consumed(sid)' in src)
    check('no candidate XY assignment', 'cand["camera_x"] =' not in src and 'cand["camera_y"] =' not in src)
    menu=ROOT/'content/menu.json'; entry=_find(json.loads(menu.read_text())) if menu.exists() else None
    check('Game Objects Test labelled V2.25.3',entry is not None and entry.get('title')=='Game Objects Test (V2.25.3)')
    scene=(ROOT/'content/games/game_objects_test_v250.py').read_text(); check('scene logs V2.25.3 ObjectHit','[V2.25.3 OBJECT-HIT]' in scene)
    print('\nV2.25.3 installation verification passed.')
if __name__=='__main__':main()
