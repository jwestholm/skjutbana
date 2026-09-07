from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from src.engine.shot_cross_thread_novelty_v253 import (
    CandidateSignatureV253, CrossThreadShotBridgeV253, annotate_cross_shot_novelty_v253,
    balance_confirmed_v253, shot_bridge_v253, _patch_v252_ready_bridge,
)
ROOT=Path(__file__).resolve().parents[1]

def check(label,cond):
    if not cond: raise AssertionError(label)
    print(f'[PASS] {label}')

def cand(group,x,y=100,physical=20,compact=8,center=12,dark=6,source='region_registered'):
    return {'camera_x':float(x),'camera_y':float(y),'score':10.0,'v251_shot_id':2,
      'v251_region_group':group,'v251_region_objects':group,'v251_region_roles':'target',
      'v252_physical_score':physical,'v252_center_abs':center,'v252_compact_abs':compact,
      'v252_dark_compact':dark,'v252_fresh_physical':1.0,'v252_authority_source':source,
      'v2225_confirm_center_abs':5.0,'v2225_confirm_compact':2.0,'v2225_confirm_peak_abs':8.0,
      'v2225_confirm_darkening':2.0}

class Scanner:
    def __init__(self,peak=100.0):
        self.audio_events=[SimpleNamespace(shot_id=2,peak_ts=peak)]
        self._v244_roi_diag={'crop':(1000,500,1400,700),'scale':(1.0,1.0),'work_shape':(700,1400)}

def test_cross_thread_bridge():
    import src.engine.shot_region_freshness_v252 as v252
    shot_bridge_v253.reset(); _patch_v252_ready_bridge()
    worker=Scanner(100.0); main=Scanner(100.0)
    v252._mark_ready(worker,2,100.20)
    check('worker ready is visible from different main scanner instance', v252._is_ready(main,2))
    wrong=Scanner(101.0)
    check('peak timestamp prevents stale shot-id readiness', not v252._is_ready(wrong,2))

def test_full_camera_history_and_novelty():
    shot_bridge_v253.reset(); scanner=Scanner(200.0)
    old=cand('a',100); old['v253_full_camera_x']=1100.0; old['v253_full_camera_y']=600.0
    shot_bridge_v253.store_confirmed(1,190.0,[old])
    vals=[cand('a',100),cand('a',180)]
    before=[(c['camera_x'],c['camera_y']) for c in vals]
    out=annotate_cross_shot_novelty_v253(scanner,2,200.0,vals)
    recurrent=min(out,key=lambda c:c['camera_x']); novel=max(out,key=lambda c:c['camera_x'])
    check('history comparison uses canonical full-camera XY', recurrent['v253_history_distance_px'] < 0.1 and novel['v253_history_distance_px'] > 18)
    check('spatially novel candidate outranks recurrent hotspot', novel['v253_novelty_score'] > recurrent['v253_novelty_score'])
    check('novelty annotation never changes candidate XY', before==[(c['camera_x'],c['camera_y']) for c in out])

def test_rehit_recovery():
    shot_bridge_v253.reset(); scanner=Scanner(300.0)
    old=cand('a',100,compact=5,center=10,dark=3); old['v253_full_camera_x']=1100; old['v253_full_camera_y']=600
    shot_bridge_v253.store_confirmed(1,290,[old])
    weak=annotate_cross_shot_novelty_v253(scanner,2,300,[cand('a',100,compact=5,center=10,dark=3)])[0]
    strong=annotate_cross_shot_novelty_v253(scanner,2,300,[cand('a',100,compact=13,center=20,dark=10)])[0]
    check('same-hole stronger registered signature recovers soft recurrence penalty', strong['v253_novelty_score'] > weak['v253_novelty_score'] and strong['v253_signature_gain'] > 0)

def test_sparse_region_prior():
    shot_bridge_v253.reset(); scanner=Scanner(400.0)
    vals=[cand('crate',100)] + [cand('noisy',300+i*8) for i in range(8)]
    out=annotate_cross_shot_novelty_v253(scanner,2,400,vals)
    crate=next(c for c in out if c['v251_region_group']=='crate')
    noisy=max((c for c in out if c['v251_region_group']=='noisy'),key=lambda c:c['v253_novelty_score'])
    check('isolated physical region peak gets soft sparsity advantage on first shot', crate['v253_novelty_score'] > noisy['v253_novelty_score'])

def test_confirm_balance():
    vals=[]
    for g,base in [('a',10),('b',100),('c',200)]:
        for i in range(4):
            c=cand(g,base+i); c['v253_authority_ok']=1.0; c['v253_novelty_score']=10-i
            vals.append(c)
    out=balance_confirmed_v253(vals)
    check('V2.25.3 confirmation preserves each physical group', {c['v251_region_group'] for c in out}=={'a','b','c'})
    check('V2.25.3 confirmation is globally bounded', len(out)<=8)

def test_source_contracts():
    src=(ROOT/'src/engine/shot_cross_thread_novelty_v253.py').read_text(encoding='utf-8')
    forbidden=['target_bonus','role_weight','owner_weight','damage_weight','projectile_weight','game_score_weight']
    check('no gameplay semantic authority weights', not any(x in src for x in forbidden))
    check('global rescue bypasses V2.25.x local selector', '_v251_previous_best_track' in src and 'was_consumed(sid)' in src)
    check('local fail-open requests FULL rescue instead of object-local base hit', 'RESCUE-REQUEST' in src and 'rescue_router_v2225.request(sid)' in src)

def test_install_and_menu():
    main=(ROOT/'main.py').read_text(encoding='utf-8')
    check('V2.25.3 runtime wired after V2.25.2', main.rfind('install_v253_runtime(App)')>main.rfind('install_v252_runtime(App)')>=0)
    scene=(ROOT/'content/games/game_objects_test_v250.py').read_text(encoding='utf-8')
    check('test scene logs V2.25.3 ObjectHit','[V2.25.3 OBJECT-HIT]' in scene)
    from automation.v253_apply_menu import patch_menu_text
    entry=json.loads((ROOT/'menu_games_entry_v253.json').read_text())
    old=json.loads((ROOT/'menu_games_entry_v252.json').read_text())
    sample=json.dumps({'children':[old]},ensure_ascii=False,indent=2)+'\n'
    patched,changed,found=patch_menu_text(sample,entry)
    check('V2.25.2 menu entry updates to V2.25.3',found and changed and 'Game Objects Test (V2.25.3)' in patched)
    patched2,changed2,found2=patch_menu_text(patched,entry)
    check('V2.25.3 menu patch is idempotent',found2 and not changed2 and patched2==patched)

def main():
    print('V2.25.3 SELFTEST\n===============')
    test_cross_thread_bridge(); test_full_camera_history_and_novelty(); test_rehit_recovery(); test_sparse_region_prior()
    test_confirm_balance(); test_source_contracts(); test_install_and_menu()
    print('\nAll V2.25.3 selftests passed.')
if __name__=='__main__': main()
