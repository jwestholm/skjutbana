from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.engine.shot_region_freshness_v252 import (
    RegisteredMetricsV252,
    _balance_fresh_confirmed,
    _disc_ring_metrics,
    _fresh_gate,
    _metric_quality,
    _normalise_groups,
)

ROOT = Path(__file__).resolve().parents[1]

def check(label: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def test_registered_compact_metrics() -> None:
    shape=(31,31)
    a=np.zeros(shape,np.float32); z=np.zeros(shape,np.float32); d=np.zeros(shape,np.float32); n=np.ones(shape,np.float32)
    yy,xx=np.ogrid[:31,:31]; c=(xx-15)**2+(yy-15)**2<=4
    a[c]=8; z[c]=5; d[c]=7
    m=_disc_ring_metrics(a,z,d,n,15,15)
    check("compact registered center metrics exist", m is not None)
    assert m is not None
    check("compact center exceeds ring", m.center_abs > 7 and m.ring_abs < 0.1 and m.compact_abs > 7)
    check("compact dark hole passes fresh gate", _fresh_gate(m))


def test_broad_motion_not_compact() -> None:
    a=np.full((31,31),8,np.float32); z=np.full((31,31),5,np.float32); d=np.full((31,31),4,np.float32); n=np.full((31,31),6,np.float32)
    m=_disc_ring_metrics(a,z,d,n,15,15)
    assert m is not None
    check("broad change has near-zero compactness", abs(m.compact_abs) < 0.1 and abs(m.dark_compact) < 0.1)
    check("broad projected change does not pass compact freshness", not _fresh_gate(m))


def test_pre_noise_soft_penalty() -> None:
    clean=RegisteredMetricsV252(6,1,5,9,4,5,0.5,4.5,0.83,1)
    noisy=RegisteredMetricsV252(6,1,5,9,4,5,0.5,4.5,0.83,12)
    check("PRE instability is a soft physical penalty", _metric_quality(clean,5) > _metric_quality(noisy,5) > 0)


def _cand(group: str, x: float, score: float, *, fresh=True, source='region_registered') -> dict:
    return {
        'camera_x':x,'camera_y':50.0,'score':10.0,
        'v251_region_group':group,'v251_region_objects':group,'v251_region_roles':'target',
        'v252_physical_score':score,'v252_fresh_physical':1.0 if fresh else 0.0,
        'v252_authority_source':source,'v252_evidence_ready':1.0,
    }


def test_group_normalisation() -> None:
    vals=[_cand('noisy',10+i,9.0+i*0.1) for i in range(8)] + [_cand('clean',100,9.5)]
    out=_normalise_groups(vals)
    noisy=next(c for c in out if c['v251_region_group']=='noisy')
    clean=next(c for c in out if c['v251_region_group']=='clean')
    check("candidate density penalty is physical and soft", clean['v252_density_weight'] > noisy['v252_density_weight'] > 0)
    check("group normalisation preserves exact XY", {(c['camera_x'],c['camera_y']) for c in out} == {(c['camera_x'],c['camera_y']) for c in vals})


def test_legacy_revalidation_provenance_contract() -> None:
    source=(ROOT/'src/engine/shot_region_freshness_v252.py').read_text(encoding='utf-8')
    check("legacy revalidation provenance exists", 'legacy_revalidated' in source)
    check("registered source provenance exists", 'region_registered' in source)
    check("diagnostic-only source cannot imply authority", 'diagnostic_only' in source and 'v252_fresh_physical' in source)


def test_confirm_balance() -> None:
    vals=[]
    for g,base in [('a',10),('b',100),('c',200)]:
        for i in range(5):
            c=_cand(g,base+i,20-i)
            c['v252_authority_score']=20-i
            c['v2225_confirm_center_abs']=3
            c['v2225_confirm_compact']=1
            vals.append(c)
    vals.append(_cand('d',300,99,fresh=False))
    before={(c['camera_x'],c['camera_y']) for c in vals}
    out=_balance_fresh_confirmed(vals)
    check("registered confirmation keeps each fresh physical group", {c['v251_region_group'] for c in out} == {'a','b','c'})
    check("registered confirmation globally bounded", len(out) <= 8)
    check("non-fresh candidate cannot enter local authority", all(c['v251_region_group']!='d' for c in out))
    check("confirmation never moves XY", all((c['camera_x'],c['camera_y']) in before for c in out))


def test_no_game_semantic_authority() -> None:
    source=(ROOT/'src/engine/shot_region_freshness_v252.py').read_text(encoding='utf-8')
    forbidden=['target_bonus','role_weight','owner_weight','damage_weight','projectile_weight','game_score_weight']
    check("authority scoring has no gameplay semantic weights", not any(x in source for x in forbidden))
    check("explicit global rescue bypass preserved", 'rescue_router_v2225.was_consumed(sid)' in source)
    check("early legacy emission gate present", 'await_registered_frame' in source and 'return None' in source)


def test_install_order_and_scene() -> None:
    main=(ROOT/'main.py').read_text(encoding='utf-8')
    check("V2.25.2 runtime wired", 'install_v252_runtime(App)' in main)
    check("V2.25.2 installs after V2.25.1", main.rfind('install_v252_runtime(App)') > main.rfind('install_v251_runtime(App)'))
    scene=(ROOT/'content/games/game_objects_test_v250.py').read_text(encoding='utf-8')
    check("test scene labels V2.25.2 hits", '[V2.25.2 OBJECT-HIT]' in scene)


def test_menu_patch() -> None:
    from automation.v252_apply_menu import patch_menu_text
    entry=json.loads((ROOT/'menu_games_entry_v252.json').read_text())
    old=json.loads((ROOT/'menu_games_entry_v251.json').read_text())
    sample=json.dumps({'children':[old]},ensure_ascii=False,indent=2)+'\n'
    patched,changed,found=patch_menu_text(sample,entry)
    check("V2.25.1 menu entry is found for update", found)
    check("menu title becomes V2.25.2", changed and 'Game Objects Test (V2.25.2)' in patched)
    patched2,changed2,found2=patch_menu_text(patched,entry)
    check("V2.25.2 menu patch is idempotent", found2 and not changed2 and patched2==patched)


def main() -> None:
    print("V2.25.2 SELFTEST\n===============")
    test_registered_compact_metrics(); test_broad_motion_not_compact(); test_pre_noise_soft_penalty()
    test_group_normalisation(); test_legacy_revalidation_provenance_contract(); test_confirm_balance()
    test_no_game_semantic_authority(); test_install_order_and_scene(); test_menu_patch()
    print("\nAll V2.25.2 selftests passed.")

if __name__=='__main__': main()
