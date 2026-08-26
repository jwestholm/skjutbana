from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214, HolePatchAIConfigV214
from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleConfigV215
from src.engine.ai.new_hole_ai_v217 import NewHoleAIConfigV217, NewHoleAIV217
from src.engine.offline.candidate_pack_v216 import CandidateCaptureConfigV216, CandidatePackV216, CandidateShadowRecorderV216
from src.engine.offline.hole_training_v213 import roc_auc
from src.engine.offline.new_hole_training_v217 import TrainingConfigV217, discover_samples_v217


def _pass(message: str) -> None:
    print(f"[PASS] {message}")


def _draw_dot(image: np.ndarray, xy: tuple[int,int], radius: int=3, value: int=25) -> None:
    cv2.circle(image,xy,radius,int(value),-1,lineType=cv2.LINE_AA)
    cv2.circle(image,xy,radius+2,min(255,int(value)+45),1,lineType=cv2.LINE_AA)


def _make_ensemble(root: Path) -> Path:
    cfg=HolePatchAIConfigV214(crop_size=64,input_size=18,hidden_size=24)
    standard=HolePatchAIV214(cfg,seed=11); mild=HolePatchAIV214(cfg,seed=12)
    std_path=root/'standard.npz'; mild_path=root/'mild.npz'; standard.save(std_path,metadata={'threshold':0.5}); mild.save(mild_path,metadata={'threshold':0.5})
    payload={'ensemble_config':HolePatchEnsembleConfigV215(standard_model_path=str(std_path),mild_model_path=str(mild_path),standard_weight=0.5,fused_threshold=0.5,standard_threshold=0.5,mild_threshold=0.5,shadow_only=True).__dict__}
    path=root/'ensemble.json'; path.write_text(json.dumps(payload,indent=2),encoding='utf-8'); return path


def _test_temporal_learning() -> None:
    rng=np.random.default_rng(217); model=NewHoleAIV217(NewHoleAIConfigV217(input_size=16,hidden_size=40),seed=217)
    pairs=[]; labels=[]; offsets=[]
    for i in range(180):
        base=np.clip(175+rng.normal(0,4,size=(64,64)),0,255).astype(np.uint8); pre=base.copy(); post=base.copy()
        if i%2==0:
            dx=int(rng.integers(-8,9)); dy=int(rng.integers(-8,9)); _draw_dot(post,(32+dx,32+dy),radius=3,value=24); labels.append(1); offsets.append((dx,dy))
        else:
            mode=i%6
            if mode==1:
                ox=int(rng.integers(20,45)); oy=int(rng.integers(20,45)); _draw_dot(pre,(ox,oy),radius=3,value=24); post=pre.copy()
            elif mode==3:
                post=np.clip(pre.astype(np.int16)+int(rng.integers(-18,19)),0,255).astype(np.uint8)
            else:
                x0=int(rng.integers(8,48)); cv2.rectangle(post,(x0,0),(min(63,x0+9),63),int(rng.integers(120,235)),-1)
            labels.append(0); offsets.append((0,0))
        pairs.append((pre,post,[post]))
    y=np.asarray(labels,dtype=np.float32); off=np.asarray(offsets,dtype=np.float32); features=model.feature_batch(pairs)
    for _ in range(28):
        order=rng.permutation(len(y))
        for start in range(0,len(order),48):
            idx=order[start:start+48]; model.train_batch(features[idx],y[idx],off[idx])
    probs,pred_off=model.predict_features(features); auc=roc_auc(y.astype(np.int32),probs)
    if auc is None or auc<0.95: raise AssertionError(f'temporal learning AUC too low: {auc}')
    pos=y>0.5; median_offset=float(np.median(np.linalg.norm(pred_off[pos]-off[pos],axis=1)))
    if median_offset>11.0: raise AssertionError(f'offset localisation too weak: {median_offset:.2f}px')
    _pass(f'NEW-hole AI learns change while old holes/flicker/large scene changes stay negative (AUC={auc:.3f})')


def _test_pack_semantics_and_recent_pre() -> None:
    with tempfile.TemporaryDirectory(prefix='v217_selftest_') as td:
        root=Path(td); ensemble_path=_make_ensemble(root); data_root=root/'candidate_shadow'
        recorder=CandidateShadowRecorderV216(CandidateCaptureConfigV216(enabled=True,data_root=str(data_root),patch_size=64,max_post_frames=2,max_candidates=16),background='white',benchmark_seed=123,sampling_mode='uniform')
        pre=np.full((128,128),180,dtype=np.uint8); _draw_dot(pre,(98,98),radius=3,value=20)
        legacy_pre=np.full_like(pre,235); post=pre.copy(); _draw_dot(post,(30,30),radius=3,value=20)
        raw=[{'camera_x':31.0,'camera_y':30.0,'score':8.0},{'camera_x':98.0,'camera_y':98.0,'score':9.0,'near_known_hole_dist':1.2},{'camera_x':110.0,'camera_y':20.0,'score':7.0}]
        result=recorder.capture_shot(round_id=1,raw_candidates=raw,ranked_candidates=list(raw),pre_gray=legacy_pre,recent_pre_gray=pre,recent_pre_timestamp=0.90,post_gray=post,post_frames=[(post,1.0),(post,1.1)],gt_camera_xy=(30.0,30.0),gt_screen_xy=(100.0,100.0),extra_metadata={'known_holes_before_shot':[{'camera_x':98.0,'camera_y':98.0}]})
        recorder.finalize()
        if not result.get('saved'): raise AssertionError(result)
        pack=CandidatePackV216.load(Path(result['json_path']))
        if pack.recent_pre_patches is None or pack.gt_recent_pre_patch is None: raise AssertionError('recent-pre capture extension missing')
        if abs(float(pack.recent_pre_timestamp or 0)-0.90)>1e-6: raise AssertionError('recent-pre timestamp did not roundtrip')
        manifest=root/'hard.jsonl'; manifest.write_text(json.dumps({'source_pack':result['json_path'],'capture_index':1,'label':None,'new_hole_label':0,'static_hole_label':None})+'\n',encoding='utf-8')
        samples,summary=discover_samples_v217(data_root,ensemble_path=ensemble_path,hardnegative_manifest=manifest,config=TrainingConfigV217(max_negatives_per_shot=4))
        positives=[s for s in samples if s.label==1 and s.source=='gt_patch']; old=[s for s in samples if s.label==0 and s.capture_index==1]
        if not positives: raise AssertionError('missing NEW-hole positive')
        # Recent-pre (around 180) must win over deliberately bright legacy reference (235).
        if float(np.median(positives[0].pre))>210.0: raise AssertionError('V2.17 did not prefer true recent-pre data')
        if not old: raise AssertionError('old-hole candidate not loaded as NOT-NEW')
        if 'does not mean non-hole' not in summary['semantic_note']: raise AssertionError('semantic contract missing')
        _pass('candidate far from GT may be an old hole; loader labels only NOT-NEW, never static non-hole')
        _pass('future candidate packs preserve true recent-pre patches/timestamp without changing legacy pre data')


def _test_save_load() -> None:
    with tempfile.TemporaryDirectory(prefix='v217_model_') as td:
        path=Path(td)/'model.npz'; model=NewHoleAIV217(NewHoleAIConfigV217(input_size=14,hidden_size=20),seed=5)
        model.save(path,metadata={'shadow_only':True,'semantic_contract':{'negative':'NOT NEW'}}); loaded,meta=NewHoleAIV217.load(path)
        if loaded.config.input_dim!=model.config.input_dim: raise AssertionError('config changed after save/load')
        if meta.get('metadata',{}).get('shadow_only') is not True: raise AssertionError('shadow flag missing')
        _pass('V2.17 model save/load preserves shadow-only semantic metadata')


def main() -> int:
    print('V2.17 SELFTEST'); print('=============='); _test_temporal_learning(); _test_pack_semantics_and_recent_pre(); _test_save_load(); print('\nAll V2.17 selftests passed.'); return 0
if __name__=='__main__': raise SystemExit(main())
