from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from src.engine.ai.training_v223.patch_v2234 import (
    PATCH_CHANNELS, PATCH_SIZE, _extract_patches_from_channels, _make_channel_images,
)
from src.engine.ai.training_v223.patch_model_v2234 import (
    PatchModelV2234, SampledPatchShotV2234, _init_model, _pair_ds, _grads,
)


def _assert(cond, msg):
    if not cond: raise AssertionError(msg)
    print(f"[PASS] {msg}")


def _synthetic_patch_dataset(seed=7):
    rng=np.random.default_rng(seed)
    shots=[]
    for s in range(24):
        nneg=40
        pos=np.zeros((5,PATCH_CHANNELS,PATCH_SIZE,PATCH_SIZE),dtype=np.uint8)
        neg=np.zeros((nneg,PATCH_CHANNELS,PATCH_SIZE,PATCH_SIZE),dtype=np.uint8)
        base=rng.integers(70,190,size=(PATCH_SIZE,PATCH_SIZE),dtype=np.uint8)
        for i in range(len(pos)):
            pre=np.clip(base.astype(np.int16)+rng.normal(0,3,base.shape),0,255).astype(np.uint8)
            post=pre.copy(); yy=PATCH_SIZE//2+rng.integers(-1,2); xx=PATCH_SIZE//2+rng.integers(-1,2)
            post[max(0,yy-1):yy+2,max(0,xx-1):xx+2]=np.maximum(0,post[max(0,yy-1):yy+2,max(0,xx-1):xx+2].astype(np.int16)-45).astype(np.uint8)
            diff=post.astype(np.int16)-pre.astype(np.int16)
            pos[i,0]=pre;pos[i,1]=post;pos[i,2]=np.clip(np.abs(diff)*8,0,255);pos[i,3]=np.clip(128+diff*4,0,255);pos[i,4]=(np.abs(diff)>2)*255
        for i in range(nneg):
            pre=np.clip(base.astype(np.int16)+rng.normal(0,5,base.shape),0,255).astype(np.uint8)
            post=np.clip(pre.astype(np.int16)+rng.normal(0,2,base.shape),0,255).astype(np.uint8)
            if i%3==0:
                y=rng.integers(1,PATCH_SIZE-2);x=rng.integers(1,PATCH_SIZE-2);post[y:y+2,x:x+2]=np.maximum(0,post[y:y+2,x:x+2].astype(np.int16)-25).astype(np.uint8)
            diff=post.astype(np.int16)-pre.astype(np.int16)
            neg[i,0]=pre;neg[i,1]=post;neg[i,2]=np.clip(np.abs(diff)*8,0,255);neg[i,3]=np.clip(128+diff*4,0,255);neg[i,4]=(np.abs(diff)>2)*255
        patches=np.concatenate([pos,neg]);dist=np.concatenate([np.zeros(len(pos),np.float32),np.full(nneg,80,np.float32)])
        shots.append(SampledPatchShotV2234(f"s{s//8}",f"{s}",patches,dist))
    return shots


def _tiny_train(samples, kind='tiny_cnn', epochs=8, seed=9):
    model=_init_model(kind,seed=seed,hidden=20,filters=4);rng=np.random.default_rng(seed+1)
    m={k:np.zeros_like(v) for k,v in model.arrays.items()};vv={k:np.zeros_like(v) for k,v in model.arrays.items()};step=0
    for _ in range(epochs):
        for si in rng.permutation(len(samples[:18])):
            sm=samples[int(si)];scores=model._forward(sm.patches);pair=_pair_ds(scores,sm.distances)
            if pair is None:continue
            ds,_=pair;gs=_grads(model,sm.patches,ds,l2=0.0005);step+=1
            for k,g in gs.items():
                m[k]=.9*m[k]+.1*g;vv[k]=.999*vv[k]+.001*g*g
                model.arrays[k]-=.002*(m[k]/(1-.9**step))/(np.sqrt(vv[k]/(1-.999**step))+1e-7)
    return model


def main()->int:
    pre=np.full((80,96),160,np.uint8);post=pre.copy();post[39:42,47:50]=70
    ch=_make_channel_images(pre,[post,post]);xy=np.asarray([[48,40],[10,10]],np.float32);patches=_extract_patches_from_channels(ch,xy)
    _assert(patches.shape==(2,PATCH_CHANNELS,PATCH_SIZE,PATCH_SIZE),'candidate patch tensor contract is stable')
    _assert(float(patches[0,2].mean())>float(patches[1,2].mean()),'PRE/POST diff remains visible in candidate-centred patch')
    samples=_synthetic_patch_dataset();model=_tiny_train(samples)
    correct=0
    for sm in samples[18:]:
        order=model.rank_indices(sm.patches);correct+=int(int(order[0])<5)
    _assert(correct>=4,'tiny learned patch model ranks synthetic NEW-hole patch first on held-out shots')
    _assert(model.metadata.get('live_authority',False) is False if model.metadata else True,'selftest path grants no live authority')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);model.metadata={'live_authority':False};model.save(p);loaded=PatchModelV2234.load(p)
        a=model.score_patches(samples[-1].patches[:7]);b=loaded.score_patches(samples[-1].patches[:7])
        _assert(np.allclose(a,b,atol=1e-5),'patch model NPZ+JSON round-trip is deterministic and pickle-free')
    print('\nAll V2.23.4 selftests passed.')
    return 0
if __name__=='__main__': raise SystemExit(main())
