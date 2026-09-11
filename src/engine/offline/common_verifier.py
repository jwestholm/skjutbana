"""Research-only source-independent impact verifier primitives."""
from __future__ import annotations
from dataclasses import dataclass
import cv2,numpy as np
from .registered_impact import sample

@dataclass
class CommonFrameContext:
    pre: np.ndarray
    post: tuple[np.ndarray,...]
    cutoff_ts: float
    timestamps: tuple[float,...]
    origin: tuple[int,int]=(0,0)

def extract_patch_context(ctx: CommonFrameContext, xy, *, causal_only=True, radius=16):
    pairs=[]; pre=sample(ctx.pre,xy,ctx.origin,radius)
    for img,ts in zip(ctx.post,ctx.timestamps):
        if causal_only and ts>ctx.cutoff_ts: continue
        post=sample(img,xy,ctx.origin,radius); signed=pre.astype(np.float32)-post.astype(np.float32)
        pairs.append(dict(timestamp=float(ts),pre=pre,post=post,signed=signed,absolute=np.abs(signed)))
    return dict(pre=pre,pairs=tuple(pairs),causal_only=bool(causal_only))

def patch_representation(bundle, kind='difference'):
    if not bundle['pairs']: return None
    stack=np.stack([p['signed'] for p in bundle['pairs']])
    if kind=='difference': return stack[0].astype(np.float32).ravel()
    if kind=='median_difference': return np.median(stack,axis=0).astype(np.float32).ravel()
    if kind=='gradient':
        gy,gx=np.gradient(stack[0]); return np.hypot(gx,gy).astype(np.float32).ravel()
    if kind=='pca':
        v=stack[0].astype(np.float32); small=cv2.resize(v,(8,8),interpolation=cv2.INTER_AREA); return small.ravel()
    if kind=='texture':
        v=stack[0].astype(np.float32); gy,gx=np.gradient(v); return np.asarray([v.mean(),v.std(),np.abs(v).mean(),np.hypot(gx,gy).mean(),np.percentile(v,10),np.percentile(v,90)],np.float32)
    raise ValueError(kind)

class CommonVerifier:
    """Interface only; disabled unless explicitly invoked by research code."""
    enabled=False
    def verify_event(self, tracks, context):
        if not self.enabled: return []
        return [self.score_track(t,context) for t in tracks]
    def score_track(self, track, context): raise NotImplementedError
