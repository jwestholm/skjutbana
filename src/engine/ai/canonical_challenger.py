"""Frozen V2.23 rank model adapter. Diagnostics only; no emission/authority API."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from src.engine.ai.training_v223.model import RankModelV223
from src.engine.ai.training_v223.schema import extract_physical_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_matrix(x, transform):
    x=np.asarray(x,dtype=np.float32)
    if transform=='identity': return x
    if transform=='signed_log': return np.sign(x)*np.log1p(np.abs(x))
    if transform=='within_shot_percentile':
        return np.column_stack([np.searchsorted(np.sort(x[:,i]),x[:,i],side='left')/len(x) for i in range(x.shape[1])]).astype(np.float32) if len(x) else x
    raise ValueError('Unknown feature transform')


class CanonicalChallenger:
    """Only accepts a manifest pinning model bytes and feature preprocessing."""
    def __init__(self, manifest):
        self.path=Path(manifest); self.manifest=json.loads(self.path.read_text())
        if self.manifest.get('mode')!='SHADOW': raise ValueError('Canonical challenger requires SHADOW')
        directory=self.path.parent/self.manifest['model_directory']
        for name in ('model.json','model.npz'):
            if digest(directory/name)!=self.manifest['hashes'][name]: raise ValueError('Model hash mismatch')
        self.model=RankModelV223.load(directory)
        self.manifest_sha256=digest(self.path)

    def rank(self,candidates,top_n=10):
        features=[extract_physical_features(c) for c in candidates]
        x=np.asarray([[f[k] for k in self.model.feature_names] for f in features],dtype=np.float32).reshape(len(features),len(self.model.feature_names))
        scores=self.model.score_matrix(transform_matrix(x,self.manifest['transform']))
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite challenger scores')
        order=np.argsort(-scores,kind='stable').tolist()
        gap=float(scores[order[0]]-scores[order[1]]) if len(order)>1 else None
        return {'mode':'SHADOW','status':self.manifest['status'],'manifest_sha256':self.manifest_sha256,
                'confidence':{'score_margin':gap,'calibrated_probability':None},
                'order':order,'top_n':order[:max(0,int(top_n))],
                'candidates':[{'input_index':i,'camera_x':c['camera_x'],'camera_y':c['camera_y'],
                    'score':float(scores[i]),'features':{k:features[i][k] for k in self.model.feature_names}} for i,c in enumerate(candidates)]}
