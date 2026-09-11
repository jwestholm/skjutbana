"""Read-only common-evidence rank replay over a complete eligible pool."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def methods(x):
    f=x.get('image_features',{})
    return {'raw':x['track'].get('best_score',0.0),'compact':f.get('ring_affine_compact',float('-inf')),
            'dark':f.get('ring_affine_dark_contrast',float('-inf')),'center_dark':f.get('ring_affine_center_dark',float('-inf')),
            'concentration':f.get('ring_affine_concentration',float('-inf'))}

def run(inp,out):
    d=json.loads(inp.read_text()); by={}
    for x in d['tracks']: by.setdefault(int(x['shot']),[]).append(x)
    result=[]
    for sid,pool in sorted(by.items()):
        pool=[x for x in pool if x['track'].get('eligible')]
        for name in methods(pool[0]).keys() if pool else []:
            order=sorted(pool,key=lambda x:methods(x)[name],reverse=True)
            for rank,x in enumerate(order,1): result.append(dict(shot=sid,method=name,rank=rank,track_id=x['track']['track_id'],source=x['track'].get('source'),value=methods(x)[name]))
    out.write_text(json.dumps(dict(status='RESEARCH_ONLY',methods=sorted(set(r['method'] for r in result)),rows=result),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
