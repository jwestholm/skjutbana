"""Forensic replay of frozen evidence collisions; no proposal/label changes."""
import argparse
import csv
from pathlib import Path
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_fusion as fusion
from automation import physical_ai_selector_gap as gap,physical_ai_event_selector as selector
from src.engine.offline.common_verifier import camera_local_patches,query_physical_witnesses
from src.engine.offline.clean_physical_change import local_contrast_batch,_LOCAL_MASKS

AUDIT=Path('/data/skjutbana/evaluation_runs/selector_gap_20261007_01/audit.json')
EVENTS=('KOMP:1','KOMP:5','POST_FIX:3','S02:13')


def forensic(output):
    output=ds.output_root(output);output.mkdir(exist_ok=True,parents=True)
    audit=ds.read(AUDIT);base,x=fusion.load(fusion.DEFAULT)
    identity=dict(audit_hash=ds.digest(AUDIT),dataset=base['dataset_hash'],code=ds.digest(__file__),events=list(EVENTS))
    allrows=[];pairs=[];source_hashes={};records=[]
    for event in EVENTS:
        saved=output/(event.replace(':','_')+'.json')
        if saved.exists():
            record=ds.read(saved)
            if record['identity']!=identity:raise ValueError('Forensic identity changed')
        else:
            a=next(e for e in audit['gaps'] if e['event']==event);e=next(e for e in base['events'] if e['event']==event)
            good=a['good_index'];wrong=a['exact_collisions'][str(good)];indices=[good]+wrong
            candidates=[a['candidates'][i] for i in indices];points=[c['xy'] for c in candidates]
            frames=candidates[0]['provenance']['witness']['frames']
            if frames!=e['frames']:raise ValueError('Frozen frame prefix changed')
            measured=[];core=[];ring=[];hashes={};shape=None
            for f in frames:
                path=f['frame_id'];actual=ds.digest(path)
                if actual!=base['source_hashes'][path]:raise ValueError('Source pixels changed')
                hashes[path]=actual;im=np.load(path,mmap_mode='r',allow_pickle=False);shape=list(im.shape)
                patches=camera_local_patches(im,points);measured.append(local_contrast_batch(patches[None])[0])
                core.append(patches[:,_LOCAL_MASKS[0]].astype(np.float64).mean(1));ring.append(patches[:,_LOCAL_MASKS[1]].astype(np.float64).mean(1))
            measured=np.asarray(measured);previous=[None]*len(points);prefix=[]
            for n in range(1,len(frames)+1):
                ids=[dict(candidate_id=c['id'],coordinate_version=c['id'],camera_xy=c['xy'],source_provenance=c['provenance']['witness']['source_provenance'],previous=previous[j]) for j,c in enumerate(candidates)]
                bundles=query_physical_witnesses(measured[:n],frames[:n],event_id=event,candidates=ids,cutoff_frame_id=frames[n-1]['frame_id'],cutoff_ts=frames[n-1]['receipt_timestamp'])
                previous=[b.witnesses[0] for b in bundles]
                prefix.append([dict(state=w['state'],measurements=w['measurements'],revocation=w['revocation']) for w in previous])
            rows=[]
            for j,c in enumerate(candidates):
                old=c['provenance']['witness'];w=previous[j];v=ds.vector(dict(physical_witness=w));raw=np.array([c['values'][n] for n in ds.FEATURE_NAMES])
                if not np.array_equal(measured[:,j],old['signal']):raise ValueError('Source signal not reproduced exactly')
                if w!=old:raise ValueError('Witness lifecycle not reproduced exactly')
                if not np.array_equal(np.asarray(v,np.float32),raw):raise ValueError('Frozen cache not reproduced')
                first=next((n for n,p in enumerate(prefix) if p[j]['state']=='SUPPORTED'),None)
                rows.append(dict(candidate=c,signal=measured[:,j].tolist(),core=np.asarray(core)[:,j].tolist(),ring=np.asarray(ring)[:,j].tolist(),
                    prefix=[p[j] for p in prefix],first_supported_ordinal=None if first is None else frames[first]['ordinal'],
                    supported_prefix_count=sum(p[j]['state']=='SUPPORTED' for p in prefix),revocation_count=sum(p[j]['revocation']['revoked_now'] for p in prefix),
                    cache=v,projected=selector.relative_features(gap.physical_matrix(np.asarray(v,np.float32)[None]))[0].tolist()))
            # Actual event-relative representation (all candidates, same frozen event).
            projection=selector.relative_features(gap.physical_matrix(x[e['indices']]))
            for row,i in zip(rows,indices):row['projected']=projection[i].tolist()
            record=dict(identity=identity,event=event,frames=frames,shape=shape,source_hashes=hashes,rows=rows,registration='NOT_APPLIED_IN_THIS_WITNESS_PATH',source_pixels_signal_witness_cache_exact=True)
            ds.atomic_json(saved,record)
        records.append(record);source_hashes.update(record['source_hashes']);g=record['rows'][0]
        for j,row in enumerate(record['rows']):
            for i,f in enumerate(record['frames']):
                p=row['prefix'][i]
                allrows.append(dict(event=event,candidate_id=row['candidate']['id'],role='AUDIT_GOOD' if j==0 else 'AUDIT_WRONG',ordinal=f['ordinal'],frame_id=f['frame_id'],receipt=f['receipt_timestamp'],camera_sequence=f['camera_sequence'],core=row['core'][i],ring=row['ring'][i],contrast=row['signal'][i],state=p['state'],raw_state=p['measurements']['state'],support=p['measurements']['support'],choices=len(p['measurements']['choices']),revoked=p['revocation']['revoked_now']))
            if j:
                a=np.array(g['signal']);b=np.array(row['signal']);m0=g['prefix'][-1]['measurements'];m1=row['prefix'][-1]['measurements']
                pairs.append(dict(event=event,good_xy=g['candidate']['xy'],good_error=g['candidate']['error'],wrong_xy=row['candidate']['xy'],wrong_error=row['candidate']['error'],
                    core_equal=g['core']==row['core'],ring_equal=g['ring']==row['ring'],signal_equal=g['signal']==row['signal'],max_abs_signal_difference=float(np.max(abs(a-b))),
                    final_measurements_equal=m0==m1,bundle_signal_equal=False,cache_equal=g['cache']==row['cache'],projection_equal=g['projected']==row['projected'],
                    last_distinct='EvidenceBundle.signal',first_equal_measurement_branch=('temporal_step_evidence unsupported/ambiguous measurements' if m0==m1 else 'physical_ai_dataset.vector omits distinct raw onset states'),first_irrecoverable_selector_stage='physical_ai_dataset.vector drops signal',
                    raw_final_state=m0['state'],good_supported_prefixes=g['supported_prefix_count'],wrong_supported_prefixes=row['supported_prefix_count'],good_revocations=g['revocation_count'],wrong_revocations=row['revocation_count'],certification=m0['certification']))
    for filename,rows in [('transitions.csv',allrows),('loss_ledger.csv',pairs)]:
        with (output/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ds.atomic_json(output/'forensics.json',dict(identity=identity,pairs=pairs,source_hashes=source_hashes))
    return pairs


def main():
    ds.install_guard();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for r in forensic(a.output):print(r['event'],r['good_xy'],r['wrong_xy'],'cache identical',r['cache_equal'],'raw delta',round(r['max_abs_signal_difference'],3),'prefix support',r['good_supported_prefixes'],r['wrong_supported_prefixes'])

if __name__=='__main__':main()
