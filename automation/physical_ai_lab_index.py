"""Incremental, rebuildable Evidence-lab index; immutable trial files own results.

SQLite transactions cover champion/count/journal changes together. Public JSON
views are bounded; an interrupted view write never controls resume or status.
"""
import json
import sqlite3
import time
from pathlib import Path
from automation import physical_ai_dataset as ds

NAME='trial_index.sqlite3'
LIMIT=100


def connect(root,readonly=False):
    path=Path(root)/NAME;ds.guard(path)
    if readonly:return sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(path);db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS trials (number INTEGER PRIMARY KEY, trial TEXT UNIQUE, status TEXT, selected5 INTEGER, worst REAL, summary TEXT NOT NULL, failures TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS trial_ranking ON trials(status,selected5 DESC,worst DESC,number)')
    return db


def state(db):
    row=db.execute('SELECT value FROM state WHERE id=1').fetchone()
    return json.loads(row[0]) if row else None


def save(db,value):db.execute('INSERT OR REPLACE INTO state VALUES(1,?)',(json.dumps(value,allow_nan=False),))


def read_state(root):
    with connect(root,True) as db:return state(db)


def compact(result,accepted,reason):
    metrics=result.get('metrics',{})
    return dict(trial=result['trial'],status=result['status'],accepted=accepted,reason=reason,
                finished_at=result['finished_at'],metrics={k:v for k,v in metrics.items() if k not in ('per_event','per_session')},
                result=f"trials/{result['trial']}/result.json")


def ingest(root,db,result,data,compare):
    current=state(db);number=int(result['trial'].split('_')[-1])
    if number<current['processed_trials']:return current  # Already committed, idempotent recovery.
    if number!=current['processed_trials']:raise ValueError('Noncontiguous trial journal')
    win=False;why=result.get('error','trial failed')
    if result['status']=='COMPLETE':
        win,why=compare(result['metrics'],current['metrics']);current['completed_trials']+=1
        if win:
            entry=dict(trial=result['trial'],reason=why,metrics=result['metrics'],model_directory=str(root/'trials'/result['trial']),dataset_hash=data['dataset_hash'],live_authority=False)
            path=root/'champions'/(result['trial']+'.json')
            if path.exists() and ds.read(path)!=entry:raise ValueError('Immutable champion changed')
            if not path.exists():ds.atomic_json(path,entry)
            current.update(champion=result['trial'],metrics=result['metrics'],history=current['history']+[result['trial']],champion_finished_at=result['finished_at'])
    item=compact(result,win,why);m=result.get('metrics',{})
    failures=[r['event'] for r in m.get('per_event',[]) if r['truth']=='HIT' and (r['error'] is None or r['error']>5)]
    db.execute('INSERT INTO trials VALUES(?,?,?,?,?,?,?)',(number,result['trial'],result['status'],m.get('selected',{}).get('5',-1),m.get('worst_session_selected5') or 0,json.dumps(item,allow_nan=False),json.dumps(failures)))
    current['processed_trials']+=1;save(db,current)
    return current


def refresh(root,data,compare,*,metadata=None,initial_metrics=None,initial_champion='frozen_exact_witness_baseline'):
    root=Path(root)
    with connect(root) as db:
        current=state(db)
        if current is None:
            current=dict(champion=initial_champion,metrics=initial_metrics or ds.evaluate(data['events'],{e['event']:e['baseline_xy'] for e in data['events']}),
                         history=[],live_authority=False,repeated_cv_is_development=True,processed_trials=0,completed_trials=0,dataset_hash=data['dataset_hash'],**(metadata or {}))
            save(db,current)
        if current['dataset_hash']!=data['dataset_hash']:raise ValueError('Index belongs to another dataset')
        db.commit()
        # One-time streaming migration when no index exists; otherwise just the
        # durable result(s) written after the last committed index transaction.
        while True:
            path=root/'trials'/f"trial_{current['processed_trials']:06d}"/'result.json'
            if not path.is_file():break
            with db:current=ingest(root,db,ds.read(path),data,compare)
        top=[json.loads(r[0]) for r in db.execute("SELECT summary FROM trials WHERE status='COMPLETE' ORDER BY selected5 DESC,worst DESC,number LIMIT 20")]
        recent=[json.loads(r[0]) for r in db.execute('SELECT summary FROM trials ORDER BY number DESC LIMIT ?',(LIMIT,))]
        failed=[set(json.loads(r[0])) for r in db.execute("SELECT failures FROM trials WHERE status='COMPLETE' ORDER BY selected5 DESC,worst DESC,number LIMIT 5")]
        common=sorted(set.intersection(*failed)) if failed else []
        milestones={}
        for n in (100,1000,current['processed_trials']):
            if n and current['processed_trials']>=n:
                if n==current['processed_trials']:
                    best=db.execute("SELECT selected5 FROM trials WHERE status='COMPLETE' ORDER BY selected5 DESC,worst DESC,number LIMIT 1").fetchone()
                    milestones[str(n)]=best[0] if best else None
                else:
                    milestones[str(n)]=db.execute("SELECT max(selected5) FROM trials WHERE status='COMPLETE' AND number<?",(n,)).fetchone()[0]
    # All role metadata is present in the single champion publication.
    ds.atomic_json(root/'champion.json',current)
    ds.atomic_json(root/'leaderboard.json',dict(format='bounded-sqlite-index-1',index=NAME,processed_trials=current['processed_trials'],top=top,recent=recent,limit=LIMIT,
                   history=current['history'],full_metrics='Immutable trials/<id>/result.json'))
    ds.atomic_json(root/'journal.json',dict(format='sqlite-journal-1',index=NAME,processed_trials=current['processed_trials'],completed_trials=current['completed_trials']))
    ds.atomic_json(root/'failure_wall.json',dict(completed_trials=current['completed_trials'],best_after=milestones,best_after_semantics='RAW Selected@5, not safety-filtered champion',
        champion_selected=current['metrics']['selected'],top_model_common_failures=common,top_model_count=len(failed),
        seconds_since_champion_improvement=time.time()-current['champion_finished_at'] if current.get('champion_finished_at') else None,
        metadata={e['event']:dict(session=e['session'],truth=e['truth'],status=e['status'],caveats=e.get('caveats',[])) for e in data['events'] if e['event'] in common}))
    return current


def publish(root,result,data,compare):
    with connect(root) as db:
        current=ingest(root,db,result,data,compare)
    ds.atomic_json(root/'champion.json',current)
    return current


def trial_count(root):
    path=Path(root)/NAME
    if path.is_file():return read_state(root)['processed_trials']
    journal=Path(root)/'journal.json'
    if journal.is_file():
        value=ds.read(journal)
        return len(value) if isinstance(value,list) else value['processed_trials']
    return 0
