"""CC0 bitmap fallback from supplied model atlases, with exact crop provenance.

These are repurposed model scans, NOT downloaded dedicated plaster/tile PBR sets.
Use the report's requested texture list for a later room-material upgrade.
"""
from pathlib import Path
import hashlib,json
from PIL import Image

def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=Path('content/games/pang_polyhaven_sources');extra=Path('content/games/pang_polyhaven_r8_sources')
    crops=[('oak',base/'modern_coffee_table_01/textures/modern_coffee_table_01', (811,150,845,590)),
           ('stone',base/'modern_coffee_table_01/textures/modern_coffee_table_01',(45,780,180,900)),
           ('plaster',base/'ceramic_vase_01/textures/ceramic_vase_01',(64,200,475,950)),
           ('fabric',extra/'Sofa_01/textures/Sofa_01',(300,530,700,680))]
    rows=[]
    for name,prefix,box in crops:
        maps={}
        for source_kind,dest_kind in [('diff','color'),('nor_gl','normal'),('rough','rough'),('arm','arm')]:
            p=Path(str(prefix)+'_'+source_kind+'_1k.jpg')
            if not p.exists():continue
            dst=out/(name+'_'+dest_kind+'.png');Image.open(p).crop(box).save(dst)
            maps[dest_kind]=dict(file=dst.name,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                crop_xyxy=box,sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
        rows.append(dict(id=name,license='CC0-1.0',url='https://polyhaven.com/a/'+prefix.parent.parent.name,maps=maps))
    (out/'SOURCES.json').write_text(json.dumps(dict(derivation='UV atlas crops, not new standalone texture downloads',assets=rows),indent=2))
    (out/'LICENSE.txt').write_text('All bitmap inputs and derived crops: Poly Haven CC0 1.0.\nhttps://polyhaven.com/license\nhttps://creativecommons.org/publicdomain/zero/1.0/\nExact sources/hashes/crops: SOURCES.json.\n')
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();build(a.out)
