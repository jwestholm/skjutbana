"""Copy only the selected local CC0 glTF dependencies; never alters the source depot."""
import argparse,hashlib,json,shutil
from pathlib import Path
from urllib.parse import unquote
ASSETS=('sofa_02','modern_coffee_table_01','Shelf_01','side_table_01','fancy_picture_frame_01',
        'Television_01','book_encyclopedia_set_01','ceramic_vase_01','ceramic_vase_02','ceramic_vase_03',
        'ceramic_pot','potted_plant_02','alarm_clock_01','desk_lamp_arm_01','metal_jug')

def stage(source,destination):
    source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=False);rows=[]
    for name in ASSETS:
        root=source/name;gltf=next(root.glob('*.gltf'));g=json.loads(gltf.read_text())
        files=[gltf]
        for entry in g.get('buffers',[])+g.get('images',[]):
            uri=entry.get('uri','')
            if uri.startswith('data:'):continue
            p=(root/unquote(uri)).resolve()
            if not p.is_relative_to(root.resolve()):raise ValueError('External glTF URI')
            files.append(p)
        hashes={}
        for p in dict.fromkeys(files):
            relative=p.relative_to(root);out=destination/name/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
            hashes[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(dict(id=name,url='https://polyhaven.com/a/'+name,license='CC0-1.0',files_sha256=hashes))
    (destination/'manifest.json').write_text(json.dumps(dict(schema='pang-polyhaven-sources-1',license_url='https://polyhaven.com/license',assets=rows),indent=2)+'\n')
    (destination/'SOURCES.md').write_text('# PANG! selected Poly Haven assets\n\nAll selected originals are CC0 1.0: https://polyhaven.com/license .\nSource depot is read-only; only referenced glTF 1k files are copied here.\nRuntime loads rendered sprites, not these models. Render variants are documented\nin automation/assets/pang_polyhaven.py and the sprite manifest.\n\n'+''.join('- '+r['id']+': '+r['url']+' — CC0-1.0\n' for r in rows))
    (destination/'LICENSE.txt').write_text('Poly Haven models and textures: CC0 1.0 Universal.\nhttps://creativecommons.org/publicdomain/zero/1.0/\nhttps://polyhaven.com/license\n')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination');a=p.parse_args();rows=stage(a.source,a.destination);print('Selected:',len(rows))
