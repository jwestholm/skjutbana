"""Package RAW natural Blender outputs and nine rooms, with no color correction.

python3 -m automation.build_pang_natural MODELS EXTRA_MODELS ROOMS NEW_RAW NEW_ASSETS
All output directories must be new; R7 baseline remains untouched.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
from automation.build_light_room_parts import build


def package(models,extra_models,rooms,raw,output):
    models,extra_models,rooms,raw,output=map(Path,(models,extra_models,rooms,raw,output))
    raw.mkdir(parents=True,exist_ok=False);meta=None
    for name in ('vase','jar','bowl','bottle','lamp','tv','books','plant','clock','mug','frame'):
        root=extra_models/('tv_on' if name=='tv' else name) if name in ('tv','frame') else models/name
        info=json.loads((root/'parts.json').read_text())
        if not info.get('natural_materials'):raise ValueError('Not a natural export: '+str(root))
        if meta is None:meta={**info,'objects':{}}
        spec=info['objects'][name]
        if len(spec['parts'])<1 or any(len(p['frames'])!=12 for p in spec['parts']):raise ValueError('Incomplete '+name)
        meta['objects'][name]=spec
        for p in root.glob(name+'*.png'):shutil.copyfile(p,raw/p.name)
    asset_rows={}
    for base in ('content/games/pang_polyhaven_sources','content/games/pang_polyhaven_r8_sources'):
        for row in json.loads((Path(base)/'manifest.json').read_text())['assets']:asset_rows[row['id']]=row
    used={n for spec in meta['objects'].values() for n in spec.get('source_assets',())}
    used|={'Sofa_01','CoffeeTable_01','side_table_01','Shelf_01','metal_jug','modern_coffee_table_01','ceramic_vase_01'}
    meta['asset_provenance']=dict(license_url='https://polyhaven.com/license',assets=[asset_rows[n] for n in sorted(used)])
    (raw/'parts.json').write_text(json.dumps(meta,indent=2))
    for n in ('room.png','layout.json'):shutil.copyfile(rooms/'living_0'/n,raw/n)
    m=build(raw,output,realistic=True);m['rooms']=[]
    for template,label in (('living','Vardagsrum'),('kitchen','Kök'),('study','Arbetsrum')):
        for n in (0,1,2):
            name=template+'_'+str(n);src=rooms/name;file='room_'+name+'.png';shutil.copyfile(src/'room.png',output/file)
            kinds={'living':list(m['objects']),'kitchen':['mug','jar','bowl','bottle','plant','clock','vase'],
                'study':['books','mug','clock','lamp','frame','plant','vase','jar','bottle']}[template]
            repeats={'living':['mug','frame','books'],'kitchen':['mug','jar','bowl','bottle'],'study':['books','mug','frame']}[template]
            m['rooms'].append(dict(id=name,template=template,label=label,furnishing=n,image=file,
                anchors=json.loads((src/'layout.json').read_text())['anchors'],kinds=kinds,repeat_kinds=repeats,
                raw_sha256={n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in ('room.png','layout.json')}))
    m.update(natural_materials=True,precision_mode='object_glow',room_plate='natural_Cycles_no_lift_no_floor_no_glazes',
        textures=json.loads(Path('content/games/pang_room_textures/SOURCES.json').read_text()))
    m['recipe_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'automation/build_pang_natural.py','automation/build_light_room_parts.py','automation/assets/render_light_room_parts.py',
        'automation/assets/pang_rooms_r8.py','automation/assets/pang_polyhaven.py','automation/assets/pang_texture_crops.py')}
    m['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.png'))}
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    with (output/'SOURCES.md').open('a') as f:f.write('\nR8 natural mode: no RGB floor, display gain or palette recoloring. Original PBR/UVs\nplus authored ceramic mug/room/kitchen/desk geometry. Modern TV screen has an\nauthored static gradient image; parts fracture only after measurement.\nArchitectural bitmap fallback: CC0 atlas crops; exact crops/hashes in\ncontent/games/pang_room_textures/SOURCES.json. Dedicated architectural PBR\nsets still requested in GAME_NEW_REPORT.md.\n')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','extra_models','rooms','raw','output'):p.add_argument(name)
    a=p.parse_args();m=package(a.models,a.extra_models,a.rooms,a.raw,a.output);print('Natural objects',len(m['objects']),'rooms',len(m['rooms']))
