"""Package headless Blender renders, bright palette and raster-derived collision polygons.
No live rendering/camera/projector access. Refuses to overwrite an existing asset bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def brighten(image, floor=128):
    a=np.array(image.convert('RGBA'))
    rgb=a[:,:,:3].astype(np.float32)
    a[:,:,:3]=np.rint(floor+(255-floor)*rgb/255).astype(np.uint8)
    a[:,:,1]=np.rint(192+63*rgb[:,:,1]/255).astype(np.uint8)
    return Image.fromarray(a)


def build(raw, output):
    raw,output=Path(raw),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='light-room-assets-1',renderer='Blender CPU Cycles',
                  blender_version=(raw/'blender_version.txt').read_text().strip(),
                  recipe='automation/assets/render_light_room.py + automation/build_light_room_assets.py',
                  alpha_threshold=128,collision='external alpha silhouette; <=1 source pixel boundary uncertainty',objects={})
    for name,label in [('vase','Vasen'),('jar','Krukan'),('bowl','Skålen')]:
        img=brighten(Image.open(raw/f'{name}.png'))
        a=np.array(img);mask=(a[:,:,3]>=128).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=1:raise ValueError('Expected a single solid ceramic silhouette')
        x,y,w,h=cv2.boundingRect(contours[0]);bbox=(x,y,x+w,y+h)
        points=[[(float(px)-x)/w,(float(py)-y)/h] for px,py in contours[0][:,0,:]]
        a[:,:,3]=mask;Image.fromarray(a).crop(bbox).save(output/f'{name}.png')
        Image.fromarray(mask).crop(bbox).save(output/f'{name}_mask.png')
        paths=[]
        for i in range(10):
            frame=brighten(Image.open(raw/f'{name}_break_{i:02d}.png'))
            # Baked soft dust, shown only after the query and protected POST window.
            dust=Image.new('RGBA',frame.size);draw=ImageDraw.Draw(dust)
            t=(i+1)/10
            for j in range(12):
                import math
                xx=x+w*.5+math.sin(j*2.4)*(10+30*t)
                yy=y+h*.7-40*t+math.cos(j*1.7)*16*t
                r=(5+4*t)*(1+j%3*.3)
                draw.ellipse((xx-r,yy-r,xx+r,yy+r),fill=(241,240,203,round(40*(1-t))))
            frame=Image.alpha_composite(frame,dust.filter(ImageFilter.GaussianBlur(4)))
            fn=f'{name}_break_{i:02d}.png';frame.save(output/fn);paths.append(fn)
        manifest['objects'][name]=dict(label=label,sprite=f'{name}.png',mask=f'{name}_mask.png',
                                       points=points,source_bbox=list(bbox),source_size=list(img.size),
                                       break_frames=paths,break_duration_s=1.0)
    brighten(Image.open(raw/'room.png'),floor=160).convert('RGB').save(output/'room.png')
    manifest['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.png'))}
    manifest['recipe_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('automation/assets/render_light_room.py'),Path(__file__)]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw');p.add_argument('output');a=p.parse_args()
    m=build(a.raw,a.output);print('Packaged',len(m['objects']),'objects',len(m['files_sha256']),'images')

if __name__=='__main__':main()
