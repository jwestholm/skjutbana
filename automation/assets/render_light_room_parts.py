"""Headless CPU Blender room and functional breakable parts; no live scene access.
blender -noaudio -b --factory-startup -P automation/assets/render_light_room_parts.py -- --out NEW_DIR
Original procedural meshes; no third-party models included. --room-only for room iteration.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import os
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

if not bpy.app.background:
    raise RuntimeError('Export requires a separate headless Blender process')

def finish():
    # All write_still and JSON writes have completed synchronously. Avoid a known
    # PulseAudio teardown hang in confined headless Blender; never used in GUI.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--room-only',action='store_true');p.add_argument('--only-kind')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=16;s.cycles.use_denoising=True
s.render.threads_mode='FIXED';s.render.threads=2;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA';s.render.film_transparent=False
s.view_settings.view_transform='Standard';s.view_settings.look='None'
s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.86,.94,1);s.world.node_tree.nodes['Background'].inputs[1].default_value=.5

def mat(name,color,emission=.12):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    b=m.node_tree.nodes.get('Principled BSDF');b.inputs['Base Color'].default_value=(*color,1);b.inputs['Roughness'].default_value=.54
    b.inputs['Emission Color'].default_value=(*color,1);b.inputs['Emission Strength'].default_value=emission
    return m

def cube(name,loc,scale,m,bevel=.04):
    bpy.ops.mesh.primitive_cube_add(size=2,location=loc);o=bpy.context.object;o.name=name;o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(m)
    if bevel:
        mod=o.modifiers.new('soft edges','BEVEL');mod.width=bevel;mod.segments=3
        o.modifiers.new('weighted normals','WEIGHTED_NORMAL')
    return o

def cam(loc,target,scale):
    bpy.ops.object.camera_add(location=loc);o=bpy.context.object;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler();o.data.type='ORTHO';o.data.ortho_scale=scale;s.camera=o
    return o

def render(path):
    s.render.filepath=str(out/path);bpy.ops.render.render(write_still=True)

camera=cam((6,-14,8),(0,.3,1.6),10.5)
for loc,power,size in [((-3,-5,8),600,6),((4,1,7),350,5)]:
    bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.rotation_euler=(Vector((0,0,1))-o.location).to_track_quat('-Z','Y').to_euler()
room=[]
def room_cube(*args,**kw):
    o=cube(*args,**kw);room.append(o);return o
cream=mat('warm plaster',(.82,.78,.61));sage=mat('sage floor',(.47,.63,.54));blue=mat('chalk teal wood',(.36,.69,.67));ivory=mat('ivory trim',(.86,.88,.75));sky=mat('soft blue daylight',(.49,.82,.91),.6)
room_cube('floor',(0,0,-.14),(5,5,.14),sage)
room_cube('back wall',(0,3.2,2.8),(5,.14,3),cream)
room_cube('left wall',(-4.7,-.6,2.8),(.14,3.8,3),ivory)
room_cube('skirting',(0,3.0,.09),(4.7,.06,.09),ivory)
# An airy, broad window; no grille or fine detail behind its sill targets.
room_cube('window daylight',(1.7,3.02,3.1),(1.65,.06,1.25),sky)
for x in (.0,3.4):room_cube('window upright',(x,2.93,3.1),(.065,.10,1.32),ivory)
for z in (1.78,4.42):room_cube('window lintel',(1.7,2.92,z),(1.76,.11,.065),ivory)
room_cube('window sill',(1.7,2.67,1.73),(1.90,.42,.09),ivory)
room_cube('display shelf',(-2.7,2.65,1.17),(1.75,.48,.08),blue)
for x in (-4,-1.4):room_cube('shelf bracket',(x,2.93,.87),(.09,.2,.25),blue)
room_cube('work table top',(1.2,-.25,1.05),(1.9,.82,.11),ivory)
for x in (-.45,2.85):
    for y in (-.85,.35):room_cube('table leg',(x,y,.48),(.085,.085,.48),blue)
# A low left-hand bench adds a second height, but targets stay large and uncluttered.
room_cube('left bench top',(-2.6,-.35,.64),(1.15,.68,.09),blue)
for x in (-3.5,-1.7):
    for y in (-.85,.15):room_cube('bench leg',(x,y,.28),(.08,.08,.28),ivory)
s.render.resolution_x=1280;s.render.resolution_y=960
bpy.context.view_layer.update()
anchors=[('shelf_left','shelf',(-3.6,2.5,1.25)),('shelf_right','shelf',(-1.7,2.5,1.25)),
 ('sill_left','sill',(.7,2.5,1.82)),('sill_right','sill',(2.8,2.5,1.82)),
 ('table_left','table',(.0,-.25,1.16)),('table_right','table',(2.25,-.25,1.16)),
 ('bench','bench',(-2.6,-.35,.73)),
 ('floor_left','floor',(-2.65,-2.7,.025)),('floor_middle','floor',(-1.0,-2.7,.025)),
 ('floor_right','floor',(1.25,-2.7,.025)),('floor_far_right','floor',(3.4,-2.7,.025))]
layout=[]
for name,support,xyz in anchors:
    q=world_to_camera_view(s,camera,Vector(xyz));layout.append(dict(id=name,support=support,world=list(xyz),anchor=[q.x,1-q.y]))
(out/'layout.json').write_text(json.dumps(dict(anchors=layout),indent=2))
s.view_settings.exposure=-1.2
if not a.only_kind:render('room.png')
s.view_settings.exposure=0.0
if a.room_only:
    print('ROOM_COMPLETE');finish()
for o in room:o.hide_render=True
s.render.film_transparent=True;s.render.resolution_x=320;s.render.resolution_y=384
camera.location=(3,-9,4.5);camera.rotation_euler=(Vector((0,0,1.1))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.ortho_scale=3.8
bpy.context.view_layer.update();pivot=world_to_camera_view(s,camera,Vector((0,0,0)))
profiles={
 'vase': [('body',[(0,.48),(.15,.65),(.6,.82),(1.05,.72),(1.3,.4)],'shatter'),('neck',[(1.3,.4),(1.57,.3),(1.78,.43)],'fall')],
 'jar': [('body',[(0,.6),(.15,.82),(.75,.87),(1.22,.72),(1.35,.53)],'shatter'),('lid',[(1.35,.57),(1.45,.6),(1.53,.24),(1.66,.14)],'fall')],
 'bowl': [('body',[(0,.3),(.12,.39),(.4,.7),(.82,1.0)],'shatter'),('rim',[(.82,1.0),(.91,1.04),(1.0,.97)],'fall')],
 'bottle': [('body',[(0,.5),(.1,.57),(.9,.57),(1.2,.32)],'shatter'),('neck',[(1.2,.32),(1.75,.21),(1.9,.27)],'fall')],
 'lamp': [('base',[(0,.55),(.12,.60),(.24,.25),(.8,.15),(1.03,.22)],'shatter'),('shade',[(1.03,.85),(1.13,.83),(1.9,.44),(2.0,.43)],'fall')],
}
colors={'vase':(.25,.76,.75),'jar':(.87,.73,.26),'bowl':(.37,.72,.91),'bottle':(.48,.78,.37),'lamp':(.90,.61,.35)}
meta=dict(schema='light-room-raw-parts-1',blender_version=bpy.app.version_string,pivot=[pivot.x*320,(1-pivot.y)*384],objects={})
if a.only_kind and (out/'parts.json').exists():meta=json.loads((out/'parts.json').read_text())

def lathe(name,profile,m):
    prof=[(profile[0][0],0)]+profile+[(profile[-1][0],0)];n=64
    vs=[(r*math.cos(2*math.pi*j/n),r*math.sin(2*math.pi*j/n),z) for z,r in prof for j in range(n)]
    fs=[];groups=[]
    for i in range(len(prof)-1):
        for j in range(n):
            fs.append((i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j));groups.append((i//3,j//8))
    me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);me.update();me.materials.append(m)
    o=bpy.data.objects.new(name,me);s.collection.objects.link(o)
    for f in me.polygons:f.use_smooth=True
    return o,vs,fs,groups

for name,definition in profiles.items():
    if a.only_kind and name!=a.only_kind:continue
    parts=[];part_meta=[];main_mat=mat(name,colors[name],.2)
    for part,profile,motion in definition:
        o,vs,fs,groups=lathe(part,profile,main_mat);parts.append((part,o,vs,fs,groups,motion))
    if name=='vase':
        flower_mat=mat('large pale flower',(.94,.78,.36),.18)
        meshes=[]
        # Chunky bloom and stem, no fine leaf textures or multiple tiny flowers.
        stem,vs,fs,gs=lathe('stem',[(1.75,.06),(2.15,.07)],flower_mat);meshes.append(stem)
        for angle in range(6):
            t=angle*math.tau/6
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=10,location=(.25*math.cos(t),0,2.20+.25*math.sin(t)))
            o=bpy.context.object;o.scale=(.23,.15,.23);o.data.materials.append(flower_mat);meshes.append(o)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=10,location=(0,-.04,2.20))
        o=bpy.context.object;o.scale=(.18,.16,.18);o.data.materials.append(flower_mat);meshes.append(o)
        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:o.select_set(True)
        bpy.context.view_layer.objects.active=stem;bpy.ops.object.join();bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        me=stem.data;vs=[tuple(v.co) for v in me.vertices];fs=[tuple(p.vertices) for p in me.polygons];gs=[(0,i//16) for i in range(len(fs))]
        parts.append(('flower',stem,vs,fs,gs,'fall'))
    render(name+'.png')
    # Opaque visible-part ownership from the SAME camera, not 3D depth guessed from physical XY.
    saved=[]
    for idx,(part,o,*_) in enumerate(parts):
        saved.append(list(o.data.materials))
        idmat=bpy.data.materials.new('id');idmat.use_nodes=True;nodes=idmat.node_tree.nodes;nodes.clear()
        e=nodes.new('ShaderNodeEmission');e.inputs[0].default_value=(*[(1.0 if j==idx else 0.0) for j in range(3)],1)
        target=nodes.new('ShaderNodeOutputMaterial');idmat.node_tree.links.new(e.outputs[0],target.inputs['Surface'])
        o.data.materials.clear();o.data.materials.append(idmat)
        for poly in o.data.polygons:poly.material_index=0
    old_samples=s.cycles.samples;s.cycles.samples=2;s.cycles.use_denoising=False;s.view_settings.view_transform='Raw'
    render(name+'_ids.png')
    s.cycles.samples=old_samples;s.cycles.use_denoising=True;s.view_settings.view_transform='Standard'
    for (_,o,*_),materials in zip(parts,saved):
        o.data.materials.clear()
        for m in materials:o.data.materials.append(m)
        o.hide_render=True
    for part,ob,vs,fs,groups,motion in parts:
        info=dict(id=part,motion=motion,frames=[])
        if motion=='shatter':
            chunks=[]
            for group in sorted(set(groups)):
                faces=[f for f,g in zip(fs,groups) if g==group];used=sorted({i for f in faces for i in f})
                c=sum((Vector(vs[i]) for i in used),Vector())/len(used);index={v:i for i,v in enumerate(used)}
                me=bpy.data.meshes.new('chip');me.from_pydata([Vector(vs[i])-c for i in used],[],[tuple(index[i] for i in f) for f in faces]);me.materials.append(main_mat)
                sh=bpy.data.objects.new('chip',me);s.collection.objects.link(sh)
                for f in me.polygons:f.use_smooth=True
                chunks.append((sh,c,len(chunks)))
        else:
            center=sum((Vector(v) for v in vs),Vector())/len(vs)
            ob.data.transform(__import__('mathutils').Matrix.Translation(-center));ob.hide_render=False
            chunks=[(ob,center,0)]
        for frame in range(12):
            t=(frame+1)/12
            for sh,c,k in chunks:
                direction=Vector((c.x+.1,c.y+.1,0));direction.normalize()
                travel=(.65*t if motion=='shatter' else .45*t)
                sh.location=c+direction*travel+Vector((0,0,.45*t-1.4*t*t))
                sh.rotation_euler=(t*.8*math.sin(k*2.1+1),t*1.1*math.cos(k*1.3),t*.6)
            fn=f'{name}_{part}_{frame:02d}.png';render(fn);info['frames'].append(fn)
        for sh,_,_ in chunks:
            if sh is ob:sh.hide_render=True
            else:bpy.data.objects.remove(sh,do_unlink=True)
        info['detach']=({'body':['neck','flower'],'neck':['flower']}.get(part,[]) if name=='vase' else
                        [p[0] for p in parts if p[0]!=part] if part in ('body','base') else [])
        info['points']={'flower':200,'neck':150,'lid':150,'rim':125,'shade':150}.get(part,100)
        info['cue']='petal' if part=='flower' else 'topple' if motion=='fall' else 'shatter'
        part_meta.append(info)
    meta['objects'][name]=dict(parts=part_meta)
    for _,ob,*_ in parts:bpy.data.objects.remove(ob,do_unlink=True)
(out/'parts.json').write_text(json.dumps(meta,indent=2))
print('PARTS_ROOM_COMPLETE',out)
finish()
