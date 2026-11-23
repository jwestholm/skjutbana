"""Headless CPU Blender room and functional breakable parts; no live scene access.
blender -noaudio -b --factory-startup -P automation/assets/render_light_room_parts.py -- --out NEW_DIR
Optional --polyhaven-root imports selected local CC0/PBR models. --only-part supports bounded exports.
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

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--room-only',action='store_true');p.add_argument('--living-room',action='store_true');p.add_argument('--polyhaven-root',type=Path);p.add_argument('--only-kind');p.add_argument('--only-part');p.add_argument('--resume',action='store_true',help='Reuse completed renders in an interrupted raw export')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if a.polyhaven_root:
    from automation.assets import pang_polyhaven as ph
    a.living_room=True
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=32;s.cycles.use_denoising=True
s.render.threads_mode='FIXED';s.render.threads=2;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA';s.render.film_transparent=False
s.view_settings.view_transform='AgX' if a.polyhaven_root else 'Standard';s.view_settings.look='None'
s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.86,.94,1);s.world.node_tree.nodes['Background'].inputs[1].default_value=.65

def mat(name,color,emission=.025):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    b=m.node_tree.nodes.get('Principled BSDF');b.inputs['Base Color'].default_value=(*color,1);b.inputs['Roughness'].default_value=.23 if name in ('vase','jar','bowl','bottle') else .48
    b.inputs['IOR'].default_value=1.46
    b.inputs['Coat Weight'].default_value=.42 if name in ('vase','jar','bowl','bottle') else .08
    b.inputs['Coat Roughness'].default_value=.16
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
    if a.resume and (out/path).is_file():
        print('REUSE_RENDER',path,flush=True);return
    s.render.filepath=str(out/path);print('RENDER_BEGIN',path,flush=True);bpy.ops.render.render(write_still=True)

camera=cam((6,-14,8),(0,.3,1.6),10.5)
if a.living_room:
    camera.location=(6,-14,6.3);camera.rotation_euler=(Vector((0,.3,1.6))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='PERSP';camera.data.lens=50
for loc,power,size in [((-3,-5,8),750,4),((4,1,7),450,5),((1,-6,4),130,2)]:
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
if a.living_room:
    for ob in room:bpy.data.objects.remove(ob,do_unlink=True)
    room.clear()
    plaster=mat('warm mineral plaster',(.72,.67,.56),.01)
    oak=mat('honey oak satin',(.53,.40,.23),.01)
    linen=mat('linen upholstery',(.53,.61,.53),.01)
    porcelain=mat('warm white lacquer',(.79,.76,.65),.015)
    # Broad, physically shaded forms. No black grout, writing or fine grain.
    room_cube('continuous oak floor',(0,0,-.15),(15,15,.15),oak,.02)
    room_cube('plaster back wall',(0,3.2,3),(7,.14,3.3),plaster,.015)
    room_cube('plaster left wall',(-5.5,-1,3),(.14,4.2,3.3),plaster,.015)
    room_cube('back skirting',(0,3.0,.10),(7,.065,.10),porcelain,.02)
    room_cube('daylight glass',(1.7,3.02,3.1),(1.65,.06,1.25),sky,.02)
    for x in (.0,3.4):room_cube('window jamb',(x,2.93,3.1),(.065,.10,1.32),porcelain,.02)
    for z in (1.78,4.42):room_cube('window lintel',(1.7,2.92,z),(1.76,.11,.065),porcelain,.02)
    room_cube('deep window sill',(1.7,2.67,1.73),(1.90,.42,.09),porcelain,.04)
    # Curtains have large folds rather than a noisy textile bitmap.
    for side in (-.23,3.63):
        for j in range(4):
            room_cube('linen curtain',(side+(j-1.5)*.07,2.77,3.1),(.08,.10,1.46),ivory,.07)
    # Bookshelf, with two clear broad target shelves; no tiny spine lettering.
    for x in (-4.5,-.9):room_cube('bookcase side',(x,2.65,1.9),(.09,.44,1.8),oak,.025)
    for z in (.25,1.17,2.25,3.65):room_cube('bookcase shelf',(-2.7,2.65,z),(1.88,.48,.08),oak,.025)
    # One large neutral framed canvas high above the floor targets.
    room_cube('canvas frame',(-4.98,.4,3.1),(.09,1.0,.77),oak,.05)
    room_cube('ivory canvas',(-4.86,.4,3.1),(.04,.90,.67),ivory,.02)
    # Sofa: rounded cushions, armrests, a visible seam between broad panels.
    room_cube('sofa plinth',(1.65,1.3,.31),(2.02,.82,.19),oak,.12)
    for x in (.7,2.6):
        room_cube('seat cushion',(x,1.05,.67),(.9,.72,.22),linen,.18)
        room_cube('back cushion',(x,1.80,1.17),(.9,.23,.60),linen,.18)
    for x in (-.42,3.72):room_cube('sofa arm',(x,1.25,.86),(.23,.86,.48),linen,.18)
    # Real target support anchors deliberately match the R4 camera-space contract.
    room_cube('coffee tabletop',(1.2,-.25,1.05),(1.9,.82,.11),oak,.11)
    for x in (-.45,2.85):
        for y in (-.85,.35):room_cube('coffee table leg',(x,y,.48),(.075,.075,.48),oak,.04)
    room_cube('media console top',(-2.6,-.35,.64),(1.15,.68,.09),oak,.055)
    room_cube('media console cabinet',(-2.6,-.28,.30),(1.06,.58,.25),porcelain,.05)
    room_cube('rug',(0,-1.25,.012),(3.9,2.5,.012),linen,.015)
    # Additional diffuse daylight points from the actual window direction.
    bpy.ops.object.light_add(type='AREA',location=(1.7,2.7,3.5));o=bpy.context.object;o.name='room daylight'
    o.data.energy=400;o.data.size=3.2;o.rotation_euler=(Vector((0,-1,.8))-o.location).to_track_quat('-Z','Y').to_euler()

if a.polyhaven_root:
    for ob in room:bpy.data.objects.remove(ob,do_unlink=True)
    room.clear()
    room,ph_anchors=ph.room(a.polyhaven_root,cube,mat)
    camera.location=(3.8,-7.2,2.8);camera.rotation_euler=(Vector((0,.1,1.0))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='PERSP';camera.data.lens=50
    for ob in list(bpy.context.scene.objects):
        if ob.type=='LIGHT':bpy.data.objects.remove(ob,do_unlink=True)
    for loc,power,size in [((1.05,1.35,2.25),500,1.8),((-1.5,-3,3.0),100,3.5)]:
        bpy.ops.object.light_add(type='AREA',location=loc);ob=bpy.context.object;ob.name='PBR daylight'
        ob.data.energy=power;ob.data.size=size;ob.rotation_euler=(Vector((0,-.5,.6))-ob.location).to_track_quat('-Z','Y').to_euler()
    s.world.node_tree.nodes['Background'].inputs[1].default_value=.35

s.render.resolution_x=1280;s.render.resolution_y=960
bpy.context.view_layer.update()
anchors=[('shelf_left','shelf',(-3.6,2.5,1.25)),('shelf_right','shelf',(-1.7,2.5,1.25)),
 ('sill_left','sill',(.7,2.5,1.82)),('sill_right','sill',(2.8,2.5,1.82)),
 ('table_left','table',(.0,-.25,1.16)),('table_right','table',(2.25,-.25,1.16)),
 ('bench','bench',(-2.6,-.35,.73)),
 ('floor_left','floor',(-2.65,-2.7,.025)),('floor_middle','floor',(-1.0,-2.7,.025)),
 ('floor_right','floor',(1.25,-2.7,.025)),('floor_far_right','floor',(3.4,-2.7,.025))]
if a.polyhaven_root:anchors=ph_anchors
layout=[]
for name,support,xyz in anchors:
    q=world_to_camera_view(s,camera,Vector(xyz));layout.append(dict(id=name,support=support,world=list(xyz),anchor=[q.x,1-q.y]))
(out/'layout.json').write_text(json.dumps(dict(anchors=layout),indent=2))
s.view_settings.exposure=.5 if a.polyhaven_root else -.35
s.cycles.samples=16 if a.living_room else 32
if not a.only_kind:render('room.png')
s.view_settings.exposure=.5 if a.polyhaven_root else 0.0
if a.room_only:
    print('ROOM_COMPLETE');finish()
for o in room:o.hide_render=True
if a.living_room and not a.polyhaven_root:
    bpy.data.objects['room daylight'].hide_render=True
if a.polyhaven_root:
    for ob in list(bpy.context.scene.objects):
        if ob.type=='LIGHT':bpy.data.objects.remove(ob,do_unlink=True)
    for loc,power,size in [((-3,-5,8),750,4),((4,1,7),450,5),((1,-6,4),130,2)]:
        bpy.ops.object.light_add(type='AREA',location=loc);ob=bpy.context.object;ob.data.energy=power;ob.data.size=size;ob.rotation_euler=(Vector((0,0,1))-ob.location).to_track_quat('-Z','Y').to_euler()
s.render.film_transparent=True;s.render.resolution_x=320;s.render.resolution_y=384
camera.data.type='ORTHO';camera.location=(3,-9,4.5);camera.rotation_euler=(Vector((0,0,1.1))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.ortho_scale=3.8
bpy.context.view_layer.update();pivot=world_to_camera_view(s,camera,Vector((0,0,0)))
profiles={
 'vase': [('body',[(0,.48),(.15,.65),(.6,.82),(1.05,.72),(1.3,.4)],'shatter'),('neck',[(1.3,.4),(1.57,.3),(1.78,.43)],'fall')],
 'jar': [('body',[(0,.6),(.15,.82),(.75,.87),(1.22,.72),(1.35,.53)],'shatter'),('lid',[(1.35,.57),(1.45,.6),(1.53,.24),(1.66,.14)],'fall')],
 'bowl': [('body',[(0,.3),(.12,.39),(.4,.7),(.82,1.0)],'shatter'),('rim',[(.82,1.0),(.91,1.04),(1.0,.97)],'fall')],
 'bottle': [('body',[(0,.5),(.1,.57),(.9,.57),(1.2,.32)],'shatter'),('neck',[(1.2,.32),(1.75,.21),(1.9,.27)],'fall')],
 'lamp': [('base',[(0,.55),(.12,.60),(.24,.25),(.8,.15),(1.03,.22)],'shatter'),('shade',[(1.03,.85),(1.13,.83),(1.9,.44),(2.0,.43)],'fall')],
}
colors={'vase':(.12,.62,.47),'jar':(.78,.59,.20),'bowl':(.19,.55,.72),'bottle':(.20,.63,.29),'lamp':(.85,.67,.34)}
meta=dict(scene='living_room' if a.living_room else 'atelier',schema='light-room-raw-parts-1',blender_version=bpy.app.version_string,pivot=[pivot.x*320,(1-pivot.y)*384],objects={})
if a.only_kind and (out/'parts.json').exists():meta=json.loads((out/'parts.json').read_text())
if a.polyhaven_root:
    meta['asset_provider']='polyhaven';meta['recipe']='automation/assets/render_light_room_parts.py'
    meta['asset_provenance']=json.loads((a.polyhaven_root/'manifest.json').read_text())

def lathe(name,profile,m):
    # Smooth ceramic silhouette, with shared part endpoints preserved. Radius is
    # bounded by the neighbouring knots, preventing narrow ringing/extra ledges.
    knots=profile;profile=[]
    for i in range(len(knots)-1):
        p0=Vector(knots[max(0,i-1)]);p1=Vector(knots[i]);p2=Vector(knots[i+1]);p3=Vector(knots[min(len(knots)-1,i+2)])
        for j in range(6):
            t=j/6
            q=.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)
            z=p1.x+(p2.x-p1.x)*t
            profile.append((z,max(min(p1.y,p2.y),min(max(p1.y,p2.y),q.y))))
    profile.append(knots[-1])
    prof=[(profile[0][0],0)]+profile+[(profile[-1][0],0)];n=64
    vs=[(r*math.cos(2*math.pi*j/n),r*math.sin(2*math.pi*j/n),z) for z,r in prof for j in range(n)]
    fs=[];groups=[]
    for i in range(len(prof)-1):
        for j in range(n):
            fs.append((i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j));groups.append((min(2,i*3//(len(prof)-1)),j//8))
    me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);me.update();me.materials.append(m)
    o=bpy.data.objects.new(name,me);s.collection.objects.link(o)
    for f in me.polygons:f.use_smooth=True
    return o,vs,fs,groups


def mesh_part(part,ob,motion):
    # Freeze evaluated bevels and transforms into the exact mesh used by the ID render.
    bpy.context.view_layer.update()
    ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());me=bpy.data.meshes.new_from_object(ev)
    me.transform(ob.matrix_world);ob.data=me;ob.location=(0,0,0);ob.rotation_euler=(0,0,0);ob.scale=(1,1,1)
    vs=[tuple(v.co) for v in me.vertices];fs=[tuple(p.vertices) for p in me.polygons]
    gs=[(min(2,int(max(0,sum(vs[i][2] for i in f)/len(f))*1.5)),j%8) for j,f in enumerate(fs)]
    return part,ob,vs,fs,gs,motion

def combine(objects):
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objects:ob.select_set(True)
    bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join();return objects[0]

def custom_parts(name):
    ceramic=mat(name,(.65,.72,.48),.025);parts=[]
    if name=='tv':
        border=mat('champagne aluminium',(.66,.67,.57),.015)
        picture=mat('bright television landscape',(.55,.82,.83),.28)
        nodes=picture.node_tree.nodes;links=picture.node_tree.links;b=nodes.get('Principled BSDF')
        tex=nodes.new('ShaderNodeTexCoord');sep=nodes.new('ShaderNodeSeparateXYZ');ramp=nodes.new('ShaderNodeValToRGB')
        links.new(tex.outputs['Generated'],sep.inputs[0]);links.new(sep.outputs['Z'],ramp.inputs[0])
        ramp.color_ramp.elements[0].color=(.35,.69,.37,1);ramp.color_ramp.elements[1].color=(.48,.77,.95,1)
        mid=ramp.color_ramp.elements.new(.45);mid.color=(.82,.83,.56,1)
        links.new(ramp.outputs[0],b.inputs['Base Color']);links.new(ramp.outputs[0],b.inputs['Emission Color'])
        emission=nodes.new('ShaderNodeEmission');links.new(ramp.outputs[0],emission.inputs['Color']);emission.inputs['Strength'].default_value=1.0
        links.new(emission.outputs[0],nodes.get('Material Output').inputs['Surface'])
        frame=cube('TV frame',(0,0,1.10),(1.20,.10,.73),border,.055)
        screen=cube('TV picture',(0,-.115,1.10),(1.12,.025,.65),picture,.025)
        parts.append(mesh_part('screen',combine([frame,screen]),'crack'))
        stand=cube('TV stand',(0,.02,.20),(.52,.30,.10),border,.045)
        stem=cube('TV support',(0,.04,.38),(.13,.10,.19),border,.03)
        parts.append(mesh_part('stand',combine([stand,stem]),'fall'))
    elif name=='books':
        for j,(x,c) in enumerate([(-.52,(.24,.64,.51)),(0,(.78,.65,.24)),(.52,(.35,.66,.75))]):
            cover=mat('cloth book cover',c,.02);paper=mat('cream paper',(.83,.80,.67),.01)
            body=cube('book pages',(x,0,.94),(.21,.28,.78),paper,.025)
            panels=[body]
            for dx in (-.235,.235):panels.append(cube('cloth cover',(x+dx,0,.94),(.027,.32,.82),cover,.015))
            panels.append(cube('plain spine',(x,-.305,.94),(.23,.028,.81),cover,.03))
            parts.append(mesh_part(('left','middle','right')[j],combine(panels),'fall'))
    elif name=='plant':
        ob,vs,fs,gs=lathe('pot',[(0,.42),(.12,.50),(.82,.64),(1.0,.66)],ceramic)
        parts.append(('body',ob,vs,fs,gs,'shatter'))
        leafmat=mat('broad jade foliage',(.26,.65,.30),.02);meshes=[]
        stem,*_=lathe('plant stem',[(.9,.07),(1.95,.06)],leafmat);meshes.append(stem)
        for j in range(6):
            theta=j*math.tau/6
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,location=(.40*math.cos(theta),.28*math.sin(theta),1.45+.10*(j%3)))
            ob=bpy.context.object
            for poly in ob.data.polygons:poly.use_smooth=True
            ob.scale=(.25,.12,.53);ob.rotation_euler[1]=.65*math.cos(theta);ob.data.materials.append(leafmat)
            bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);meshes.append(ob)
        parts.append(mesh_part('foliage',combine(meshes),'fall'))
    elif name=='mug':
        glaze=mat('mug turquoise glaze',(.18,.58,.64),.02)
        ob,vs,fs,gs=lathe('mug body',[(0,.42),(.12,.47),(1.12,.49),(1.22,.52)],glaze)
        parts.append(('body',ob,vs,fs,gs,'shatter'))
        vertices=[];faces=[]
        # An open C handle: its actual opening remains a miss, never filled by a mask.
        for j in range(25):
            phi=-math.pi/2+math.pi*j/24
            for k in range(12):
                theta=k*math.tau/12;r=.36+.075*math.cos(theta)
                vertices.append((.48+r*math.cos(phi),.075*math.sin(theta),.65+r*math.sin(phi)))
        for j in range(24):
            for k in range(12):faces.append((j*12+k,j*12+(k+1)%12,(j+1)*12+(k+1)%12,(j+1)*12+k))
        me=bpy.data.meshes.new('open handle');me.from_pydata(vertices,[],faces);me.materials.append(glaze)
        ob=bpy.data.objects.new('handle',me);s.collection.objects.link(ob)
        for poly in me.polygons:poly.use_smooth=True
        parts.append(mesh_part('handle',ob,'fall'))
    elif name=='clock':
        shell=mat('clock glazed brass',(.74,.66,.34),.025);face=mat('clock pearl dial',(.85,.83,.69),.03)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,location=(0,0,.98));ob=bpy.context.object;ob.scale=(.73,.22,.77);ob.data.materials.append(shell)
        for poly in ob.data.polygons:poly.use_smooth=True
        # Broad, bright hands; no dark fine tick marks in the analysed area.
        dial=cube('clock face',(0,-.23,.98),(.55,.02,.57),face,.18)
        hands=[cube('hour hand',(-.10,-.26,1.02),(.15,.012,.035),ceramic,.02),cube('minute hand',(0,-.26,1.19),(.035,.012,.25),ceramic,.02)]
        parts.append(mesh_part('face',combine([ob,dial]+hands),'crack'))
    return parts

if a.living_room:
    profiles.update({k:[] for k in ('tv','books','plant','clock','mug')})

for name,definition in profiles.items():
    if a.only_kind and name!=a.only_kind:continue
    parts=[];part_meta=[];main_mat=mat(name,colors.get(name,(.65,.70,.45)),.025)
    if a.polyhaven_root:parts=ph.make_parts(a.polyhaven_root,name)
    elif not definition:parts=custom_parts(name)
    for part,profile,motion in ([] if a.polyhaven_root else definition):
        o,vs,fs,groups=lathe(part,profile,main_mat);parts.append((part,o,vs,fs,groups,motion))
    if name=='vase' and not a.polyhaven_root:
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
    if a.polyhaven_root:camera.data.ortho_scale=5.6 if name=='bowl' else 3.8
    object_pivot=world_to_camera_view(s,camera,Vector((0,0,0)))
    object_pivot_px=[object_pivot.x*s.render.resolution_x,(1-object_pivot.y)*s.render.resolution_y]
    s.cycles.samples=32
    render(name+'.png')
    # Opaque visible-part ownership from the SAME camera, not 3D depth guessed from physical XY.
    saved=[]
    for idx,(part,o,*_) in enumerate(parts):
        saved.append((list(o.data.materials),[p.material_index for p in o.data.polygons]))
        if a.polyhaven_root:
            for index,original in enumerate(list(o.data.materials)):
                o.data.materials[index]=ph.id_material(original,[(1.0 if j==idx else 0.0) for j in range(3)])
        else:
            idmat=bpy.data.materials.new('id');idmat.use_nodes=True;nodes=idmat.node_tree.nodes;nodes.clear()
            e=nodes.new('ShaderNodeEmission');e.inputs[0].default_value=(*[(1.0 if j==idx else 0.0) for j in range(3)],1)
            target=nodes.new('ShaderNodeOutputMaterial');idmat.node_tree.links.new(e.outputs[0],target.inputs['Surface'])
            o.data.materials.clear();o.data.materials.append(idmat)
            for poly in o.data.polygons:poly.material_index=0
    old_samples=s.cycles.samples;s.cycles.samples=2;s.cycles.use_denoising=False;s.view_settings.view_transform='Raw'
    render(name+'_ids.png')
    s.cycles.samples=old_samples;s.cycles.use_denoising=True;s.view_settings.view_transform='AgX' if a.polyhaven_root else 'Standard'
    for (_,o,*_),(materials,indices) in zip(parts,saved):
        o.data.materials.clear()
        for m in materials:o.data.materials.append(m)
        for poly,index in zip(o.data.polygons,indices):poly.material_index=index
        o.hide_render=True
    for part,ob,vs,fs,groups,motion in parts:
        if a.only_part and part!=a.only_part:continue
        info=dict(id=part,motion=motion,frames=[])
        if motion=='shatter':
            chunks=[]
            for group in sorted(set(groups)):
                face_indices=[i for i,g in enumerate(groups) if g==group]
                faces=[fs[i] for i in face_indices];used=sorted({i for f in faces for i in f})
                c=sum((Vector(vs[i]) for i in used),Vector())/len(used);index={v:i for i,v in enumerate(used)}
                me=bpy.data.meshes.new('chip');me.from_pydata([Vector(vs[i])-c for i in used],[],[tuple(index[i] for i in f) for f in faces]);me.materials.append(main_mat)
                if a.polyhaven_root:
                    me.materials.clear()
                    for material in ob.data.materials:me.materials.append(material)
                    for dest,index in zip(me.polygons,face_indices):dest.material_index=ob.data.polygons[index].material_index
                    for layer in ob.data.uv_layers:
                        uv=me.uv_layers.new(name=layer.name)
                        for dest,index in zip(me.polygons,face_indices):
                            for dl,sl in zip(dest.loop_indices,ob.data.polygons[index].loop_indices):uv.data[dl].uv=layer.data[sl].uv
                sh=bpy.data.objects.new('chip',me);s.collection.objects.link(sh)
                for f in me.polygons:f.use_smooth=True
                chunks.append((sh,c,len(chunks)))
        elif motion=='crack':
            ob.hide_render=False;chunks=[]
            # Light cracks are part of the post-decision sprite, never a live overlay.
            for j in range(7):
                curve=bpy.data.curves.new('fracture line','CURVE');curve.dimensions='3D';curve.bevel_depth=.012;curve.bevel_resolution=2
                sp=curve.splines.new('POLY');sp.points.add(2)
                ang=j*math.tau/7
                for pt,(x,z) in zip(sp.points,[(0,1.1),(.35*math.cos(ang+.12),1.1+.28*math.sin(ang)),(.85*math.cos(ang),1.1+.5*math.sin(ang))]):pt.co=(x if name=='tv' else x*.58,-.16 if name=='tv' else -.28,z if name=='tv' else .98+(z-1.1)*.9,1)
                if a.polyhaven_root:
                    lo,hi=ph.bounds([ob]);center=(lo+hi)/2
                    if name=='tv':center.x-=(hi.x-lo.x)*.12
                    for pt in sp.points:
                        px,_,pz,_=pt.co;pt.co=(center.x+px*(hi.x-lo.x)*.30,lo.y-.015,center.z+(pz-1.1)*(hi.z-lo.z)*.5,1)
                crack=bpy.data.objects.new('glass fracture',curve);s.collection.objects.link(crack);curve.materials.append(mat('bright fracture',(.88,.88,.79),.08));chunks.append((crack,crack.location.copy(),j))
        else:
            center=sum((Vector(v) for v in vs),Vector())/len(vs)
            ob.data.transform(__import__('mathutils').Matrix.Translation(-center));ob.hide_render=False
            chunks=[(ob,center,0)]
        if name=='plant' and part=='body':
            earth=mat('light clay soil',(.55,.40,.24),.015)
            for k in range(3):
                bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8)
                soil=bpy.context.object;soil.scale=(.16,.12,.10);soil.data.materials.append(earth)
                bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
                chunks.append((soil,Vector(((k-1)*.22,0,.6)),len(chunks)))
        s.cycles.samples=8  # moving fragments only, after the protected window
        for frame in range(12):
            t=(frame+1)/12
            for sh,c,k in chunks:
                if motion=='crack':
                    sh.hide_render=frame<k;continue
                direction=Vector((c.x+.1,c.y+.1,0));direction.normalize()
                travel=(.65*t if motion=='shatter' else .45*t)
                sh.location=c+direction*travel+Vector((0,0,.45*t-(max(1.4,c.z+1.) if a.polyhaven_root else 1.4)*t*t))
                sh.rotation_euler=(t*.8*math.sin(k*2.1+1),t*1.1*math.cos(k*1.3),t*.6)
                if name=='books':sh.rotation_euler=(t*1.48,t*.12,t*.35)
                if (a.polyhaven_root or name in ('books','plant','clock','tv','mug')) and sh.type=='MESH':
                    rotation=sh.rotation_euler.to_matrix()
                    bottom=min((rotation@v.co).z for v in sh.data.vertices)+sh.location.z
                    if bottom<.025:sh.location.z+=.025-bottom
            fn=f'{name}_{part}_{frame:02d}.png';render(fn);info['frames'].append(fn)
        if motion=='crack':ob.hide_render=True
        for sh,_,_ in chunks:
            if sh is ob:sh.hide_render=True
            else:bpy.data.objects.remove(sh,do_unlink=True)
        info['detach']=({'body':['neck','flower'],'neck':['flower']}.get(part,[]) if name=='vase' else
                        [p[0] for p in parts if p[0]!=part] if part in ('body','base') else [])
        if name=='plant' and part=='body':info['detach']=['foliage']
        if name=='mug' and part=='body':info['detach']=['handle']
        if name=='tv' and part=='stand':info['detach']=['screen']
        info['points']={'flower':200,'neck':150,'lid':150,'rim':125,'shade':150}.get(part,100)
        info['cue']='petal' if part=='flower' else 'topple' if motion=='fall' else 'shatter'
        part_meta.append(info)
        previous={p['id']:p for p in meta['objects'].get(name,{}).get('parts',[])}
        previous.update({p['id']:p for p in part_meta})
        meta['objects'][name]=dict(pivot=object_pivot_px,parts=[previous[p[0]] for p in parts if p[0] in previous])
        if a.polyhaven_root:meta['objects'][name]['source_assets']=ph.SOURCE_BY_KIND[name]
        checkpoint=out/'parts.checkpoint.json';checkpoint.write_text(json.dumps(meta,indent=2));checkpoint.replace(out/'parts.json')
    for _,ob,*_ in parts:bpy.data.objects.remove(ob,do_unlink=True)
(out/'parts.json').write_text(json.dumps(meta,indent=2))
print('PARTS_ROOM_COMPLETE',out)
finish()
