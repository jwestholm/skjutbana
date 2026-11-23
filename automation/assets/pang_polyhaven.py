"""Headless-only PANG asset recipe. Local CC0 glTFs, preserved PBR/UVs, manual parts."""
from pathlib import Path
import math
import bpy,bmesh
from mathutils import Matrix,Vector

SOURCE_BY_KIND={'vase':['ceramic_vase_02','potted_plant_02'],'jar':['ceramic_vase_01'],
 'bowl':['ceramic_pot'],'bottle':['ceramic_vase_03'],'lamp':['desk_lamp_arm_01'],
 'tv':['Television_01'],'books':['book_encyclopedia_set_01'],'plant':['potted_plant_02'],
 'clock':['alarm_clock_01'],'mug':['metal_jug']}
NATURAL_MATERIALS=False
ROOM_SOURCES=['sofa_02','modern_coffee_table_01','Shelf_01','side_table_01','fancy_picture_frame_01']


def bounds(objects):
    bpy.context.view_layer.update()
    p=[o.matrix_world@Vector(v) for o in objects for v in o.bound_box]
    return Vector(tuple(min(v[i] for v in p) for i in range(3))),Vector(tuple(max(v[i] for v in p) for i in range(3)))


def lighten(material,amount):
    """Painted/light diffuse variant; keep roughness, metallic, normal and alpha links."""
    m=material.copy();m.name+=' light variant'
    if not m.use_nodes:return m
    tree=m.node_tree;b=next((n for n in tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
    if b is None:return m
    socket=b.inputs['Base Color'];mix=tree.nodes.new('ShaderNodeMixRGB');mix.blend_type='MIX'
    mix.inputs[0].default_value=amount*.5;mix.inputs[2].default_value=(.86,.84,.76,1)
    gamma=tree.nodes.new('ShaderNodeGamma');gamma.inputs['Gamma'].default_value=.55
    if socket.is_linked:tree.links.new(socket.links[0].from_socket,gamma.inputs['Color'])
    else:gamma.inputs['Color'].default_value=socket.default_value
    tree.links.new(gamma.outputs[0],mix.inputs[1])
    tree.links.new(mix.outputs[0],socket)
    for n in tree.nodes:
        if n.type=='NORMAL_MAP':n.inputs['Strength'].default_value=.4 if amount>.6 else .75
    return m


def load(root,name,height=None,location=(0,0,0),rotation=0,light=.65):
    if NATURAL_MATERIALS:light=None
    root=Path(root)
    if NATURAL_MATERIALS and not (root/name).exists():root=root.with_name('pang_polyhaven_r8_sources')
    before=set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(next((Path(root)/name).resolve().glob('*.gltf'))))
    imported=list(set(bpy.context.scene.objects)-before);objects=[o for o in imported if o.type=='MESH']
    lo,hi=bounds(objects);scale=height/(hi.z-lo.z) if height is not None else 1.
    offset=Vector(((lo.x+hi.x)/2,(lo.y+hi.y)/2,lo.z))
    transform=Matrix.Translation(Vector(location))@Matrix.Rotation(rotation,4,'Z')@Matrix.Scale(scale,4)@Matrix.Translation(-offset)
    cache={}
    for o in objects:
        world=o.matrix_world.copy();o.data=o.data.copy();o.data.transform(transform@world)
        o.parent=None;o.matrix_world=Matrix.Identity(4)
        for i,m in enumerate(o.data.materials):
            if m is None:continue
            if m not in cache:cache[m]=lighten(m,light) if light is not None else m
            o.data.materials[i]=cache[m]
    for o in imported:
        if o.type!='MESH':bpy.data.objects.remove(o,do_unlink=True)
    return objects


def join(objects,name):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:o.hide_render=False;o.select_set(True)
    bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join();ob=objects[0];ob.name=name
    return ob


def sliced(ob,name,axis,lower=None,upper=None):
    """Real mesh cut, UVs interpolated by bmesh; no fictional screen-space seam."""
    mesh=ob.data.copy();bm=bmesh.new();bm.from_mesh(mesh)
    for plane,low in ((lower,True),(upper,False)):
        if plane is None:continue
        point=[0.,0.,0.];point[axis]=plane;normal=[0.,0.,0.];normal[axis]=1
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),dist=1e-6,
            plane_co=point,plane_no=normal,clear_inner=low,clear_outer=not low)
    bm.to_mesh(mesh);bm.free();mesh.update()
    if not mesh.polygons:raise ValueError('Empty manual part '+name)
    o=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(o);return o


def part(name,ob,motion):
    me=ob.data;vs=[tuple(v.co) for v in me.vertices];fs=[tuple(p.vertices) for p in me.polygons];lo,hi=bounds([ob]);groups=[]
    for f in fs:
        c=sum((Vector(vs[i]) for i in f),Vector())/len(f)
        groups.append((min(2,int(3*(c.z-lo.z)/max(1e-5,hi.z-lo.z))),int((math.atan2(c.y,c.x)+math.pi)*8/math.tau)%8))
    return name,ob,vs,fs,groups,motion


def id_material(original,color):
    """Use original opacity graph in the part-ID render (leaves must not turn solid)."""
    m=original.copy();m.name='part ID with source alpha';tree=m.node_tree
    b=next((n for n in tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
    out=next(n for n in tree.nodes if n.type=='OUTPUT_MATERIAL')
    e=tree.nodes.new('ShaderNodeEmission');e.inputs[0].default_value=(*color,1)
    t=tree.nodes.new('ShaderNodeBsdfTransparent');mix=tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(t.outputs[0],mix.inputs[1]);tree.links.new(e.outputs[0],mix.inputs[2])
    alpha=b.inputs['Alpha'] if b else None
    if alpha and alpha.is_linked:tree.links.new(alpha.links[0].from_socket,mix.inputs[0])
    else:mix.inputs[0].default_value=alpha.default_value if alpha else 1
    tree.links.new(mix.outputs[0],out.inputs['Surface']);return m


def television_on(ob,modern=False):
    # The screen rectangle is in this model's documented diffuse UV atlas, not game/GT XY.
    for m in ob.data.materials:
        b=next((n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
        if b is None:continue
        tree=m.node_tree;nodes=tree.nodes;links=tree.links
        uv=nodes.new('ShaderNodeTexCoord');sep=nodes.new('ShaderNodeSeparateXYZ');links.new(uv.outputs['UV'],sep.inputs[0])
        gates=[]
        for socket,op,threshold in [(sep.outputs['X'],'GREATER_THAN',.690 if modern else .033),(sep.outputs['X'],'LESS_THAN',.985 if modern else .352),
                                    (sep.outputs['Y'],'GREATER_THAN',.765 if modern else .05),(sep.outputs['Y'],'LESS_THAN',.985 if modern else .277)]:
            n=nodes.new('ShaderNodeMath');n.operation=op;links.new(socket,n.inputs[0]);n.inputs[1].default_value=threshold;gates.append(n.outputs[0])
        mask=gates[0]
        for g in gates[1:]:
            n=nodes.new('ShaderNodeMath');n.operation='MULTIPLY';links.new(mask,n.inputs[0]);links.new(g,n.inputs[1]);mask=n.outputs[0]
        ramp=nodes.new('ShaderNodeValToRGB');links.new(sep.outputs['Y'],ramp.inputs[0])
        ramp.color_ramp.elements[0].position=.765 if modern else .05;ramp.color_ramp.elements[0].color=(.36,.68,.32,1)
        ramp.color_ramp.elements[1].position=.985 if modern else .277;ramp.color_ramp.elements[1].color=(.5,.78,.94,1)
        mid=ramp.color_ramp.elements.new(.87 if modern else .15);mid.color=(.85,.82,.58,1)
        mix=nodes.new('ShaderNodeMixRGB');links.new(mask,mix.inputs[0]);links.new(b.inputs['Base Color'].links[0].from_socket,mix.inputs[1]);links.new(ramp.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],b.inputs['Base Color'])
        emit=nodes.new('ShaderNodeMixRGB');links.new(mask,emit.inputs[0]);emit.inputs[1].default_value=(0,0,0,1);links.new(ramp.outputs[0],emit.inputs[2]);links.new(emit.outputs[0],b.inputs['Emission Color']);b.inputs['Emission Strength'].default_value=.7


def prune_leaves(ob,count=3):
    """Keep three complete real leaf meshes, preserving UVs; never alter a hit mask."""
    bm=bmesh.new();bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5)
    seen=set();components=[]
    for seed in bm.verts:
        if seed in seen:continue
        stack=[seed];vertices=set()
        while stack:
            v=stack.pop()
            if v in vertices:continue
            vertices.add(v);stack.extend(e.other_vert(v) for e in v.link_edges if e.other_vert(v) not in vertices)
        seen.update(vertices);faces={f for v in vertices for f in v.link_faces}
        components.append((sum(f.calc_area() for f in faces),vertices))
    selected=set().union(*(v for _,v in sorted(components,key=lambda c:-c[0])[:count]))
    bmesh.ops.delete(bm,geom=[v for v in bm.verts if v not in selected],context='VERTS')
    bm.to_mesh(ob.data);bm.free();ob.data.update();return ob


def make_parts(root,name):
    if name in ('vase','jar','bowl','bottle'):
        asset={'vase':'ceramic_vase_02','jar':'ceramic_vase_01','bowl':'ceramic_pot','bottle':'ceramic_vase_03'}[name]
        ob=join(load(root,asset,2.1),'imported '+name)
        cut={'vase':1.45,'jar':1.87,'bowl':1.58,'bottle':1.4}[name]
        second={'vase':'neck','jar':'lid','bowl':'rim','bottle':'neck'}[name]
        low=sliced(ob,'body',2,upper=cut);high=sliced(ob,second,2,lower=cut);bpy.data.objects.remove(ob,do_unlink=True)
        parts=[part('body',low,'shatter'),part(second,high,'fall')]
        if name=='vase':
            # A real imported leafy sprig; stem/foliage stays a separate historical flower ID.
            leaves=load(root,'potted_plant_02',1.1,location=(0,0,1.65))
            chosen=[o for o in leaves if 'plant' in o.name.lower() and o.name.split('.')[0].endswith('_leaves')]
            for o in leaves:
                if o not in chosen:bpy.data.objects.remove(o,do_unlink=True)
            parts.append(part('flower',prune_leaves(join(chosen,'vase foliage')),'fall'))
        return parts
    if name=='plant':
        objects=load(root,'potted_plant_02',2.3)
        leaves=[o for o in objects if 'plant' in o.name.lower() and o.name.split('.')[0].endswith('_leaves')]
        soil=[o for o in objects if o not in leaves]
        return [part('body',join(soil,'pot and soil'),'shatter'),part('foliage',prune_leaves(join(leaves,'plant foliage')),'fall')]
    if name=='books':
        objects=load(root,'book_encyclopedia_set_01',2.)
        selected=sorted(objects,key=lambda o:o.name)[8:11]
        for o in objects:
            if o not in selected:bpy.data.objects.remove(o,do_unlink=True)
        lo,hi=bounds(selected);center=Vector(((lo.x+hi.x)/2,(lo.y+hi.y)/2,0))
        for o in selected:o.data.transform(Matrix.Translation(-center))
        return [part(p,o,'fall') for p,o in zip(('left','middle','right'),selected)]
    if name=='tv':
        ob=join(load(root,'television_02' if NATURAL_MATERIALS else 'Television_01',2.),'TV')
        television_on(ob,modern=NATURAL_MATERIALS)
        lower=sliced(ob,'TV base',2,upper=.24);upper=sliced(ob,'TV body',2,lower=.24);bpy.data.objects.remove(ob,do_unlink=True)
        return [part('screen',upper,'crack'),part('stand',lower,'fall')]
    if name=='frame':
        ob=join(load(root,'standing_picture_frame_01',1.8),'standing picture frame')
        return [part('picture',ob,'fall')]
    if name=='clock':
        return [part('face',join(load(root,'alarm_clock_01',1.9),'clock'),'crack')]
    if name=='lamp':
        ob=join(load(root,'desk_lamp_arm_01',2.2),'lamp')
        low=sliced(ob,'lamp support',2,upper=1.55);high=sliced(ob,'lamp shade',2,lower=1.55);bpy.data.objects.remove(ob,do_unlink=True)
        return [part('base',low,'shatter'),part('shade',high,'fall')]
    if name=='mug':
        ob=join(load(root,'metal_jug',2.),'jug');lo,hi=bounds([ob]);cut=lo.x+(hi.x-lo.x)*.78
        left=sliced(ob,'jug main body',0,upper=cut)
        right=sliced(ob,'jug lower side',0,lower=cut)
        lower=sliced(right,'jug lower side',2,upper=1.30);bpy.data.objects.remove(right,do_unlink=True)
        high=sliced(ob,'jug right handle with mount',0,lower=cut)
        high2=sliced(high,'jug right handle',2,lower=1.30);bpy.data.objects.remove(high,do_unlink=True);high=high2
        low=join([left,lower],'jug body');bpy.data.objects.remove(ob,do_unlink=True)
        return [part('body',low,'shatter'),part('handle',high,'fall')]
    raise ValueError(name)


def room(root,cube,mat):
    """Scanned furnishings in a room measured in metres; diffuse daylight and warm whites."""
    objects=[]
    def add(asset,location,rotation=0,light=.4):
        obs=load(root,asset,location=location,rotation=rotation,light=light);objects.extend(obs);return bounds(obs)
    oak=mat('limed oak floor',(.61,.54,.4),0);plaster=mat('warm lime plaster',(.79,.77,.69),0)
    white=mat('window painted timber',(.84,.83,.76),0)
    def box(*args,**kw):o=cube(*args,**kw);objects.append(o);return o
    box('continuous floor',(0,0,-.08),(8,8,.08),oak,.01)
    box('back plaster',(0,1.8,1.6),(3,.08,1.6),plaster,.01)
    box('left plaster',(-2.5,-.5,1.6),(.08,2.3,1.6),plaster,.01)
    box('skirting',(0,1.7,.065),(3,.035,.065),white,.015)
    add('sofa_02',(.65,.95,0),light=.48)
    add('Shelf_01',(-1.55,1.53,0),light=.5)
    table_lo,table_hi=add('modern_coffee_table_01',(.5,-.55,0),math.pi/2,light=.55)
    bench_lo,bench_hi=add('side_table_01',(-1.3,-.55,0),light=.5)
    frame=load(root,'fancy_picture_frame_01',height=.75,location=(-.6,1.69,1.7),light=.5);objects.extend(frame)
    # Existing imported canvas remains textured, but receives a bright diffuse variant.
    sky=mat('daylight window',(.62,.78,.9),.25)
    box('window glass',(1.05,1.705,1.95),(1.0,.025,.65),sky,.005)
    for x in (.02,2.08):box('window jamb',(x,1.65,1.95),(.035,.055,.71),white,.015)
    for z in (1.25,2.65):box('window lintel',(1.05,1.65,z),(1.07,.055,.035),white,.015)
    box('window sill',(1.05,1.54,1.24),(1.12,.20,.045),white,.02)
    anchors=[('shelf_left','shelf',(-1.78,1.38,.9423)),('shelf_right','shelf',(-1.32,1.38,.9423)),
             ('sill_left','sill',(.55,1.50,1.285)),('sill_right','sill',(1.55,1.50,1.285)),
             ('table_left','table',(.18,-.55,table_hi.z)),('table_right','table',(.83,-.55,table_hi.z)),
             ('bench','bench',(-1.3,-.55,bench_hi.z)),
             ('floor_left','floor',(-1.9,-1.55,.01)),('floor_middle','floor',(-.62,-1.75,.01)),
             ('floor_right','floor',(.7,-1.85,.01)),('floor_far_right','floor',(1.9,-1.4,.01))]
    return objects,anchors
