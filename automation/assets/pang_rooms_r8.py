"""Distinct natural PBR rooms for the opt-in local-glow experiment.

Bitmap inputs are CC0 atlas crops (SOURCES.json). Dedicated architectural scans
are still requested in GAME_NEW_REPORT; no claim that these crops are new scans.
"""
from pathlib import Path
import math
import bpy
from . import pang_polyhaven as ph


def textured(name,texture,color=(1,1,1),scale=(3,3,1),roughness=.7):
    root=Path(__file__).resolve().parents[2]/'content/games/pang_room_textures'
    m=bpy.data.materials.new(name);m.use_nodes=True;t=m.node_tree;n=t.nodes;l=t.links;b=n.get('Principled BSDF')
    coord=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY';mapping.inputs[1].default_value=scale;l.new(coord.outputs['Generated'],mapping.inputs[0])
    def image(kind,noncolor=False):
        p=root/(texture+'_'+kind+'.png')
        if not p.exists():return None
        im=n.new('ShaderNodeTexImage');im.image=bpy.data.images.load(str(p),check_existing=True)
        if noncolor:im.image.colorspace_settings.name='Non-Color'
        l.new(mapping.outputs[0],im.inputs['Vector']);return im
    im=image('color');mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1
    l.new(im.outputs['Color'],mix.inputs[1]);mix.inputs[2].default_value=(*color,1);l.new(mix.outputs[0],b.inputs['Base Color'])
    normal=image('normal',True)
    if normal:
        normal_map=n.new('ShaderNodeNormalMap');normal_map.inputs['Strength'].default_value=.30
        l.new(normal.outputs['Color'],normal_map.inputs['Color']);l.new(normal_map.outputs['Normal'],b.inputs['Normal'])
    rough=image('rough',True);arm=image('arm',True)
    if rough:l.new(rough.outputs['Color'],b.inputs['Roughness'])
    elif arm:
        sep=n.new('ShaderNodeSeparateColor');l.new(arm.outputs['Color'],sep.inputs[0]);l.new(sep.outputs[1],b.inputs['Roughness'])
    else:b.inputs['Roughness'].default_value=roughness
    return m


def room(root,cube,mat,template='living',furnishing=0):
    if template not in ('living','kitchen','study') or furnishing not in (0,1,2):raise ValueError('room/layout')
    objects=[];anchors=[];shift=(furnishing-1)*.28
    def box(name,location,size,material,bevel=.025):
        o=cube(name,location,size,material,bevel);objects.append(o);return o
    def add(name,location,height=None,rotation=0):
        obs=ph.load(root,name,height=height,location=location,rotation=rotation,light=None);objects.extend(obs);return ph.bounds(obs)
    def anchor(name,support,xyz):anchors.append((name,support,xyz))
    oak=textured('real oak atlas grain','oak',scale=(9,2,1));wall=textured('painted mineral wall','plaster',
        color={'living':(.85,.69,.48),'kitchen':(.92,.94,.88),'study':(.31,.43,.54)}[template],scale=(2,2,1))
    stone=textured('scanned stone countertop','stone',scale=(4,4,1));linen=textured('upholstery bitmap','fabric',scale=(8,8,1))
    cream=textured('ivory painted wood','plaster',(.83,.81,.73));wood_dark=textured('walnut desk','oak',(.42,.30,.19),scale=(5,2,1))
    metal=mat('brushed steel',(.32,.34,.35),0);metal.node_tree.nodes.get('Principled BSDF').inputs['Metallic'].default_value=.8
    glass=mat('window daylight',(.65,.79,.92),.25)
    floor=oak if template=='living' else stone if template=='kitchen' else linen
    box('floor',(0,0,-.08),(8,8,.08),floor,.01)
    box('back wall',(0,2.0,2.5),(6,.08,2.5),wall,.01)
    box('left wall',(-2.65,-.5,2.5),(.08,4,2.5),wall,.01)
    box('skirting',(0,1.88,.075),(3,.035,.075),cream,.01)
    def window(x,z,width,height):
        box('daylight glass',(x,1.89,z),(width,.018,height),glass)
        for xx in (x-width,x+width):box('timber jamb',(xx,1.82,z),(.045,.065,height+.06),oak)
        for zz in (z-height,z+height):box('timber lintel',(x,1.82,zz),(width+.045,.065,.045),oak)
        box('sill',(x,1.68,z-height-.06),(width+.10,.25,.045),cream)
        anchor('sill_left','sill',(x-width*.55,1.60,z-height-.015))
        anchor('sill_right','sill',(x+width*.55,1.60,z-height-.015))
    if template=='living':
        window(-1.25+shift,2.05,.92,.65)
        add('Sofa_01',(.8,1.03,0),height=.92)
        add('side_table_01',(2.03,-.05,0));add('potted_plant_02',(2.03,-.05,.55),height=.75)
        add('CoffeeTable_01',(.60+shift,-.65,0),height=.43)
        box('wool rug',(.6,-.7,.012),(1.45,1.12,.012),linen,.008)
        # Wide natural oak TV console; target itself remains a separate sprite.
        tx=-1.45+shift
        box('oak TV console',(tx,-.78,.28),(.83,.38,.28),oak)
        for x in (-.43,.43):
            box('console door',(tx+x,-1.175,.29),(.40,.025,.22),wood_dark)
            box('console handle',(tx+x,-1.209,.43),(.10,.012,.008),metal,.008)
        anchor('tv_console','tv_bench',(tx,-.78,.56))
        for i,x in enumerate((.15,.70,1.24)):anchor('coffee_'+str(i),'table',(x+shift,-.65,.43))
        # Long low bookcase on the far side, populated below the target bays.
        sx=2.25
        add('Shelf_01',(sx,1.58,0))
        for i,z in enumerate((.3794,.9423,1.5052)):
            add('book_encyclopedia_set_01',(sx-.22,1.48,z),height=.23)
            anchor('shelf_'+str(i),'shelf',(sx+.18,1.41,z))
        add('standing_picture_frame_01',(2.03,-.05,.55),height=.20)
    elif template=='kitchen':
        # L-shaped kitchen, island, backsplash and actual coffee-maker geometry.
        tile=textured('ceramic tile scans','plaster',(.67,.79,.76),scale=(10,5,1))
        box('backsplash',(0,1.82,1.25),(2.45,.025,.30),tile,.006)
        for i,x in enumerate((-1.95,-1.20,-.45,.30,1.05,1.80)):
            box('base cabinet',(x,1.40,.43),(.36,.46,.43),cream)
            box('cabinet door',(x,.91,.44),(.33,.025,.38),oak)
            box('pull',(x,.875,.72),(.11,.015,.011),metal,.006)
        box('countertop',(0,1.36,.90),(2.56,.51,.045),stone)
        for i,x in enumerate((-1.55,-.60,.40,1.55)):anchor('counter_'+str(i),'table',(x,1.15,.945))
        # Upper cupboards replace R7's identical bookcase/window backdrop.
        for x in (-1.8,-.95,-.1):
            box('wall cupboard',(x,1.62,2.22),(.40,.30,.37),cream)
            box('cupboard glass',(x,1.29,2.22),(.34,.02,.30),glass)
        window(1.55,2.16,.79,.55)
        ix=.25+shift
        box('island',(ix,-.9,.43),(1.07,.60,.43),oak)
        box('island stone',(ix,-.9,.91),(1.13,.66,.055),stone)
        for i,x in enumerate((-.50,.2,.9)):anchor('island_'+str(i),'table',(x+shift,-1.0,.965))
        # Authored CC0 coffee machine, glass carafe and metal hotplate.
        black=mat('coffee machine black polymer',(.022,.025,.023),0)
        box('coffee maker',(-2.0,1.37,1.15),(.17,.20,.21),black)
        box('coffee maker crown',(-2.0,1.28,1.40),(.20,.26,.055),black)
        box('carafe',(-2.0,1.10,1.08),(.13,.12,.12),glass,.06)
        box('carafe handle',(-1.82,1.10,1.11),(.035,.08,.09),black)
        add('metal_jug',(1.95,1.42,.945),height=.28)
        box('kitchen shelf',(-1.95,-.70,.69),(.50,.25,.045),oak)
        for i,x in enumerate((-2.15,-1.73)):anchor('side_'+str(i),'shelf',(x,-.70,.735))
    else:
        window(1.45,2.15,.85,.63)
        # A study with two filled shelves, a proper desk and upholstered chair.
        for n,sx in enumerate((-1.95,-.83)):
            add('Shelf_01',(sx,1.58,0))
            for i,z in enumerate((.3794,.9423,1.5052)):
                add('book_encyclopedia_set_01',(sx-.21,1.44,z),height=.23)
                anchor('books_'+str(n)+'_'+str(i),'shelf',(sx+.17,1.34,z))
        dx=.75+shift
        box('walnut desktop',(dx,-.50,.80),(1.12,.54,.055),wood_dark)
        for x in (-.9,.9):
            box('desk pedestal',(dx+x,-.45,.39),(.19,.45,.39),oak)
            for z in (.20,.43,.65):box('drawer face',(dx+x,-.925,z),(.17,.02,.10),wood_dark)
        for i,x in enumerate((-.0,.7,1.4)):anchor('desk_'+str(i),'table',(x+shift,-.55,.855))
        # Empty chair behind desk, not an unmeasurable active target.
        box('chair seat',(.70, .43,.47),(.30,.29,.065),linen)
        box('chair back',(.70,.70,.86),(.31,.07,.38),linen,.08)
        for x in (.44,.96):
            for y in (.20,.64):box('chair leg',(x,y,.21),(.035,.035,.21),wood_dark,.012)
        add('standing_picture_frame_01',(1.55,-.45,.855),height=.22)
        box('side credenza',(-1.65,-1.05,.38),(.64,.30,.38),oak)
        for i,x in enumerate((-1.95,-1.35)):anchor('credenza_'+str(i),'bench',(x,-1.05,.76))
    # Floor supports cover the safe left/right/bottom board areas without corners.
    for i,(x,y) in enumerate(((-2.15,-1.05),(-1.25,-1.65),(-.35,-1.95),(.65,-2.0),(1.80,-1.70),(2.45,-.8))):
        anchor('floor_'+str(i),'floor',(x+shift*.4,y,.015))
    return objects,anchors
