"""PANG R7: natural-contrast room plates; all geometry in metres, local glTFs only."""
import math
from . import pang_polyhaven as ph

TEMPLATES=('living','kitchen','study')

def room(root,cube,mat,template='living',furnishing=0):
    if template not in TEMPLATES or furnishing not in (0,1):raise ValueError('Unknown room/layout')
    objects=[];anchors=[]
    def box(name,loc,half,color,rough=0):
        o=cube(name,loc,half,mat(name,color,0),.025);objects.append(o);return o
    def add(name,loc,height=None,rotation=0):
        obs=ph.load(root,name,height=height,location=loc,rotation=rotation,light=None);objects.extend(obs);return ph.bounds(obs)
    def anchor(name,support,xyz):anchors.append((name,support,xyz))
    oak=(.34,.22,.12);white=(.76,.74,.67)
    wall={'living':(.66,.62,.52),'kitchen':(.62,.70,.65),'study':(.57,.65,.70)}[template]
    box('oak floor',(0,0,-.08),(8,8,.08),oak)
    box('plaster',(0,1.8,1.6),(3,.08,1.6),wall)
    box('side wall',(-2.5,-.5,1.6),(.08,2.3,1.6),wall)
    box('skirting',(0,1.69,.065),(3,.04,.065),white)
    # Broad daylight opening with an unpatterned pale surround.
    wx=1.0 if furnishing==0 else .65
    box('window',(wx,1.705,1.98),(.95,.025,.61),(.74,.83,.9))
    for x in (wx-.99,wx+.99):box('window frame',(x,1.64,1.98),(.035,.055,.67),white)
    for z in (1.32,2.64):box('window frame',(wx,1.64,z),(1.025,.055,.035),white)
    box('window sill',(wx,1.49,1.28),(1.08,.22,.04),white)
    for i,x in enumerate((wx-.60,wx+.52)):anchor('sill_'+str(i),'sill',(x,1.43,1.32))
    # Bright shelf carcass with original PBR fascia; static decorative books stay
    # away from the blank shooting bays at the middle and upper levels.
    sx=-1.62 if furnishing==0 else -1.25
    add('Shelf_01',(sx,1.5,0))
    box('shelf bright backing',(sx,1.55,1.0),(.49,.025,.91),(.77,.75,.64))
    for j,z in enumerate((.39,.94,1.49)):
        if j in (0,2):
            add('book_encyclopedia_set_01',(sx-.22,1.42,z),height=.24)
        anchor('shelf_'+str(j),'shelf',(sx+.20,1.30,z))
    if template=='living':
        first=len(objects);add('sofa_02',(.75,1.0,0),height=.88)
        # Warm beige leather is a material variant, not a room-wide image lift.
        # Retain the source roughness/normal maps and real cushion geometry.
        mats={m for o in objects[first:] for m in o.data.materials if m and m.use_nodes}
        for material in mats:
            tree=material.node_tree;b=next(n for n in tree.nodes if n.type=='BSDF_PRINCIPLED')
            socket=b.inputs['Base Color'];mix=tree.nodes.new('ShaderNodeMixRGB')
            mix.inputs[0].default_value=.80;mix.inputs[2].default_value=(.48,.42,.28,1)
            if socket.is_linked:tree.links.new(socket.links[0].from_socket,mix.inputs[1])
            else:mix.inputs[1].default_value=socket.default_value
            tree.links.new(mix.outputs[0],socket)
        add('modern_coffee_table_01',(.65,-.63,0),rotation=math.pi/2)
        box('woven wool rug',(.65,-.65,.008),(1.45,1.05,.008),(.53,.48,.39))
        # TV now belongs on a wide media cabinet, not the coffee table.
        tx=-1.20 if furnishing==0 else -.75
        box('media cabinet',(tx,-.68,.29),(.82,.38,.29),(.30,.22,.14))
        box('media top',(tx,-.68,.60),(.87,.41,.035),(.80,.76,.64))
        for dx in (-.41,.41):
            box('media door',(tx+dx,-1.068,.31),(.39,.015,.23),(.41,.30,.19))
            box('media pull',(tx+dx,-1.091,.44),(.10,.012,.009),(.65,.58,.43))
        anchor('tv_console','tv_bench',(tx,-.68,.635))
        for i,x in enumerate((.18,1.0)):anchor('coffee_'+str(i),'table',(x,-.63,.39))
        add('side_table_01',(2.0,.30,0))
        add('potted_plant_02',(2.0,.30,.551),height=.58)
        # Foreground right is furnished, with a low pale sideboard for small targets.
        box('right low cabinet',(1.85,-1.43,.30),(.53,.34,.30),(.72,.69,.58))
        for i,x in enumerate((1.50,2.12)):anchor('right_'+str(i),'bench',(x,-1.43,.60))
    elif template=='kitchen':
        # Simple fitted pale cabinetry: broad fronts, no fine dark tile grid.
        for i,x in enumerate(tuple(x+.16*furnishing for x in (-.05,.65,1.35,2.05))):
            box('kitchen base '+str(i),(x,1.23,.44),(.34,.40,.44),(.68,.74,.64))
            box('kitchen door '+str(i),(x,.82,.44),(.32,.015,.40),(.52,.61,.49))
            box('kitchen handle '+str(i),(x,.794,.76),(.10,.014,.012),(.56,.52,.40))
        box('continuous worktop',(1.0+.16*furnishing,1.18,.91),(1.46,.48,.045),(.83,.80,.68))
        for i,x in enumerate((-.02,.76,1.63)):anchor('counter_'+str(i),'table',(x+.16*furnishing,1.12,.955))
        box('kitchen island',(.45-.22*furnishing,-.75,.40),(1.1,.60,.40),(.60,.67,.60))
        box('island worktop',(.45-.22*furnishing,-.75,.85),(1.15,.65,.055),(.80,.76,.63))
        for i,x in enumerate((-.25,.50,1.22)):anchor('island_'+str(i),'table',(x-.22*furnishing,-.9,.905))
        box('small display bench',(1.9,-1.65,.28),(.48,.31,.28),(.72,.69,.57))
        anchor('tv_console','tv_bench',(1.9,-1.65,.56))
        add('side_table_01',(-1.65,-.8,0))
        anchor('side','bench',(-1.65,-.8,.551))
    else:
        add('modern_coffee_table_01',(.4,-.6,0),rotation=math.pi/2)
        box('writing desk',(0.8-.25*furnishing,.75,.83),(1.1,.48,.055),(.67,.59,.43))
        for x in (-.18,1.78):box('desk leg',(x-.25*furnishing,.75,.4),(.055,.40,.4),(.48,.44,.35))
        for i,x in enumerate((.05,.75,1.45)):anchor('desk_'+str(i),'table',(x-.25*furnishing,.70,.885))
        add('side_table_01',(1.9,-1.5,0))
        anchor('side','bench',(1.9,-1.5,.551))
        box('studio rug',(.25,-.65,.008),(1.38,1.02,.008),(.42,.49,.46))
        box('media cabinet',(-1.2,-.70,.31),(.82,.38,.31),(.66,.64,.52))
        anchor('tv_console','tv_bench',(-1.2,-.70,.62))
        for i,x in enumerate((.1,.9)):anchor('coffee_'+str(i),'table',(x,-.6,.39))
    add('fancy_picture_frame_01',(-.4,1.68,1.85),height=.55)
    # Several safe, different floor positions, including previously empty lower right.
    for i,(x,y) in enumerate(((-1.85,-1.5),(-.7,-1.75),(.65,-1.9),(1.9,-2.0))):
        anchor('floor_'+str(i),'floor',(x,y,.015))
    anchor('floor_outer_left','floor',(-2.05,-1.05,.015))
    anchor('floor_outer_right','floor',(2.55,-.70,.015))
    # Second furnishing uses genuine furniture/world-coordinate variations, not a
    # mirrored bitmap (book text/materials stay correct).
    if furnishing:
        for i,(name,support,xyz) in enumerate(anchors):
            if support=='floor':anchors[i]=(name,support,(xyz[0]+.12,xyz[1]+.06,xyz[2]))
    return objects,anchors
