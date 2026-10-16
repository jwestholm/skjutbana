"""Invisible first-person soldier target, with conservative trench sight testing.

Camera and physical projector are distinct coordinate systems. Player collision
lives exactly at the virtual camera X/Z; it is NEVER rendered or displaced by
Skjutbanan's 6.3m physical shooting distance.
"""
import math


def player_body(eye, eye_height):
    """Sphere centres, radii, aligned to the actual last PRESENTED eye position.

    World Y points up; standing 1.80m, kneeling ~1.04m, prone ~0.46m.
    The eye is at the top-ish of the body. A small constant head margin sits
    above it. Body grows/shrinks with posture, never a floating 1.8m dummy.
    """
    h = max(.43, min(1.80, float(eye_height) + .23))
    base = float(eye[1]) - float(eye_height)
    x, z = float(eye[0]), float(eye[2])
    if h < .7:
        return ((x, base + .29, z, .235), (x, base + .17, z, .27))
    return ((x, base + h - .15, z, .17),
            (x, base + h * .64, z, .27),
            (x, base + h * .36, z, .21))


def line_clear(origin, target, route_start, route_end, *, route, sides,
               wall_width, wall_height, floor_y, obstacles_in_range):
    """Reject shots through solid trench banks and existing solid crates.

    Approximate geometry test, not a final rendered-mesh triangle raycast.
    This favors blocking doubtful shots, NEVER claiming impossible X-ray hits.
    """
    x0,y0,z0=origin
    x1,y1,z1=target
    dist=math.dist(origin,target)
    if dist > 90.0 or dist < 1e-6:
        return False
    lo=max(0.0,min(route_start,route_end)-5.0)
    hi=max(route_start,route_end)+5.0
    centres=[]
    s=lo
    while s<=hi+.001:
        cx,cz=sides(s,0.0)
        centres.append((s,cx,cz))
        s+=.55
    if len(centres)<2:return False
    obstacles=[ob for ob in obstacles_in_range(lo,hi)
               if ob.kind in ('crate','rubble')]
    # Every segment of the 3D shot line must be inside the excavated corridor
    # when below the banks. Above banks is unobstructed by bank geometry.
    for j in range(1,max(2,math.ceil(dist/.34))):
        t=j/max(2,math.ceil(dist/.34))
        px=x0+(x1-x0)*t
        py=y0+(y1-y0)*t
        pz=z0+(z1-z0)*t
        best=(1e30,None,0.0)
        for i in range(len(centres)-1):
            sa,ax,az=centres[i];sb,bx,bz=centres[i+1]
            dx=bx-ax;dz=bz-az
            ratio=max(0.0,min(1.0,((px-ax)*dx+(pz-az)*dz)/max(1e-9,dx*dx+dz*dz)))
            cx=ax+dx*ratio;cz=az+dz*ratio
            r2=(px-cx)**2+(pz-cz)**2
            if r2<best[0]:best=(r2,sa+(sb-sa)*ratio,ratio)
        near_s=best[1]
        if near_s is None:return False
        heading=route(near_s)[2]
        rx,rz=sides(near_s,0.)
        across=(px-rx)*math.cos(heading)-(pz-rz)*math.sin(heading)
        side=1 if across>=0 else -1
        if py < floor_y(near_s) - .13:
            return False
        if abs(across) > wall_width(near_s,side) + .055 and py < wall_height(near_s,side)-.025:
            return False
        for ob in obstacles:
            bx,bz=sides(ob.s,ob.lateral)
            if abs(px-bx)<ob.width*.48 and abs(pz-bz)<ob.length*.52:
                ground=floor_y(ob.s,ob.lateral)
                if ground < py < ground + ob.height:
                    return False
    return True


def mesh_clear_body(origin,body,meshes):
    """Sight samples against PRESENTED solid triangles for off-trunk sentries.

    Unlike the old route cylinder, this respects excavated doorways. Never
    claim sight when the current world mesh has not been loaded yet.
    """
    if not meshes:return ()
    from shot_geometry import trace_world
    visible=[]
    for x,y,z,radius in body:
        dx,dy,dz=x-origin[0],y-origin[1],z-origin[2]
        length=math.sqrt(dx*dx+dy*dy+dz*dz)
        if length<.09 or length>55.:continue
        direction=(dx/length,dy/length,dz/length)
        if trace_world(origin,direction,length,meshes) is None:
            visible.append((x,y,z,radius))
    return tuple(visible)


def enemy_shown_to_player_mesh(frame,enemy_id,eye):
    """Last-presented frame: projection AND shared world mesh both agree."""
    if frame is None or not frame.world_shot_meshes:return False
    from combat_tactics import projected_visible
    for ident,part,center,radius in frame.hit_spheres:
        if ident!=enemy_id:continue
        if not projected_visible(center,frame.view_projection,frame.width,frame.height):
            continue
        if mesh_clear_body(eye,((*center,radius),),frame.world_shot_meshes):
            return True
    return False


def can_see_body(origin,body,route_start,route_end,**geometry):
    visible=[]
    for x,y,z,radius in body:
        # Ray to sphere centre. A small profile tolerance accounts for broad
        # body visibility, not a hit/aim-assist tolerance.
        if line_clear(origin,(x,y,z),route_start,route_end,**geometry):
            visible.append((x,y,z,radius))
    return tuple(visible)


def enemy_faces_player(enemy_position, player_eye, facing_yaw, *, half_fov_degrees=72.0, near_metres=4.3):
    """Vision cone for initial detection, with short-range peripheral awareness.

    This is gameplay visibility, not a physical camera calibration. Enemies
    investigating gunshots can point toward the source using their current yaw.
    """
    dx=player_eye[0]-enemy_position[0]
    dz=player_eye[2]-enemy_position[2]
    dist=math.hypot(dx,dz)
    if dist<=near_metres:return True
    if dist<1e-9:return True
    forward=(math.sin(facing_yaw),math.cos(facing_yaw))
    return (dx*forward[0]+dz*forward[1])/dist>=math.cos(math.radians(half_fov_degrees))


def enemy_shown_to_player(frame, enemy_id, eye, *, geometry):
    """Visible-to-player opportunity based on the EXACT LAST PRESENTED frame.

    Hostile body detection and the player's actual chance to shoot are separate
    authorities. No enemy can deliver a lethal round before a genuine visible
    shooting opportunity. This is an ARCADE fairness rule, not CV aim assist.
    """
    if frame is None:
        return False
    from combat_tactics import projected_visible
    for ident, part, centre, radius in frame.hit_spheres:
        if ident != enemy_id:
            continue
        if not projected_visible(centre, frame.view_projection,
                                 frame.width,frame.height):
            continue
        if line_clear(eye,centre,frame.route_m,
                      dict(frame.actor_routes).get(enemy_id,frame.route_m),
                      **geometry):
            return True
    return False
