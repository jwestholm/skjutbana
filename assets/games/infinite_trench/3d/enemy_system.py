"""V3.3 First Contact: deterministic enemy encounters and screen-space 3D hit query.

No PyBullet, no OpenGL dependency in the decision logic. 3D display is an
interchangeable mesh-prototype renderer; future GLB rigged avatars plug in here.
Only the player can resolve a living threat.  Enemy fire is visual in V3.3.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from combat_tactics import difficulty_at, encounter_lateral


@dataclass(frozen=True)
class EnemyProfile:
    kind: str
    react_time: float
    aim_time: float
    fire_interval: float
    crouch_during_reload: bool = False


PROFILES = {
    'unaware': EnemyProfile('unaware', .95, .90, 1.15),
    'ambush': EnemyProfile('ambush', .38, .52, .92),
    'cover': EnemyProfile('cover', .65, .60, 1.25, True),
    # Rusher behaviour is reserved until collision/approach-path support exists.
    'rush': EnemyProfile('rush', .50, .68, .85),
}


@dataclass(frozen=True)
class EnemySpec:
    enemy_id: int
    route_m: float
    lateral: float
    profile: str
    branch_index: int | None = None
    branch_side: int = 0
    branch_depth: float = 0.0
    exterior_side: int = 0
    exterior_clearance: float = 1.15
    initial_gaze_phase: float = 0.0
    visual_variant: int | None = None


def encounter_unit(seed,index,salt=0):
    """Independent encounter stream; stable across processes and chunk order."""
    v=(int(seed)^0xC8013EA4^(int(index)*0x9E3779B9)^(salt*0x85EBCA6B))&0xffffffff
    v=((v^(v>>16))*0x7FEB352D)&0xffffffff
    v=((v^(v>>15))*0x846CA68B)&0xffffffff
    return (v^(v>>16))/4294967296.


def encounter_spec(index,seed=None):
    """Stable, bounded lookups; no million-enemy registry or RNG state."""
    if index < 0:
        raise ValueError('encounter index must be non-negative')
    # Explicit world seed drives gameplay; no-seed calls preserve older
    # authored scenery/cover anchors and legacy fixtures.
    if seed is not None:
        # Bounded random intervals, analytically indexed: no growing plan cache.
        # The no-seed path below remains available for legacy scenery anchors.
        if index<4:base,jitter=34.+52.*index,7.
        elif index<9:base,jitter=238.+44.*(index-4),6.
        else:base,jitter=470.+34.*(index-9),5.
        s=base+(encounter_unit(seed,index)*2.-1.)*jitter
        lateral=(.30+.28*encounter_unit(seed,index,1))*(1 if encounter_unit(seed,index,2)>.5 else -1)
        kinds=('unaware',) if s<100 else ('unaware','ambush','cover')
        kind=kinds[min(len(kinds)-1,int(encounter_unit(seed,index,3)*len(kinds)))]
        return EnemySpec(index,s,lateral,kind,
                         initial_gaze_phase=encounter_unit(seed,index,4)*math.tau,
                         visual_variant=int(encounter_unit(seed,index,5)*3))
    first = (
        (32.0, -.43, 'unaware'),
        (164.0, .51, 'ambush'),
        (258.0, .38, 'cover'),
        (384.0, -.48, 'unaware'),
    )
    if index < len(first):
        s, lateral, kind = first[index]
    else:
        # Main-line patrols after the four stable, manually tested contacts.
        # 72m spacing puts patrols BETWEEN recurring 72m junction markers:
        # first is 516 (junctions 496/568), then 588, 660, ...
        # Side-branch spawns and hiding are deliberately not faked here.
        s = 516.0 + (index-4)*72.0
        lateral = encounter_lateral(index)
        kind = ('ambush','cover','unaware')[index % 3]
    return EnemySpec(index, s, lateral, kind)


# Later encounters occasionally have a second real soldier, not a second
# hitbox on the same actor. High IDs cannot overlap infinite primary IDs in
# any realistically addressable route window; no infinite pair registry.
PAIR_ID_BASE = 1000000000
PAIR_FIRST = 4
PAIR_PERIOD = 5


def encounter_partner_spec(primary_index,seed=None):
    """Seed-stable companion beside selected encounters, capped by distance.

    The legacy no-seed path preserves old cover anchors. A companion is
    its own independently simulated, shot-frozen enemy with a distinct ID.
    """
    lead=encounter_spec(primary_index,seed)
    if seed is None:
        if primary_index < PAIR_FIRST or (primary_index-PAIR_FIRST) % PAIR_PERIOD:return None
    else:
        chance=min(.55,max(0.,(lead.route_m-100.)/1100.))
        if encounter_unit(seed,primary_index,6)>=chance:return None
    flank=-1 if lead.lateral>=0 else 1
    return EnemySpec(PAIR_ID_BASE+primary_index,lead.route_m+2.45,
                     flank*.57,'cover' if primary_index%2 else 'ambush',
                     initial_gaze_phase=lead.initial_gaze_phase,
                     visual_variant=lead.visual_variant)


BRANCH_ENEMY_ID_BASE = 2000000000


def side_room_enemy_spec(index,side):
    """A REAL side-room sentry in a sparse subset of existing dugouts.

    Route metre identifies its junction for streaming, not a fake world
    position. Rendering, frozen hit spheres and sight resolve branch geometry.
    """
    if index<7 or index%8!=7:
        return None
    return EnemySpec(BRANCH_ENEMY_ID_BASE+index,64.0+72.0*index,0.0,
                     'ambush',branch_index=index,
                     branch_side=1 if side>=0 else -1,
                     branch_depth=2.8)


EXTERIOR_PATROL_ID_BASE = 3000000000


def exterior_patrol_spec(index,seed=None):
    """Sparse REAL above-bank patrols away from openings and tutorial combat.

    One lookout near the midpoint between 72m-spaced junctions every eight
    main encounters; side is deterministic, not a screen-space placement.
    """
    if index < 7 or index % 8 != 7:
        return None
    route_m = encounter_spec(index,seed).route_m
    side = 1 if (index // 8) % 2 else -1
    return EnemySpec(EXTERIOR_PATROL_ID_BASE + index,route_m,0.,'unaware',
                     exterior_side=side,exterior_clearance=1.30)


def off_trunk(spec):
    """Real exterior and side-room enemies require *triangle* LOS authority."""
    return spec.branch_index is not None or spec.exterior_side != 0


def primary_for_partner(enemy_id):
    index=int(enemy_id)-PAIR_ID_BASE
    return (index if index>=PAIR_FIRST and
            (index-PAIR_FIRST)%PAIR_PERIOD==0 else None)


def _vec_sub(a,b):
    return tuple(ai-bi for ai,bi in zip(a,b))


def _dot(a,b):
    return sum(ai*bi for ai,bi in zip(a,b))


def _normalize(a):
    length = math.sqrt(max(0.0,_dot(a,a)))
    if length < 1e-9: return (0.,0.,0.)
    return tuple(x/length for x in a)


def ray_sphere(origin,direction,center,radius):
    """Closest positive ray/sphere intersection distance or None."""
    oc = _vec_sub(origin,center)
    b = _dot(oc,direction)
    c = _dot(oc,oc) - radius*radius
    delta = b*b-c
    if delta < 0: return None
    root = math.sqrt(delta)
    for t in (-b-root,-b+root):
        if t > .045: return t
    return None


def ray_from_screen(x,y,width,height,view_projection):
    """Unproject Pygame TOP-left pixel coordinates through the same render MVP.

    This is the future verified physical Game Space -> camera ray adapter.
    Only input changes when integrated with Skjutbanan; physical detection
    stays completely outside this module.
    """
    import numpy as np
    if width <= 0 or height <= 0: raise ValueError('invalid viewport')
    nx = 2.0*(x+.5)/width - 1.0
    ny = 1.0-2.0*(y+.5)/height
    inv = np.linalg.inv(np.asarray(view_projection,dtype='f8'))
    points=[]
    for depth in (-1.,1.):
        p=inv @ np.array([nx,ny,depth,1.],dtype='f8')
        p/=p[3]
        points.append(tuple(float(t) for t in p[:3]))
    return points[0],_normalize(_vec_sub(points[1],points[0]))


class EnemyAgent:
    """Owns one enemy's behavior; no magical knowledge of other combatants."""
    def __init__(self,spec):
        self.spec=spec
        self.profile=PROFILES[spec.profile]
        self.state='idle'
        self.state_time=0.0
        self.total_time=0.0
        self.shots_fired=0
        self.dead=False
        self.injured_part=None
        self.anim_revision=0
        # Continuous visual pose channels; logical combat states are discrete,
        # but weapon/stance must NEVER snap when switching states.
        self.pose_aim=0.0
        self.pose_crouch=0.0
        self.blast_timer=0.0
        self.blast_level=0
        # Current WORLD-route position. Must be used by model AND frozen hit
        # geometry; the seed-original EnemySpec is immutable metadata only.
        self.route_m=float(spec.route_m)
        self.investigation_goal=None
        self.heard_shots=0
        # Tactical stage is visible in the real actor and the frozen hitboxes.
        # No ghost peeking: lateral displacement modifies ONE shared world pose.
        self.lateral_offset=0.0
        self.peek_count=0
        self.search_sweeps=0
        self.tactical=False
        self.tactical_difficulty=difficulty_at(spec.route_m)
        # V3.9.2: animation follows ACTUAL route displacement, never a "running"
        # state label that could animate the feet while blocked by the world.
        self.gait_phase=0.0
        self.gait_strength=0.0
        self.gait_speed_mps=0.0
        self.head_look=0.0
        # World-space cover target; actual visible position AND frozen hit
        # spheres always use route_m + lateral_offset from this same agent.
        self.cover_goal=None          # (route_m, lateral, object_id)
        self.cover_obstacle_id=None
        self.cover_reactions=0
        self.ally_lost=False
        self.last_cover_reaction_time=-1000.0
        self.patrol_origin=float(spec.route_m)

    def hear_shot(self,origin_route_m,max_range=78.0):
        """Enter route-following investigation if a NEW gunshot is ahead.

        This early main-route-only perception uses path distance, not magical
        X-ray awareness of off-route dugouts. Do not chase from behind player.
        """
        if self.dead or self.state!='idle' or off_trunk(self.spec):return False
        ahead=self.route_m-float(origin_route_m)
        if not (1.5<ahead<=max_range):return False
        self.investigation_goal=max(float(origin_route_m)+10.,self.route_m-42.)
        self.heard_shots+=1
        self._transition('investigating')
        return True

    def react_to_blast(self,level):
        """A brief suppression reflex. Does NOT spawn/kill/engage a target."""
        level=int(level)
        if level<=0 or self.dead:return False
        self.blast_level=max(self.blast_level,level)
        self.blast_timer=max(self.blast_timer,{1:.40,2:1.05,3:1.65}[self.blast_level])
        return True

    def react_to_fire(self, cover=None, *, ally_lost=False):
        """React to audible nearby player fire without acquiring player LOS.

        A REAL static box or sandbag site may be supplied by the director.
        Otherwise this is suppression, not an invented invulnerable cover.
        """
        if self.dead or self.state in ('idle','investigating'):
            return False
        # Do not restart a cover maneuver at 60 Hz or on every rapid shot.
        if (self.state in ('seeking_cover','sheltered','suppressed') or
            self.total_time-self.last_cover_reaction_time<2.2):
            return False
        self.last_cover_reaction_time=self.total_time
        if ally_lost:
            self.ally_lost=True
        self.cover_reactions+=1
        if cover is not None:
            # An obstacle must be between the player and this actor; the
            # director is responsible for that physical world-space check.
            s=float(cover.s)
            lateral=float(cover.lateral)
            self.cover_goal=(s+max(.61,cover.length*.5+.28),lateral,cover.index)
            self.cover_obstacle_id=cover.index
            self._transition('seeking_cover')
        else:
            self.cover_goal=None
            self.cover_obstacle_id=None
            self._transition('suppressed')
        return True

    def _cover_movement(self,dt):
        """Finite, continuous displacement; never jump rendered or hit pose."""
        state=self.state
        if state=='suppressed':
            goal_crouch=1.0
            goal_aim=.05
            if self.state_time>=1.00:
                self._transition('locating' if self.tactical else 'turning')
        elif state=='seeking_cover':
            if self.cover_goal is None:
                self._transition('suppressed')
                return True
            target_route,target_lane,_=self.cover_goal
            ds=target_route-self.route_m
            self.route_m+=math.copysign(min(abs(ds),2.1*dt),ds) if abs(ds)>1e-6 else 0.
            lateral_target=target_lane-self.spec.lateral
            diff=lateral_target-self.lateral_offset
            self.lateral_offset+=math.copysign(min(abs(diff),1.15*dt),diff) if abs(diff)>1e-6 else 0.
            goal_crouch=.52
            goal_aim=.03
            if abs(ds)<.11 and abs(diff)<.10:
                self._transition('sheltered')
        elif state=='sheltered':
            goal_crouch=1.0
            goal_aim=.04
            if self.state_time>=1.35:
                self._transition('cover_peek')
        elif state=='cover_peek':
            # Deliberately expose a real shoulder out from the side of the
            # static prop, not a magic 'fire through cover' exemption.
            lane=(self.cover_goal[1] if self.cover_goal else self.spec.lateral)
            desired=lane-math.copysign(.43,lane)
            diff=(desired-self.spec.lateral)-self.lateral_offset
            self.lateral_offset+=math.copysign(min(abs(diff),.95*dt),diff) if abs(diff)>1e-6 else 0.
            goal_crouch=.22
            goal_aim=.44
            if self.state_time>=.84:
                self._transition('aiming')
        else:
            return False
        self.pose_crouch+=(goal_crouch-self.pose_crouch)*(1.-math.exp(-dt*6.0))
        self.pose_aim+=(goal_aim-self.pose_aim)*(1.-math.exp(-dt*5.0))
        return True

    def engage(self):
        if not self.dead and self.state in ('idle','investigating','alert'):
            self._transition('turning')

    def begin_tactical_encounter(self):
        """A fair staged contact: scan, peek/retract, advance, locate, fire.

        This is not line-of-sight authority. The director must still verify
        actual sight before starting; the game separately requires the player
        to see the actor before hostile shots may be lethal.
        """
        if self.dead or self.tactical or self.state not in ('idle','alert','investigating'):
            return False
        self.tactical=True
        self._transition('scanning')
        return True

    def _tactical_pose(self, dt, player_route):
        """Advance staged combat without teleporting or acquiring magic vision."""
        cfg=self.tactical_difficulty
        side=1 if self.spec.enemy_id%2 else -1
        offset=0.0
        crouch=0.0
        phase=self.state
        if phase=='scanning':
            crouch=.35
            if self.state_time>=cfg.scan_seconds:
                self._transition('peeking')
        elif phase=='peeking':
            offset=side*.20
            crouch=.30
            if self.state_time>=.55:
                self.peek_count+=1
                self._transition('hiding')
        elif phase=='hiding':
            offset=-side*.16
            crouch=1.0
            if self.state_time>=.38:
                self._transition('peeking' if self.peek_count<cfg.peek_cycles else 'rushing')
                if self.state=='rushing':self.rush_from=self.route_m
        elif phase in ('rushing','repositioning'):
            offset=0.0
            crouch=.14
            goal=max(0.0, float(player_route) + 6.8) if player_route is not None else -1e8
            speed=cfg.rush_speed*(.80 if phase=='repositioning' else 1.0)
            self.route_m=max(goal,self.route_m-speed*dt)
            distance_moved=getattr(self,'rush_from',self.route_m)-self.route_m
            if distance_moved>= (2.8 if phase=='rushing' else 2.5) or self.route_m<=goal+1e-5:
                self._transition('locating')
        elif phase=='locating':
            crouch=.2
            if self.state_time>=cfg.locate_seconds:
                self._transition('turning')
        self.pose_crouch+=(crouch-self.pose_crouch)*(1.0-math.exp(-dt*4.4))
        self.pose_aim+=(0.0-self.pose_aim)*(1.0-math.exp(-dt*5.5))
        self.lateral_offset+=(offset-self.lateral_offset)*(1.0-math.exp(-dt*5.8))
        return True

    def update(self,dt,visible=True,player_route=None):
        """Advance behavior, then animation from the *actual* travelled metres.

        Separate from draw frequency and hit tests: the same deterministic
        actor pose is consumed by rendering and shot-frozen hit spheres.
        """
        before=self.route_m
        self._update_behavior(dt,visible,player_route)
        dt=min(.1,max(0.0,float(dt)))
        travelled=abs(self.route_m-before)
        speed=travelled/max(dt,1.e-6)
        # Smooth actual velocity independently of the 0..1 gait intensity.
        # This allows walk/run animation differences without fake route motion.
        gait_target=0.0 if self.dead else min(speed,5.0)
        self.gait_speed_mps+=(gait_target-self.gait_speed_mps)*(1.-math.exp(-dt*7.))
        target=min(1.0,speed/2.4) if not self.dead else 0.0
        k=1.0-math.exp(-dt*(9.0 if target>self.gait_strength else 6.0))
        self.gait_strength+=(target-self.gait_strength)*k
        if travelled>.00001:
            # Roughly one left+right step cycle per 1.25 m of travel.
            self.gait_phase=(self.gait_phase+travelled*math.tau/1.25)%math.tau
        if self.state in ('scanning','peeking','hiding','locating','searching','alert',
                          'seeking_cover','sheltered','cover_peek','suppressed'):
            target_look=.44*math.sin(self.total_time*(1.8+.15*self.tactical_difficulty.level)
                                       +self.spec.enemy_id*.61)
        else:
            target_look=.06*math.sin(self.total_time*1.2+self.spec.enemy_id)
        if self.dead:target_look=0.0
        self.head_look+=(target_look-self.head_look)*(1.0-math.exp(-dt*5.8))

    def _update_behavior(self,dt,visible=True,player_route=None):
        if dt<0:raise ValueError('negative timestep')
        dt=min(dt,.1)
        self.total_time+=dt
        if self.dead:
            # Death pose is driven by state_time. Freezing this clock left
            # eliminated enemies standing forever despite correct hit results.
            self.state_time+=dt
            return
        if self.blast_timer>0.0:
            self.blast_timer=max(0.0,self.blast_timer-dt)
            # Hold the existing combat phase, but suppress firing and duck.
            target_crouch=1.0 if self.blast_level>=2 else .35
            self.pose_crouch+=(target_crouch-self.pose_crouch)*(1.-math.exp(-dt*5.5))
            self.pose_aim+=(0.0-self.pose_aim)*(1.-math.exp(-dt*6.0))
            if self.blast_timer>0.0:
                return
            self.blast_level=0
        self.state_time+=dt
        if self.spec.exterior_side != 0 and self.state=='idle':
            # Patrol is bounded and frame-time independent. Animation, shots,
            # and world pose all consume this SAME continuously moving route_m.
            self.route_m=(self.patrol_origin + 3.0*math.sin(
                self.total_time*.32)*self.spec.exterior_side)
            self.pose_aim+=(0.0-self.pose_aim)*(1.-math.exp(-dt*5.))
            return
        if self.state in ('suppressed','seeking_cover','sheltered','cover_peek'):
            self._cover_movement(dt)
            return
        if not visible and self.state in ('turning','aiming','firing','cover'):
            self._transition('searching')
        elif visible and self.state=='searching':
            self._transition('locating' if self.tactical else 'turning')
        if self.state=='searching':
            self.pose_aim+=(0.0-self.pose_aim)*(1.-math.exp(-dt*5.0))
            if self.tactical and self.state_time>=1.35:
                self.search_sweeps+=1
                self.rush_from=self.route_m
                self._transition('repositioning')
            return
        if self.tactical and self.state in ('scanning','peeking','hiding',
                                            'rushing','repositioning','locating'):
            self._tactical_pose(dt,player_route)
            return
        if self.state=='investigating':
            # Single-lane route-following patrol: no position snapping and no
            # movement before an actual heard shot. Runs at 2.55 m/s until the
            # audible source area, then waits alert, not firing through banks.
            previous=self.route_m
            self.route_m=max(self.investigation_goal,self.route_m-2.55*dt)
            target_crouch=.06
            self.pose_crouch+=(target_crouch-self.pose_crouch)*(1.-math.exp(-dt*5.))
            self.pose_aim+=(0.0-self.pose_aim)*(1.-math.exp(-dt*5.))
            if self.route_m<=self.investigation_goal+1e-5:
                self._transition('alert')
            return
        if self.state=='turning' and self.state_time>=self.profile.react_time:
            self._transition('aiming')
        elif self.state=='aiming' and self.state_time>=self.profile.aim_time:
            # The muzzle flash is visible upon entering FIRING, not when the
            # state expires a second later. Count precisely that visual shot.
            self.shots_fired+=1
            self._transition('firing')
        elif self.state=='firing' and self.state_time>=self.profile.fire_interval:
            # Cover agent actually ducks on reload, other agents re-aim.
            self._transition('cover' if self.profile.crouch_during_reload and self.shots_fired%3==0 else 'aiming')
        elif self.state=='cover' and self.state_time>=1.55:
            self._transition('aiming')
        self.lateral_offset+=(0.0-self.lateral_offset)*(1.-math.exp(-dt*5.8))
        # Separate target stance from discrete AI. A continuous channel prevents
        # teleporting between crouch and upright or dropping the weapon each
        # time FIRING becomes AIMING. Rates are in inverse seconds.
        target_aim=(1.0 if self.state in ('firing','aiming') else
                    0.38 if self.state=='cover' else 0.0)
        target_crouch=1.0 if self.state=='cover' else 0.0
        self.pose_aim+=(target_aim-self.pose_aim)*(1.-math.exp(-dt*4.8))
        self.pose_crouch+=(target_crouch-self.pose_crouch)*(1.-math.exp(-dt*3.5))

    def _transition(self,state):
        self.state=state
        self.state_time=0.0
        self.anim_revision+=1

    def eliminate(self,part):
        if self.dead:return False
        self.injured_part=part
        self.dead=True
        self._transition('dead')
        return True

    @property
    def threat(self):
        return self.state not in ('idle','alert','dead')

    def turn_fraction(self):
        if self.state=='idle':return 0.
        if self.state=='turning':
            x=max(0.,min(1.,self.state_time/self.profile.react_time))
            return x*x*(3-2*x)
        return 1.

    def aim_fraction(self):
        return self.pose_aim

    def crouch_fraction(self):
        return self.pose_crouch

    def hit_regions(self,center,heading=0.0):
        """Frozen proxy centres sampled from the SAME animated joint pose.

        No artificially enlarged aim assist. This remains sphere-based collision;
        it is not a replacement for future triangle/skinned rig picking.
        """
        if self.dead:return ()
        joints=actor_joints(self)
        x,y,z=center
        c,si=math.cos(heading),math.sin(heading)
        def pt(local):
            dx,dy,dz=local
            return (x+c*dx+si*dz,y+dy,z-si*dx+c*dz)
        return (
            ('head',pt(joints['head']),.179),
            ('torso',pt(joints['torso']),.307),
            ('pelvis',pt(joints['pelvis']),.221),
            ('left_leg',pt(joints['knees'][0]),.150),
            ('right_leg',pt(joints['knees'][1]),.150),
        )


@dataclass(frozen=True)
class FrozenEnemyFrame:
    """The exact 3D pose presented for one frame, independent of later AI updates.

    Store this with a shot-time snapshot when Skjutbanan integrates the game.
    One frame consumes only a handful of tiny tuples, never a texture/mesh.
    Pygame mouse tests use the latest PRESENTED frame, not the next update.
    """
    width: int
    height: int
    view_projection: tuple
    route_m: float
    hit_spheres: tuple  # (enemy_id, part_name, world_center, radius)
    actor_routes: tuple = ()  # immutable (enemy_id, frozen route_m) for LOS
    world_shot_meshes: tuple = ()  # actual PRESENTED static triangles; refs, no copy


class EnemyDirector:
    """Activates only nearby deterministic encounters; remembers resolved IDs.

    There is no autonomous kill or expiry. Once a threat engages, it holds the
    soldier until a validated Game Space hit eliminates the agent.
    """
    def __init__(self,world_seed=None):
        self.world_seed=world_seed
        self.active={}
        self.eliminated=set()
        self.contact_count=0
        self.last_hit=None
        self.gunshots_this_frame=()
        self.manual_presentation=False
        self.manual_waiting={}
        self.manual_target=None

    def reset(self):
        self.active.clear()
        self.manual_waiting.clear()
        self.manual_target=None
        self.eliminated.clear()
        self.contact_count=0
        self.last_hit=None
        self.gunshots_this_frame=()

    def set_manual_presentation(self,enabled,route_m,preferred=()):
        """Admission gate only: the encounter plan and actor FSM are unchanged.

        Deferred actors are absent from BOTH rendering and frozen hit regions.
        Existing actor objects retain their state when toggling either way.
        """
        self.manual_presentation=bool(enabled)
        if enabled:
            self._present_one(route_m,preferred)
        else:
            self.active.update(self.manual_waiting)
            self.manual_waiting.clear()
            self.manual_target=None

    def _present_one(self,route_m,preferred=()):
        if not self.manual_presentation:return
        candidates=dict(self.manual_waiting)
        candidates.update({i:a for i,a in self.active.items() if not a.dead})
        # Same spatial lifetime as normal actors. Eviction never eliminates an
        # ID: its seed-defined encounter remains available on returning here.
        candidates={i:a for i,a in candidates.items() if not a.dead
                    and i not in self.eliminated and route_m-85<=a.route_m<=route_m+100}
        target=self.manual_target
        # Do not let an already passed/hidden side patrol monopolise the
        # presentation while playable main-route encounters wait ahead.
        ahead=[i for i,a in candidates.items() if a.route_m>=route_m-3 and not off_trunk(a.spec)]
        if target in candidates and ahead and (candidates[target].route_m<route_m-3 or off_trunk(candidates[target].spec)):
            target=None
        if target not in candidates:
            preferred=set(preferred)
            target=min(candidates,key=lambda i:(i not in preferred,off_trunk(candidates[i].spec),
                candidates[i].route_m<route_m-3,
                abs(candidates[i].route_m-route_m),i)) if candidates else None
        self.active={i:a for i,a in self.active.items() if a.dead}
        if target is not None:self.active[target]=candidates.pop(target)
        self.manual_waiting=candidates
        self.manual_target=target

    def react_to_player_fire(self,route_m,cover_candidates=None,fallen_enemy_id=None):
        """Bounded player-shot stimulus to nearby ALREADY alerted combatants.

        Physical cover candidates come from the same deterministic prop list
        that the game renders and its raycast intersects. Hearing a shot does
        not grant visual awareness or a license to fire through earth.
        """
        responded=[]
        for idx,agent in sorted(self.active.items()):
            if agent.dead or not (0.0 < agent.route_m-route_m < 25.):
                continue
            if agent.state in ('idle','investigating'):
                continue
            cover=None
            if cover_candidates is not None and not off_trunk(agent.spec):
                candidates=cover_candidates(agent.route_m-5.0,agent.route_m+.3)
                options=[ob for ob in candidates
                         if ob.kind=='crate' and ob.height>=.78 and
                         route_m+.7<ob.s<agent.route_m-.5 and
                         agent.route_m-ob.s<=4.0]
                if options:
                    cover=min(options,key=lambda ob:(abs(agent.route_m-ob.s),
                                abs(agent.spec.lateral-ob.lateral)))
            ally_lost=(fallen_enemy_id is not None and
                       (primary_for_partner(idx)==fallen_enemy_id or
                        primary_for_partner(fallen_enemy_id)==idx))
            if agent.react_to_fire(cover,ally_lost=ally_lost):
                responded.append(idx)
        return tuple(responded)

    def hear_shot(self,route_m,max_range=78.0):
        """Gunfire is a local cue. Returns IDs of newly alerted patrols.

        An existing patrol only hears after it has been generated into the
        bounded active window. Repeated shots do not re-trigger or teleport.
        """
        alerted=[]
        for idx,agent in sorted(self.active.items()):
            if agent.hear_shot(route_m,max_range):
                alerted.append(idx)
        return tuple(alerted)

    def update(self,dt,route_m,can_see_player=None,branch_side_at=None):
        # Legacy no-seed fixtures use the old 72m grid. Seeded gameplay uses
        # the bounded progression window below; never allocate an infinite plan.
        first=max(4,4+int(math.floor((route_m-105.-516.)/72.)))
        last=max(first+1,4+int(math.ceil((route_m+105.-516.)/72.)))
        primary_indices=list(range(4))+list(range(first,last+1))
        seed=self.world_seed
        if seed is not None:
            # At most a small fixed window, even millions of metres in.
            first=max(9,9+int(math.floor((route_m-110.-470.)/34.)))
            last=max(first,9+int(math.ceil((route_m+110.-470.)/34.)))
            primary_indices=list(range(9))+list(range(first,last+1))
        # Companions are generated by the same bounded active window, not an
        # infinite global registry. Their IDs and positions are deterministic.
        specs=[]
        for idx in primary_indices:
            specs.append(encounter_spec(idx,seed))
            partner=encounter_partner_spec(idx,seed)
            if partner is not None:
                specs.append(partner)
        # Sparse stationary sentries in REAL side rooms. The game supplies
        # junction_side() so this cannot diverge from world excavation sides.
        if branch_side_at is not None:
            first_branch=max(7,int(math.floor((route_m-100.-64.)/72.)))
            last_branch=max(first_branch,int(math.ceil((route_m+100.-64.)/72.)))
            for branch_i in range(first_branch,last_branch+1):
                if branch_i%8!=7:continue
                side=branch_side_at(branch_i)
                specs.append(side_room_enemy_spec(branch_i,side))
        # Sparse above-bank patrols: genuine world-ground positions and
        # independent actors. They NEVER become blocking main-route threats.
        ext_first=max(7,int(math.floor((route_m-100.-516.)/72.))+4)
        ext_last=max(ext_first,int(math.ceil((route_m+100.-516.)/72.))+4)
        for i in (range(ext_first,ext_last+1) if seed is None else primary_indices):
            if i%8==7:
                specs.append(exterior_patrol_spec(i,seed))
        for spec in specs:
            if spec is None:continue
            idx=spec.enemy_id
            if idx in self.eliminated:continue
            if abs(spec.route_m-route_m)<=90.0:
                if self.manual_presentation:
                    if idx not in self.active and idx not in self.manual_waiting:
                        self.manual_waiting[idx]=EnemyAgent(spec)
                else:self.active.setdefault(idx,EnemyAgent(spec))
        self._present_one(route_m)
        gunshots=[]
        shot_ids=[]
        self.gunshots_this_frame=()
        for idx,agent in list(self.active.items()):
            if agent.dead:
                self.eliminated.add(idx)
            if agent.route_m<route_m-85 or agent.route_m>route_m+100:
                del self.active[idx]
                continue
            # Visible/nearby patrol reaches combat distance and stops moving
            # even when it was previously following a distant sound.
            close=not agent.dead and 0.0<agent.route_m-route_m<=16.0
            visible=(can_see_player(agent) if can_see_player is not None and close else
                     can_see_player(agent) if can_see_player is not None and
                     agent.state in ('turning','aiming','firing','cover','searching',
                                     'suppressed','seeking_cover','sheltered','cover_peek') else True)
            if close and visible and agent.state in ('idle','investigating','alert'):
                if off_trunk(agent.spec):
                    # A room sentry uses ordinary reaction/aiming, not the
                    # main-trench rushing FSM that would move him through dirt.
                    agent.engage()
                    self.contact_count+=1
                elif agent.begin_tactical_encounter():
                    self.contact_count+=1
            before=agent.shots_fired
            agent.update(dt,visible=visible,player_route=route_m)
            if agent.shots_fired>before and not agent.dead:
                gunshots.append(agent.route_m)
                shot_ids.append(idx)
        # Fired enemy rifles can alert other patrols, but only AFTER this
        # frame's states were updated, preventing order-dependent cascades.
        for source_m in gunshots:
            self.hear_shot(source_m,max_range=82.0)
        self.gunshots_this_frame=tuple(shot_ids)

    def react_to_blast(self,impact_position,strength,level_for_distance,position_at):
        """Apply one blast signal to ACTUALLY nearby active enemies.

        Uses world-space X/Z distance, never the route metre difference.
        Does not change kill state, encounter ids or hit precision.
        """
        x,_,z=impact_position
        count=0
        for agent in self.active.values():
            if agent.dead:continue
            ax,_,az=position_at(agent.spec)
            distance=math.hypot(ax-x,az-z)
            level=level_for_distance(distance,strength)
            if agent.react_to_blast(level):count+=1
        return count

    def current_threat(self,route_m):
        candidates=[a for a in self.active.values()
                    if a.threat and not off_trunk(a.spec)
                    and -3 < a.route_m-route_m < 24]
        return min(candidates,key=lambda a:a.route_m-route_m) if candidates else None

    def clamp_advance(self,route_m,travel):
        """No skipping a live encounter at high FPS/sprint speed."""
        threat=self.current_threat(route_m)
        if threat is None:return travel
        stop=threat.route_m-8.0
        return min(travel,max(0.,stop-route_m))

    def capture_frame(self,view_projection,width,height,route_m,position_at,heading_at,visible_radius=65.):
        """Freeze the SHOWN poses for a later verified game coordinate.

        No live AI positions are consulted for the geometry of this frame.
        Dead actors get no hittable spheres, even though their corpses render.
        """
        mat=tuple(tuple(float(x) for x in row) for row in view_projection)
        spheres=[]
        routes=[]
        for agent in sorted(self.active.values(),key=lambda a:a.spec.enemy_id):
            if agent.dead or abs(agent.route_m-route_m)>=visible_radius:continue
            routes.append((agent.spec.enemy_id,float(agent.route_m)))
            p=position_at(agent.spec)
            yaw=actor_facing(agent,heading_at(agent.route_m))
            for name,point,r in agent.hit_regions(p,yaw):
                spheres.append((agent.spec.enemy_id,name,point,float(r)))
        return FrozenEnemyFrame(int(width),int(height),mat,float(route_m),
                                tuple(spheres),tuple(routes))

    def resolve_frozen_hit(self,x,y,frame,occluded=None):
        """Resolve an EXACT game-space point against the captured visible pose.

        The closest intersection is used; optional occlusion protects solid
        level geometry. occluded receives (agent,t,origin,direction).
        This method does not detect physical shots and does not snap coordinates.
        """
        origin,direction=ray_from_screen(x,y,frame.width,frame.height,
                                         frame.view_projection)
        nearest=None
        for enemy_id,part,point,r in frame.hit_spheres:
            agent=self.active.get(enemy_id)
            if agent is None or agent.dead:continue
            t=ray_sphere(origin,direction,point,r)
            if t is None or t>150:continue
            if nearest is None or t<nearest[0]:nearest=(t,agent,part)
        if nearest is None:return None
        t,agent,part=nearest
        if occluded is not None and occluded(agent,t,origin,direction):return None
        if not agent.eliminate(part):return None
        self.eliminated.add(agent.spec.enemy_id)
        self.last_hit=(agent.spec.enemy_id,part)
        return self.last_hit

    def resolve_ray(self,origin,direction,position_at,occluded=None):
        """Nearest living body's 3D proxy. No screen-radius forgiving hit.

        occluded is a pluggable visibility callback(agent, ray_t, origin,
        direction) for geometry depth/cover; DO NOT claim true LOS without one.
        """
        nearest=None
        direction=_normalize(direction)
        for agent in self.active.values():
            if agent.dead:continue
            center=position_at(agent.spec)
            for part,p,radius in agent.hit_regions(center):
                t=ray_sphere(origin,direction,p,radius)
                if t is None or t>150:continue
                if nearest is None or t<nearest[0]:
                    nearest=(t,agent,part)
        if nearest is None:return None
        t,agent,part=nearest
        if occluded and occluded(agent,t,origin,direction):return None
        if agent.eliminate(part):
            self.eliminated.add(agent.spec.enemy_id)
            self.last_hit=(agent.spec.enemy_id,part)
            return self.last_hit
        return None

    def resolve_screen_hit(self,x,y,width,height,view_projection,position_at,occluded=None):
        origin,direction=ray_from_screen(x,y,width,height,view_projection)
        return self.resolve_ray(origin,direction,position_at,occluded)


# ----- Renderer-neutral skeleton pose. A future GLB rig can replace geometry. -----
def _smoothstep01(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _rotated_support_bottom(part, angle):
    """Lowest local-space point of a dead actor's protective/body equipment.

    Revolved/box shapes use conservative bounds; rods use their endpoint
    radii. The weapon and muzzle flash are excluded from floor contact.
    This is a *visual* pose adjustment, not a rigid-body physics claim.
    """
    if part[1] not in (10, 23, 24, 25):
        return None
    if part[0] in ('cylinder','limb'):
        return min(part[2][1], part[3][1]) - part[4]
    cx,cy,cz=part[2]
    if part[0] == 'jacket':
        # Ring mesh: y [-.39,+.36], z [-.21,+.21] in unit jacket coordinates.
        ry,rz=.40*part[3][1],.22*part[3][2]
    elif part[0] == 'helmet':
        ry,rz=.20*part[3][1],.245*part[3][2]
    elif part[0] == 'face':
        ry,rz=.19*part[3][1],.18*part[3][2]
    else:
        _,ry,rz=part[3]
    return cy - abs(math.cos(angle))*ry - abs(math.sin(angle))*rz


def corpse_ground_clearance(parts, angle):
    """Positive translation needed to place the lowest BODY point on ground.

    Computed from animated parts themselves. No hard-coded backpack pivot or
    magic death height; the exact same translation applies to all parts.
    """
    lows=[v for p in parts if (v:=_rotated_support_bottom(p,angle)) is not None]
    return max(0.0, .012 - min(lows)) if lows else 0.0


def actor_joints(agent):
    """One lightweight, deterministic local-space joint pose for mesh AND hitboxes.

    A measured travelled distance animates feet, not simply being in the rush
    state. Crouch bends knees rather than shrinking all limbs uniformly.
    Joint coordinates are in metres relative to a stationary ground anchor.
    """
    crouch=max(0.,min(1.,agent.crouch_fraction()))
    move=max(0.,min(1.,getattr(agent,'gait_strength',0.0)))
    cycle=getattr(agent,'gait_phase',0.0)
    bob=.022*move*(1.0-math.cos(cycle*2.0))
    lean=.045*move+.055*crouch
    pelvis=(0.,.88-.18*crouch+bob,.015+.025*move)
    torso=(0.,1.15-.26*crouch+bob,.045+lean)
    head=(0.,1.67-.28*crouch+bob,.045+lean*.7)
    hips=[]; knees=[];ankles=[];feet=[]
    for i,side in enumerate((-1.,1.)):
        wave=math.sin(cycle+math.pi*i)
        stride=.205*move*wave
        rise=.105*move*max(0.,wave)
        foot_z=.075+stride+.11*crouch
        # A bent knee travels forward under cover, with smooth ankle extension.
        hip=(side*.16,pelvis[1]-.015,pelvis[2])
        knee=(side*.17,.49-.125*crouch+bob*.35+rise*.34,
              .085+.055*wave*move+.18*crouch)
        ankle=(side*.175,.165+rise,foot_z-.065)
        boot=(side*.175,.108+rise,foot_z+.045)
        hips.append(hip);knees.append(knee);ankles.append(ankle);feet.append(boot)
    return {'pelvis':pelvis,'torso':torso,'head':head,
            'hips':tuple(hips),'knees':tuple(knees),
            'ankles':tuple(ankles),'feet':tuple(feet),
            'move':move,'cycle':cycle,'crouch':crouch}


def enemy_parts(agent):
    """V3.9.2 articulated procedural stand-in for a future rigged GLB soldier.

    This returns shared-mesh rigid components, never dynamic mesh allocations.
    Every weapon, uniform, helmet, knee, boot and head movement belongs to the
    SAME sampled pose as shot-frozen head/torso/leg hit regions.
    """
    pose=actor_joints(agent)
    crouch=pose['crouch']
    aim=agent.aim_fraction()
    move=pose['move'];cycle=pose['cycle']
    dying=_smoothstep01(agent.state_time/1.18) if agent.dead else 0.0
    torso=pose['torso'];head=pose['head'];pelvis=pose['pelvis']
    torso_y=torso[1];head_y=head[1];torso_z=torso[2]
    recoil=(.056*math.exp(-agent.state_time*18.) if agent.state=='firing' else 0.)
    parts=[]
    def oval(mat,center,radii,angles=None):
        parts.append(('oval',mat,center,radii) if angles is None
                     else ('oval',mat,center,radii,angles))
    def cube(mat,center,radii,angles=None):
        parts.append(('cube',mat,center,radii) if angles is None
                     else ('cube',mat,center,radii,angles))
    def cyl(mat,a,b,r,shape='cylinder'):
        parts.append((shape,mat,a,b,r))
    def jacket(mat,center):parts.append(('jacket',mat,center,(1.,1.,1.)))
    def helmet(mat,center,angles=None):
        parts.append(('helmet',mat,center,(1.,1.,1.)) if angles is None
                     else ('helmet',mat,center,(1.,1.,1.),angles))

    # Actual stride-driven joints. Knees bend and feet lift instead of rigid legs
    # swinging through the floor. Trousers use sculpted cloth-tapered limbs.
    for i,side in enumerate((-1.,1.)):
        hip=pose['hips'][i];knee=pose['knees'][i];ankle=pose['ankles'][i];foot=pose['feet'][i]
        cyl(23,hip,knee,.141,'limb')
        oval(23,knee,(.139,.123,.138))
        cyl(23,knee,ankle,.110,'limb')
        # Visible knee pad and ankle gaiter follow the ACTUAL knee and ankle.
        oval(24,(knee[0],knee[1]-.010,knee[2]+.100),(.096,.095,.050))
        oval(24,(ankle[0],ankle[1]+.016,ankle[2]+.002),(.112,.118,.115))
        parts.append(('boot',10,foot,(.136,.108,.229)))
        # A separate toe-cap layer is intentionally avoided: it reads as a toy.

    oval(23,pelvis,(.247,.176,.169))
    jacket(23,(torso[0],torso_y,torso_z))
    # Cloth tunic, protective webbing and asymmetric pouches.
    cube(24,(0.,torso_y+.015,torso_z+.215),(.244,.232,.051))
    cube(24,(0.,torso_y-.22,torso_z+.198),(.255,.040,.065))
    for side in (-1.,1.):
        x=side*.16
        cube(24,(x,torso_y-.195,torso_z+.282),(.080,.112,.058))
        cyl(24,(side*.226,torso_y+.28,torso_z+.172),
               (side*.125,torso_y-.265,torso_z+.242),.016)
    cube(10,(0.,torso_y-.32,torso_z+.14),(.247,.024,.025))
    # Backpack follows upper body. Smaller silhouette than old oversized cube.
    cube(24,(0.,torso_y-.10,torso_z-.224),(.226,.275,.120))
    cube(24,(0.,torso_y+.156,torso_z-.216),(.185,.052,.110))
    for side in (-1.,1.):
        cube(24,(side*.206,torso_y-.13,torso_z-.225),(.047,.155,.102))
        cyl(24,(side*.20,torso_y+.255,torso_z-.14),
               (side*.171,torso_y-.20,torso_z-.238),.018)

    kit=agent.spec.enemy_id%3
    if kit==0:
        oval(24,(.285,.79-.17*crouch,-.07),(.080,.118,.072))
        cyl(24,(.285,.88-.17*crouch,-.07),(.285,.91-.17*crouch,-.07),.024)
    elif kit==1:
        cyl(13,(-.17,torso_y+.25,torso_z-.252),(.17,torso_y+.25,torso_z-.252),.063)
        for x in (-.11,.11):cube(24,(x,torso_y+.252,torso_z-.25),(.016,.077,.081))
    else:
        cube(24,(-.30,torso_y-.23,torso_z-.12),(.086,.158,.105))
        cube(10,(-.30,torso_y-.082,torso_z-.12),(.084,.019,.106))
        cyl(24,(-.30,torso_y-.10,torso_z-.12),(-.30,torso_y+.20,torso_z-.17),.013)

    # Hands are connected to a rifle held at a meaningful height. A small
    # inverse-phase gait sway settles as soon as physical travel stops.
    gy=torso_y-.17+aim*.32
    gz=torso_z+.26+aim*.16-recoil
    for i,side in enumerate((-1.,1.)):
        wave=math.sin(cycle+math.pi*i)
        shoulder=(side*.304,torso_y+.22,torso_z-.010)
        elbow=(side*.337,torso_y-.055+aim*.045,
               torso_z+.06+aim*.080+.024*move*wave)
        hand=(side*.115,gy+.035+(.012*move*wave),gz+.045)
        oval(23,shoulder,(.135,.128,.139))
        cyl(23,shoulder,elbow,.103,'limb')
        oval(23,elbow,(.103,.10,.106))
        cyl(23,elbow,hand,.081,'limb')
        oval(24,hand,(.083,.069,.076))

    # Anatomical face, rather than a full round face and separate "teddy" muzzle.
    # Look yaw is independent of entire body yaw; attached eyes/straps/helmet
    # rotate around the cervical joint without teleporting a visible head.
    look=getattr(agent,'head_look',0.0)
    co,si=math.cos(look),math.sin(look)
    def hp(dx,dy,dz):
        return (dx*co+dz*si,head_y+dy,head[2]-dx*si+dz*co)
    head_rot=(0.,look,0.)
    cyl(25,(0.,torso_y+.305,torso_z+.02),hp(0.,-.145,-.01),.077)
    parts.append(('face',25,hp(0.,0.,.025),(1.,1.,1.),head_rot))
    for side in (-1.,1.):
        oval(25,hp(side*.135,-.035,.023),(.026,.050,.033),head_rot)
        oval(10,hp(side*.055,.028,.166),(.016,.013,.012),head_rot)
    # Nose bridge and subtle lip separation are attached to turning face.
    oval(25,hp(0.,-.038,.164),(.028,.042,.033),head_rot)
    cyl(10,hp(-.035,-.109,.169),hp(.035,-.109,.169),.0045)
    helmet(24,hp(0.,.160,.015),head_rot)
    for side in (-1.,1.):
        cyl(24,hp(side*.157,.027,.015),hp(side*.123,-.128,.122),.012)

    # Rifle: fixed local part identities let us animate the recoil and drop
    # without updating GL vertex buffers for each shot.
    weapon_begin=len(parts)
    cyl(10,(-.045,gy+.010,gz-.11),(.005,gy+.035,gz+.48),.041)
    cyl(10,(.005,gy+.035,gz+.41),(.005,gy+.035,gz+.96),.016)
    cyl(13,(-.055,gy,gz-.09),(-.124,gy-.10,gz-.36),.042)
    cyl(13,(.004,gy-.008,gz+.28),(.004,gy-.008,gz+.55),.034)
    cube(10,(-.026,gy+.035,gz+.12),(.053,.071,.144))
    cyl(10,(-.022,gy-.032,gz+.10),(-.008,gy-.19,gz+.17),.033)
    cyl(10,(-.008,gy-.19,gz+.17),(.006,gy-.24,gz+.27),.029)
    cyl(10,(.004,gy+.077,gz+.73),(.004,gy+.104,gz+.73),.012)
    if agent.state=='firing' and agent.state_time<.115:
        oval(11,(.004,gy+.035,gz+1.0),(.064,.066,.129))

    if dying:
        # First fold knees, then lean and settle forward; the old immediate
        # plank rotation was visually rigid. Keep all body parts grounded.
        buckle=.14*_smoothstep01(agent.state_time/.34)
        angle=1.52*dying
        pivot_y=.13;pivot_z=.10
        co,si=math.cos(angle),math.sin(angle)
        def rotate(point):
            x,y,z=point
            y=max(.09,y-buckle*max(0.0,min(1.0,(y-.2)*1.5)))
            y-=pivot_y;z-=pivot_z
            return (x,pivot_y+y*co-z*si,pivot_z+y*si+z*co)
        rotated=[]
        for part in parts[:weapon_begin]:
            if part[0] in ('cylinder','limb'):
                rotated.append((part[0],part[1],rotate(part[2]),rotate(part[3]),part[4]))
            else:
                rotated.append((part[0],part[1],rotate(part[2]),part[3],angle))
        lift=corpse_ground_clearance(rotated,angle)
        corrected=[]
        def up(point):return (point[0],point[1]+lift,point[2])
        for part in rotated:
            if part[0] in ('cylinder','limb'):
                corrected.append((part[0],part[1],up(part[2]),up(part[3]),part[4]))
            else:
                corrected.append((part[0],part[1],up(part[2]),part[3],angle))
        release=_smoothstep01((agent.state_time-.19)/.58)
        def dropped(point):
            return (point[0]+.38,.14+(point[1]-gy)*.18,1.08+(point[2]-gz))
        def weapon_point(point):
            left=up(rotate(point));right=dropped(point)
            return tuple(left[i]*(1.-release)+right[i]*release for i in range(3))
        for part in parts[weapon_begin:]:
            if part[0] in ('cylinder','limb'):
                corrected.append((part[0],part[1],weapon_point(part[2]),
                                  weapon_point(part[3]),part[4]))
            else:
                corrected.append((part[0],part[1],weapon_point(part[2]),
                                  part[3],angle*(1.-release)))
        return corrected
    return parts


def actor_facing(agent,heading):
    """Facing world yaw; patrol running DOWN the route faces sound source."""
    if agent.spec.exterior_side != 0:
        # Patroller follows a real bank-side route and periodically watches
        # DOWN into the trench; only an actual visibility ray detects us.
        sweep=.60*math.sin(agent.total_time*.74+agent.spec.enemy_id*.23)
        return heading-agent.spec.exterior_side*(math.pi*.5+sweep)
    if agent.spec.branch_index is not None:
        # Stationary sentry looks out through the real side-branch portal.
        return heading-agent.spec.branch_side*math.pi*.5 + (
            .13*math.sin(agent.total_time*.8+agent.spec.enemy_id) if agent.state=='idle' else 0.)
    if agent.state=='investigating':return heading+math.pi
    if agent.state in ('idle','alert') and agent.spec.profile=='unaware':
        # A sentry gradually sweeps the trench. It must LOOK toward the player
        # before visual contact; never instant omniscience or idle blindness.
        if agent.heard_shots:
            return heading+math.pi*.94
        fraction=.70+.67*math.sin(agent.total_time*.67+agent.spec.enemy_id*.43+agent.spec.initial_gaze_phase)
        return heading+math.pi*fraction
    if agent.state in ('scanning','peeking','hiding','locating'):
        sweep=.79*math.sin(agent.total_time*(1.65+agent.tactical_difficulty.level*.18)
                            +agent.spec.enemy_id*.47)
        return heading+math.pi+sweep
    if agent.spec.profile=='unaware':return heading+math.pi*agent.turn_fraction()
    if agent.spec.profile=='ambush':
        # No sudden 58-degree snap on engage: the lookout had already turned
        # most of the way toward the player before finishing the last sweep.
        return heading+math.pi*(.68+.32*agent.turn_fraction())
    return heading+math.pi
