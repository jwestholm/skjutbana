"""Ljusjakten: bright pre-rendered ceramics, frozen OBJECT targets and post-decision break effects."""
from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import uuid
import pygame
from src.engine.events import event_bus
from src.engine.game_objects import ObjectManager, ObjectGeometry, HitShapeSpec, make_breakable_object
from src.engine.game_objects.events import EffectAction, ReactionRule
from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.input.hit_regions import object_hit_regions, hit_context_snapshot_for_shot, object_hit_registry_v2223
from src.engine.input.hit_input import hit_input
from src.engine.settings import load_range_projection_settings
from content.games.light_room_art import LightRoomArt
from content.games.light_room_audio import build_sounds

BACKGROUND = (255, 255, 192)
PALETTE = ((160, 240, 205), (150, 240, 245), (220, 245, 150))
INK = (32, 66, 65)
BREAK_FPS = 20
RESULT_DWELL_S = .95
OBJECTS_PER_ROOM = 10
PRE_SETTLE_S = 1.7  # >=1.5 s plus 0.2 s projector/camera-receipt guard


class LightTargetsGame:
    owns_hit_presentation = True
    shot_critical_updates_required = True  # timers run, but pending render is immutable
    hit_precision = 'OBJECT'
    precision_flash = 'off'

    def __init__(self, game_root, viewport):
        self.viewport = viewport.copy()
        self.object_highlight=os.environ.get('SKJUTBANA_PANG_OBJECT_HIGHLIGHT',
            os.environ.get('SKJUTBANA_PANG_OBJECT_GLOW','0'))=='1'
        self.precision_glow=dict(color=(255,248,175),halo_color=(248,244,214),halo_mix=.82,
            radius_px=12,pastel_mix=.45,reference_s=.60,reference_frames=3,
            measurement_s=.40,min_interval_s=2.)
        self._glow_ready=False;self._glow_regions=();self._glow_state='show_targets'
        self.manager = ObjectManager()
        self.rng = random.Random()
        self.owner = 'light_targets:' + uuid.uuid4().hex
        self.players = 1
        self.shots_per_player = 10
        self.scores = [0, 0]
        self.used = [0, 0]
        self.streak = 0
        self.player = 0
        self.token = 0
        self.phase = 'intro'
        self.phase_at = time.monotonic()
        self.presented_ts = None
        self.presented_mono = None
        self.active_shot = None
        self.active_peak = None
        self._frozen = None
        self._last_frame = None
        self._render_key = None
        self._seen = set()
        self._slots = []
        self._last_slot = None
        self._subscribed = False
        self.feedback = ''
        self.font = None
        self.big = None
        self.art = None
        self.items = []
        self.broken_ids = set()
        self.reaction_at = None
        self.reaction_wall_ts = None
        self._reaction_visible = False
        self._reaction_presented = False
        self._reaction_start_mono = None
        self._effect_ids = set()
        self.effect_sides = {}
        self._effects_before = 0
        self.layout_id = 0
        self._anchor_uses = {}
        self._recent_rooms=[];self._recent_positions=[]
        self._history_path=Path(os.environ.get('SKJUTBANA_PANG_ROOM_HISTORY',os.path.expanduser('~/.local/state/skjutbana/pang_room_recent.json')))
        self._history_loaded=False
        self.ready_wall_ts = None
        self.sound_enabled = True
        self._sounds = {}
        self._cue = None
        self._sfx = None
        self._clear_bonus = 0
        self._impact_xy = None
        self.last_outcome = None

    def on_enter(self):
        if self._subscribed: return
        self.font = pygame.font.Font(None, 23)
        self.big = pygame.font.Font(None, 38)
        self.art = LightRoomArt(self.viewport.size,natural=self._uses_glow())
        self._load_room_history()
        self._sounds = build_sounds()
        event_bus.subscribe(self._on_object_event)
        hit_input.subscribe(self._on_hit)
        self._subscribed = True

    def on_exit(self):
        for sound in self._sounds.values():
            sound.stop()
        self._sounds.clear()
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_exit')
        if self._subscribed:
            event_bus.unsubscribe(self._on_object_event)
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    def start(self):
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_restart')
        self.owner = 'light_targets:' + uuid.uuid4().hex
        self._seen.clear()
        self.scores = [0, 0]; self.used = [0, 0];self.last_outcome=None
        self.player = 0; self.streak = 0; self._slots = []
        self._next_target()

    @property
    def object_highlight(self):
        return self.precision_flash == "object_glow"

    @object_highlight.setter
    def object_highlight(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("object_highlight requires a boolean")
        self.precision_flash = "object_glow" if enabled else "off"

    def _uses_glow(self):
        from src.engine.precision_flash import resolve_policy
        return self.precision_flash=='object_glow' and resolve_policy(self)[0]

    def get_precision_glow_context(self):
        if not self._uses_glow() or not self.items or self.phase in ('intro','finished','unsupported'):return None
        settings=load_range_projection_settings();width_mm=10.*float(settings.get('viewport_physical_width_cm',90))
        height_mm=10.*float(settings.get('viewport_physical_height_cm',width_mm/10*self.viewport.h/self.viewport.w))
        return dict(key=self.owner+':'+str(self.layout_id),viewport=list(self.viewport),
            regions=self._glow_regions,active_regions=tuple(self.manager.get_hit_regions()),mm_per_px=(width_mm/self.viewport.w,height_mm/self.viewport.h))

    def on_precision_glow_state(self,state):
        self._glow_ready=bool(state.get('ready'));self._glow_state=str(state.get('state','unknown'))
        if self.phase in ('settling','live','waiting'):self._render_key=None

    def _load_room_history(self):
        if self._history_loaded:return
        self._history_loaded=True
        try:
            data=json.loads(self._history_path.read_text())
            if data.get('schema')!='pang-room-history-1':return
            self._recent_rooms=[str(r) for r in data.get('rooms',[])][-12:]
            self._recent_positions=[(float(x),float(y)) for x,y in data.get('positions',[])][-96:]
            self._recent_positions=[(x,y) for x,y in self._recent_positions if 0<=x<=1 and 0<=y<=1]
        except (OSError,ValueError,TypeError):pass

    def _remember_room(self):
        if not self.art.rooms:return
        self._recent_rooms=(self._recent_rooms+[self.art.room_id])[-12:]
        self._recent_positions=(self._recent_positions+[(i['geometry'].center[0]/self.viewport.w,
            i['geometry'].center[1]/self.viewport.h) for i in self.items])[-96:]
        tmp=None
        try:
            self._history_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=self._history_path.with_name(self._history_path.name+'.'+uuid.uuid4().hex+'.tmp')
            tmp.write_text(json.dumps(dict(schema='pang-room-history-1',rooms=self._recent_rooms,positions=self._recent_positions)))
            os.replace(tmp,self._history_path)
        except OSError as exc:
            print('[PANG-ROOM] history not saved: '+str(exc),flush=True)
            if tmp is not None:
                try:tmp.unlink(missing_ok=True)
                except OSError:pass

    def _choose_room_layout(self):
        if not self.art.rooms:return self._choose_layout()
        recent=self._recent_rooms
        # Prefer another template, then an unused furnishing, then oldest use.
        candidates=sorted(self.art.rooms,key=lambda r:(
            r['template'] in [s.rsplit('_',1)[0] for s in recent[-2:]],
            r['id'] in recent[-4:],
            max((i for i,s in enumerate(recent) if s==r['id']),default=-1),self.rng.random()))
        for room in candidates:
            self.art.select_room(room)
            items=self._choose_layout()
            if items:return items
        return []

    def _choose_layout(self):
        w,h=self.viewport.size
        if w<640 or h<480:return []
        if self.art.rooms and not self.art.natural:
            import numpy as np
            from content.games.light_room_safety import camera_proxy
            plate_dn=camera_proxy(pygame.surfarray.array3d(self.art._plate))
        base=[k for k in (('vase','bottle','plant') if self.art.rooms else ('vase','jar','bowl','bottle','lamp')) if k in self.art.objects]
        allowed=getattr(self.art,'room_spec',{}).get('kinds',tuple(self.art.objects))
        base=[k for k in base if k in allowed]
        extras=[k for k in self.art.objects if k not in base and k in allowed]
        supports={'books':('shelf',),'tv':(('tv_bench',) if self.art.rooms else ('table','bench')),'lamp':('table','bench','shelf'),
                  'clock':('shelf','sill','table','bench'),'mug':('shelf','sill','table','bench'),'frame':('shelf','sill','table','bench')}
        # Place constrained items first, then backtrack locally if a wide object
        # blocks a later anchor. Shapes remain large and non-overlapping.
        for _ in range(32):
            extra=[k for k in ('tv','books') if k in extras]
            pool=[k for k in extras if k not in extra]
            extra+=self.rng.sample(pool,min(len(pool),OBJECTS_PER_ROOM-len(base)-len(extra)))
            order=base+extra
            order+=[self.rng.choice(getattr(self.art,'room_spec',{}).get('repeat_kinds',base)) for _ in range(OBJECTS_PER_ROOM-len(order))]
            self.rng.shuffle(order);order.sort(key=lambda k:len(supports.get(k,range(8))))
            choices=[]
            for kind in order:
                candidates=[]
                for anchor in self.art.manifest['anchors']:
                    if kind in supports and anchor['support'] not in supports[kind]:continue
                    if self.art.rooms and anchor['support']=='tv_bench' and kind!='tv':continue
                    if self.art.rooms and kind=='bowl' and anchor['support']=='floor':continue
                    factor={'bowl':.68,'lamp':.95,'bottle':1.04,'tv':.82,'books':.85}.get(kind,1.)
                    for size in self.rng.sample([.82,.96,1.08],3):
                        height=max(78,min(132,min(124,h*.21)*factor*size))
                        if self.art.rooms:
                            metres={'vase':.46,'jar':.33,'bowl':.24,'bottle':.40,'lamp':.52,'tv':.85,'books':.31,'plant':.55,'clock':.29,'mug':(.22 if self.art.natural else .34),'frame':.28}[kind]
                            minimum=60 if kind in ('books','bottle','vase') else 40
                            height=max(minimum,min(125,h*anchor['unit_height']*metres*size))
                        g=self.art.geometry(kind,anchor,height)
                        if g.x<w*.075 or g.right>w*.93 or g.y<104 or g.bottom>h*.94:continue
                        lighting_cost=0.
                        if self.art.rooms and not self.art.natural:
                            area=pygame.Rect(g.aabb).inflate(24,24).clip(pygame.Rect(0,104,w,h-104))
                            patch=plate_dn[area.x:area.right,area.y:area.bottom]
                            # A game may choose naturally bright display bays. This
                            # is spawn geometry, never a detector/known-hole veto.
                            lighting_cost=8.*float(np.maximum(124-patch,0).mean())/30.
                        cx,cy=g.center[0]/w,g.center[1]/h
                        wear=sum(max(0,1-((cx-x)**2+(cy-y)**2)**.5/.13) for x,y in self._recent_positions)
                        candidates.append((self._anchor_uses.get(str(self.art.room_id)+anchor['id'],0)+wear+lighting_cost+self.rng.random()*1.2,
                            dict(kind=kind,geometry=g,anchor=anchor,parts={})))
                choices.append([item for _,item in sorted(candidates,key=lambda c:c[0])])
            budget=[600];items=[];used=set()
            def place(index):
                if index==len(order):return True
                for item in choices[index]:
                    budget[0]-=1
                    if budget[0]<0:return False
                    aid=item['anchor']['id'];rect=pygame.Rect(item['geometry'].aabb)
                    if aid in used or any(rect.inflate(10,10).colliderect(pygame.Rect(i['geometry'].aabb)) for i in items):continue
                    used.add(aid);items.append(item)
                    if place(index+1):return True
                    items.pop();used.remove(aid)
                return False
            if place(0):return items
        return []

    def _arm_shot(self):
        object_hit_registry_v2223.cancel_object_queries(self.owner,'light_targets_surface_changed')
        self.token+=1;self.active_shot=None;self.active_peak=None;self.ready_wall_ts=None
        self.presented_ts=None;self.presented_mono=None;self._frozen=None;self._last_frame=None
        self.reaction_at=None;self.reaction_wall_ts=None;self._reaction_start_mono=None
        self._reaction_visible=False;self._reaction_presented=False;self._effect_ids=set()
        self._cue=None;self._sfx=None;self._clear_bonus=0;self._impact_xy=None
        self.phase='settling';self.phase_at=time.monotonic()
        for obj in self.manager.objects:obj.metadata['lamp_token']=self.token
        print('[LAMP-TARGET] '+json.dumps(dict(owner=self.owner,token=self.token,layout_id=self.layout_id,
            player=self.player+1,shape='visible_part_alpha_polygons',game_version='polyhaven-object-glow-8' if self.art.natural else 'polyhaven-rooms-7',room_id=self.art.room_id,
            asset_manifest_sha256=self.art.manifest_sha256,
            objects=[dict(entity_id=i['entity_id'],kind=i['kind'],support=i['anchor']['support'],
                anchor_id=i['anchor']['id'],geometry=list(i['geometry'].aabb),parts=i['parts']) for i in self.items],
            wall_ts=time.time())),flush=True)

    def _next_target(self):
        # A fresh room is created at a new player/cleared room, not on every shot.
        object_hit_registry_v2223.cancel_object_queries(self.owner,'light_targets_new_room')
        self.items=self._choose_room_layout()
        if not self.items:
            self.phase='unsupported';self.feedback='För liten säker spelyta – välj minst 640 × 480.';return
        self.layout_id+=1;self.manager=ObjectManager();self.broken_ids=set();self.effect_sides={}
        w=self.viewport.w
        width_cm=max(1,float(load_range_projection_settings().get('viewport_physical_width_cm',90)))
        touch=2.25*w/(width_cm*10)
        for index,item in enumerate(self.items):
            kind=item['kind'];entity=f'{kind}_{self.layout_id}_{index+1}';item['entity_id']=entity
            anchor=item['anchor'];aid=str(self.art.room_id)+anchor['id'];self._anchor_uses[aid]=self._anchor_uses.get(aid,0)+1
            for part,spec in self.art.objects[kind]['parts'].items():
                pieces=[HitShapeSpec.polygon(p) for p in spec['polygons']]
                shape=pieces[0] if len(pieces)==1 else HitShapeSpec.union(*pieces)
                shape=replace(shape,touch_radius=touch)
                oid=entity+':'+part;item['parts'][part]=oid
                obj=make_breakable_object(oid,replace(item['geometry']),shape=shape,owner=self.owner,
                    entity_id=entity,part_id=part,break_effect='part_'+spec['motion'],break_sound=spec['cue'])
                if spec['detach']:
                    obj.reactions.append(ReactionRule.on('object.broken',EffectAction.break_parts(*spec['detach']),once=True))
                obj.metadata.update(lamp_owner=self.owner,player=self.player+1,sprite_kind=kind,
                    points=spec['points'],support=anchor['support'],asset_schema=self.art.manifest['schema'],
                    asset_manifest_sha256=self.art.manifest_sha256)
                self.manager.add(obj)
        self.target=self.items[0]['geometry'];self.color=PALETTE[self.layout_id%len(PALETTE)]
        self._glow_regions=deepcopy(tuple(self.manager.get_hit_regions()))
        self._glow_ready=False;self._glow_state="show_targets"
        self.art.prepare_round(self.items)
        self._remember_room();self._arm_shot()

    def entity_states(self):
        states={}
        for item in self.items:
            objects=[self.manager.get(oid) for oid in item['parts'].values()]
            alive=sum(bool(o and o.active) for o in objects)
            states[item['entity_id']]='intact' if alive==len(objects) else 'hit' if alive else 'broken'
        return states

    def get_hit_regions(self):
        if self.phase not in ('live', 'waiting'): return ()
        regions = tuple(replace(r, metadata={**r.metadata, 'lamp_owner':self.owner,
                           'lamp_token':self.token}) for r in self.manager.get_hit_regions())
        return object_hit_regions(regions, owner=self.owner, evidence_method='object_glow' if self.art.natural else 'dense', top_k=3,
            margin_camera_px=36, surface_after_ts=self.presented_ts, require_presented_surface=True)

    def on_frame_presented(self, timestamp):
        if self.phase in ('settling', 'live') and self._last_frame is not None and self.presented_ts is None:
            self.presented_ts = float(timestamp)
            self.presented_mono = time.monotonic()
            print('[LAMP-PRESENTED] '+json.dumps(dict(owner=self.owner, token=self.token, wall_ts=self.presented_ts)), flush=True)
        if self.phase == 'result' and self.active_shot is not None and self._reaction_visible and not self._reaction_presented:
            snap = hit_context_snapshot_for_shot(self.active_shot)
            if snap is not None:
                snap.object_gameplay.update(reaction_presented_ts=timestamp)
                self._reaction_presented = True
        if self.phase=='result' and self._reaction_visible and not self._reaction_presented:
            self._reaction_presented=True  # simulated preview has no scanner snapshot
        if self.phase=='result' and self._reaction_visible and self._sfx is None:
            sound=self._sounds.get(self._cue) if self.sound_enabled else None
            if sound is not None:
                self._sfx=audio_peak_detector.play_feedback(sound,owner=self.owner,shot_id=int(self.active_shot or 0),cue=self._cue)
            else:self._sfx={'status':'MUTED_OR_UNAVAILABLE'}
            print('[LAMP-REACTION] '+json.dumps(dict(owner=self.owner,shot_id=self.active_shot,
                presented_ts=float(timestamp),effects=sorted(self._effect_ids),sound=self._sfx)),flush=True)

    def on_object_shot_frozen(self, snapshot):
        if self.phase != 'live' or self.active_shot is not None: return
        if self.ready_wall_ts is not None and snapshot.peak_ts < self.ready_wall_ts:
            return  # delayed game-feedback/old-window audio cannot take a new attempt
        self._effects_before=len(self.manager.effect_requests)
        self.active_shot = int(snapshot.shot_id); self.active_peak = float(snapshot.peak_ts)
        self.phase = 'waiting'; self.phase_at = time.monotonic()
        print('[LAMP-SHOT] '+json.dumps(dict(owner=self.owner, token=self.token, shot_id=self.active_shot, peak_ts=self.active_peak)), flush=True)
        self._frozen = self._last_frame.copy() if self._last_frame is not None else None

    def _on_object_event(self, message):
        if message.get('event') != 'physical.object_result': return
        result = message['data']; sid = int(result.get('shot_id', 0))
        if not result.get('final', False) or sid != self.active_shot or self.phase != 'waiting' or sid in self._seen: return
        if result.get('gameplay_owner') is False or result.get('paired_follower_of'): return
        snap = hit_context_snapshot_for_shot(sid)
        if snap is None or not any(r.metadata.get('lamp_owner') == self.owner and
                                  r.metadata.get('lamp_token') == self.token for r in snap.game_regions): return
        self._seen.add(sid)
        resolution, hypothesis = self.manager.resolve_object_query(result)
        if hypothesis is None and result.get('evidence_method')=='object_glow' and result.get('object_decision',{}).get('state')=='NO_OBJECT_HIT':
            hypothesis=result['object_decision'].get('hypothesis')
        hit = bool(resolution is not None and resolution.interactions)
        decision = result.get('object_decision', {}).get('state', 'UNKNOWN')
        outcome = 'HIT' if hit else ('MISS' if decision == 'NO_OBJECT_HIT' else 'UNKNOWN')
        snap.object_gameplay = dict(owner=self.owner, accepted=True, result=outcome, reaction_expected=True,
                                    player=self.player+1, token=self.token)
        self._finish(outcome, sid, result, hypothesis)

    def _on_hit(self, hit):
        if self.phase!='live' or not getattr(hit,'is_simulated',False):return
        # Even the preview never resolves a part seam using arbitrary depth/order.
        hits=self.manager._current_hits(hit.game_x,hit.game_y)
        self._effects_before=len(self.manager.effect_requests)
        if len(hits)!=1:
            self._finish('MISS' if not hits else 'UNKNOWN',None,None,None,simulated=True);return
        resolution=self.manager.resolve_hit(hit,active_only=True)
        oid=resolution.interactions[0].object_id if resolution.interactions else None
        self._finish('HIT' if oid else 'MISS',None,dict(object_decision=dict(object_id=oid)),
                     dict(game_xy=[hit.game_x,hit.game_y]),simulated=True)

    def _finish(self, outcome, sid, result, hypothesis, simulated=False):
        self.phase = 'result'; self.phase_at = time.monotonic();self.last_outcome=outcome
        # Even an early positive decision must not alter the protected POST projection.
        hold = 0.0 if simulated or self.active_peak is None or (result or {}).get('evidence_method')=='object_glow' else max(0.0, self.active_peak + 1.2 - time.time())
        self.reaction_at = self.phase_at + hold
        self.reaction_wall_ts = time.time() + hold
        self._reaction_visible=False;self._reaction_presented=False;self._reaction_start_mono=None
        requests=self.manager.effect_requests[self._effects_before:]
        self._effect_ids={e.object_id for e in requests if e.payload.get('kind')=='spawn_effect'}
        self.broken_ids={o.object_id for o in self.manager.objects if not o.visible}
        oid=(result or {}).get('object_decision',{}).get('object_id') if outcome=='HIT' else None
        obj=self.manager.get(oid) if oid else None
        xy=(hypothesis or {}).get('game_xy')
        self._impact_xy=tuple(xy) if outcome=='HIT' and xy is not None else None
        side=-1 if obj is not None and xy is not None and xy[0]<obj.geometry.center[0] else 1
        self.effect_sides.update({part:side for part in self._effect_ids})
        cues=[e.payload.get('name') for e in requests if e.payload.get('kind')=='play_sound']
        self._cue=next((cue for cue in ('shatter','topple','petal') if cue in cues),None)
        if outcome in ('MISS','UNKNOWN'):self._cue='miss' if outcome=='MISS' else 'unknown'
        self._clear_bonus=500 if outcome=='HIT' and self.manager.objects and not any(o.active for o in self.manager.objects) else 0
        if self._frozen is None and self._last_frame is not None:
            self._frozen = self._last_frame.copy()
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_decision_complete')
        self.used[self.player] += 1
        if outcome == 'HIT':
            self.streak += 1;points=int(obj.metadata.get('points',100) if obj else 100)+25*min(self.streak-1,4)+self._clear_bonus
            self.scores[self.player]+=points
            part_label={'body':'KROSSAT!','base':'KROSSAT!','neck':'HALSEN!','flower':'BLOMMAN!','lid':'LOCKET!','rim':'KANTEN!','shade':'LAMPSKÄRMEN!','screen':'SKÄRMEN SPRACK!','face':'GLASET SPRACK!','stand':'TV-FOTEN!','foliage':'BLADEN!','left':'BOKEN FÖLL!','middle':'BOKEN FÖLL!','right':'BOKEN FÖLL!','handle':'HANDTAGET!'}.get(obj.part_id if obj else '', 'TRÄFF!')
            if self.art.manifest.get('asset_provider')=='polyhaven':
                if obj and obj.part_id=='flower':part_label='BLADEN!'
                if obj and obj.metadata.get('sprite_kind')=='jar' and obj.part_id=='lid':part_label='MYNNINGEN!'
                if obj and obj.part_id=='stand':part_label='TV-BASEN!'
                if obj and obj.metadata.get('sprite_kind')=='bowl' and obj.part_id=='rim':part_label='LOCKET!'
                if obj and obj.metadata.get('sprite_kind')=='mug' and obj.part_id=='handle':part_label='HANDTAG/FÄSTE!'
            self.feedback=f'TRÄFF! {part_label} +{points}  KOMBO ×{self.streak}'
            if self._clear_bonus:self.feedback+=' • RUMMET RENSAT!'
        elif outcome == 'MISS':
            self.streak = 0; points = 0; self.feedback = 'MISS – ingen verifierad målträff'
            distance=(result or {}).get('object_decision',{}).get('distance_mm')
            if distance is not None:self.feedback=f'MISS – {distance:.1f} mm från objektet'
        else:
            self.streak = 0; points = 0; self.feedback = 'OSÄKERT ? – ingen träff utdelad'
        print('[LAMP-RESULT] '+json.dumps(dict(owner=self.owner, token=self.token, shot_id=sid,
            shot_peak_ts=self.active_peak, player=self.player+1, outcome=outcome, points=points,
            shots_used=self.used[self.player], source='SIMULATED' if simulated else 'OBJECT',
            hypothesis_xy=(hypothesis or {}).get('camera_xy'), miss_distance_mm=(result or {}).get('object_decision',{}).get('distance_mm') if outcome=='MISS' else None, reason=(result or {}).get('reason'),
            game_version='polyhaven-object-glow-8' if self.art.natural else 'polyhaven-rooms-7',room_id=self.art.room_id,object_id=oid,entity_id=obj.entity_id if obj else None,
            part_id=obj.part_id if obj else None,impact_game_xy=xy,effect_side=side,clear_bonus=self._clear_bonus,
            broken_object_ids=sorted(self.broken_ids), reaction_not_before_wall_ts=self.reaction_wall_ts,
            wall_ts=time.time())), flush=True)

    def update(self, dt):
        now = time.monotonic()
        if self.phase == 'settling' and self.presented_ts is not None and self.presented_mono is not None and now-self.presented_mono >= PRE_SETTLE_S and (not self._uses_glow() or self._glow_ready):
            self.phase = 'live';self.ready_wall_ts=time.time()
        elif self.phase == 'waiting' and now-self.phase_at > (8.5 if self._uses_glow() else 4):
            # Engine result lost/cancelled: explicitly UNKNOWN, never an invented miss.
            self._finish('UNKNOWN', self.active_shot, dict(reason='game_wait_timeout'), None)
        elif self.phase == 'result' and self._reaction_start_mono is not None and now-self._reaction_start_mono >= RESULT_DWELL_S:
            if self.used[self.player] >= self.shots_per_player:
                if self.player+1 < self.players:
                    self.player += 1; self.streak = 0; self._next_target()
                else: self.phase = 'finished'
            elif not any(o.active for o in self.manager.objects):self._next_target()
            else:self._arm_shot()
        return None

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN:return None
        # The mode is chosen in the intro.  Toggling it during settling used
        # to cancel the glow context and silently start a non-glow room, so
        # shots then arrived with no object regions.  Keep the active round's
        # hit policy and target layout frozen once play has begun.
        if event.key==pygame.K_g and self.phase in ('intro','finished'):
            self.object_highlight = not self.object_highlight
            self.art=LightRoomArt(self.viewport.size,natural=self._uses_glow())
            self._last_frame=None;self._render_key=None
            return None
        if event.key==pygame.K_m:
            self.sound_enabled=not self.sound_enabled
            if not self.sound_enabled:
                for sound in self._sounds.values():sound.stop()
            return None
        if self.phase in ('intro', 'finished', 'unsupported'):
            if event.key in (pygame.K_1, pygame.K_2): self.players = 1 if event.key == pygame.K_1 else 2
            elif event.key in (pygame.K_5, pygame.K_0): self.shots_per_player = 5 if event.key == pygame.K_5 else 10
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE): self.start()
        return None

    def render(self, screen):
        now = time.monotonic()
        hold_result = self.phase == 'result' and self.reaction_at is not None and now < self.reaction_at
        if (self.phase == 'waiting' or hold_result) and self._frozen is not None:
            screen.blit(self._frozen, self.viewport.topleft); return
        effect_frame=None
        if self.phase=='result' and self.reaction_at is not None:
            if self._reaction_start_mono is None:self._reaction_start_mono=now
            effect_frame=int(max(0.0,now-self._reaction_start_mono)*BREAK_FPS)
            self._reaction_visible=True
        key = (self.phase, self.token, self.players, self.shots_per_player, self.player, tuple(self.scores), tuple(self.used), self.feedback, effect_frame,self.sound_enabled)
        if key == self._render_key and self._last_frame is not None:
            screen.blit(self._last_frame, self.viewport.topleft); return
        surf = pygame.Surface(self.viewport.size); surf.fill(BACKGROUND)
        if self.phase in ('intro', 'finished', 'unsupported'):
            title = 'PANG! • HA SÖNDER RUMMET' if self.phase == 'intro' else ('KLART! ' + ' / '.join(f'Spelare {i+1}: {self.scores[i]}' for i in range(self.players)))
            if self.phase == 'unsupported': title = self.feedback
            if self.phase=='finished' and self.items:
                self.art.render(surf,self.items,self.broken_ids,effect_sides=self.effect_sides)
            else:surf.blit(self.art.background,(0,0))
            card=pygame.Surface((self.viewport.w-48,300));card.fill((250,250,224))
            surf.blit(card,(24,100))
            lines = [title, f'{self.players} spelare • {self.shots_per_player} skott var',
                     '1 / 2 spelare    5 / 0 = fem / tio skott', 'ENTER startar. Ett skott på REDO. ESC: meny.',
                     'G: målupplysning '+('PÅ' if self._uses_glow() else 'AV')+' • Delträffar ger bonus.',
                     f'Luftgevär 20 m. M: ljud {"på" if self.sound_enabled else "av"}.']
            if self.phase=='finished' and self.last_outcome:lines.append('Senast: '+self.feedback)
            for i,line in enumerate(lines):surf.blit(self.font.render(line,True,INK),(42,120+i*38))
        else:
            self.art.render(surf,self.items,self.broken_ids,self._effect_ids if self.phase=='result' else (),effect_frame,self.effect_sides)
            if self.phase=='result' and self._impact_xy is not None and effect_frame is not None and effect_frame<4:
                pygame.draw.circle(surf,(255,234,154),tuple(round(v) for v in self._impact_xy),10+5*effect_frame,3)
            status = 'REDO' if self.phase == 'live' else ('VÄNTA' if self.phase == 'settling' else self.feedback)
            states=self.entity_states()
            remaining=sum(v!='broken' for v in states.values())
            header=f'PANG! P{self.player+1} • {self.used[self.player]}/{self.shots_per_player} • {self.scores[self.player]} p • {remaining} kvar'
            colors={'HIT':((35,140,80),(255,255,245)),'MISS':((235,165,45),INK),'UNKNOWN':((40,90,160),(255,255,245))}
            banner,text=colors.get(self.last_outcome,((192,196,190),INK)) if self.phase=='result' else ((192,196,190),INK)
            pygame.draw.rect(surf,banner,(0,0,self.viewport.w,64))
            if self._uses_glow() and self.phase=='settling':status='VISA MÅLEN – '+{'show_targets':'titta på målen','ready':'referens klar','measuring':'mäter','no_frames':'kamerabilder saknas'}.get(self._glow_state,'väntar på stabil mätbild')
            if self.art.rooms:status+=' • '+{'living':'Vardagsrum','kitchen':'Kök','study':'Arbetsrum'}[self.art.room_template]
            if self.phase=='live' and self.last_outcome:status+=' • senast '+{'HIT':'TRÄFF','MISS':'MISS','UNKNOWN':'OSÄKERT'}[self.last_outcome]
            surf.blit(self.font.render(header,True,text),(18,8))
            surf.blit(self.font.render(status,True,text),(18,33))
            counts=f'Hela {list(states.values()).count("intact")} / Träffade {list(states.values()).count("hit")} / Krossade {list(states.values()).count("broken")}'
            if self.phase!='result':surf.blit(self.font.render(counts,True,text),(self.viewport.w-310,45))
            sound_label='M: ljud på' if self.sound_enabled else 'M: tyst'
            surf.blit(self.font.render(sound_label,True,text),(self.viewport.w-116,12))
        self._last_frame = surf
        self._render_key = key
        screen.blit(surf,self.viewport.topleft)


def create_game(game_root, viewport):
    return LightTargetsGame(game_root, viewport)
