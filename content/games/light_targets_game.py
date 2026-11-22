"""Ljusjakten: static bright OBJECT targets; physical acceptance is measured separately."""
from __future__ import annotations
from dataclasses import replace
import json
import random
import time
import uuid
import pygame
from src.engine.events import event_bus
from src.engine.game_objects import ObjectManager, ObjectGeometry, HitShapeSpec, make_breakable_object
from src.engine.input.hit_regions import object_hit_regions, hit_context_snapshot_for_shot, object_hit_registry_v2223
from src.engine.input.hit_input import hit_input
from src.engine.settings import load_range_projection_settings

BACKGROUND = (255, 255, 192)
PALETTE = ((160, 240, 205), (150, 240, 245), (220, 245, 150))
INK = (32, 66, 65)


class LightTargetsGame:
    owns_hit_presentation = True
    shot_critical_updates_required = True  # timers run, but pending render is immutable
    hit_precision = 'OBJECT'
    precision_flash = 'off'

    def __init__(self, game_root, viewport):
        self.viewport = viewport.copy()
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

    def on_enter(self):
        if self._subscribed: return
        self.font = pygame.font.Font(None, 23)
        self.big = pygame.font.Font(None, 38)
        event_bus.subscribe(self._on_object_event)
        hit_input.subscribe(self._on_hit)
        self._subscribed = True

    def on_exit(self):
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_exit')
        if self._subscribed:
            event_bus.unsubscribe(self._on_object_event)
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    def start(self):
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_restart')
        self.owner = 'light_targets:' + uuid.uuid4().hex
        self._seen.clear()
        self.scores = [0, 0]; self.used = [0, 0]
        self.player = 0; self.streak = 0; self._slots = []
        self._next_target()

    def _next_target(self):
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_surface_changed')
        self.token += 1; self.active_shot = None; self.active_peak = None
        self.presented_ts = None; self.presented_mono = None; self._frozen = None; self._last_frame = None
        self.phase = 'settling'; self.phase_at = time.monotonic()
        w, h = self.viewport.size
        # Large projected target, independent of virtual-distance shrinkage.
        diameter = min(120, w * .20, h * .24)
        radius = diameter / 2
        if not self._slots:
            self._slots = [(x, y) for y in range(3) for x in range(4)]
            self.rng.shuffle(self._slots)
        if self._last_slot is not None:
            # Prefer a distant remaining cell; every cell is used before reuse.
            self._slots.sort(key=lambda c: (c[0]-self._last_slot[0])**2+(c[1]-self._last_slot[1])**2)
        slot = self._slots.pop(); self._last_slot = slot
        xmin, xmax = radius + w*.10, w-radius-w*.10
        ymin, ymax = radius + max(108, h*.17), h-radius-h*.10
        # Tiny viewports abstain rather than placing a target over the HUD.
        if xmax <= xmin or ymax <= ymin:
            self.phase = 'unsupported'; self.feedback = 'För liten viewport – välj minst 480 × 360.'; return
        cx = xmin + (xmax-xmin)*slot[0]/3
        cy = ymin + (ymax-ymin)*slot[1]/2
        cx, cy, radius = round(cx), round(cy), round(radius)
        diameter = 2 * radius
        geometry = ObjectGeometry(cx-radius, cy-radius, diameter, diameter)
        settings = load_range_projection_settings()
        width_cm = max(1, float(settings.get('viewport_physical_width_cm', 90)))
        touch = 2.25 * w / (width_cm * 10)  # actual 4.5 mm projectile, same edge convention as cans
        shape = replace(HitShapeSpec(kind='ellipse'), touch_radius=touch)
        self.manager = ObjectManager()
        obj = make_breakable_object('lamp', geometry, shape=shape, break_effect='', owner=self.owner)
        obj.metadata.update(lamp_owner=self.owner, lamp_token=self.token, player=self.player+1)
        self.manager.add(obj)
        self.target = geometry
        self.color = PALETTE[self.token % len(PALETTE)]
        print('[LAMP-TARGET] '+json.dumps(dict(owner=self.owner, token=self.token, player=self.player+1,
            virtual_distance_m=20, geometry=list(geometry.aabb), shape='ellipse', touch_radius=touch,
            color=self.color, wall_ts=time.time())), flush=True)

    def get_hit_regions(self):
        if self.phase not in ('live', 'waiting'): return ()
        regions = tuple(replace(r, metadata={**r.metadata, 'lamp_owner':self.owner,
                           'lamp_token':self.token}) for r in self.manager.get_hit_regions())
        return object_hit_regions(regions, owner=self.owner, evidence_method='dense', top_k=3,
            margin_camera_px=36, surface_after_ts=self.presented_ts, require_presented_surface=True)

    def on_frame_presented(self, timestamp):
        if self.phase in ('settling', 'live') and self._last_frame is not None and self.presented_ts is None:
            self.presented_ts = float(timestamp)
            self.presented_mono = time.monotonic()
            print('[LAMP-PRESENTED] '+json.dumps(dict(owner=self.owner, token=self.token, wall_ts=self.presented_ts)), flush=True)
        if self.phase == 'result' and self.active_shot is not None:
            snap = hit_context_snapshot_for_shot(self.active_shot)
            if snap is not None:
                snap.object_gameplay.update(reaction_presented_ts=timestamp)

    def on_object_shot_frozen(self, snapshot):
        if self.phase != 'live' or self.active_shot is not None: return
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
        hit = bool(resolution is not None and resolution.interactions)
        decision = result.get('object_decision', {}).get('state', 'UNKNOWN')
        outcome = 'HIT' if hit else ('MISS' if decision == 'NO_OBJECT_HIT' else 'UNKNOWN')
        snap.object_gameplay = dict(owner=self.owner, accepted=True, result=outcome, reaction_expected=True,
                                    player=self.player+1, token=self.token)
        self._finish(outcome, sid, result, hypothesis)

    def _on_hit(self, hit):
        # Physical CURRENT XY never scores. Mouse is an explicitly labelled preview.
        if self.phase != 'live' or not getattr(hit, 'is_simulated', False): return
        resolution = self.manager.resolve_hit(hit)
        self._finish('HIT' if resolution.interactions else 'MISS', None, None, None, simulated=True)

    def _finish(self, outcome, sid, result, hypothesis, simulated=False):
        self.phase = 'result'; self.phase_at = time.monotonic(); self._frozen = None
        object_hit_registry_v2223.cancel_object_queries(self.owner, 'light_targets_decision_complete')
        self.used[self.player] += 1
        if outcome == 'HIT':
            self.streak += 1; points = 100 + 25 * min(self.streak-1, 4)
            self.scores[self.player] += points; self.feedback = f'TRÄFF! +{points}  Svit {self.streak}'
        elif outcome == 'MISS':
            self.streak = 0; points = 0; self.feedback = 'INGEN MÅLTRÄFF – nästa lampa!'
        else:
            self.streak = 0; points = 0; self.feedback = 'OSÄKERT – inga poäng (inte verifierad miss)'
        print('[LAMP-RESULT] '+json.dumps(dict(owner=self.owner, token=self.token, shot_id=sid,
            shot_peak_ts=self.active_peak, player=self.player+1, outcome=outcome, points=points,
            shots_used=self.used[self.player], source='SIMULATED' if simulated else 'OBJECT',
            hypothesis_xy=(hypothesis or {}).get('camera_xy'), reason=(result or {}).get('reason'),
            wall_ts=time.time())), flush=True)

    def update(self, dt):
        now = time.monotonic()
        if self.phase == 'settling' and self.presented_ts is not None and self.presented_mono is not None and now-self.presented_mono >= 1.1:
            self.phase = 'live'
        elif self.phase == 'waiting' and now-self.phase_at > 4:
            # Engine result lost/cancelled: explicitly UNKNOWN, never an invented miss.
            self._finish('UNKNOWN', self.active_shot, dict(reason='game_wait_timeout'), None)
        elif self.phase == 'result' and now-self.phase_at >= 1.4:
            if self.used[self.player] >= self.shots_per_player:
                if self.player+1 < self.players:
                    self.player += 1; self.streak = 0; self._next_target()
                else: self.phase = 'finished'
            else: self._next_target()
        return None

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN: return None
        if self.phase in ('intro', 'finished', 'unsupported'):
            if event.key in (pygame.K_1, pygame.K_2): self.players = 1 if event.key == pygame.K_1 else 2
            elif event.key in (pygame.K_5, pygame.K_0): self.shots_per_player = 5 if event.key == pygame.K_5 else 10
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE): self.start()
        return None

    def render(self, screen):
        if self.phase == 'waiting' and self._frozen is not None:
            screen.blit(self._frozen, self.viewport.topleft); return
        key = (self.phase, self.token, self.players, self.shots_per_player, self.player, tuple(self.scores), tuple(self.used), self.feedback)
        if key == self._render_key and self._last_frame is not None:
            screen.blit(self._last_frame, self.viewport.topleft); return
        surf = pygame.Surface(self.viewport.size); surf.fill(BACKGROUND)
        if self.phase in ('intro', 'finished', 'unsupported'):
            title = 'LJUSJAKTEN • LUFTGEVÄR 20 m' if self.phase == 'intro' else ('KLART! ' + ' / '.join(f'Spelare {i+1}: {self.scores[i]}' for i in range(self.players)))
            if self.phase == 'unsupported': title = self.feedback
            lines = [title, f'{self.players} spelare • {self.shots_per_player} skott var',
                     '1 / 2 spelare    5 / 0 = fem / tio skott', 'ENTER startar. Ett skott per lampa. ESC: meny.',
                     'Skjut när REDO visas. Inga tidspoäng – lugnt och säkert.']
            for i, line in enumerate(lines): surf.blit(self.font.render(line, True, INK), (24, 50+i*36))
        else:
            # Smooth broad bright boundary; no dark outline, string, text or shadow by the target.
            g = self.target; center = tuple(round(v) for v in g.center); radius = round(g.width/2)
            for r in range(radius, radius-6, -1):
                t = (radius-r)/5; color=tuple(round(BACKGROUND[j]*(1-t)+self.color[j]*t) for j in range(3))
                pygame.draw.circle(surf,color,center,r)
            pygame.draw.circle(surf,self.color,center,max(1,radius-6))
            status = 'REDO' if self.phase == 'live' else ('VÄNTA' if self.phase == 'settling' else self.feedback)
            header=f'Spelare {self.player+1} • {self.used[self.player]}/{self.shots_per_player} • {self.scores[self.player]} poäng'
            surf.blit(self.font.render(header,True,INK),(14,10))
            surf.blit(self.font.render(status,True,INK),(14,38))
            if self.phase == 'result' and self.feedback.startswith('TRÄFF'):
                # Gentle local success ring only AFTER queries have closed; no white flash.
                pygame.draw.circle(surf,(160,230,160),center,radius+8,5)
        self._last_frame = surf
        self._render_key = key
        screen.blit(surf,self.viewport.topleft)


def create_game(game_root, viewport):
    return LightTargetsGame(game_root, viewport)
