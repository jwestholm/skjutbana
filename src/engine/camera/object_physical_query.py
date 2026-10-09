"""Bounded, shot-local OBJECT query; never a global coordinate/learning authority.

Owned by ObjectShotSnapshotV2223, delivered on the existing EventBus. Reuses
V2.24 region merging/working maps, V2.22 analysis crops, V2 registration and
V2.25 coordinate-local disc/ring measurements. Thresholds are engineering
defaults, not calibrated physical probabilities or an accuracy claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np

from src.engine.camera.analysis_geometry_v2221 import AnalysisGeometryV2221
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.camera.demo_object_zone_extension import _regions_in_work_space
from src.engine.game_objects.geometry import shape_contains
from src.engine.shot_object_local_v241 import merge_camera_regions_v241
from src.engine.shot_region_freshness_v252 import _disc_ring_metrics, RegisteredMetricsV252

MAX_REGIONS = 64
MAX_PIXELS = 1_000_000
MAX_FRAMES = 8
MAX_COMPONENTS = 24
PER_OBJECT_HYPOTHESES = 3
PER_OBJECT_COMPONENTS = 2
DEADLINE_S = .35
CONTEXT_PX = 8
_yy,_xx=np.ogrid[-8:9,-8:9]
_d2=_xx**2+_yy**2
_CENTER=_d2<=4
_RING=(_d2>=16)&(_d2<=64)


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()


class ObjectPhysicalQuery:
    """One bounded query, stored in the existing registry's shot snapshot.

    Coordinates are actual POST pixels, including when PRE is registered into
    POST. Persistence is remeasured at this SAME XY in the previous causal POST;
    no representative, region confidence or old-hole confirmation is inherited.
    """

    def __init__(self, snapshot, pre_frames, *, camera_to_game=None,
                 known_holes=(), top_k=3, margin_px=36., capture_crop=None, capture_result=None):
        self.shot_id = snapshot.shot_id
        self.peak_ts = snapshot.peak_ts
        self.regions = deepcopy(tuple(snapshot.camera_regions))
        self.game_regions = deepcopy(tuple(snapshot.game_regions))
        self.object_hit_requested = any(r.metadata.get('physical_query',{}).get(
            'result_semantics') == 'OBJECT_HIT' for r in self.game_regions)
        self.camera_to_game = camera_to_game
        self.capture_crop=capture_crop
        self.capture_result=capture_result
        if int(top_k)<1: raise ValueError('OBJECT top_k must be positive')
        self.top_k = min(3, int(top_k))
        self.margin = max(0., min(128., float(margin_px)))
        self.known_holes = tuple(deepcopy(known_holes))[:512]
        self.tiles = []
        self.frames = 0
        self.first_post_ts = None
        self.first_hypothesis_ts = None
        self.last_frame_ts = -math.inf
        self.final = False
        self.published = False
        self.saw_change = False
        self.pre_activity = False
        self.processing_ms = []
        self.roi_prepare_ms = self.pre_prepare_ms = 0.
        self.post_analysis_ms = []
        self.merge_dedupe_ms = []
        self.proposal_ms = []
        self.local_confirmation_ms = []
        self.freshness_ms = []
        self.candidate_count = 0
        self.examined_candidate_count = 0
        self.result = None
        self.pre_times = [float(f.timestamp) for f in (pre_frames[-3:] if self.object_hit_requested else pre_frames)]
        self.search_pixels = self.read_pixels = self.frame_pixels = 0
        self.hypotheses = []
        self.per_object = []
        self.candidate_ledger = []
        started = time.perf_counter()
        if not self.regions or not snapshot.transform_available:
            self.finish('UNCERTAIN', 'missing_frozen_camera_regions_or_transform')
        elif len(self.regions) > MAX_REGIONS:
            self.finish('UNCERTAIN', 'region_budget_exceeded')
        elif not pre_frames or any(t >= self.peak_ts for t in self.pre_times):
            self.finish('UNCERTAIN', 'no_causal_recent_surface')
        elif self.object_hit_requested and any(a>=b for a,b in zip(self.pre_times,self.pre_times[1:])):
            self.finish('UNCERTAIN','pre_history_not_strictly_ordered')
        elif self.peak_ts - max(self.pre_times) > .5:
            self.finish('UNCERTAIN', 'recent_surface_too_old')
        else:
            self._prepare(pre_frames[-3:])
        self.prepare_ms = (time.perf_counter() - started) * 1000

    def _prepare(self, pre_frames):
        roi_started=time.perf_counter()
        h, w = pre_frames[-1].frame_bgr.shape[:2]
        if any(f.frame_bgr.shape[:2] != (h,w) for f in pre_frames):
            self.finish('UNCERTAIN','pre_camera_geometry_changed')
            return
        self.frame_shape = (h, w)
        self.frame_pixels = h*w
        windows = list(merge_camera_regions_v241(self.regions,
            margin_px=self.margin+CONTEXT_PX, max_regions=MAX_REGIONS))
        # A union's AABB can overlap another group's AABB even when the original
        # rectangles did not. Merge to a fixed point so no crop is scanned twice.
        changed = True
        while changed:
            changed = False
            for i in range(len(windows)):
                for j in range(i+1, len(windows)):
                    if windows[i].overlaps(windows[j]):
                        windows[i] = windows[i].union(windows.pop(j))
                        changed = True
                        break
                if changed:
                    break
        bounds = [(max(0, math.floor(q.x0)), max(0, math.floor(q.y0)),
                   min(w, math.ceil(q.x1)), min(h, math.ceil(q.y1))) for q in windows]
        self.read_pixels = sum(max(0,x1-x0)*max(0,y1-y0) for x0,y0,x1,y1 in bounds)
        if self.read_pixels > MAX_PIXELS:
            self.finish('UNCERTAIN', 'pixel_budget_exceeded')
            return
        self.roi_prepare_ms += (time.perf_counter()-roi_started)*1000.
        for window, (x0,y0,x1,y1) in zip(windows,bounds):
            roi_started=time.perf_counter()
            if x1-x0 < 9 or y1-y0 < 9:
                continue
            shape = (y1-y0,x1-x0)
            geometry = AnalysisGeometryV2221(h,w,x0,y0,x1,y1,
                np.empty((0,2)),np.empty((0,2)),np.ones(shape,np.uint8),
                'object_query',0.,CONTEXT_PX)
            scanner = SimpleNamespace(_v2221_active_geometry=geometry)
            work_regions = _regions_in_work_space(self.regions, scanner, shape)
            mask = np.zeros(shape,np.uint8)
            for region in work_regions:
                a,b = max(0,math.floor(region.x-self.margin)),max(0,math.floor(region.y-self.margin))
                c,d = min(shape[1],math.ceil(region.x+region.width+self.margin)),min(shape[0],math.ceil(region.y+region.height+self.margin))
                if c>a and d>b:
                    mask[b:d,a:c] = 255
            mask[:CONTEXT_PX,:] = mask[-CONTEXT_PX:,:] = 0
            mask[:,:CONTEXT_PX] = mask[:,-CONTEXT_PX:] = 0
            if not np.any(mask):
                continue
            self.roi_prepare_ms += (time.perf_counter()-roi_started)*1000.
            pre_started=time.perf_counter()
            # Slice before conversion: no grayscale/difference scan of the board.
            stack = np.stack([_gray(geometry.crop_array(f.frame_bgr)) for f in pre_frames]).astype(np.float32)
            if self.capture_crop:
                for f,pixels in zip(pre_frames,stack):
                    self.capture_crop(self.shot_id,'pre',f.timestamp,(x0,y0,x1,y1),pixels)
            reference = stack[-1].copy()  # latest current surface; not a clean-board median
            noise = np.median(np.abs(stack-reference),axis=0)
            envelope=None;pre_registration=[];photometric_offsets=[]
            if self.object_hit_requested and len(stack)>=2:
                freshness_started=time.perf_counter()
                low,high=reference.copy(),reference.copy()
                for prior in stack[:-1]:
                    aligned,reg=CandidateGeneratorV2._register_current(None,reference,prior,
                        roi=mask,cfg={'registration_max_dimension':256})
                    matched,offset=CandidateGeneratorV2._normalise_photometry(reference,aligned,roi=mask)
                    np.minimum(low,matched,out=low);np.maximum(high,matched,out=high)
                    pre_registration.append(reg);photometric_offsets.append(offset)
                envelope=(low,high)
                self.freshness_ms.append((time.perf_counter()-freshness_started)*1000.)
            self.pre_activity |= np.count_nonzero((noise>3.)&(mask>0))>=2
            valid_pixels=int(np.count_nonzero(mask))
            self.tiles.append(dict(geometry=geometry,mask=mask,valid_pixels=valid_pixels,pre=reference,
                noise=noise,previous=None,previous_ts=None,object_ids=window.object_ids,
                pre_envelope=envelope,pre_registration=pre_registration,
                pre_photometric_offsets=photometric_offsets,previous_new=None))
            self.search_pixels += valid_pixels
            self.pre_prepare_ms += (time.perf_counter()-pre_started)*1000.
        if not self.tiles:
            self.finish('UNCERTAIN','no_observable_search_pixels')

    @staticmethod
    def _evidence(absolute, noise, signed, x, y):
        # PRE variability scales evidence gently; it is never a validity veto.
        x0,y0=max(0,x-8),max(0,y-8)
        a=absolute[y0:y+9,x0:x+9]
        n=noise[y0:y+9,x0:x+9]
        d=signed[y0:y+9,x0:x+9]
        z = a / np.maximum(1., np.minimum(n,4.))
        m = _disc_ring_metrics(a,z,np.maximum(-d,0),n,x-x0,y-y0)
        return ObjectPhysicalQuery._local_result(m)

    @staticmethod
    def _local_result(m):
        if m is None:
            return None,False,0.
        localized = m.center_abs >= 1.6 and m.compact_abs >= .30 and m.center_z >= 1.3
        score = max(0.,m.compact_abs) * min(1.,m.center_abs/6.)
        return m,localized,score

    @staticmethod
    def _evidence_many(absolute,noise,signed,points):
        """Same r2/r4..8 measurements, vectorized over this bounded component list.

        Boolean selection order, float32 means and Python-float subtraction match
        _disc_ring_metrics. No pruning, map smoothing or new physical threshold.
        Clipped boundary patches retain the scalar implementation.
        """
        if not points:return []
        h,w=absolute.shape
        if (any(a.dtype!=np.float32 for a in (absolute,noise,signed)) or
                any(x<8 or y<8 or x+8>=w or y+8>=h for x,y in points)):
            return [ObjectPhysicalQuery._evidence(absolute,noise,signed,x,y) for x,y in points]
        xy=np.asarray(points,dtype=int);offset=np.arange(-8,9)
        ix=xy[:,0,None,None]+offset[None,None,:]
        iy=xy[:,1,None,None]+offset[None,:,None]
        a=absolute[iy,ix];n=noise[iy,ix];d=signed[iy,ix]
        z=a/np.maximum(1.,np.minimum(n,4.));dark=np.maximum(-d,0)
        # Boolean indexing yields a column-contiguous array. Copy rows to keep
        # NumPy's pairwise reduction order identical to scalar a[mask].mean().
        def mean(v,mask):return np.ascontiguousarray(v[:,mask]).mean(axis=1)
        values=np.stack([mean(a,_CENTER),mean(a,_RING),
            mean(z,_CENTER),mean(dark,_CENTER),mean(dark,_RING),
            a[:,_CENTER].max(axis=1),np.median(n[:,_CENTER],axis=1)],axis=1)
        result=[]
        for center,ring,center_z,center_dark,ring_dark,peak,pre_noise in values.tolist():
            m=RegisteredMetricsV252(center_abs=center,ring_abs=ring,compact_abs=center-ring,
                peak_abs=peak,center_z=center_z,center_dark=center_dark,ring_dark=ring_dark,
                dark_compact=center_dark-ring_dark,dark_fraction=max(0.,min(1.5,center_dark/max(.5,center))),
                pre_noise=max(0.,pre_noise))
            result.append(ObjectPhysicalQuery._local_result(m))
        return result

    @staticmethod
    def _new_change(signed, other_signed):
        """Signed distance beyond ALL observed PRE levels, in the POST plane.

        Inputs are POST minus registered/photometrically matched PRE bounds.
        Zero means POST overlaps the observed PRE envelope. No learned threshold,
        polarity preference, candidate identity or game geometry enters here.
        This qualifies observed novelty, not projectile causation or clean PRE.
        """
        low,high=signed.copy(),signed.copy()
        for delta in other_signed:
            np.minimum(low,delta,out=low);np.maximum(high,delta,out=high)
        return np.maximum(low,0.)+np.minimum(high,0.)

    @staticmethod
    def _same_polarity(a,b):
        # mean(signed) == mean(abs) - 2*mean(darkening), same existing r2 core.
        return a is not None and b is not None and (
            (a.center_abs-2*a.center_dark)*(b.center_abs-2*b.center_dark)>0.)

    def _component_reserve(self, labels, sizes, means, geometry):
        """Keep the existing crop budget, reserving two components per object.

        Only label IDs are inspected here; crops are still processed once. For
        unusually many merged objects the explicit bound is 2*regions (<=128).
        No score, threshold, confirmation or coordinate is supplied by geometry.
        """
        order=np.flatnonzero((sizes>=2)&(np.arange(len(sizes))>0))
        order=order[np.argsort(-means[order],kind='stable')]
        pools={}
        h,w=labels.shape
        for r in sorted(self.regions,key=lambda r:r.object_id):
            x,y=geometry.camera_to_local(r.x-self.margin,r.y-self.margin)
            right,bottom=geometry.camera_to_local(r.right+self.margin,r.bottom+self.margin)
            a,b=max(0,math.floor(x)),max(0,math.floor(y))
            c,d=min(w,math.ceil(right)),min(h,math.ceil(bottom))
            ids=np.unique(labels[b:d,a:c]) if c>a and d>b else np.array([],dtype=int)
            ids=ids[(ids>0)&(sizes[ids]>=2)]
            ids=ids[np.argsort(-means[ids],kind='stable')]
            pools[r.object_id]=ids
        reserved=set()
        for ids in pools.values():reserved.update(map(int,ids[:PER_OBJECT_COMPONENTS]))
        budget=max(MAX_COMPONENTS,len(reserved))
        selected=set(reserved)
        for cid in order:
            if len(selected)>=budget:break
            selected.add(int(cid))
        diagnostics={oid:dict(raw_components=len(ids),retained_components=sum(int(i) in selected for i in ids),
            best_components=[dict(component_id=int(i),mean_change=float(means[i]),
                reason='per_object_reserve' if int(i) in reserved else
                    ('crop_budget_retained' if int(i) in selected else 'crop_component_capacity'))
                for i in ids[:PER_OBJECT_HYPOTHESES]]) for oid,ids in pools.items()}
        return [int(i) for i in order if int(i) in selected],diagnostics

    @staticmethod
    def _physical_order(c):
        return (not c['confirmed'],-c['evidence_score'],c['camera_xy'])

    def _select_hypotheses(self,candidates,production):
        """Evaluate each frozen shape BEFORE truncating the public top-k.

        Scores remain physical. Exact collision defines which supported points
        answer the OBJECT query; it cannot confirm change. Preserve observed XY
        and all contributors when deduplicating, including across shape edges.
        """
        candidates.sort(key=self._physical_order)
        for i,c in enumerate(candidates):c['global_evidence_rank']=i+1
        rows=[];pool={}
        for r in sorted(self.regions,key=lambda r:r.object_id):
            associated=[c for c in candidates if any(a['object_id']==r.object_id for a in c['regions'])]
            for i,c in enumerate(associated):
                a=next(a for a in c['regions'] if a['object_id']==r.object_id)
                a['local_evidence_rank']=i+1
            # Preserve one supported exact-shape point, then physical-score
            # alternatives. Nearby high-amplitude change cannot erase that point.
            exact=[c for c in associated if c['confirmed'] and any(
                a['object_id']==r.object_id and a['inside_exact_shape'] is True for a in c['regions'])]
            retained=[]
            for c in exact[:1]+associated:
                if any(c is old for old in retained):continue
                retained.append(c)
                if len(retained)>=PER_OBJECT_HYPOTHESES:break
            for c in retained:pool[tuple(c['camera_xy'])]=c
            rows.append(dict(object_id=r.object_id,observed_count=len(associated),
                production=production.get(r.object_id,{}),
                decision='supported_exact_shape' if exact else
                    ('near_or_unconfirmed_only' if associated else 'no_observed_candidate'),
                candidates=retained))
        # Collisions here are pure frozen geometry checks, with no game damage.
        def exact_supported(c):
            return c['confirmed'] and any(a['inside_exact_shape'] is True for a in c['regions'])
        ordered=sorted(pool.values(),key=lambda c:(not exact_supported(c),*self._physical_order(c)))
        chosen=[];reasons={}
        for c in ordered:
            inside={a['object_id'] for a in c['regions'] if a['inside_exact_shape'] is True}
            duplicate=next((old for old in chosen if math.dist(c['camera_xy'],old['camera_xy'])<4.
                and inside=={a['object_id'] for a in old['regions'] if a['inside_exact_shape'] is True}),None)
            key=tuple(c['camera_xy'])
            if duplicate is not None:reasons[key]='spatial_duplicate_same_exact_membership'
            elif len(chosen)>=self.top_k:reasons[key]='public_top_k_after_object_evaluation'
            else:chosen.append(c);reasons[key]='returned'
        for row in rows:
            selected_ids={id(c) for c in row['candidates']}
            row['candidates']=[dict(c,retention_reason=reasons[tuple(c['camera_xy'])]) for c in row['candidates']]
            row['dropped_by_object_capacity']=row['observed_count']-len(selected_ids)
        self.per_object=rows
        self.candidate_ledger.append(dict(frame_ts=self.last_frame_ts,objects=deepcopy(rows)))
        return chosen

    def observe(self, frame, *, delivered_ts=None):
        if self.final:
            return self.result
        ts = float(frame.timestamp)
        now = time.time() if delivered_ts is None else float(delivered_ts)
        if now >= self.peak_ts+DEADLINE_S:
            return self.finish_deadline(now)
        if ts > now or ts < self.peak_ts or ts <= self.last_frame_ts:
            return self.result
        if ts > self.peak_ts+DEADLINE_S:
            return self.finish_deadline(now)
        if frame.frame_bgr.shape[:2] != self.frame_shape:
            return self.finish('UNCERTAIN','camera_geometry_changed')
        started = time.perf_counter()
        stop_after=started+self.peak_ts+DEADLINE_S-now
        self.first_post_ts = ts if self.first_post_ts is None else self.first_post_ts
        self.last_frame_ts = ts
        self.frames += 1
        candidates = []
        production = {}
        def over_budget():
            elapsed=time.perf_counter()-started
            if time.perf_counter()<stop_after: return False
            self.processing_ms.append(elapsed*1000.)
            self.hypotheses=[]
            self.finish('UNCERTAIN','processing_deadline_exceeded',now+elapsed)
            return True
        for tile in self.tiles:
            if over_budget(): return self.result
            geometry,mask = tile['geometry'],tile['mask']
            post = _gray(geometry.crop_array(frame.frame_bgr)).astype(np.float32)
            if self.capture_crop:
                self.capture_crop(self.shot_id,'post',ts,
                    (*geometry.local_to_camera(0,0),*geometry.local_to_camera(post.shape[1],post.shape[0])),post)
            # Move PRE into the observed POST plane, never move a hypothesis to
            # an object or emit a coordinate interpolated between observations.
            pre, registration = CandidateGeneratorV2._register_current(None,post,tile['pre'],
                roi=mask,cfg={'registration_max_dimension':256})
            current, _ = CandidateGeneratorV2._normalise_photometry(pre,post,roi=mask)
            signed = current-pre
            absolute = np.abs(signed).astype(np.float32)
            new_signed=None
            if self.object_hit_requested and tile['pre_envelope'] is not None:
                freshness_started=time.perf_counter()
                low,high=tile['pre_envelope']
                if registration['applied']:
                    matrix=np.float32([[1,0,-registration['dx']],[0,1,-registration['dy']]])
                    low,high=(cv2.warpAffine(a,matrix,(post.shape[1],post.shape[0]),
                        flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE) for a in (low,high))
                # Positive interpolation weights preserve a conservative envelope:
                # warped minimum <= every warped PRE <= warped maximum.
                new_signed=self._new_change(current-low,(current-high,))
                self.freshness_ms.append((time.perf_counter()-freshness_started)*1000.)
            new_absolute=None if new_signed is None else np.abs(new_signed)
            values = absolute[mask>0]
            median = float(np.median(values))
            threshold = max(3.,median+3.*1.4826*float(np.median(np.abs(values-median))))
            proposal_started=time.perf_counter()
            binary = ((absolute>=threshold)&(mask>0)).astype(np.uint8)
            count,labels,stats,_ = cv2.connectedComponentsWithStats(binary,8)
            self.saw_change |= count>1
            smooth = cv2.boxFilter(absolute,-1,(3,3))
            ranked = []
            sizes=stats[:,cv2.CC_STAT_AREA]
            means=np.bincount(labels.ravel(),weights=absolute.ravel(),minlength=count)/np.maximum(1,sizes)
            components,component_diagnostics=self._component_reserve(labels,sizes,means,geometry)
            raw_ids={c['component_id'] for d in component_diagnostics.values() for c in d['best_components']}
            raw_positions={}
            for component in raw_ids:
                x,y,width,height,area=stats[component]
                support=np.where(labels[y:y+height,x:x+width]==component,smooth[y:y+height,x:x+width],-1.)
                py,px=np.unravel_index(np.argmax(support),support.shape)
                cx,cy=geometry.local_to_camera(x+px,y+py)
                raw_positions[component]=[cx,cy]
            for oid,diag in component_diagnostics.items():
                for raw in diag['best_components']:
                    raw['camera_xy']=raw_positions[raw['component_id']]
                    raw['nomination_threshold']=threshold
            for oid,diag in component_diagnostics.items():
                if diag['raw_components'] or oid not in production:production[oid]=diag
            for component in components:
                x,y,width,height,area = stats[component]
                if area < 2:
                    continue
                local = labels[y:y+height,x:x+width] == component
                support = np.where(local,smooth[y:y+height,x:x+width],-1.)
                py,px = np.unravel_index(np.argmax(support),support.shape)
                ranked.append((float(support[py,px]),x+px,y+py,int(area),int(component)))
            ranked.sort(reverse=True)
            self.proposal_ms.append((time.perf_counter()-proposal_started)*1000.)
            evidence_started=time.perf_counter()
            self.candidate_count += len(ranked)
            batches=None
            if self.object_hit_requested:
                points=[(r[1],r[2]) for r in ranked]
                def measure(a,d):return self._evidence_many(a,tile['noise'],d,points)
                missing=[(None,False,0.)]*len(points)
                batches=(measure(absolute,signed),missing if tile['previous'] is None else measure(*tile['previous']),
                    missing if new_signed is None else measure(new_absolute,new_signed),
                    missing if tile['previous_new'] is None else measure(*tile['previous_new']))
            for index,(nomination_score,x,y,area,component) in enumerate(ranked):
                if over_budget(): return self.result
                self.examined_candidate_count+=1
                metric,local_ok,score = batches[0][index] if batches is not None else self._evidence(absolute,tile['noise'],signed,x,y)
                if metric is None:
                    continue
                previous = tile['previous']
                old_metric,previous_ok,old_score = (batches[1][index] if batches is not None else
                    (None,False,0.) if previous is None else self._evidence(previous[0],tile['noise'],previous[1],x,y))
                # Large coherent changes remain UNCERTAIN, not NO_HIT. This is
                # per-frame morphology/context, never a permanent edge blacklist.
                compact_component = area <= max(64,int(tile['valid_pixels']*.1))
                persistent = bool(local_ok and previous_ok and compact_component)
                confirmed=persistent
                freshness=None
                if self.object_hit_requested:
                    nm,nok,_=batches[2][index]
                    pm,pok,_=batches[3][index]
                    polarity=self._same_polarity(nm,pm)
                    confirmed=bool(persistent and nok and pok and polarity)
                    why=('new_change_beyond_observed_pre' if confirmed else
                        'pre_history_unavailable' if new_signed is None else
                        'not_persistent_local_change' if not persistent else
                        'post_overlaps_pre_variation' if not nok else
                        'previous_post_not_new' if not pok else 'post_polarity_reversal')
                    freshness=dict(state='SUPPORTED' if confirmed else 'UNKNOWN',reason=why,
                        criterion='same_xy_signed_pre_envelope_v1',pre_frame_ts=list(self.pre_times),
                        current=asdict(nm) if nm else None,previous=asdict(pm) if pm else None,
                        same_polarity=bool(polarity),pre_registration=tile['pre_registration'],
                        pre_photometric_offsets=tile['pre_photometric_offsets'],clean_pre_certified=False,
                        physical_cause_certified=False)
                cx,cy = geometry.local_to_camera(x,y)
                game_xy = self.camera_to_game(cx,cy) if self.camera_to_game else None
                associations = []
                for region in self.regions:
                    dx=max(region.x-cx,0.,cx-region.right)
                    dy=max(region.y-cy,0.,cy-region.bottom)
                    distance=math.hypot(dx,dy)
                    if distance>self.margin:
                        continue
                    exact=None
                    original=next((r for r in self.game_regions if r.object_id==region.object_id),None)
                    shape=(original.metadata.get('v250_object_snapshot',{}).get('shape') if original else None)
                    if game_xy is not None and shape:
                        exact=shape_contains(shape,*game_xy)
                    elif self.object_hit_requested and game_xy is not None and original is not None:
                        # Plain HitRegion has rectangular semantics. Context
                        # margin never enters this exact game-space membership.
                        gx,gy=game_xy
                        exact=(original.x<=gx<original.x+original.width and
                               original.y<=gy<original.y+original.height)
                    associations.append(dict(object_id=region.object_id,inside_camera_aabb=distance==0.,
                        camera_aabb_distance_px=distance,inside_exact_shape=exact))
                if not associations:
                    continue
                nearest=min((math.hypot(cx-h['camera_x'],cy-h['camera_y']) for h in self.known_holes
                    if 'camera_x' in h and 'camera_y' in h),default=None)
                physical_score=min(score,old_score) if persistent else score
                candidates.append(dict(camera_xy=[cx,cy],game_xy=list(game_xy) if game_xy else None,
                    evidence_score=physical_score,confidence=1.-math.exp(-physical_score/8.),
                    confidence_semantics='uncalibrated_evidence_strength',confirmed=confirmed,
                    persistent_local_change=persistent,event_freshness=freshness,
                    evidence_decision=(freshness['reason'] if self.object_hit_requested else
                        'persistent_local_change' if confirmed else
                        'current_xy_not_localized' if not local_ok else
                        'component_not_compact' if not compact_component else
                        'no_previous_same_xy_confirmation'),
                    producer='OBJECT_LOCAL_CHANGE',frame_ts=ts,previous_frame_ts=tile['previous_ts'],
                    component_id=component,nomination_score=nomination_score,nomination_threshold=threshold,
                    tile_camera_bbox=[geometry.crop_x0,geometry.crop_y0,geometry.crop_x1,geometry.crop_y1],
                    pre_frame_ts=max(self.pre_times),delivered_ts=now,component_pixels=area,
                    evidence=asdict(metric),previous_same_xy_evidence=asdict(old_metric) if old_metric else None,
                    registration=registration,regions=associations,nearest_known_hole_px=nearest,
                    known_hole_semantics='context_only_no_truth_update'))
            tile['previous']=(absolute,signed)
            tile['previous_new']=None if new_signed is None else (new_absolute,new_signed)
            tile['previous_ts']=ts
            self.local_confirmation_ms.append((time.perf_counter()-evidence_started)*1000.)
        if over_budget(): return self.result
        merge_started=time.perf_counter()
        self.post_analysis_ms.append((merge_started-started)*1000.)
        chosen=self._select_hypotheses(candidates,production)
        if over_budget():return self.result
        self.hypotheses=chosen
        self.merge_dedupe_ms.append((time.perf_counter()-merge_started)*1000.)
        elapsed=time.perf_counter()-started
        finished_at=now+elapsed
        if chosen and self.first_hypothesis_ts is None:
            self.first_hypothesis_ts=finished_at
        self.processing_ms.append(elapsed*1000.)
        confirmed=[c for c in chosen if c['confirmed']]
        if confirmed:
            self.hypotheses=confirmed
            return self.finish('PHYSICAL_HIT','persistent_local_change',finished_at)
        if self.frames>=MAX_FRAMES or ts-self.peak_ts>=.20:
            return self.finish_deadline(finished_at)
        return self.payload('UNCERTAIN','awaiting_persistence',finished_at)

    def finish_deadline(self, now=None):
        enough=self.frames>=2
        status='NO_PHYSICAL_HIT' if enough and not self.saw_change and not self.pre_activity else 'UNCERTAIN'
        return self.finish(status,'no_local_change' if status=='NO_PHYSICAL_HIT' else 'insufficient_or_unresolved_change',now)

    def payload(self,status,reason,now=None):
        now=time.time() if now is None else float(now)
        hypotheses=[dict(c,rank=i+1) for i,c in enumerate(self.hypotheses)]
        return dict(mode='OBJECT',shot_id=self.shot_id,status=status,reason=reason,final=self.final,
            hypotheses=hypotheses,scope='frozen_object_regions_plus_technical_margin',
            per_object=deepcopy(self.per_object),candidate_ledger=deepcopy(self.candidate_ledger),
            global_coordinate_authority=False,truth_or_learning_input=False,
            frozen_object_count=len(self.regions),top_k=self.top_k,margin_camera_px=self.margin,
            searched_pixels=self.search_pixels,processed_crop_pixels=self.read_pixels,
            full_camera_pixels=self.frame_pixels,search_fraction=self.search_pixels/max(1,self.frame_pixels),
            candidate_count=self.candidate_count,post_frames=self.frames,pre_frame_ts=self.pre_times,
            examined_candidate_count=self.examined_candidate_count,
            pre_activity_context=self.pre_activity,
            peak_ts=self.peak_ts,first_post_ts=self.first_post_ts,decision_ts=now,
            timing=dict(prepare_ms=getattr(self,'prepare_ms',0.),processing_ms=list(self.processing_ms),
                roi_prepare_ms=self.roi_prepare_ms,pre_prepare_ms=self.pre_prepare_ms,
                post_analysis_ms=list(self.post_analysis_ms),merge_dedupe_ms=list(self.merge_dedupe_ms),
                proposal_ms=list(self.proposal_ms),local_confirmation_ms=list(self.local_confirmation_ms),
                freshness_ms=list(self.freshness_ms),
                audio_to_first_frame_ms=None if self.first_post_ts is None else (self.first_post_ts-self.peak_ts)*1000,
                audio_to_first_hypothesis_ms=None if self.first_hypothesis_ts is None else (self.first_hypothesis_ts-self.peak_ts)*1000,
                audio_to_decision_ms=(now-self.peak_ts)*1000,
                first_frame_to_decision_ms=None if self.first_post_ts is None else (now-self.first_post_ts)*1000))

    def finish(self,status,reason,now=None):
        self.final=True
        self.result=self.payload(status,reason,now)
        if self.object_hit_requested:
            started=time.perf_counter()
            self.result['result_semantics']='OBJECT_HIT'
            self.result['object_decision']=self._object_decision(status,reason)
            self.result['timing']['object_decision_ms']=(time.perf_counter()-started)*1000.
        self.tiles.clear()  # immediately release PRE/POST arrays; only bounded diagnostics survive
        if self.capture_result:
            self.capture_result(self.shot_id,self.result)
        return self.result

    def _object_decision(self,status,reason):
        """Resolve supported local observations before public top-k truncation.

        More than one supported requested object is
        ambiguous, including overlapping shapes. Retain diagnostics, never pick
        an object by depth, priority, distance or arbitrary list order.
        """
        supported={}
        if status=='PHYSICAL_HIT':
            for row in self.per_object:
                oid=row['object_id']
                points=[c for c in row['candidates'] if c['confirmed'] and any(
                    a['object_id']==oid and a['inside_exact_shape'] is True for a in c['regions'])]
                if points: supported[oid]=min(points,key=self._physical_order)
        oid=next(iter(supported)) if len(supported)==1 else None
        state='OBJECT_HIT' if oid is not None else ('NO_OBJECT_HIT' if status=='NO_PHYSICAL_HIT' else 'UNKNOWN')
        chosen=supported.get(oid)
        if chosen is not None:
            chosen=next((h for h in self.result['hypotheses'] if h['camera_xy']==chosen['camera_xy']),chosen)
        return dict(state=state,object_id=oid,hypothesis=deepcopy(chosen),
            supported_object_ids=sorted(supported),
            reason=('unique_supported_object' if oid is not None else
                'multiple_supported_objects' if len(supported)>1 else
                'no_confirmed_point_in_requested_shape' if status=='PHYSICAL_HIT' else reason),
            scope='requested_frozen_objects_only',global_rescue_used=False,
            no_physical_impact_established=False)

    def compact_record(self, *, gameplay=None, current=None):
        """Bounded per-object diagnostics; no pixel arrays or frame ledgers."""
        result=self.result or {}
        def point(c):
            if c is None:return None
            return {k:c.get(k) for k in ('camera_xy','game_xy','confirmed','evidence_score',
                'evidence_decision','evidence','previous_same_xy_evidence','frame_ts',
                'previous_frame_ts','pre_frame_ts','registration','producer','component_id','tile_camera_bbox',
                'persistent_local_change','event_freshness','regions','component_pixels','nomination_threshold')}
        decision=result.get('object_decision',{})
        return dict(shot_id=self.shot_id,semantic='OBJECT_HIT',status=decision.get('state','UNKNOWN'),
            support_policy='same_xy_signed_pre_envelope_v1',peak_ts=self.peak_ts,
            pre_frame_ts=list(self.pre_times),post_frames=self.frames,
            timestamp_semantics='camera_receipt_not_exposure; PRE_is_not_certified_pre_damage',
            physical_query_status=result.get('status'),
            object_id=decision.get('object_id'),reason=decision.get('reason'),
            supported_object_ids=decision.get('supported_object_ids',[]),
            diagnostic_xy=point(decision.get('hypothesis')),
            frozen_regions=[dict(object_id=r.object_id,camera_aabb=[r.x,r.y,r.width,r.height]) for r in self.regions],
            per_object=[dict(object_id=r.object_id,game_aabb=[r.x,r.y,r.width,r.height],
                shape=r.metadata.get('v250_object_snapshot',{}).get('shape'),
                support_state=('SUPPORTED' if r.object_id in decision.get('supported_object_ids',[]) else 'UNKNOWN'),
                decision=next((row['decision'] for row in self.per_object if row['object_id']==r.object_id),'no_observed_candidate'),
                candidates=[point(c) for row in self.per_object if row['object_id']==r.object_id for c in row['candidates']],
                best_candidate=point(next((row['candidates'][0] for row in self.per_object
                    if row['object_id']==r.object_id and row['candidates']),None))) for r in self.game_regions],
            timing=result.get('timing',{}),searched_pixels=self.search_pixels,
            global_rescue_used_for_object=False,exact_coordinate_result=dict(current or {'state':'pending'}),
            gameplay=dict(gameplay or {}),physical_accuracy_claim=False)
