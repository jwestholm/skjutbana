"""Research-only source-independent impact verifier primitives."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from copy import deepcopy
import math
import time
import cv2,numpy as np
from .registered_impact import sample

@dataclass
class CommonFrameContext:
    pre: np.ndarray
    post: tuple[np.ndarray,...]
    cutoff_ts: float
    timestamps: tuple[float,...]
    origin: tuple[int,int]=(0,0)


@dataclass(frozen=True)
class PhysicalTemporalFrame:
    """An observed camera state. timestamp is receipt time, not exposure time."""
    gray: np.ndarray
    timestamp: float
    frame_id: str
    camera_sequence: int | None
    provenance: dict


class CommonTemporalHistory:
    """Bounded OFFLINE/SHADOW companion to CommonFrameContext.

    Accept the camera owner's already computed uint8 grayscale once per frame.
    This prototype owns one immutable copy; a future shared owner can supply the
    same lifetime guarantee without duplicate rings. No live hook is installed.
    It deliberately stores no audio-sided PRE, candidate, object or change map.
    """
    def __init__(self, *, max_frames=48, max_bytes=512*1024*1024):
        if type(max_frames) is not int or max_frames < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive integer history limits required')
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self._frames = deque()
        self._bytes = 0
        self._stream = None
        self._shape = None

    @property
    def bytes_used(self):
        return self._bytes

    def __len__(self):
        return len(self._frames)

    def append(self, gray, *, timestamp, frame_id, stream_id, camera_sequence=None, provenance=None):
        gray = np.asarray(gray)
        if gray.ndim != 2 or gray.dtype != np.uint8 or gray.nbytes > self.max_bytes:
            raise ValueError('A bounded uint8 grayscale camera frame required')
        if not np.isfinite(timestamp) or not frame_id or not stream_id:
            raise ValueError('Finite receipt timestamp and explicit frame/stream identity required')
        if camera_sequence is not None and (type(camera_sequence) is not int or camera_sequence < 0):
            raise ValueError('Camera sequence must be an integer or unknown')
        if self._frames:
            last = self._frames[-1]
            if self._stream != stream_id or self._shape != gray.shape:
                raise ValueError('New stream or shape requires a new history')
            if timestamp <= last.timestamp or any(f.frame_id == frame_id for f in self._frames):
                raise ValueError('Strict receipt ordering and unique frame identity required')
            if (camera_sequence is None) != (last.camera_sequence is None):
                raise ValueError('Do not mix known and unknown camera sequences')
            if camera_sequence is not None and camera_sequence <= last.camera_sequence:
                raise ValueError('Camera sequences must increase in one stream')
        owned = gray.copy()
        owned.flags.writeable = False
        entry = PhysicalTemporalFrame(owned, float(timestamp), str(frame_id), camera_sequence,
                                      deepcopy(provenance or {}))
        while self._frames and (len(self._frames) >= self.max_frames or self._bytes+owned.nbytes > self.max_bytes):
            self._bytes -= self._frames.popleft().gray.nbytes
        self._frames.append(entry); self._bytes += owned.nbytes
        self._stream, self._shape = stream_id, gray.shape

    def select(self, frame_ids, *, cutoff):
        """Exact required history, never silently shortened after eviction.

        Missing data raises LookupError; callers must report unavailable/UNCERTAIN.
        Future frames are rejected, even when already present in the ring.
        Returned images are read-only views; metadata is detached from the owner.
        """
        if not np.isfinite(cutoff) or not frame_ids or len(set(frame_ids)) != len(frame_ids):
            raise ValueError('Finite cutoff and unique required frame IDs needed')
        by_id = {f.frame_id:f for f in self._frames}
        if any(k not in by_id for k in frame_ids):
            raise LookupError('Required camera history unavailable or evicted')
        frames = [by_id[k] for k in frame_ids]
        if any(f.timestamp > cutoff for f in frames) or any(a.timestamp >= b.timestamp for a,b in zip(frames,frames[1:])):
            raise ValueError('Only ordered observations at/before cutoff allowed')
        return tuple(PhysicalTemporalFrame(f.gray.view(), f.timestamp, f.frame_id, f.camera_sequence,
                                         deepcopy(f.provenance)) for f in frames)

    def causal_query(self, frame_ids, points, *, cutoff, snapshot_frame_id=None, early_times=()):
        """Coordinate-local replay query; all comparison diagnostics retained.

        Returns evidence and separately measurable stages. One patch stack is
        temporary per event, never retained in the continuous history. Candidate
        coordinates enter only here, after global frame maintenance.
        """
        from .clean_physical_change import causal_local_evidence
        start = time.perf_counter()
        frames = self.select(frame_ids, cutoff=cutoff)
        times = [f.timestamp for f in frames]
        snap = self.select([snapshot_frame_id], cutoff=cutoff)[0] if snapshot_frame_id else None
        setup = time.perf_counter()
        pixels = np.stack([camera_local_patches(f.gray, points) for f in frames])
        snapshot = None
        if snap is not None:
            sample = pixels[times.index(snap.timestamp)] if snap.frame_id in frame_ids else camera_local_patches(snap.gray, points)
            snapshot = ({'timestamp':snap.timestamp}, sample)
        sampled = time.perf_counter()
        rows = causal_local_evidence(times, pixels, snapshot, cutoff, early_times)
        end = time.perf_counter()
        return rows, dict(setup_ms=(setup-start)*1000, patch_ms=(sampled-setup)*1000,
            inference_ms=(end-sampled)*1000, total_ms=(end-start)*1000,
            frames=[dict(timestamp=f.timestamp, frame_id=f.frame_id, camera_sequence=f.camera_sequence,
                         provenance=f.provenance) for f in frames],
            temporary_patch_bytes=int(pixels.nbytes))

    def witness_query(self, frame_ids, *, cutoff, camera_xy, event_id,
                      candidate_id, coordinate_version, source_provenance, previous=None):
        """Clocked replay adapter; reuse this history, never create a second ring."""
        from .clean_physical_change import local_contrast
        frames=self.select(frame_ids,cutoff=cutoff)
        values=[float(local_contrast(camera_local_patches(f.gray,[camera_xy]))[0]) for f in frames]
        metadata=[dict(frame_id=f.frame_id,ordinal=i,receipt_timestamp=f.timestamp,
            camera_sequence=f.camera_sequence,provenance=f.provenance) for i,f in enumerate(frames)]
        return query_physical_witness(values,metadata,event_id=event_id,candidate_id=candidate_id,
            coordinate_version=coordinate_version,camera_xy=camera_xy,source_provenance=source_provenance,
            cutoff_frame_id=frame_ids[-1],cutoff_ts=cutoff,previous=previous)

    def witness_queries(self, frame_ids, *, cutoff, event_id, candidates, profile=None):
        """Batch adapter over this existing owner, without another image ring."""
        if len(candidates) > 32 or len(frame_ids) > 50:
            raise ValueError('At most 32 coordinates and 50 observations required')
        started = time.perf_counter()
        frames = self.select(frame_ids, cutoff=cutoff)
        points = [c['camera_xy'] for c in candidates]
        values = np.stack([camera_local_contrasts(f.gray, points) for f in frames])
        metadata = [dict(frame_id=f.frame_id, ordinal=i, receipt_timestamp=f.timestamp,
            camera_sequence=f.camera_sequence, provenance=f.provenance) for i,f in enumerate(frames)]
        extracted = time.perf_counter()
        output = query_physical_witnesses(values, metadata, event_id=event_id, candidates=candidates,
            cutoff_frame_id=frame_ids[-1], cutoff_ts=cutoff, profile=profile)
        if profile is not None:
            profile['history_select_and_extraction_ms'] = (extracted-started)*1000
        return output


def camera_local_contrasts(gray, points):
    """Same sampler/reduction order, sharing masks across exact coordinates."""
    from .clean_physical_change import local_contrast_batch
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    return local_contrast_batch(camera_local_patches(gray, points)[None])[0]


def camera_local_patches(gray, points):
    """Reuse accuracy_verifier's subpixel sampler at full-camera XY.

    The 52-pixel source crop and 49-pixel result are identical to the frozen
    causal research sampler. No full-frame float conversion per coordinate.
    """
    from .accuracy_verifier import patch
    values = []
    for x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError('Finite camera coordinates required')
        ix, iy = int(math.floor(x)), int(math.floor(y))
        xa, ya = ix-25, iy-25
        if xa < 0 or ya < 0 or ix+27 > gray.shape[1] or iy+27 > gray.shape[0]:
            raise ValueError('Full patch unavailable')
        values.append(patch(gray[ya:iy+27, xa:ix+27], (x, y), (xa, ya)))
    if not values:
        raise ValueError('At least one candidate required')
    return np.stack(values)

def extract_patch_context(ctx: CommonFrameContext, xy, *, causal_only=True, radius=16):
    pairs=[]; pre=sample(ctx.pre,xy,ctx.origin,radius)
    for img,ts in zip(ctx.post,ctx.timestamps):
        if causal_only and ts>ctx.cutoff_ts: continue
        post=sample(img,xy,ctx.origin,radius); signed=pre.astype(np.float32)-post.astype(np.float32)
        pairs.append(dict(timestamp=float(ts),pre=pre,post=post,signed=signed,absolute=np.abs(signed)))
    return dict(pre=pre,pairs=tuple(pairs),causal_only=bool(causal_only))

def patch_representation(bundle, kind='difference'):
    if not bundle['pairs']: return None
    stack=np.stack([p['signed'] for p in bundle['pairs']])
    if kind=='difference': return stack[0].astype(np.float32).ravel()
    if kind=='median_difference': return np.median(stack,axis=0).astype(np.float32).ravel()
    if kind=='gradient':
        gy,gx=np.gradient(stack[0]); return np.hypot(gx,gy).astype(np.float32).ravel()
    if kind=='pca':
        v=stack[0].astype(np.float32); small=cv2.resize(v,(8,8),interpolation=cv2.INTER_AREA); return small.ravel()
    if kind=='texture':
        v=stack[0].astype(np.float32); gy,gx=np.gradient(v); return np.asarray([v.mean(),v.std(),np.abs(v).mean(),np.hypot(gx,gy).mean(),np.percentile(v,10),np.percentile(v,90)],np.float32)
    raise ValueError(kind)

class CommonVerifier:
    """Interface only; disabled unless explicitly invoked by research code."""
    enabled=False
    def verify_event(self, tracks, context):
        if not self.enabled: return []
        return [self.score_track(t,context) for t in tracks]
    def score_track(self, track, context): raise NotImplementedError


def coherent_change_evidence(encoded, names, *, pre_context=False):
    """Fixed offline conjunction over existing accuracy_verifier measurements.

    Require change/localization/temporal consistency at the SAME scale. Scales
    are alternatives, not independent votes. The score is an evidence strength,
    not a probability. No fitting, producer metadata, XY or labels are consumed.
    PRE variability can soften strength; neither texture nor old holes veto it.
    Returns per-scale reasons so every candidate's ordering can be inspected.
    """
    x = np.asarray(encoded, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Unique feature schema and two-dimensional matrix required')
    keys = ('abs_late', 'center_ring_ratio', 'spatial_consistency', 'pre_std')
    columns = [names.index(f'r{r}_{key}') for r in (2, 4, 8) for key in keys]
    used = x[:, columns].reshape(len(x), 3, 4)
    if not np.isfinite(used).all() or np.any(used < 0) or np.any(used > 50):
        raise ValueError('Invalid nonnegative signed-log physical evidence')
    raw = np.expm1(used)
    amplitude, ratio, consistency, noise = (raw[:, :, j] for j in range(4))
    if np.any(consistency > 1 + 1e-6):
        raise ValueError('Temporal consistency must be in [0,1]')
    consistency = np.clip(consistency, 0, 1)
    strength = amplitude / (1 + amplitude + (noise if pre_context else 0))
    localization = ratio / (1 + ratio)
    joint = np.minimum(strength, localization) * consistency
    reasons = dict(radii=[2, 4, 8], strength=strength, localization=localization,
                   temporal_consistency=consistency, joint=joint,
                   winning_scale=np.argmax(joint, axis=1))
    return np.max(joint, axis=1), reasons


def coherent_event_decision(scores, coordinates, *, abstain=False):
    """Research event semantics; lack of evidence never proves NO_PHYSICAL_HIT.

    Optional fixed dominance check compares distinct locations (>16 camera px,
    twice the largest patch radius). It is NOT calibrated on physical negatives.
    XY only separates competing observations; it never supplies coordinate truth.
    Exact score ties use input order, which callers must log.
    """
    scores = np.asarray(scores, dtype=float)
    points = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    if scores.ndim != 1 or len(scores) != len(points):
        raise ValueError('Scores and coordinates must correspond')
    if not np.isfinite(scores).all() or not np.isfinite(points).all() or np.any(scores < 0):
        raise ValueError('Finite nonnegative evidence and camera XY required')
    if not len(scores) or np.max(scores) <= 0:
        return dict(state='UNCERTAIN', selected=None, reason='NO_SUPPORTED_COORDINATE')
    winner = int(np.argmax(scores))
    far = np.linalg.norm(points - points[winner], axis=1) > 16.
    runner = float(np.max(scores[far])) if np.any(far) else 0.
    if abstain and scores[winner] <= 2. * runner:
        return dict(state='UNCERTAIN', selected=None, reason='COMPETING_PHYSICAL_EVIDENCE',
                    proposed=winner, best_score=float(scores[winner]), competitor_score=runner)
    return dict(state='PHYSICAL_HIT', selected=winner, reason='CONDITIONAL_COORDINATE_SELECTION',
                best_score=float(scores[winner]), competitor_score=runner)


def _physical_witness_context(frames, cutoff_frame_id, cutoff_ts):
    ids=[f['frame_id'] for f in frames]
    if any(not i for i in ids) or len(set(ids))!=len(ids) or cutoff_frame_id!=ids[-1]:
        raise ValueError('Unique ordered frame identity prefix ending at cutoff required')
    ordinals=[f['ordinal'] for f in frames]
    if any(type(i) is not int for i in ordinals) or any(a>=b for a,b in zip(ordinals,ordinals[1:])):
        raise ValueError('Strict observation ordinals required; these are not camera sequences')
    times=[f.get('receipt_timestamp') for f in frames]
    if any(t is None for t in times) and not all(t is None for t in times):
        raise ValueError('Do not mix unknown and known receipt clocks')
    ordinal_only=times[0] is None
    if ordinal_only:
        if cutoff_ts is not None:raise ValueError('Unknown clocks cannot certify a receipt cutoff')
        times=None
    elif cutoff_ts is None or not np.isfinite(cutoff_ts) or times[-1]>cutoff_ts:
        raise ValueError('Recorded observations must precede explicit receipt cutoff')
    seq=[f.get('camera_sequence') for f in frames]
    if all(v is None for v in seq):seq=None
    elif any(v is None for v in seq) or ordinal_only:raise ValueError('Sequence provenance is inconsistent')
    return ids, ordinals, times, seq


def _physical_witness_identity(x, frames, context, event_id, candidate_id,
                               coordinate_version, camera_xy, source_provenance, previous):
    ids, ordinals, _times, _seq = context
    xy=tuple(float(v) for v in camera_xy)
    if len(xy)!=2 or not np.isfinite(xy).all() or not all((event_id,candidate_id,coordinate_version)):
        raise ValueError('Exact finite camera XY and explicit event/candidate/version required')
    if previous is not None:
        identity=(str(event_id),str(candidate_id),str(coordinate_version),list(xy))
        if identity!=(previous['event_id'],previous['candidate_id'],previous['coordinate_version'],previous['camera_xy']):
            raise ValueError('Cannot transfer evidence across coordinate/version/event identity')
        old_ids=[f['frame_id'] for f in previous['frames']]
        if ids[:len(old_ids)]!=old_ids:raise ValueError('Proof history changed/evicted; cannot inherit prior support')
        if frames[:len(old_ids)]!=previous['frames'] or x[:len(old_ids)].tolist()!=previous['signal']:
            raise ValueError('Existing frame provenance or observations changed; cannot inherit prior support')
    born=source_provenance.get('born_ordinal')
    if born is not None and born>ordinals[-1]:
        raise ValueError('Candidate observation is beyond the available prefix')
    return xy


def _physical_witness_bundle(x, frames, context, *, event_id, candidate_id,
                              coordinate_version, xy, source_provenance,
                              cutoff_frame_id, cutoff_ts, previous, raw):
    from .evidence import EvidenceBundle
    ids, _ordinals, times, _seq = context
    state='SUPPORTED' if raw['supported'] else 'UNKNOWN';reason=None
    revoked=bool(previous and previous['state']=='SUPPORTED' and not raw['supported'])
    if revoked:
        old_onset=previous['measurements']['onset_ordinal']
        reverted=any(c['index']==old_onset and not c['persists'] for c in raw['choices'])
        state='CONTRADICTED' if reverted else 'UNKNOWN'
        reason='PERSISTENCE_REVERTED' if reverted else raw['state']
    witness=dict(event_id=str(event_id),candidate_id=str(candidate_id),coordinate_version=str(coordinate_version),
        camera_xy=list(xy),family='TEMPORAL_NEWNESS',proposition='unique_persistent_coordinate_local_step',
        dependency_family='raw_same_xy_disc_annulus_contrast',state=state,physical_state='UNKNOWN',
        strength_units='minimum_same_polarity_change_over_local_noise_not_probability',
        measurements=deepcopy(raw),signal=x.tolist(),frames=deepcopy(frames),
        source_provenance=deepcopy(source_provenance),criterion='frozen_three_before_three_after_v1',
        availability=dict(cutoff_frame_id=cutoff_frame_id,receipt_cutoff=cutoff_ts,
            latest_receipt=None if times is None else times[-1],producer_delivery=None,acquisition=None,
            certification=raw['certification']),
        revocation=dict(revoked_now=revoked,reason=reason,previous_state=previous['state'] if previous else None,
            last_revocation_frame_id=ids[-1] if revoked else (previous or {}).get('revocation',{}).get('last_revocation_frame_id'),
            last_revocation_reason=reason if revoked else (previous or {}).get('revocation',{}).get('last_revocation_reason')))
    return EvidenceBundle(None,{},None,metadata=dict(scope='OFFLINE_SHADOW_NO_EMISSION_AUTHORITY',
        family_count_is_not_independent_votes=True),witnesses=(witness,))


def query_temporal_observation(witness):
    """Describe an existing bounded signal without certifying new damage.

    Same r4 core-minus-r8..12 ring channel for every producer. First/last three
    observations are fixed by the supplied prefix, NOT selected by onset or GT.
    Neither the first window nor its median is claimed to be clean pre-impact.
    This optional query leaves the exact witness support/revocation unchanged.
    """
    from .clean_physical_change import persistent_local_support
    from .evidence import EvidenceBundle
    x = np.asarray(witness['signal'], dtype=np.float64)
    frames = witness['frames']; availability = witness['availability']
    if x.ndim != 1 or len(x) != len(frames) or not 1 <= len(x) <= 50 or not np.isfinite(x).all():
        raise ValueError('One bounded finite observation per frame required')
    context = _physical_witness_context(frames, availability['cutoff_frame_id'], availability['receipt_cutoff'])
    ids, ordinals, times, sequences = context
    _physical_witness_identity(x, frames, context, witness['event_id'], witness['candidate_id'],
        witness['coordinate_version'], witness['camera_xy'], witness['source_provenance'], None)
    if times is not None and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Strict receipt ordering required')
    if sequences is not None and (not np.isfinite(sequences).all() or np.any(np.diff(sequences) <= 0)):
        raise ValueError('Strict camera sequence ordering required')
    result = dict(available=len(x) >= 6, reason='OBSERVED_WINDOW_ONLY' if len(x) >= 6 else 'INSUFFICIENT_OBSERVATIONS',
        signed_change=None, coherent_strength=None, noise=None,
        reference_level=None, last_level=None,
        first_frame_ids=ids[:3], last_frame_ids=ids[-3:], observation_count=len(x),
        receipt_span=None if times is None else float(times[-1]-times[0]),
        max_receipt_gap=None if times is None or len(times)<2 else float(np.max(np.diff(times))),
        certification=availability['certification'], cutoff_frame_id=ids[-1],
        receipt_cutoff=availability['receipt_cutoff'])
    if result['available']:
        before, after = x[:3], x[-3:]
        reference, last = float(np.median(before)), float(np.median(after))
        noise = max(.75, 1.4826*float(np.median(np.abs(before-reference))),
                    1.4826*float(np.median(np.abs(after-last))))
        measured = persistent_local_support(reference, after, noise)
        result.update(signed_change=measured['signed_change'], coherent_strength=measured['support'],
                      noise=noise, reference_level=reference, last_level=last)
    return EvidenceBundle(None, {}, None, metadata=dict(observed_window=result,
        family='LOCAL_PHYSICAL_CHANGE', dependency_family='raw_same_xy_disc_annulus_contrast',
        source_independent=True, physical_state='UNKNOWN',
        units='signed grayscale contrast; weakest same-polarity last3 change / first3-and-last3 MAD floor0.75',
        clean_pre_certified=False, onset_certified=False, scope='OFFLINE_SHADOW_NO_EMISSION_AUTHORITY'))


def query_physical_witness(signal, frames, *, event_id, candidate_id,
                           coordinate_version, camera_xy, source_provenance,
                           cutoff_frame_id, cutoff_ts=None, previous=None):
    """Scalar compatibility path; identical validation and witness lifecycle."""
    from .clean_physical_change import temporal_step_evidence
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or not 1 <= len(x) <= 50 or len(frames) != len(x) or not np.isfinite(x).all():
        raise ValueError('One to fifty finite observations and matching frame identities required')
    context = _physical_witness_context(frames, cutoff_frame_id, cutoff_ts)
    xy = _physical_witness_identity(x, frames, context, event_id, candidate_id,
        coordinate_version, camera_xy, source_provenance, previous)
    raw = temporal_step_evidence(x, context[2], context[3])
    return _physical_witness_bundle(x, frames, context, event_id=event_id,
        candidate_id=candidate_id, coordinate_version=coordinate_version, xy=xy,
        source_provenance=source_provenance, cutoff_frame_id=cutoff_frame_id,
        cutoff_ts=cutoff_ts, previous=previous, raw=raw)


def query_physical_witnesses(signals, frames, *, event_id, candidates,
                            cutoff_frame_id, cutoff_ts=None, profile=None):
    """Bounded shared-prefix query; one detached EvidenceBundle per exact point.

    Signals are T x candidates in caller order. Metadata validation and temporal
    arithmetic are shared, never coordinate identity, revocation or authority.
    Caller owns the existing history; this function retains no cache or images.
    """
    from .clean_physical_change import temporal_step_evidence_batch
    started = time.perf_counter()
    x = np.asarray(signals, dtype=np.float64)
    if (x.ndim != 2 or not 1 <= len(x) <= 50 or len(frames) != len(x)
            or x.shape[1] != len(candidates) or len(candidates) > 32 or not np.isfinite(x).all()):
        raise ValueError('One to fifty frames and at most 32 matching finite candidate signals required')
    context = _physical_witness_context(frames, cutoff_frame_id, cutoff_ts)
    keys = [(str(c['candidate_id']), str(c['coordinate_version'])) for c in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate candidate/version identity in batch')
    context_done = time.perf_counter()
    points = [_physical_witness_identity(x[:, j], frames, context, event_id,
        c['candidate_id'], c['coordinate_version'], c['camera_xy'],
        c['source_provenance'], c.get('previous')) for j, c in enumerate(candidates)]
    validated = time.perf_counter()
    evidence = temporal_step_evidence_batch(x, context[2], context[3])
    measured = time.perf_counter()
    output = tuple(_physical_witness_bundle(x[:, j], frames, context, event_id=event_id,
        candidate_id=c['candidate_id'], coordinate_version=c['coordinate_version'], xy=points[j],
        source_provenance=c['source_provenance'], cutoff_frame_id=cutoff_frame_id,
        cutoff_ts=cutoff_ts, previous=c.get('previous'), raw=evidence[j]) for j,c in enumerate(candidates))
    ended = time.perf_counter()
    if profile is not None:
        profile.update(context_ms=(context_done-started)*1000,
            temporal_arithmetic_ms=(measured-validated)*1000,
            witness_update_ms=((validated-context_done)+(ended-measured))*1000,
            total_ms=(ended-started)*1000, candidates=len(candidates), frames=len(frames))
    return output
