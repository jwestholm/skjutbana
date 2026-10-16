"""Capture-launcher presentation only. runtime.log remains the full forensic sink.

Runs in the parent capture process, never in camera/detector/game threads. Small
JSONL updates reference raw byte ranges rather than duplicating candidate payloads.
Unkeyed legacy text stays in runtime.log; we never assign it to the latest shot.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time


def _field(text, name):
    match = re.search(r'"' + re.escape(name) + r'"\s*:\s*', text)
    if match:
        try:
            return json.JSONDecoder().raw_decode(text, match.end())[0]
        except ValueError:
            pass  # Interleaved legacy stdout is preserved, not repaired/invented.
    return None


def _short(value, limit=48):
    return re.sub(r'\s+', ' ', str(value)).strip()[:limit]


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


class CaptureConsole:
    """One small view/index on the existing launcher-owned runtime log."""

    def __init__(self, root, events, *, verbose=False, object_crops=False, mature=False):
        self.root, self.events = Path(root), events
        self.verbose, self.object_crops, self.mature = verbose, object_crops, mature
        self.shots = {}
        self.warnings = 0
        self.records = 0

    def _append(self, source, shot_id, raw, **values):
        row = dict(source=source, shot_id=shot_id, observed_at=time.time(),
                   timestamp_semantics='parent_log_receipt; not detector/evidence time',
                   raw=raw, **values)
        self.events.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        self.events.flush()  # Parent only; each completed update survives Ctrl+C.
        self.records += 1

    def warning(self, text, *, shot_id=None, raw=None):
        self.warnings += 1
        message = _short(text, 180)
        print(f'[CAPTURE WARNING] {message} (details: runtime.log)', flush=True)
        self._append('CAPTURE_WARNING', shot_id, raw, message=message)

    def _announce(self, sid):
        s = self.shots[sid]
        if s.get('announced'):
            return
        s['announced'] = True
        ms = s.get('decision_ms')
        timing = '?' if ms is None else f'{ms:.0f}'
        line = (f"[SHOT {sid}] OBJECT_HIT={s.get('object_state', 'UNAVAILABLE')}"
                f" supports={s.get('supports', '?')}"
                f" mature={s.get('mature_proposals', '?')}/{s.get('mature_confirmed', '?')}"
                f" exact={s.get('exact_state', 'pending')} decision={timing}ms")
        if s.get('object_id'):
            line += ' object=' + _short(s['object_id'], 24)
        if not self.verbose:
            print(line[:160], flush=True)

    def observe(self, text, offset, length):
        """Called after these exact bytes have been persisted to runtime.log."""
        raw = dict(path='runtime.log', byte_offset=offset, byte_length=length)
        if self.verbose:
            print(text, end='', flush=True)
        if text.startswith('[GT-CAPTURE]'):
            # Research pause/READY instructions must survive compact capture.
            # Full text remains in runtime.log; no image/candidate payload here.
            if not self.verbose:
                print(text.strip()[:160], flush=True)
            match=re.search(r'\bshot=(\d+)\b',text)
            self._append('GT-CAPTURE', int(match[1]) if match else None, raw)
            return
        source, sid = None, None
        fragment = text
        for tag in ('OBJECT-HIT-CURRENT', 'OBJECT-HIT'):
            marker = '[' + tag + '] '
            if marker in text:
                source, fragment = tag, text.split(marker, 1)[1]
                sid = _field(fragment, 'shot_id')
                break
        if source is None:
            for tag in ('OBJECT-MATURE-CAPTURE', 'V2.22.3 SHOT', 'V2.22.4 CV'):
                if '[' + tag + ']' in text:
                    source = tag
                    m = re.search(r'\bshot=(\d+)\b', text)
                    sid = int(m[1]) if m else None
                    break
        if isinstance(sid, int) and not isinstance(sid, bool) and sid > 0:
            s = self.shots.setdefault(sid, {})
            values = {}
            if source in ('OBJECT-HIT', 'OBJECT-HIT-CURRENT'):
                state = _field(fragment, 'status')
                supported = _field(fragment, 'supported_object_ids')
                exact = _field(fragment, 'exact_coordinate_result')
                timing = _field(fragment, 'timing')
                s['peak_ts'] = _number(_field(fragment, 'peak_ts'))
                s['object_state'] = _short(state, 24) if state else 'UNAVAILABLE'
                s['object_id'] = _field(fragment, 'object_id')
                s['supports'] = len(supported) if isinstance(supported, list) else '?'
                s['decision_ms'] = _number(timing.get('audio_to_decision_ms')) if isinstance(timing, dict) else None
                if isinstance(exact, dict):
                    s['exact_state'] = _short(exact.get('state', 'unavailable'), 16)
                values.update(peak_ts=s['peak_ts'], object_state=s['object_state'],
                    object_id=_short(s['object_id']) if s['object_id'] else None,
                    support_count=s['supports'], decision_ms=s['decision_ms'],
                    exact_state=s.get('exact_state'), exact_timestamp=_number(exact.get('timestamp')) if isinstance(exact, dict) else None,
                    reason=_short(_field(fragment, 'reason'), 96),
                    parse_status='EXTRACTED_FIELDS' if state and isinstance(supported, list) else 'PARTIAL',
                    object_trace=(f'object_crops/shots/shot_{sid:08d}/trace.json' if self.object_crops else None))
                if source == 'OBJECT-HIT':
                    s['object_complete'] = True
                    self._announce(sid)
                if values['parse_status'] == 'PARTIAL' or (source=='OBJECT-HIT-CURRENT' and not isinstance(exact, dict)):
                    self.warning(f'shot={sid} {source} partially parsed/interleaved; raw bytes retained', shot_id=sid, raw=raw)
            elif source == 'OBJECT-MATURE-CAPTURE':
                stage = re.search(r'\bstage=(\w+)', text)
                count = re.search(r'\bcandidates=(\d+)', text)
                cost = re.search(r'\bcopy_enqueue_ms=([\d.]+)', text)
                if stage and count:
                    key = 'mature_proposals' if stage[1]=='WORKER_DELIVERY' else 'mature_confirmed'
                    s[key] = int(count[1])
                values.update(stage=stage[1] if stage else None, candidate_count=int(count[1]) if count else None,
                    copy_enqueue_ms=_number(cost[1]) if cost else None,
                    candidate_artifacts=f'object_crops/shots/shot_{sid:08d}/mature_candidates/',
                    peak_ts=s.get('peak_ts'))
            elif source == 'V2.22.4 CV':
                values['timings'] = {k:float(v) for k,v in re.findall(r'\b(age|queue|worker|v2|other)=(-?[\d.]+)', text)}
                values['timing_semantics'] = 'native CV log units ms; age is frame age, not delivery latency'
                values['peak_ts'] = s.get('peak_ts')
            self._append(source, sid, raw, **values)
        elif source in ('OBJECT-HIT', 'OBJECT-HIT-CURRENT'):
            self.warning(source+' lacks a valid physical shot_id; not associated by guess', raw=raw)

        # Explicit failures anywhere in a line, including interleaved stdout.
        # Do not confuse diagnostic JSON error/missing-value fields with warnings.
        # Literal prefilter uses C string scans. Avoid a general warning regex
        # walking every character of otherwise uninteresting giant candidate JSON.
        lower = text.lower()
        problem = None
        if any(token in lower for token in ('warn', 'error', 'fail', 'refused',
                'traceback', 'exception:', 'unavailable:', 'limit reached',
                'invalid dimensions', 'missing artifact', 'permission denied',
                'no space left', 'oväntat fel:', 'fatal:', 'dropped export', 'invalid frame size')):
            problem = re.search(r'(?i)(?<![\w"\'])(?:warning|warn|error|failed|failure|refused)(?=[: \]])'
                            r'|Traceback \(most recent call last\)|\b\w*(?:Error|Exception):'
                            r'|(?i:unavailable:|limit reached|invalid dimensions|missing artifact|'
                            r'permission denied|no space left|Oväntat fel:|fatal:|dropped export|invalid frame size)', text)
        losses = None
        if any(token in text for token in ('writer_errors', 'queue_drops', 'metadata_failures')):
            losses = re.search(r"(?:writer_errors|queue_drops|metadata_failures)['\"]?\s*[:=]\s*[1-9]\d*", text)
        if problem or losses:
            at = (problem or losses).start()
            self.warning(text[max(0, at-50):at+180], shot_id=sid, raw=raw)

    def finish(self):
        """Final observational integrity check, after child/trace writers exited."""
        for sid, state in self.shots.items():
            if not state.get('announced'):
                self._announce(sid)
                self.warning(f'shot={sid} no completed OBJECT_HIT log before exit', shot_id=sid)
            if not self.object_crops:
                continue
            shot = self.root/'object_crops/shots'/f'shot_{sid:08d}'
            try:
                trace = json.loads((shot/'trace.json').read_text())
                completeness = trace.get('completeness', {})
                if completeness.get('trace_complete') is not True:
                    self.warning(f'shot={sid} trace incomplete: writer_errors={completeness.get("writer_errors")} queue_drops={completeness.get("queue_drops")}', shot_id=sid)
                if self.mature:
                    if not (shot/'mature_candidates/pool_0000.json').is_file():
                        self.warning(f'shot={sid} mature candidate artifacts unavailable', shot_id=sid)
                    if (shot/'mature_candidates/LIMIT.json').is_file():
                        self.warning(f'shot={sid} mature candidate export limit reached', shot_id=sid)
                    # Existing recorder bounds: 32 pools/event. Inspect only named
                    # sidecars after shutdown; no payload duplication or shot work.
                    for i in range(32):
                        path = shot/'mature_candidates'/f'pool_{i:04d}.json'
                        if not path.is_file():
                            continue
                        try:
                            packet = json.loads(path.read_text())
                            if packet.get('complete') is not True:
                                self.warning(f'shot={sid} mature_candidates/{path.name} incomplete: '
                                    +str(packet.get('unavailable_reason', 'candidate export truncated')), shot_id=sid)
                        except (OSError, ValueError) as exc:
                            self.warning(f'shot={sid} mature_candidates/{path.name} unreadable: {exc}', shot_id=sid)
            except (OSError, ValueError) as exc:
                self.warning(f'shot={sid} missing/unreadable trace.json: {exc}', shot_id=sid)
        return dict(events=len(self.shots), object_decisions=sum(bool(s.get('object_complete')) for s in self.shots.values()),
                    warnings=self.warnings, summary_records=self.records)
