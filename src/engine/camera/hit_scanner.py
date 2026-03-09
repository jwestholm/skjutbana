from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_camera_calibration,
    load_scanport_rect,
    load_viewport_rect,
)


@dataclass
class HoleTrack:
    x: float
    y: float
    camera_x: float
    camera_y: float
    hits: int = 1
    last_seen: float = 0


class HitScanner:

    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self):

        self.enabled = False
        self.state = self.STATE_OFF

        self.arm_duration = 1.0
        self.arm_until = 0

        self.history = deque(maxlen=12)

        self.tracks: list[HoleTrack] = []
        self.known_holes = []

        self.last_candidates = []
        self.last_stable = []

    def enable(self):

        self.enabled = True
        self.state = self.STATE_ARMING
        self.arm_until = time.time() + self.arm_duration

        self.history.clear()
        self.tracks.clear()
        self.known_holes.clear()

    def disable(self):

        self.enabled = False
        self.state = self.STATE_OFF

    def update(self, dt):

        if not self.enabled:
            return

        frame = camera_manager.get_latest_frame()

        if frame is None:
            return

        board = self._prepare_board(frame)

        if board is None:
            return

        self.history.append(board)

        if len(self.history) < 5:
            return

        ref = self._reference()

        candidates = self._detect(board, ref)

        self.last_candidates = candidates

        now = time.time()

        self._update_tracks(candidates, now)

        stable = [t for t in self.tracks if t.hits >= 3]

        self.last_stable = stable

        if self.state == self.STATE_ARMING:

            if now >= self.arm_until:
                self.state = self.STATE_ACTIVE
                self.tracks.clear()

            return

        if self.state != self.STATE_ACTIVE:
            return

        for t in stable:

            if self._duplicate(t.x,t.y):
                continue

            hit_input.push_camera_hit(t.camera_x,t.camera_y)

            self.known_holes.append((t.x,t.y))

    def get_debug_snapshot(self):

        return {
            "state": self.state,
            "enabled": self.enabled,
            "candidates_count": len(self.last_candidates),
            "stable_tracks_count": len(self.last_stable),
            "known_holes_count": len(self.known_holes)
        }

    def _prepare_board(self, frame):

        calibration = load_camera_calibration()

        if not calibration:
            return None

        H = np.array(calibration["homography"],dtype=np.float32)

        scanport = load_scanport_rect()
        viewport = load_viewport_rect()

        crop = frame[
            scanport.y:scanport.y+scanport.h,
            scanport.x:scanport.x+scanport.w
        ]

        warp = cv2.warpPerspective(
            crop,
            H,
            (viewport.w,viewport.h)
        )

        gray = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        return gray

    def _reference(self):

        stack = np.stack(list(self.history)[:-2])

        return np.median(stack,axis=0).astype(np.uint8)

    def _detect(self, gray, ref):

        change = cv2.subtract(ref,gray)

        blur = cv2.GaussianBlur(gray,(0,0),3)

        local = cv2.GaussianBlur(blur,(0,0),9)

        dark = cv2.subtract(local,blur)

        score = cv2.addWeighted(change,0.6,dark,0.4,0)

        _,mask = cv2.threshold(score,18,255,cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3)))

        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        candidates=[]

        for c in contours:

            area=cv2.contourArea(c)

            if area<10 or area>300:
                continue

            (x,y),r=cv2.minEnclosingCircle(c)

            candidates.append((x,y))

        return candidates

    def _update_tracks(self,candidates,now):

        for x,y in candidates:

            found=None

            for t in self.tracks:

                if np.hypot(x-t.x,y-t.y)<18:
                    found=t
                    break

            if found:

                found.hits+=1
                found.last_seen=now

            else:

                self.tracks.append(
                    HoleTrack(x,y,x,y,1,now)
                )

        self.tracks=[
            t for t in self.tracks
            if now-t.last_seen<0.5
        ]

    def _duplicate(self,x,y):

        for hx,hy in self.known_holes:

            if np.hypot(x-hx,y-hy)<25:
                return True

        return False


hit_scanner = HitScanner()