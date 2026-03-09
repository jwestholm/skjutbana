from __future__ import annotations

import time
from collections import deque
import numpy as np
import cv2

from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_scanport_rect,
    load_camera_calibration
)


class HitScanner:

    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self):

        self.enabled = False
        self.state = self.STATE_OFF

        self.history = deque(maxlen=10)

        self.arm_time = 1.5
        self.arm_until = 0

        self.tracks = []
        self.known = []

        self.last_candidates = []
        self.last_stable = []

    def enable(self):

        self.enabled = True
        self.state = self.STATE_ARMING
        self.arm_until = time.time() + self.arm_time

        self.history.clear()
        self.tracks.clear()
        self.known.clear()

    def disable(self):

        self.enabled = False
        self.state = self.STATE_OFF

    def update(self, dt):

        if not self.enabled:
            return

        frame = camera_manager.get_latest_frame()

        if frame is None:
            return

        scanport = load_scanport_rect()

        crop = frame[
            scanport.y:scanport.y + scanport.h,
            scanport.x:scanport.x + scanport.w
        ]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        self.history.append(gray)

        if len(self.history) < 5:
            return

        reference = np.median(
            np.stack(list(self.history)[:-2]),
            axis=0
        ).astype(np.uint8)

        change = cv2.subtract(reference, gray)

        blur = cv2.GaussianBlur(gray, (0,0), 2)

        local = cv2.GaussianBlur(blur, (0,0), 9)

        dark = cv2.subtract(local, blur)

        score = cv2.addWeighted(change, 0.7, dark, 0.3, 0)

        _, mask = cv2.threshold(score, 20, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3)))

        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for c in contours:

            area = cv2.contourArea(c)

            if area < 6 or area > 80:
                continue

            (x,y),r = cv2.minEnclosingCircle(c)

            candidates.append((x,y))

        self.last_candidates = candidates

        now = time.time()

        new_tracks = []

        for x,y in candidates:

            found = None

            for t in self.tracks:

                if np.hypot(x - t[0], y - t[1]) < 12:
                    found = t
                    break

            if found:

                found[2] += 1
                found[3] = now
                new_tracks.append(found)

            else:

                new_tracks.append([x,y,1,now])

        self.tracks = [
            t for t in new_tracks
            if now - t[3] < 0.5
        ]

        stable = [t for t in self.tracks if t[2] >= 3]

        self.last_stable = stable

        if self.state == self.STATE_ARMING:

            if now > self.arm_until:
                self.state = self.STATE_ACTIVE
                self.tracks.clear()

            return

        if self.state != self.STATE_ACTIVE:
            return

        calibration = load_camera_calibration()

        if not calibration:
            return

        H = np.array(calibration["homography"], dtype=np.float32)

        for t in stable:

            x,y = t[0],t[1]

            pt = np.array([[[x + scanport.x, y + scanport.y]]], dtype=np.float32)

            screen = cv2.perspectiveTransform(pt, H)[0][0]

            hit_input.push_camera_hit(
                float(screen[0]),
                float(screen[1])
            )

            self.known.append((x,y))

    def get_debug_snapshot(self):

        return {
            "state": self.state,
            "enabled": self.enabled,
            "candidates_count": len(self.last_candidates),
            "stable_tracks_count": len(self.last_stable),
            "known_holes_count": len(self.known)
        }


hit_scanner = HitScanner()