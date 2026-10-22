"""
Camera motion for follow-cam video (Veo pans and zooms).

For each frame, a similarity transform (rotation, scale, shift) taking the
previous frame's pixel positions to this frame's, fitted with RANSAC to
optical flow on background corners, so players running don't drag it. The
Veo logo is masked out: it never moves on screen, which would pull the fit
towards "no motion".

Chained together, these map any frame's positions into a shared reference
(the clip's first frame), where a ball lying still stays put even while the
camera moves, and speeds are comparable across frames.
"""

from pathlib import Path

import cv2
import numpy as np

SCALE = 0.5  # work at half size: plenty for background motion, 4x cheaper
LOGO = (1600, 930)  # Veo logo: right of x and below y (1920x1080 pixels) is masked


def _as3(a: np.ndarray) -> np.ndarray:
    return np.vstack([a, [0, 0, 1]])


class Tracker:
    """Feed frames one at a time; each call returns the step from the previous frame."""

    def __init__(self):
        self.prev = None
        self.mask = None

    def step(self, frame: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        if self.mask is None:
            self.mask = np.full(grey.shape, 255, np.uint8)
            self.mask[int(LOGO[1] * SCALE):, int(LOGO[0] * SCALE):] = 0
        step = np.eye(2, 3)
        prev = self.prev
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=8, mask=self.mask)
            if pts is not None and len(pts) >= 8:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, grey, pts, None)
                good = status.ravel() == 1
                if good.sum() >= 8:
                    a, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.5)
                    if a is not None:
                        a[:, 2] /= SCALE  # shift back to full-size pixels
                        step = a
        self.prev = grey
        return step


def estimate(path: Path) -> np.ndarray:
    """(n, 2, 3) transforms: step[0] = identity, step[f] maps frame f-1 pixels to frame f pixels."""
    cap = cv2.VideoCapture(str(path))
    t, steps = Tracker(), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        steps.append(t.step(frame))
    cap.release()
    return np.array(steps)


def load_or_estimate(path: Path) -> np.ndarray:
    """Cached next to the video as <name>.motion.npy (recomputed if the video is newer)."""
    cache = path.with_suffix(".motion.npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    steps = estimate(path)
    np.save(cache, steps)
    return steps


def to_reference(steps: np.ndarray) -> np.ndarray:
    """(n, 3, 3): maps frame f pixels into frame 0's pixels (the shared reference)."""
    out = np.empty((len(steps), 3, 3))
    acc = np.eye(3)  # frame 0 -> frame f
    for f, s in enumerate(steps):
        acc = _as3(s) @ acc
        out[f] = np.linalg.inv(acc)
    return out


def apply(m: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points through a 3x3 transform."""
    return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]
