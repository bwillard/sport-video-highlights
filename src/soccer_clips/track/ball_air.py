"""
The ball in the air: which stretches of the ball path are flights, and where the ball is over the field meanwhile.

With one camera about 4 m up (the camera centre from the field's homographies: FC Stars 4.1 m, 25 m from the
middle of the field), a ball in the air projects onto the field well beyond where it is: 2 m up, about twice as
far from the camera. FC Stars' 28:26 cross, about 1.5 m up, read over the goal line and the far touchline.
- **Steadied pixels:** Veo's view turns and zooms about a fixed point, so one frame's pixels map onto another's by
  a homography that holds for anything, in the air or not (one frame's pitch -> pixels times the other's inverse).
- **Gravity in the picture (kappa):** over WINDOW_S either way, a quadratic in time through the ball steadied
  into the middle frame. Its downward acceleration, at the scale where the map puts the ball (metres per pixel
  across the picture there), in g. A ball rolling in a straight line draws a straight line (perspective adds
  ~0.2); in the air, gravity is g / distance in pixels, and the map puts the ball further than it is, so
  kappa is 1 or more. Windows across a kick fit badly (FIT_PX) and don't count.
- **Flights:** kappa at least KAPPA_MIN, fitting within FIT_PX, for SEED_S or more; grown a detection at a time
  either way while one quadratic through the whole flight, steadied into its middle frame, fits within GROW_PX
  (rms) and GROW_MAX_PX: out to the take-off and the landing (or a header, or a catch).
- **Over the field meanwhile:** on the straight line from where the ball was last on the ground before the flight
  to where it's next on the ground after (a ball's track over the ground in flight is straight, at a steady
  speed), by time; as the map has it where either end is more than ANCHOR_S away. Only where the map puts the ball
  BULGE_M or more off that line somewhere in the flight: a ball popping up in a scramble reads only a few metres
  off, and is played where it reads better than on a line across a touch in the air (Elite Strikers' 19:52 and
  20:08 in your labelled window: placed on the line, 3 of 69 labelled touches went to someone else; from 8 m,
  none in the three labelled games). Such a flight is still in the air.
"""
from __future__ import annotations

import numpy as np

G = 9.81
WINDOW_S = 0.2  # kappa: a quadratic over this either way
MIN_ROWS = 7  # ...through at least this many detections
FIT_PX = 2.5  # ...fitting within this (rms, pixels): a window across a kick doesn't
KAPPA_MIN = 1.0  # in the air: falling at least this (in g, at the map's scale)...
SEED_S = 0.15  # ...for this long
GROW_PX = 3.0  # a flight: one quadratic through it fitting within this (rms)...
GROW_MAX_PX = 8.0  # ...and every detection within this
GAP_S = 0.25  # detections further apart than this split a window or a flight
ANCHOR_S = 1.0  # a flight's ball placed from where it was on the ground no further than this before and after...
BULGE_M = 8.0  # ...where the map puts it this far off that line somewhere in the flight (1 m up at 30 m: ~7 m)


def _apply(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.einsum("...ij,...j->...i", H, np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1))
    return q[..., :2] / q[..., 2:]


def _quad(t: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares quadratics in t through points p (..., k, 2), w (..., k) 0 / 1: (coefficients
    (..., 3, 2): constant, t, t^2; rms residual (...,) in p's units)."""
    A = np.stack([np.ones_like(t), t, t * t], axis=-1) * w[..., None]
    AtA = np.einsum("...ki,...kj->...ij", A, A) + np.eye(3) * 1e-9
    c = np.linalg.solve(AtA, np.einsum("...ki,...kd->...id", A, p * w[..., None]))
    r = (np.einsum("...ki,...id->...kd", A, c) - p * w[..., None]) ** 2
    return c, np.sqrt(r.sum(axis=(-1, -2)) / np.maximum(w.sum(-1), 1))


def kappa(frame: np.ndarray, uv: np.ndarray, M: np.ndarray, mpp: np.ndarray, fps: float,
          chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Per detection (frame order; uv frame pixels, M pitch -> pixels of its frame, mpp metres per pixel across
    the picture where the map puts it): (kappa, rms of the window's fit); NaN where too few detections or not
    placed (M not finite)."""
    n = len(frame)
    k = int(np.ceil(WINDOW_S * fps))
    off = np.arange(-k, k + 1)
    ok = np.isfinite(M).all(axis=(1, 2)) & np.isfinite(mpp)
    Mi = np.full_like(M, np.nan)
    Mi[ok] = np.linalg.inv(M[ok])
    kap, rms = np.full(n, np.nan), np.full(n, np.nan)
    for a in range(0, n, chunk):
        i = np.arange(a, min(a + chunk, n))
        j = np.clip(i[:, None] + off, 0, n - 1)
        dt = (frame[j] - frame[i, None]) / fps
        w = ((i[:, None] + off >= 0) & (i[:, None] + off < n) & (np.abs(dt) <= WINDOW_S + 1e-9)
             & ok[j] & ok[i, None]).astype(float)
        H = np.where(w[..., None, None] > 0, M[i, None] @ np.nan_to_num(Mi[j]), np.eye(3))
        st = np.nan_to_num(_apply(H, uv[j]))
        st -= st[:, k:k + 1]  # about the middle detection: better conditioned
        c, r = _quad(dt, st, w)
        good = (w.sum(1) >= MIN_ROWS) & ok[i]
        kap[i] = np.where(good, 2 * c[:, 2, 1] * mpp[i] / G, np.nan)
        rms[i] = np.where(good, r, np.nan)
    return kap, rms


def _fits(frame: np.ndarray, uv: np.ndarray, M: np.ndarray, Mi: np.ndarray, rows: np.ndarray, ref: int,
          fps: float) -> bool:
    """Whether one quadratic through these detections, steadied into detection `ref`'s frame, fits."""
    if len(rows) < MIN_ROWS or np.diff(frame[rows]).max(initial=0) > GAP_S * fps:
        return False
    st = _apply(M[ref] @ Mi[rows], uv[rows])
    _, r = _quad((frame[rows] - frame[ref]) / fps, st - st[0], np.ones(len(rows)))
    if r > GROW_PX:
        return False
    t = (frame[rows] - frame[ref]) / fps
    A = np.stack([np.ones_like(t), t, t * t], axis=-1)
    c, _ = _quad(t, st - st[0], np.ones(len(rows)))
    return float(np.linalg.norm(A @ c - (st - st[0]), axis=1).max()) <= GROW_MAX_PX


def flights(frame: np.ndarray, uv: np.ndarray, M: np.ndarray, kap: np.ndarray, rms: np.ndarray,
            fps: float) -> list[tuple[int, int]]:
    """The flights: [(first frame, last frame)] of the detections in each, in order, not overlapping."""
    n = len(frame)
    ok = np.isfinite(M).all(axis=(1, 2))
    Mi = np.full_like(M, np.nan)
    Mi[ok] = np.linalg.inv(M[ok])
    with np.errstate(invalid="ignore"):
        seed = (kap >= KAPPA_MIN) & (rms <= FIT_PX)
    out: list[tuple[int, int]] = []
    floor, i = 0, 0
    while i < n:
        if not seed[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and seed[j + 1] and frame[j + 1] - frame[j] <= GAP_S * fps:
            j += 1
        if (frame[j] - frame[i]) < SEED_S * fps or i < floor:
            i = j + 1
            continue
        ref = (i + j) // 2
        a, b = i, j
        rows = lambda a, b: np.arange(a, b + 1)[ok[a:b + 1]]  # noqa: E731
        if not _fits(frame, uv, M, Mi, rows(a, b), ref, fps):
            i = j + 1
            continue
        grew = True
        while grew:
            grew = False
            if a - 1 >= floor and ok[a - 1] and _fits(frame, uv, M, Mi, rows(a - 1, b), ref, fps):
                a, grew = a - 1, True
            if b + 1 < n and ok[b + 1] and _fits(frame, uv, M, Mi, rows(a, b + 1), ref, fps):
                b, grew = b + 1, True
        out.append((int(frame[a]), int(frame[b])))
        floor = b + 1
        i = b + 1
    return out


def place(bframe: np.ndarray, bxy: np.ndarray, placed: np.ndarray, fl: list[tuple[int, int]],
          fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xy, air, moved) per ball row (frame order; bxy where the map puts it, placed: rows whose place can be used
    as an anchor): rows inside a flight are in the air; where the map puts the ball BULGE_M or more off the straight
    line between the last placed row on the ground before the flight and the first after, its rows are moved onto
    that line, by frame (left as the map has them where either end is more than ANCHOR_S away)."""
    xy = bxy.astype(np.float64).copy()
    air = np.zeros(len(bframe), bool)
    moved = np.zeros(len(bframe), bool)
    spans = [tuple(np.searchsorted(bframe, [a, b + 1])) for a, b in fl]
    for lo, hi in spans:
        air[lo:hi] = True
    g = np.flatnonzero(placed & ~air)
    for lo, hi in spans:
        k = np.searchsorted(g, lo)
        p, q = (g[k - 1], g[k]) if 0 < k < len(g) else (None, None)  # no ground row inside: g[k] is after it
        if p is None or bframe[lo] - bframe[p] > ANCHOR_S * fps or bframe[q] - bframe[hi - 1] > ANCHOR_S * fps:
            continue  # nothing to measure it against: as the map has it
        w = (bframe[lo:hi] - bframe[p]) / (bframe[q] - bframe[p])
        line = bxy[p] + w[:, None] * (bxy[q] - bxy[p])
        dev = np.linalg.norm(line - bxy[lo:hi], axis=1)
        if np.nanmax(np.where(np.isfinite(dev), dev, 0.0), initial=0) >= BULGE_M:
            xy[lo:hi] = line
            moved[lo:hi] = True
    return xy, air, moved


# ---- from a game ----

def find(store, game, bp: dict) -> list[tuple[int, int]]:
    """The game's flights from its ball path (ball_game.load_path), on the detections (not filled frames)."""
    from soccer_clips.track import ball_track, pitch

    det = bp["kind"] == ball_track.DETECTED
    frame = bp["frame"][det].astype(np.int64)
    uv = np.column_stack([bp["x"][det] * game.width, bp["y"][det] * game.height]).astype(np.float64)
    if len(frame) < MIN_ROWS:
        return []
    M = pitch.frame_matrices(store, game, frame)
    if M is None:
        return []
    ok = np.isfinite(M).all(axis=(1, 2))
    Mi = np.full_like(M, np.nan)
    Mi[ok] = np.linalg.inv(M[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        mpp = np.linalg.norm(_apply(Mi, uv + [10.0, 0.0]) - _apply(Mi, uv), axis=1) / 10
    kap, rms = kappa(frame, uv, M, mpp, game.fps)
    return flights(frame, uv, M, kap, rms, game.fps)
