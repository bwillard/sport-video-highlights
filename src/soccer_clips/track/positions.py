"""
Where every tracked player and the ball are on the field, in metres (Phase 5.4).

pitch.py ties each frame to the field; this places what's in the frames, once per game, so the map and
everything after it (Phase 6) read positions instead of each projecting boxes again:
- **Players:** each box's feet (bottom centre) through its frame's homography, then a running median along
  its tracklet over SMOOTH_HALF rows each way (~0.5 s). Box bottoms jitter by a few pixels, which is metres
  on the far side, and a moment of hidden feet pushes the box bottom up, which reads as further away.
- **Off the field:** feet further than status.FIELD_PAD_M outside the lines, or standing still by a
  touchline (status.still_by_line: the bench and parents, who on the far side read up to 4 m inside it).
  The same rule as the Games page's count, here for every box.
- **Ball:** the ball path's positions on the field. With one camera there's no height, so a ball in the air
  projects far beyond where it is; more than BALL_OFF_M outside the lines is flagged as that.

<game>/tracks/pitch.npz: `xy` (rows, 2) float32 metres per row of the game's tracklets (people.Tracks order,
NaN where the frame isn't placed), `off` per row, `ball_frame` / `ball_xy` / `ball_off`, the field's size
it was made with, `stats` (JSON: how much was placed, seam jumps) and `key`: the files it was made from,
so a re-track, a ball fix or a new calibration means it's made again (`get`, or `build_soon` for the app).
"""
from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

VERSION = 1  # bump when what's saved changes meaning
SMOOTH_HALF = 7  # rows each side of a box in its tracklet's running median (15 rows: ~0.5 s)
BALL_OFF_M = 10.0  # a ball further outside the lines than this is in the air (as the map has it)
CHUNK = 400_000  # rows smoothed at once (memory: 15 x 2 float64 per row)


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "pitch.npz"


def _sources(store: Store, game: Game) -> list[Path]:
    from soccer_clips.track import ball_game, people, pitch, world

    d = ball_game.tracks_dir(store, game)
    return [people.path(store, game, people.DEFAULT_MODEL), pitch.cal_path(store, game), world.path(store, game),
            d / "ball_raw.npz", d / "ball.npz"]


def key(store: Store, game: Game) -> str | None:
    """What the positions are made from (file dates); None while a file they need is missing."""
    src = _sources(store, game)
    if not all(p.exists() for p in src[:4]):  # the ball path is optional
        return None
    stamps = [str(p.stat().st_mtime_ns) if p.exists() else "-" for p in src]
    return f"v{VERSION} " + " ".join(stamps)


# ---- the pieces (pure, for testing) ----

def smooth(track: np.ndarray, frame: np.ndarray, xy: np.ndarray, half: int = SMOOTH_HALF) -> np.ndarray:
    """Running median of xy (n, 2) over `half` rows each side within each tracklet (in frame order), ignoring
    NaN (unplaced frames); rows stay in their given order."""
    n = len(track)
    out = np.full((n, 2), np.nan, np.float32)
    if not n:
        return out
    order = np.lexsort((frame, track))
    k, v = track[order], xy[order].astype(np.float64)
    lo = np.searchsorted(k, k, "left")  # each row's tracklet: rows lo..hi-1 of the sorted order
    hi = np.searchsorted(k, k, "right")
    off = np.arange(-half, half + 1)
    sm = np.empty((n, 2))
    for a in range(0, n, CHUNK):
        i = np.arange(a, min(a + CHUNK, n))
        j = i[:, None] + off
        inside = (j >= lo[i, None]) & (j < hi[i, None])
        w = v[np.clip(j, 0, n - 1)]  # (rows, window, 2)
        w[~inside] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all NaN: not placed
            sm[i] = np.nanmedian(w, axis=1)
    out[order] = sm
    return out


def off_field(track: np.ndarray, frame: np.ndarray, xy: np.ndarray, length: float, width: float,
              fps: float) -> np.ndarray:
    """Per row (every frame): off the field, by the Games page's rule (status: more than FIELD_PAD_M outside
    the lines, or still by a touchline). Rows that aren't placed count as on it."""
    from soccer_clips.track import status

    with np.errstate(invalid="ignore"):
        off = (np.abs(xy[:, 0]) > length / 2 + status.FIELD_PAD_M) | (np.abs(xy[:, 1]) > width / 2 + status.FIELD_PAD_M)
    return off | status.still_by_line(track, frame, xy, width, fps, step=1)


def seam_jumps(track: np.ndarray, frame: np.ndarray, xy: np.ndarray, keyframe: np.ndarray) -> dict:
    """How far a player's raw position moves from one frame to the next, where the frame's keyframe changes
    (a seam between two views' calibrations) and where it doesn't, on the near (y < 0) and far halves.
    Medians in metres, and how many steps they're from."""
    order = np.lexsort((frame, track))
    k, f, p, kf = track[order], frame[order], xy[order], keyframe[order]
    same = (k[1:] == k[:-1]) & (f[1:] == f[:-1] + 1)
    step = np.linalg.norm(p[1:] - p[:-1], axis=1)
    ok = same & np.isfinite(step)
    switch = kf[1:] != kf[:-1]
    near = p[1:, 1] < 0
    out = {}
    for side, m in (("near", near), ("far", ~near)):
        for what, s in (("seam", switch), ("step", ~switch)):
            sel = ok & m & s
            out[f"{side}_{what}_m"] = round(float(np.median(step[sel])), 3) if sel.any() else None
            out[f"{side}_{what}_n"] = int(sel.sum())
    return out


# ---- making them ----

def _to_pitch(store: Store, game: Game, P: np.ndarray, frames: np.ndarray, x: np.ndarray, y: np.ndarray):
    """(pitch metres (n, 2), keyframe per point) for ground points in frame pixels; NaN / -1 where not placed."""
    from soccer_clips.track import pitch

    uf, inv = np.unique(np.asarray(frames, int), return_inverse=True)
    kf, Hs = pitch.frame_homographies(*pitch._registration(store, game), uf)
    ok = (kf >= 0) & np.isfinite(Hs).all(axis=(1, 2))
    ok[ok] &= np.isfinite(P[kf[ok]]).all(axis=(1, 2))
    M = np.full((len(uf), 3, 3), np.nan)
    M[ok] = np.linalg.inv(P[kf[ok]]) @ Hs[ok]  # frame pixels -> keyframe pixels -> pitch metres
    out = np.full((len(frames), 2), np.nan)
    m = ok[inv]
    out[m] = pitch.project(M[inv[m]], np.column_stack([x, y])[m])
    return out, np.where(ok, kf, -1)[inv]


def place(store: Store, game: Game, log=None) -> dict:
    """Work out and save the game's positions (seconds to a minute); returns the stats. Raises ValueError when
    the game isn't tracked, registered or calibrated."""
    import time

    from soccer_clips.track import ball_game, people, pitch

    t0 = time.time()
    k = key(store, game)
    if k is None:
        raise ValueError("needs the game's tracklets, frame registration and field calibration")
    cal = pitch.load_cal(store, game)
    if cal is None and pitch.load_points(store, game):  # clicked since the last solve
        pitch.calibrate(store, game)
        cal, k = pitch.load_cal(store, game), key(store, game)
    if cal is None or "P" not in cal:
        raise ValueError("the field isn't calibrated")
    t = people.tracks(store, game)
    fps = game.fps_num / game.fps_den
    L, W = (float(v) for v in cal["sizes"][:2])

    b = t.box
    raw, kf = _to_pitch(store, game, cal["P"], t.frame, (b[:, 0] + b[:, 2]) / 2 * game.width, b[:, 3] * game.height)
    xy = smooth(t.track, t.frame, raw)
    off = off_field(t.track, t.frame, xy, L, W, fps)

    bp = ball_game.load_path(store, game)
    if bp is not None and len(bp["frame"]):
        bf = bp["frame"].astype(np.int32)
        bxy, _ = _to_pitch(store, game, cal["P"], bf, bp["x"] * game.width, bp["y"] * game.height)
        with np.errstate(invalid="ignore"):
            boff = (np.abs(bxy[:, 0]) > L / 2 + BALL_OFF_M) | (np.abs(bxy[:, 1]) > W / 2 + BALL_OFF_M)
    else:
        bf, bxy, boff = np.empty(0, np.int32), np.empty((0, 2)), np.empty(0, bool)

    placed = np.isfinite(xy[:, 0])
    stats = {"rows": int(len(xy)), "placed": round(float(placed.mean()), 4) if len(xy) else 0.0,
             "off": round(float(off[placed].mean()), 4) if placed.any() else 0.0,
             "ball": int(len(bf)), "ball_placed": round(float(np.isfinite(bxy[:, 0]).mean()), 4) if len(bf) else 0.0,
             "ball_off": round(float(boff.mean()), 4) if len(bf) else 0.0,
             "smoothing_m": round(float(np.nanmedian(np.linalg.norm(xy - raw, axis=1))), 3) if placed.any() else None,
             **seam_jumps(t.track, t.frame, raw, kf), "seconds": 0.0}
    stats["seconds"] = round(time.time() - t0, 1)
    ball_game.save_npz(path(store, game), key=np.array(k), xy=xy.astype(np.float32), off=off,
                       ball_frame=bf, ball_xy=bxy.astype(np.float32), ball_off=boff,
                       size=np.array([L, W]), stats=np.array(json.dumps(stats)))
    _cache.pop(game.id, None)
    if log:
        log(f"{game.id}: {stats['rows']} boxes, {stats['placed']:.0%} placed, {stats['off']:.1%} off the field; "
            f"ball {stats['ball_placed']:.0%} of {stats['ball']} placed; {stats['seconds']} s")
    return stats


# ---- reading them ----

_cache: dict[str, tuple[str, dict]] = {}  # game id -> (key, positions); one game kept
_building: set[str] = set()
_lock = threading.Lock()


def load(store: Store, game: Game) -> dict | None:
    """The saved positions if they're up to date (kept in memory), else None."""
    k = key(store, game)
    if k is None:
        return None
    hit = _cache.get(game.id)
    if hit and hit[0] == k:
        return hit[1]
    p = path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        if str(z["key"]) != k:
            return None
        out = {n: z[n] for n in z.files if n not in ("key", "stats")}
        out["stats"] = json.loads(str(z["stats"]))
    _cache.clear()
    _cache[game.id] = (k, out)
    return out


def get(store: Store, game: Game, log=None) -> dict | None:
    """The positions, made first if they're missing or out of date; None if the game can't be placed yet."""
    got = load(store, game)
    if got is not None:
        return got
    try:
        place(store, game, log=log)
    except ValueError:
        return None
    return load(store, game)


def build_soon(store: Store, game: Game) -> None:
    """Make the game's positions in a background thread if they're out of date (for the app: the map places
    boxes itself meanwhile). One at a time per game."""
    with _lock:
        if game.id in _building or key(store, game) is None:
            return
        _building.add(game.id)

    def run():
        try:
            if load(store, game) is None:
                place(store, game)
        except Exception as e:  # noqa: BLE001 - a background job; the map falls back to placing in the browser
            print(f"positions for {game.id} failed: {e}", flush=True)
        finally:
            with _lock:
                _building.discard(game.id)

    threading.Thread(target=run, daemon=True, name=f"positions-{game.id}").start()


def saved_stats(store: Store, game: Game) -> tuple[dict, bool] | None:
    """(stats, up to date) of the saved positions without loading them, or None if there are none."""
    p = path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        return json.loads(str(z["stats"])), str(z["key"]) == key(store, game)
