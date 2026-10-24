"""
Drift-free positions for a whole game, for finding things that never move.

A Veo camera doesn't move: the follow-cam video is a virtual view panned and zoomed inside a
fixed wide recording. So any two frames showing the same part of the field are related by a
homography, and a point that doesn't move in the world (a flag, a chair, a bag up the hill)
lands on the same spot in a given view all game. camera.py chains small frame-to-frame steps
instead, which is accurate over a second but drifts by 100+ px over minutes of panning.

Every SAMPLE-th frame is matched directly (ORB features, RANSAC homography) to a keyframe:
the one the last frame matched, its linked neighbours, or the most similar thumbnails. A
view no keyframe covers well becomes a new keyframe, linked to the one it matched. A point
is then placed as (keyframe, position in that keyframe's pixels): accurate, because nothing
is chained over minutes. (One flat world plane for the whole game doesn't work: wide pans
stretch far views into extreme perspective and the registration falls apart.)

<game>/tracks/world.npz: frames (sampled game frames), key (keyframe per sample, -1 if
unmatched), H (n, 3, 3: frame pixels -> keyframe pixels), links (m, 2) and link_H (m, 3, 3:
keyframe a pixels -> keyframe b pixels) between keyframes with overlapping views.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

SAMPLE = 10  # register every 10th frame (3 a second)
SCALE = 0.5  # find features at 960x540
LOGO = (1600, 930)  # Veo logo: right of x and below y (full-size pixels) is masked
MIN_INLIERS = 50  # a match to a keyframe needs this many RANSAC inliers
GOOD_INLIERS = 200  # stop trying other keyframes after a match this good
NEW_KEY_OVERLAP = 0.6  # a frame whose view the matched keyframe covers less than this becomes one
NEW_KEY_INLIERS = 120  # ...or one matched with fewer inliers than this
THUMB = (64, 36)


def _as3(a: np.ndarray) -> np.ndarray:
    return np.vstack([a, [0, 0, 1]])


def _project(H: np.ndarray, pts) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def _covered(H: np.ndarray, w: int, h: int) -> float:
    """Share of a frame's view (mapped into a keyframe by H) that lies inside the keyframe."""
    quad = _project(H, [[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
    area = cv2.contourArea(quad)
    if area <= 0 or not cv2.isContourConvex(quad):
        return 0.0
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    inter, _ = cv2.intersectConvexConvex(quad, rect)
    return float(inter / area)


def _thumb(img: np.ndarray) -> np.ndarray:
    t = cv2.resize(cv2.GaussianBlur(img, (0, 0), 2), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def _frames(store: Store, game: Game, start: int, end: int):
    """Yield (game frame, grey SCALE-sized image) for every SAMPLE-th frame in [start, end)."""
    from soccer_clips import ffmpeg

    w, h = int(game.width * SCALE), int(game.height * SCALE)
    fps = game.fps_num / game.fps_den
    seek_s, dur_s = ffmpeg.cut_window(game, start / fps, end / fps)  # an exact seek: output starts at `start`
    vf = f"select='not(mod(n\\,{SAMPLE}))',scale={w}:{h},format=gray"
    cmd = [store.settings.ffmpeg, "-v", "error", "-ss", f"{seek_s:.6f}", "-t", f"{dur_s:.6f}",
           "-i", str(store.original_path(game)), "-map", "0:v:0",
           "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        f = start
        while f < end:
            buf = proc.stdout.read(w * h)
            if len(buf) < w * h:
                break
            yield f, np.frombuffer(buf, np.uint8).reshape(h, w)
            f += SAMPLE
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


class _Matcher:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def homography(self, pts_a, des_a, pts_b, des_b) -> tuple[np.ndarray | None, int]:
        """H mapping a's pixels to b's, and its inlier count."""
        if des_a is None or des_b is None or len(pts_a) < MIN_INLIERS or len(pts_b) < MIN_INLIERS:
            return None, 0
        pairs = [m for m in self.bf.knnMatch(des_a, des_b, k=2) if len(m) == 2]
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        if len(good) < MIN_INLIERS:
            return None, 0
        H, inl = cv2.findHomography(pts_a[[g.queryIdx for g in good]], pts_b[[g.trainIdx for g in good]],
                                    cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        return (H, n) if H is not None and n >= MIN_INLIERS else (None, 0)


def register(store: Store, game: Game, ranges, log=print) -> dict:
    """Keyframe and homography for every SAMPLE-th frame of each range (see the module docstring)."""
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
    match = _Matcher()
    W, Hh = game.width, game.height
    w, h = int(W * SCALE), int(Hh * SCALE)
    mask = np.full((h, w), 255, np.uint8)
    mask[int(LOGO[1] * SCALE):, int(LOGO[0] * SCALE):] = 0
    keys: list[dict] = []  # {"pts", "des", "thumb"}
    neighbours: dict[int, set[int]] = {}
    links, link_H = [], []
    frames, key_of, Hs = [], [], []
    last = -1

    def link(a: int, b: int, H_ab: np.ndarray) -> None:
        if b in neighbours.setdefault(a, set()):
            return
        neighbours[a].add(b)
        neighbours.setdefault(b, set()).add(a)
        links.append((a, b))
        link_H.append(H_ab)

    for s, e in ranges:
        for f, img in _frames(store, game, s, e):
            kp, des = orb.detectAndCompute(img, mask)
            pts = np.array([k.pt for k in kp], np.float32) / SCALE if kp else np.empty((0, 2), np.float32)
            th = _thumb(img)
            cands: list[int] = []
            if last >= 0:
                cands += [last] + sorted(neighbours.get(last, ()), key=lambda k: -float(keys[k]["thumb"] @ th))[:4]
            if keys:
                sims = np.array([k["thumb"] for k in keys]) @ th
                cands += [int(k) for k in np.argsort(-sims)[:3]]
            best = (-1, None, 0)
            for k in dict.fromkeys(cands):  # in order, without repeats
                H, n = match.homography(pts, des, keys[k]["pts"], keys[k]["des"])
                if n > best[2]:
                    best = (k, H, n)
                if n >= GOOD_INLIERS:
                    break
            k, H, n = best
            new = des is not None and len(pts) >= MIN_INLIERS and (
                k < 0 or n < NEW_KEY_INLIERS or _covered(H, W, Hh) < NEW_KEY_OVERLAP)
            if new:
                keys.append({"pts": pts, "des": des, "thumb": th})
                knew = len(keys) - 1
                if k >= 0:
                    link(knew, k, H)
                k, H = knew, np.eye(3)
            elif k >= 0 and last >= 0 and k != last:  # moved from one keyframe's view to another's
                H2, _ = match.homography(keys[last]["pts"], keys[last]["des"], keys[k]["pts"], keys[k]["des"])
                if H2 is not None:
                    link(last, k, H2)
            frames.append(f)
            key_of.append(k)
            Hs.append(H if H is not None else np.full((3, 3), np.nan))
            last = k
            if len(frames) % 1000 == 0:
                recent = np.array(key_of[-1000:])
                log(f"  {f}: {len(frames)} sampled, {np.mean(recent >= 0):.0%} placed lately, {len(keys)} keyframes")
    return {"frames": np.array(frames, np.int32), "key": np.array(key_of, np.int32), "H": np.array(Hs),
            "links": np.array(links, np.int32).reshape(-1, 2), "link_H": np.array(link_H).reshape(-1, 3, 3)}


def _chain(steps: dict[int, np.ndarray], a: int, b: int) -> np.ndarray:
    """3x3 mapping frame a pixels to frame b pixels (a <= b) from the per-frame camera steps."""
    m = np.eye(3)
    for f in range(a + 1, b + 1):
        s = steps.get(f)
        if s is not None:
            m = _as3(s) @ m
    return m


def place(world: dict, motion_frames: np.ndarray, motion: np.ndarray, f: np.ndarray, x: np.ndarray,
          y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(keyframe, kx, ky) for points (x, y) in frames f: keyframe -1 where there's none."""
    steps = dict(zip(motion_frames.tolist(), motion))
    sf = world["frames"]
    key = np.full(len(f), -1, np.int32)
    kx, ky = np.full(len(f), np.nan), np.full(len(f), np.nan)
    cache: dict[int, tuple[int, np.ndarray | None]] = {}
    for i in np.argsort(f, kind="stable"):
        fi = int(f[i])
        if fi not in cache:
            j = int(np.searchsorted(sf, fi, side="right")) - 1
            if j < 0 or fi - sf[j] >= SAMPLE * 2 or world["key"][j] < 0:
                cache[fi] = (-1, None)
            else:
                cache[fi] = (int(world["key"][j]), world["H"][j] @ np.linalg.inv(_chain(steps, int(sf[j]), fi)))
        k, H = cache[fi]
        if k >= 0:
            key[i] = k
            (kx[i], ky[i]), = _project(H, [[x[i], y[i]]])
    return key, kx, ky


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "world.npz"


def load(store: Store, game: Game) -> dict | None:
    p = path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


CELL = 35.0  # keyframe pixels: detections at a spot are counted over this cell and its neighbours. A Veo
# view isn't an exact pinhole camera (two lenses stitched), so a still object's placements wobble by
# ~30 px near a view's edge; at 15 px a ball lying still all game (fc-1 56:10) and a faint mark (PRSC
# 22:12) were split into pieces too short or too sparse to count.
MAX_HOPS = 1  # also look a spot up in keyframes this many links away (3 hops: errors add up, worse)
FRAME = (1920, 1080)  # keyframe pixels are frame pixels


def _views(world: dict, max_hops: int | None = None) -> dict[int, list[tuple[int, np.ndarray]]]:
    """For each keyframe: (keyframe, H from its pixels to that one's) for itself and every keyframe
    up to max_hops (default MAX_HOPS) links away, by the shortest chain of links."""
    max_hops = MAX_HOPS if max_hops is None else max_hops
    linked: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (a, b), H in zip(world["links"], world["link_H"]):
        linked.setdefault(int(a), []).append((int(b), H))
        linked.setdefault(int(b), []).append((int(a), np.linalg.inv(H)))
    out = {}
    for k in np.unique(world["key"][world["key"] >= 0]).tolist():
        seen = {k: np.eye(3)}
        frontier = [k]
        for _ in range(max_hops):
            nxt = []
            for a in frontier:
                for b, H in linked.get(a, []):
                    if b not in seen:
                        seen[b] = H @ seen[a]
                        nxt.append(b)
            frontier = nxt
        out[k] = list(seen.items())
    return out


def _mapped(views: list[tuple[int, np.ndarray]], x: float, y: float):
    """(keyframe, x, y) for the point in each view that shows it (inside that keyframe's picture)."""
    for k, H in views:
        p = H @ np.array([x, y, 1.0])
        if p[2] <= 0:
            continue
        px, py = p[0] / p[2], p[1] / p[2]
        if 0 <= px < FRAME[0] and 0 <= py < FRAME[1]:
            yield k, px, py


def spot_seconds(world: dict, key: np.ndarray, kx: np.ndarray, ky: np.ndarray, f: np.ndarray, fps: float) -> np.ndarray:
    """
    For each placed detection, in how many distinct seconds of the game something was detected
    at the same spot: its cell and the 8 around it, in its keyframe and in keyframes up to
    MAX_HOPS links away that show the same place (its position mapped across). A fixed object racks up hundreds; the game ball moves on.
    """
    sec = (f // round(fps)).astype(np.int64)
    ok = key >= 0
    cells: dict[tuple[int, int, int], set[int]] = {}
    ix, iy = np.floor(kx / CELL), np.floor(ky / CELL)
    for k, a, b, s in zip(key[ok], ix[ok].astype(int), iy[ok].astype(int), sec[ok]):
        cells.setdefault((int(k), int(a), int(b)), set()).add(int(s))
    views = _views(world)
    cache: dict[tuple[int, int, int], int] = {}

    def around(k: int, a: int, b: int, out: set) -> None:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                out |= cells.get((k, a + da, b + db), set())

    n = np.zeros(len(key), np.int32)
    for i in np.nonzero(ok)[0]:
        c = (int(key[i]), int(ix[i]), int(iy[i]))
        if c not in cache:
            seen: set[int] = set()
            for k2, x2, y2 in _mapped(views.get(c[0], [(c[0], np.eye(3))]), (c[1] + 0.5) * CELL, (c[2] + 0.5) * CELL):
                around(k2, int(np.floor(x2 / CELL)), int(np.floor(y2 / CELL)), seen)
            cache[c] = len(seen)
        n[i] = cache[c]
    return n


SEARCH = (320, 180, 1600, 900)  # the detector's centre crop (ball_game.CROP), full-size pixels


def spot_share(world: dict, key: np.ndarray, kx: np.ndarray, ky: np.ndarray, f: np.ndarray, fps: float,
               min_seconds: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    For each placed detection: (seconds something was detected at its spot, seconds the spot was
    inside the detector's search area). A fixed object is detected most of the time it can be
    seen; a spot on the field the ball keeps crossing is searched far longer than it has a
    ball on it. Spots with fewer than min_seconds detected seconds are left at 0 searched.
    """
    detected = spot_seconds(world, key, kx, ky, f, fps)
    searched = np.zeros(len(key), np.int32)
    sec = world["frames"] // round(fps)
    inv = np.full_like(world["H"], np.nan)
    good = np.isfinite(world["H"]).all(axis=(1, 2))
    inv[good] = np.linalg.inv(world["H"][good])
    by_key: dict[int, np.ndarray] = {}
    for k in np.unique(world["key"][world["key"] >= 0]):
        by_key[int(k)] = np.nonzero((world["key"] == k) & good)[0]
    views = _views(world)
    x0, y0, x1, y1 = SEARCH
    cache: dict[tuple[int, int, int], int] = {}
    for i in np.nonzero((key >= 0) & (detected >= min_seconds))[0]:
        c = (int(key[i]), int(np.floor(kx[i] / CELL)), int(np.floor(ky[i] / CELL)))
        if c not in cache:
            secs: set[int] = set()
            for k, px, py in _mapped(views.get(c[0], [(c[0], np.eye(3))]), (c[1] + 0.5) * CELL, (c[2] + 0.5) * CELL):
                j = by_key.get(k)
                if j is None or not len(j):
                    continue
                q = inv[j] @ np.array([px, py, 1.0])  # into each sample frame that matched this keyframe
                fx, fy = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
                inside = (fx >= x0) & (fx < x1) & (fy >= y0) & (fy < y1)
                secs.update(sec[j[inside]].tolist())
            cache[c] = len(secs)
        searched[i] = cache[c]
    return detected, searched


FIXED_SHARE = 0.6  # detected at least this share of the time the spot is searched...
FIXED_SECONDS = 60  # ...over at least this many seconds: a fixed object (PRSC flag, chair, hill item, faint
# hill mark, fc-1's ball lying still: all ~1.0; the game ball in play at labelled frames at most 0.34)


def fixed_path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "fixed.npz"


def find_fixed(store: Store, game: Game, log=print) -> int:
    """
    Register the game (world.npz, if not done yet) and mark every ball detection on a fixed
    object. Writes fixed.npz: blocked (bool, aligned with ball_raw.npz's dets), detected and
    searched seconds per detection, and the count of dets it was made for. Returns how many.
    """
    from soccer_clips.track import ball_game

    raw = ball_game.load_raw(store, game)
    w = load(store, game)
    if w is None:
        log(f"{game.id}: registering frames against keyframes...")
        w = register(store, game, raw["ranges"].tolist(), log=log)
        ball_game.save_npz(path(store, game), **w)
    if not bool(w.get("dense", False)):
        w = densify(store, game, w, log=log)
        ball_game.save_npz(path(store, game), **w)
    d = raw["dets"]
    f = d[:, 0].astype(int)
    key, kx, ky = place(w, raw["motion_frames"], raw["motion"], f, (d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2)
    detected, searched = spot_share(w, key, kx, ky, f, game.fps_num / game.fps_den)
    blocked = (detected >= FIXED_SECONDS) & (detected >= FIXED_SHARE * searched) & (searched > 0)
    ball_game.save_npz(fixed_path(store, game), blocked=blocked, detected=detected, searched=searched,
                       n_dets=np.int64(len(d)))
    return int(blocked.sum())


def load_blocked(store: Store, game: Game, n_dets: int) -> np.ndarray | None:
    """The blocked mask for the current detections (None if not made, or made for other detections)."""
    p = fixed_path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["blocked"] if int(z["n_dets"]) == n_dets else None


DENSE_CANDIDATES = 12  # per keyframe, try linking to this many of the most similar other keyframes
DENSE_INLIERS = 60


def _keyframe_frames(world: dict) -> dict[int, int]:
    """The sampled game frame each keyframe was made from (its first sample, where H is the identity)."""
    out: dict[int, int] = {}
    for f, k in zip(world["frames"].tolist(), world["key"].tolist()):
        if k >= 0 and k not in out:
            out[k] = f
    return out


def _grab(store: Store, game: Game, frame: int) -> np.ndarray | None:
    """One frame, grey, at SCALE (an exact seek)."""
    from soccer_clips import ffmpeg

    w, h = int(game.width * SCALE), int(game.height * SCALE)
    fps = game.fps_num / game.fps_den
    seek_s, _ = ffmpeg.cut_window(game, frame / fps, (frame + 1) / fps)
    out = subprocess.run([store.settings.ffmpeg, "-v", "error", "-ss", f"{seek_s:.6f}", "-i", str(store.original_path(game)),
                          "-frames:v", "1", "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w) if len(out) == w * h else None


def densify(store: Store, game: Game, world: dict, log=print) -> dict:
    """
    Link keyframes that show the same place but were never linked (registration only links a
    keyframe to the one the camera came from): match each keyframe to its most similar others
    directly and add a link wherever the match is good. Then a spot is looked up in every view
    of it one accurate hop away (fc-1 51:04: a still ball's views were 2-3 links apart).
    """
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
    match = _Matcher()
    w, h = int(game.width * SCALE), int(game.height * SCALE)
    mask = np.full((h, w), 255, np.uint8)
    mask[int(LOGO[1] * SCALE):, int(LOGO[0] * SCALE):] = 0
    feats = {}
    for k, f in sorted(_keyframe_frames(world).items()):
        img = _grab(store, game, f)
        if img is None:
            continue
        kp, des = orb.detectAndCompute(img, mask)
        pts = np.array([p.pt for p in kp], np.float32) / SCALE if kp else np.empty((0, 2), np.float32)
        feats[k] = (pts, des, _thumb(img))
    keys = sorted(feats)
    thumbs = np.array([feats[k][2] for k in keys])
    have = {(int(a), int(b)) for a, b in world["links"]} | {(int(b), int(a)) for a, b in world["links"]}
    links, link_H = list(map(tuple, world["links"].tolist())), list(world["link_H"])
    added = 0
    for i, a in enumerate(keys):
        sims = thumbs @ thumbs[i]
        for j in np.argsort(-sims)[1:DENSE_CANDIDATES + 1]:
            b = keys[int(j)]
            if (a, b) in have:
                continue
            H, n = match.homography(feats[a][0], feats[a][1], feats[b][0], feats[b][1])
            have |= {(a, b), (b, a)}
            if H is not None and n >= DENSE_INLIERS:
                links.append((a, b))
                link_H.append(H)
                added += 1
    log(f"  {game.id}: {added} keyframe links added to {len(world['links'])}")
    return {**world, "links": np.array(links, np.int32).reshape(-1, 2), "link_H": np.array(link_H).reshape(-1, 3, 3),
            "dense": np.bool_(True)}
