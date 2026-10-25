"""
Kit colour of every detected person box (Phase 4.2), for telling teams apart.

Each frame is decoded at half size and every pixel put in one of NBINS colour bins: 8 hues x
(bright, dark) for coloured pixels, plus dark / grey / light for the rest. Pixels close to
the grass colour of that frame (its most common colour low in the picture) are left out, so
a green kit still counts unless it's the exact shade of the field. A box's feature is the
histogram of its torso (the middle of the box's width, from just below the head to the
waist), square-rooted so distances between features behave (Hellinger), as uint8.

Per game, tracks/people_kit_<model>.npz: feats (m, NBINS) uint8, one row per row of
people_raw_<model>.npz's dets (same order), all zero where the torso had no usable pixels.
Takes ~15-30 min a game on this CPU (decoding is most of it), split over processes.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from soccer_clips import ffmpeg
from soccer_clips.models import Game
from soccer_clips.store import Store
from soccer_clips.track.ball_game import chunks, merge, save_npz, tracks_dir

SCALE = 0.5
NBINS = 19
TORSO = (0.25, 0.75, 0.18, 0.5)  # x from, x to, y from, y to: fractions of the box
GRASS_H, GRASS_S, GRASS_V = 10, 60, 60  # how close (OpenCV HSV units) a pixel must be to the grass to be left out
EXCLUDED = 255
GRASS_EVERY = 15  # frames between grass colour estimates (it changes slowly, with the light and the view)


def kit_path(store: Store, game: Game, model: str) -> Path:
    return tracks_dir(store, game) / f"people_kit_{model}.npz"


def grass_colour(frame_bgr: np.ndarray) -> tuple[int, int, int]:
    """The grass's hue, saturation, value: the most common hue low in the picture (sampled sparsely)."""
    low = cv2.cvtColor(np.ascontiguousarray(frame_bgr[frame_bgr.shape[0] // 2::4, ::4]), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    gh = int(np.bincount(low[:, 0], minlength=180).argmax())
    near = np.abs(low[:, 0].astype(np.int16) - gh) <= GRASS_H
    gs, gv = (int(np.median(low[near, i])) if near.any() else 128 for i in (1, 2))
    return gh, gs, gv


def bins_of(hsv: np.ndarray, grass: tuple[int, int, int]) -> np.ndarray:
    """Colour bin (0..NBINS-1) of HSV pixels (..., 3), EXCLUDED for grass."""
    gh, gs, gv = grass
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    dh = np.minimum(np.abs(h - gh), 180 - np.abs(h - gh))
    grass = (dh <= GRASS_H) & (np.abs(s - gs) <= GRASS_S) & (np.abs(v - gv) <= GRASS_V)
    coloured = (s >= 60) & (v >= 50)
    hue_bin = ((h + 11) % 180) // 23  # 8 hue bins, red centred on bin 0
    out = np.where(coloured, hue_bin * 2 + (v < 140), np.where(v < 70, 16, np.where(v < 170, 17, 18)))
    out = out.astype(np.uint8)
    out[grass] = EXCLUDED
    return out


def bin_map(frame_bgr: np.ndarray) -> np.ndarray:
    """Colour bin of every pixel of a frame (for looking at; features only bin the torsos)."""
    return bins_of(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), grass_colour(frame_bgr))


def box_features(frame_bgr: np.ndarray, boxes: np.ndarray, grass: tuple[int, int, int] | None = None) -> np.ndarray:
    """uint8 features (n, NBINS) for boxes (n, 4: x1, y1, x2, y2 in full-size pixels) on a SCALE-size frame."""
    H, W = frame_bgr.shape[:2]
    out = np.zeros((len(boxes), NBINS), np.uint8)
    if not len(boxes):
        return out
    fx0, fx1, fy0, fy1 = TORSO
    b = boxes * SCALE
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    xa = np.clip(b[:, 0] + fx0 * w, 0, W).astype(int)
    xb = np.clip(np.ceil(b[:, 0] + fx1 * w), 0, W).astype(int)
    ya = np.clip(b[:, 1] + fy0 * h, 0, H).astype(int)
    yb = np.clip(np.ceil(b[:, 1] + fy1 * h), 0, H).astype(int)
    crops, owner = [], []
    for i in np.flatnonzero((xb > xa) & (yb > ya)):
        c = frame_bgr[ya[i]:yb[i], xa[i]:xb[i]].reshape(-1, 3)
        crops.append(c)
        owner.append(np.full(len(c), i))
    if not crops:
        return out
    px = np.concatenate(crops)[:, None, :]  # all torso pixels as one image column: one colour conversion
    bins = bins_of(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[:, 0], grass or grass_colour(frame_bgr))
    owner = np.concatenate(owner)
    keep = bins != EXCLUDED
    n = np.bincount(owner[keep] * NBINS + bins[keep], minlength=len(boxes) * NBINS).reshape(len(boxes), NBINS)
    tot = n.sum(1, keepdims=True)
    ok = tot[:, 0] > 0
    out[ok] = np.round(np.sqrt(n[ok] / tot[ok]) * 255)
    return out


def read_small(store: Store, game: Game, start: int, end: int):
    """Frames start..end-1 at SCALE size (BGR), exact frame numbers as ball_game.read_frames."""
    w, h = int(game.width * SCALE), int(game.height * SCALE)
    fps = game.fps_num / game.fps_den
    seek_s, _ = ffmpeg.cut_window(game, start / fps, end / fps)
    proc = subprocess.Popen([store.settings.ffmpeg, "-v", "error", "-threads", "2", "-ss", f"{seek_s:.6f}",
                             "-i", str(store.original_path(game)), "-frames:v", str(end - start), "-map", "0:v:0",
                             "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    try:
        for _ in range(end - start):
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def features_range(store: Store, game: Game, dets: np.ndarray, start: int, end: int) -> np.ndarray:
    """Features for the dets (rows: frame, x1, y1, x2, y2, conf) with frames in [start, end), in their order."""
    out = np.zeros((len(dets), NBINS), np.uint8)
    order = np.argsort(dets[:, 0], kind="stable")
    bounds = np.searchsorted(dets[order, 0], np.arange(start, end + 1))
    grass = None
    for k, frame in enumerate(read_small(store, game, start, end)):
        if k % GRASS_EVERY == 0:
            grass = grass_colour(frame)
        rows = order[bounds[k]:bounds[k + 1]]
        if len(rows):
            out[rows] = box_features(frame, dets[rows, 1:5], grass)
    return out


def _chunk(args):
    settings, game_id, model, s, e = args
    cv2.setNumThreads(1)  # one process per core already
    from soccer_clips.track import people

    store = Store(settings)
    game = store.load_game(game_id)
    dets = people.load_raw(store, game, model)["dets"]
    sel = np.flatnonzero((dets[:, 0] >= s) & (dets[:, 0] < e))
    return sel, features_range(store, game, dets[sel], s, e)


def compute(store: Store, game: Game, model: str, workers: int = 4, log=print) -> int:
    """Kit features for every saved box of the game (all of it again, if re-run). Returns boxes done."""
    import multiprocessing as mp

    from soccer_clips.track import people

    raw = people.load_raw(store, game, model)
    feats = np.zeros((len(raw["dets"]), NBINS), np.uint8)
    todo = chunks(merge([tuple(r) for r in raw["ranges"].tolist()]), workers * 4)
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (sel, f) in enumerate(pool.imap_unordered(_chunk, [(store.settings, game.id, model, s, e) for s, e in todo]), 1):
            feats[sel] = f
            log(f"{game.id}: kit colours {n}/{len(todo)} chunks")
    save_npz(kit_path(store, game, model), feats=feats)
    return len(feats)


TEAM_K = 6  # clusters per game: us and them (each often split into a light and a shaded cluster), refs, keepers, spectators
TEAM_MIN_FRAMES = 30  # tracklets shorter than this aren't used to find the clusters (they're still assigned)


def track_kits(track: np.ndarray, det: np.ndarray, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per tracklet: ids, mean kit (unit length; zero if none known) and frames with a known kit."""
    f = feats[det].astype(np.float32) / 255
    known = f.any(1)
    ids, inv = np.unique(track, return_inverse=True)
    sums = np.zeros((len(ids), f.shape[1]), np.float32)
    np.add.at(sums, inv[known], f[known])
    n = np.bincount(inv[known], minlength=len(ids))
    means = sums / (np.linalg.norm(sums, axis=1, keepdims=True) + 1e-9)
    return ids, means, n


def cluster_teams(means: np.ndarray, n: np.ndarray, k: int = TEAM_K, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted k-means (spherical: kits are unit vectors) of tracklet kits, fitted on tracklets with
    at least TEAM_MIN_FRAMES known frames, weighted by that count. Returns (cluster of each
    tracklet, -1 without a known kit; centres).
    """
    rng = np.random.default_rng(seed)
    fit = n >= TEAM_MIN_FRAMES
    X, w = means[fit], n[fit].astype(np.float64)
    if len(X) < k:
        return np.full(len(means), -1), np.zeros((0, means.shape[1]))
    # k-means++ start
    centres = [X[rng.choice(len(X), p=w / w.sum())]]
    for _ in range(1, k):
        d = 1 - np.max(X @ np.array(centres).T, axis=1)
        pr = w * np.maximum(d, 0) ** 2
        centres.append(X[rng.choice(len(X), p=pr / pr.sum())] if pr.sum() else X[rng.integers(len(X))])
    C = np.array(centres)
    for _ in range(50):
        lab = np.argmax(X @ C.T, axis=1)
        newC = np.array([(X[lab == j] * w[lab == j, None]).sum(0) if (lab == j).any() else C[j] for j in range(k)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9
        if np.allclose(newC, C):
            break
        C = newC
    out = np.argmax(means @ C.T, axis=1)
    out[n == 0] = -1
    return out, C


def load(store: Store, game: Game, model: str) -> np.ndarray | None:
    p = kit_path(store, game, model)
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["feats"]
