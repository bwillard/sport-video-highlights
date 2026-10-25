"""
People tracks for games (Phase 3.2), in two passes like the ball:

1. detect (GPU, once): every person box in each frame, full frame at native resolution.
   Kept per model, so models can be compared on the same frames, in
   tracks/people_raw_<model>.npz: ranges (k, 2) [start, end) game frames; dets (m, 6)
   frame, x1, y1, x2, y2, conf (pixels).
2. track (CPU, re-runnable for free): the saved boxes, with the camera motion already in
   ball_raw.npz, go through a tracker into tracklets (stretches where one ID stays on one
   person), in tracks/people_<model>.npz: frame, track, x1, y1, x2, y2 (fractions of the
   frame width/height), conf, plus the ranges covered. Track ids are unique within a game.

Trackers ("tracker" in Params), both from ultralytics and fed saved boxes, no video:
  botsort    BoT-SORT's Kalman filter (box width/height) with our camera steps applied to
             the predictions, so a pan doesn't break every track (its own image-based
             camera-motion step is switched off). No re-ID: appearance comes in later, when
             tracklets are joined (Phase 4.3).
  bytetrack  ByteTrack, no camera motion: the baseline.

The yardstick (`evaluate`) is the callouts: a callout says where a player is at one moment of
a tagged clip. At each one, is there a box under it, and how much of the clip does that
box's track cover? Coverage can't tell a track that stayed on her from one that switched to
someone else, so it flatters trackers that hold on too long; the draw videos are the check.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

from soccer_clips.models import Clip, Game
from soccer_clips.store import Store
from soccer_clips.track import ball_game
from soccer_clips.track.ball_game import merge, save_npz, subtract, tracks_dir
from soccer_clips.track.benchmark import PERSON, models_dir

MODELS = ("yolo11m", "yolo26m", "yolo26l", "yolo26x")  # stock COCO weights
DEFAULT_MODEL = "yolo11m"  # best player tracking in the 2026-10-18 benchmark (m1920)
IMGSZ = 1920  # full frame at native resolution: players are only ~60-70 px tall
MIN_CONF = 0.1  # ByteTrack's second pass uses boxes down to 0.1
BATCH = 4
CALLOUT_PAD_S = 2.0
# GPU memory per worker at 1920, batch 4, full precision: rough, with room to spare. A worker
# that runs out anyway is redone on its own afterwards.
GPU_GB = {"yolo11m": 4.0, "yolo26m": 4.0, "yolo26l": 5.0, "yolo26x": 8.0}


def raw_path(store: Store, game: Game, model: str) -> Path:
    return tracks_dir(store, game) / f"people_raw_{model}.npz"


def path(store: Store, game: Game, model: str) -> Path:
    return tracks_dir(store, game) / f"people_{model}.npz"


def ranges_for(game: Game, mode: str) -> list[tuple[int, int]]:
    """'callouts': each clip with a callout, a little either side; 'clips' / 'all' as for the ball."""
    if mode != "callouts":
        return ball_game.ranges_for(game, mode)
    last = game.frame_at(game.duration_s) - 1
    return merge([(max(0, game.frame_at(c.start_s - CALLOUT_PAD_S)), min(last, game.frame_at(c.end_s + CALLOUT_PAD_S)))
                  for c in game.clips if c.callout])


# ---- 1. detection ----

def load_raw(store: Store, game: Game, model: str) -> dict:
    p = raw_path(store, game, model)
    if not p.exists():
        return {"ranges": np.empty((0, 2), int), "dets": np.empty((0, 6), np.float32)}
    with np.load(p) as z:
        return {"ranges": z["ranges"], "dets": z["dets"]}


def detect_range(store: Store, game: Game, model, start: int, end: int, log=print) -> tuple[np.ndarray, int]:
    """Person boxes (rows: game frame, x1, y1, x2, y2, conf) for [start, end), and frames read."""
    rows, batch, frames = [], [], []

    def flush():
        res = model.predict(batch, imgsz=IMGSZ, conf=MIN_CONF, classes=[PERSON], verbose=False)
        for fno, r in zip(frames, res):
            b = r.boxes.cpu().numpy()
            rows.extend((fno, *xyxy, c) for xyxy, c in zip(b.xyxy, b.conf))
        batch.clear()
        frames.clear()

    n = 0
    for n, frame in enumerate(ball_game.read_frames(store, game, start, end), 1):
        batch.append(frame)
        frames.append(start + n - 1)
        if len(batch) == BATCH:
            flush()
        if n % 3000 == 0:
            log(f"{game.id}: {n}/{end - start} frames of range {start}-{end}")
    if batch:
        flush()
    if n < end - start:
        log(f"{game.id}: only {n} of {end - start} frames could be read from {start}")
    return np.array(rows, np.float32).reshape(-1, 6), n


def _load_model(store: Store, name: str):
    from ultralytics import YOLO

    models_dir(store).mkdir(parents=True, exist_ok=True)
    return YOLO(models_dir(store) / f"{name}.pt")  # stock weights download themselves the first time


def _detect_chunk(args):
    """Worker: detect one chunk (its own process, with its own copy of the model)."""
    settings, game_id, model_name, s, e = args
    store = Store(settings)
    model = _load_model(store, model_name)
    return s, e, *detect_range(store, store.load_game(game_id), model, s, e, log=lambda m: print(m, flush=True))


def fit_workers(wanted: int, model_name: str, log=print) -> int:
    if wanted <= 1 or not ball_game._on_gpu():
        return wanted
    import torch

    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    n = max(1, min(wanted, int(free_gb // GPU_GB.get(model_name, 8.0))))
    if n < wanted:
        log(f"{free_gb:.0f} GB of GPU memory free: {n} worker(s) instead of {wanted}")
    return n


def detect_game(store: Store, game_id: str, mode: str = "callouts", model_name: str = DEFAULT_MODEL,
                workers: int = 1, log=print) -> int:
    """
    Detect people in what isn't done yet of this game's ranges; saved after each chunk, so a
    stopped run picks up where it left off. Returns frames detected.
    """
    import multiprocessing as mp

    game = store.load_game(game_id)
    raw = load_raw(store, game, model_name)
    have = merge([tuple(r) for r in raw["ranges"].tolist()])
    todo = ball_game.chunks(subtract(ranges_for(game, mode), have), workers)
    detected = 0

    def add(s, e, dets, n):
        nonlocal have, detected
        raw["dets"] = np.vstack([raw["dets"], dets])
        have = merge(have + [(s, s + n)])
        raw["ranges"] = np.array(have, int).reshape(-1, 2)
        save_npz(raw_path(store, game, model_name), **raw)
        detected += n
        log(f"{game.id}: {model_name} detected frames {s}-{e}")

    workers = fit_workers(workers, model_name, log) if todo else workers
    failed = []
    if todo and workers > 1:
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:  # spawn: a CUDA context each
            jobs = [(s, e, pool.apply_async(_detect_chunk, ((store.settings, game.id, model_name, s, e),)))
                    for s, e in todo]
            for s, e, job in jobs:
                try:
                    add(*job.get())
                except Exception as err:  # e.g. out of GPU memory: redo it below, one at a time
                    log(f"{game.id}: chunk {s}-{e} failed in a worker ({type(err).__name__}); will redo it")
                    failed.append((s, e))
    else:
        failed = todo
    if failed:
        model = _load_model(store, model_name)
        for s, e in failed:
            add(s, e, *detect_range(store, game, model, s, e, log))
    return detected


# ---- 2. tracking ----

@dataclass
class Params:
    tracker: str = "botsort"  # or "bytetrack" (see the module docstring)
    track_high_thresh: float = 0.25  # first-pass boxes
    track_low_thresh: float = 0.1  # second-pass boxes (occluded players are often weak)
    new_track_thresh: float = 0.25
    track_buffer: int = 30  # frames a lost track is kept to be found again
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5  # botsort: least box overlap to match
    # Kit colour (track/kit.py; needs people_kit_<model>.npz, else these are skipped). Distances are
    # Hellinger, 0 (same) to 1. Defaults tuned on the #38 labels, 2026-10-25 (see plans/phase4_identify.md).
    kit_gate: float = 0.45  # botsort: never match a track to a box whose kit is further than this (0 = off)
    split_kit: float = 0.35  # cut a tracklet where its kit changes by more than this (0 = off)
    link_gap_s: float = 1.0  # join a tracklet to one starting up to this long after it ends nearby (0 = off)
    link_kit: float = 0.2  # ...if their kits are at most this far apart
    link_heights: float = 1.0  # ...and the start is within this many box heights (plus LINK_SPEED a second)


class _Dets:
    """The bits of ultralytics' Boxes the trackers use."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, np.full(len(conf), float(PERSON))

    @property
    def xywh(self) -> np.ndarray:
        return np.column_stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                                self.xyxy[:, 2] - self.xyxy[:, 0], self.xyxy[:, 3] - self.xyxy[:, 1]])

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, i) -> "_Dets":
        return _Dets(self.xyxy[i], self.conf[i])


def _tracker(p: Params):
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
    from ultralytics.trackers.byte_tracker import BYTETracker, multi_gmc, parse_bboxes
    from ultralytics.trackers.utils import matching
    from ultralytics.utils import IterableSimpleNamespace

    args = IterableSimpleNamespace(**asdict(p), fuse_score=True, gmc_method="none", appearance_thresh=0.8,
                                   with_reid=False, model="auto")
    if p.tracker == "bytetrack":
        return BYTETracker(args)
    if p.tracker != "botsort":
        raise ValueError(f"unknown tracker {p.tracker!r}")

    def gate(dists, tracks, detections):
        # a box whose kit is too far from the track's (both known) can't be matched to it
        if p.kit_gate and len(tracks) and len(detections):
            emb = matching.embedding_distance(tracks, detections)  # cosine = Hellinger squared; 2 = unknown
            dists[(emb < 1.5) & (emb > p.kit_gate ** 2)] = 1.0
        return dists

    class CameraBOTSORT(BOTSORT):
        warp = None  # set before each update: our camera step into this frame

        def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
            if self.warp is not None:
                multi_gmc(strack_pool, self.warp)
                multi_gmc(unconfirmed, self.warp)

        def init_track(self, results, img=None):
            # `img` is the kit features of these boxes when update() is given feats
            if len(results) == 0:
                return []
            feats = img if isinstance(img, np.ndarray) and img.ndim == 2 else [None] * len(results)
            return [BOTrack(xywh, s, c, f if f is not None and f.any() else None)
                    for xywh, s, c, f in zip(parse_bboxes(results), results.conf, results.cls, feats)]

        def get_dists(self, tracks, detections):
            dists = matching.iou_distance(tracks, detections)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections)
            return gate(dists, tracks, detections)

        def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
            # as ByteTrack's (overlap only, for weak boxes), plus the kit gate: crossings are where it matters
            r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            if r_tracked and detections_second:
                dists = gate(matching.iou_distance(r_tracked, detections_second), r_tracked, detections_second)
                matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
                self._apply_matches(matches, r_tracked, detections_second, activated, refind)
            else:
                u_track = list(range(len(r_tracked)))
            for it in u_track:
                r_tracked[it].mark_lost()
                lost.append(r_tracked[it])

    return CameraBOTSORT(args)


def track_boxes(dets: np.ndarray, steps: np.ndarray, p: Params | None = None,
                feats: np.ndarray | None = None) -> np.ndarray:
    """
    Tracklets for one stretch of frames. dets: rows of frame (0 = first frame of the stretch),
    x1, y1, x2, y2, conf; steps: (n, 2, 3) camera step into each frame from the one before;
    feats: kit features of the dets (uint8, track/kit.py), used by the kit gate.
    Returns rows of frame, track (from 1), x1, y1, x2, y2, conf, det (index of the det the box
    came from, in `dets`).
    """
    p = p or Params()
    tracker = _tracker(p)
    order = np.argsort(dets[:, 0], kind="stable")
    sdets = dets[order]
    f32 = feats[order].astype(np.float32) / 255 if feats is not None and p.kit_gate else None
    starts = np.searchsorted(sdets[:, 0], np.arange(len(steps) + 1))
    out = []
    for f in range(len(steps)):
        a, b = starts[f], starts[f + 1]
        d = sdets[a:b]
        tracker.warp = steps[f] if f else None
        res = tracker.update(_Dets(d[:, 1:5], d[:, 5]), feats=None if f32 is None else f32[a:b])
        for x1, y1, x2, y2, tid, score, _, idx in res:
            out.append((f, tid, x1, y1, x2, y2, score, order[a + int(idx)]))
    return np.array(out, np.float64).reshape(-1, 8)


LINK_SPEED = 3.0  # box heights a second a player can cover (sprinting kids: ~4-5 m/s, ~1.4 m tall)
SPLIT_MIN = 15  # frames of kit either side of a cut


def _kit_unit(feats: np.ndarray) -> np.ndarray:
    return feats.astype(np.float32) / 255


def split_on_kit(rows: np.ndarray, feats: np.ndarray, thresh: float, next_id: int) -> tuple[np.ndarray, int]:
    """
    Cut tracklets where the kit changes: the frame where the mean kit before and after differ
    most, if by more than `thresh` (Hellinger), then the same for each part. rows as from
    track_boxes (column 7 = det index into `feats`); new pieces get ids from next_id.
    """
    rows = rows.copy()
    order = np.lexsort((rows[:, 0], rows[:, 1]))
    rows = rows[order]
    bounds = np.flatnonzero(np.diff(rows[:, 1])) + 1
    todo = [(a, b) for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(rows)])]
    while todo:
        a, b = todo.pop()
        if b - a < 2 * SPLIT_MIN:
            continue
        f = _kit_unit(feats[rows[a:b, 7].astype(int)])
        known = f.any(1)
        if known.sum() < 2 * SPLIT_MIN:
            continue
        c = np.cumsum(f * known[:, None], 0)
        n = np.cumsum(known)
        k = np.arange(SPLIT_MIN, b - a - SPLIT_MIN + 1)
        ok = (n[k - 1] >= SPLIT_MIN) & (n[-1] - n[k - 1] >= SPLIT_MIN)
        if not ok.any():
            continue
        k = k[ok]
        before = c[k - 1] / n[k - 1, None]
        after = (c[-1] - c[k - 1]) / (n[-1] - n[k - 1])[:, None]
        # means of square-rooted histograms aren't unit length; normalise before comparing
        before /= np.linalg.norm(before, axis=1, keepdims=True) + 1e-9
        after /= np.linalg.norm(after, axis=1, keepdims=True) + 1e-9
        d = np.linalg.norm(before - after, axis=1) / np.sqrt(2)
        j = int(np.argmax(d))
        if d[j] > thresh:
            cut = a + int(k[j])
            rows[cut:b, 1] = next_id
            next_id += 1
            todo += [(a, cut), (cut, b)]
    return rows[np.argsort(rows[:, 0], kind="stable")], next_id


def _cumulative(steps: np.ndarray) -> np.ndarray:
    """(n, 3, 3): frame 0's pixels to frame f's. Frame a to b is then cum[b] @ inv(cum[a])."""
    out = np.empty((len(steps), 3, 3))
    m = np.eye(3)
    for f, st in enumerate(steps):
        if f:
            m = np.vstack([st, [0, 0, 1]]) @ m
        out[f] = m
    return out


def link_tracklets(rows: np.ndarray, feats: np.ndarray, steps: np.ndarray, p: Params, fps: float) -> np.ndarray:
    """
    Join a tracklet to one that starts soon after it ends (within p.link_gap_s), near where the
    first would be by then (camera motion cancelled; within p.link_heights + LINK_SPEED a second,
    in box heights) and in the same kit (within p.link_kit, over their last / first second).
    Closest pairs first; each end and start used once. rows: frame relative to `steps`.
    """
    rows = rows.copy()
    ids = np.unique(rows[:, 1]).astype(int)
    if len(ids) < 2:
        return rows
    order = np.lexsort((rows[:, 0], rows[:, 1]))
    r = rows[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(r[:, 1])) + 1, len(r)]
    sec = int(round(fps))
    info = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        tid = int(r[a, 1])
        def kit_of(sl):
            f = _kit_unit(feats[r[sl, 7].astype(int)])
            f = f[f.any(1)]
            if not len(f):
                return None
            m = f.mean(0)
            return m / (np.linalg.norm(m) + 1e-9)
        info[tid] = {"first": int(r[a, 0]), "last": int(r[b - 1, 0]), "start": r[a, 2:6], "end": r[b - 1, 2:6],
                     "kit_start": kit_of(slice(a, min(b, a + sec))), "kit_end": kit_of(slice(max(a, b - sec), b))}
    gap = int(round(p.link_gap_s * fps))
    cum = _cumulative(steps)
    starts = sorted(info, key=lambda t: info[t]["first"])
    first_frames = np.array([info[t]["first"] for t in starts])
    pairs = []
    for ta, ia in info.items():
        if ia["kit_end"] is None:
            continue
        lo, hi = np.searchsorted(first_frames, ia["last"] + 1), np.searchsorted(first_frames, ia["last"] + gap + 1)
        for tb in starts[lo:hi]:
            ib = info[tb]
            if ib["kit_start"] is None:
                continue
            kd = float(np.linalg.norm(ia["kit_end"] - ib["kit_start"]) / np.sqrt(2))
            if kd > p.link_kit:
                continue
            x1, y1, x2, y2 = ia["end"]
            m = cum[ib["first"]] @ np.linalg.inv(cum[ia["last"]])
            fx, fy = m[0, 0] * (x1 + x2) / 2 + m[0, 1] * y2 + m[0, 2], m[1, 0] * (x1 + x2) / 2 + m[1, 1] * y2 + m[1, 2]
            bx1, by1, bx2, by2 = ib["start"]
            h = ((y2 - y1) + (by2 - by1)) / 2
            dt = (ib["first"] - ia["last"]) / fps
            d = np.hypot(fx - (bx1 + bx2) / 2, fy - by2) / max(h, 1.0)
            if d <= p.link_heights + LINK_SPEED * dt:
                pairs.append((d / (p.link_heights + LINK_SPEED * dt) + kd, ta, tb))
    parent = {t: t for t in ids.tolist()}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    used_end, used_start = set(), set()
    for _, ta, tb in sorted(pairs):
        if ta in used_end or tb in used_start or root(ta) == root(tb):
            continue
        used_end.add(ta)
        used_start.add(tb)
        parent[root(tb)] = root(ta)
    rows[:, 1] = [root(int(t)) for t in rows[:, 1]]
    return rows


def _motion(store: Store, game: Game) -> tuple[np.ndarray, np.ndarray]:
    raw = ball_game.load_raw(store, game)
    order = np.argsort(raw["motion_frames"])
    return raw["motion_frames"][order], raw["motion"][order]


def track_game(store: Store, game: Game, model: str = DEFAULT_MODEL, p: Params | None = None,
               save: bool = True, log=print) -> np.ndarray:
    """
    Tracklets for every detected range of the game: rows of game frame, track, x1, y1, x2, y2,
    conf (pixels). Camera steps come from ball_raw.npz; frames it doesn't cover count as a still
    camera. With `save`, also written to people_<model>.npz (as fractions of the frame).
    """
    from soccer_clips.track import kit

    p = p or Params()
    raw = load_raw(store, game, model)
    mframes, motion = _motion(store, game)
    feats = kit.load(store, game, model)
    if feats is None and (p.kit_gate or p.split_kit or p.link_gap_s):
        log(f"{game.id}: no kit colours yet (soccer-clips people kit), so tracking without them")
        p = Params(**{**asdict(p), "kit_gate": 0.0, "split_kit": 0.0, "link_gap_s": 0.0})
    fps = game.fps_num / game.fps_den
    out, next_id = [], 1
    for s, e in raw["ranges"].tolist():
        steps = np.repeat(np.eye(2, 3)[None], e - s, 0)
        i = np.searchsorted(mframes, np.arange(s, e))
        ok = (i < len(mframes)) & (mframes[np.minimum(i, len(mframes) - 1)] == np.arange(s, e)) if len(mframes) else \
            np.zeros(e - s, bool)
        steps[ok] = motion[i[ok]]
        if not ok.all():
            log(f"{game.id}: no camera motion for {int((~ok).sum())} of frames {s}-{e}; taken as still")
        sel = np.flatnonzero((raw["dets"][:, 0] >= s) & (raw["dets"][:, 0] < e))
        d = raw["dets"][sel].astype(np.float64)
        d[:, 0] -= s
        t = track_boxes(d, steps, p, None if feats is None or not p.kit_gate else feats[sel])
        if len(t):
            if p.split_kit:
                t, _ = split_on_kit(t, feats[sel], p.split_kit, int(t[:, 1].max()) + 1)
            if p.link_gap_s:
                t = link_tracklets(t, feats[sel], steps, p, fps)
            t[:, 7] = sel[t[:, 7].astype(int)]  # det index into the whole game's dets
            t[:, 0] += s
            ids, t[:, 1] = np.unique(t[:, 1], return_inverse=True)
            t[:, 1] += next_id
            next_id += len(ids)
        out.append(t)
    rows = np.vstack(out) if out else np.empty((0, 8))
    if save:
        W, H = game.width, game.height
        teams = _teams(rows[:, 1], rows[:, 7], feats) if feats is not None and len(rows) else {}
        save_npz(path(store, game, model), **teams, frame=rows[:, 0].astype(np.int32), track=rows[:, 1].astype(np.int32),
                 x1=(rows[:, 2] / W).astype(np.float32), y1=(rows[:, 3] / H).astype(np.float32),
                 x2=(rows[:, 4] / W).astype(np.float32), y2=(rows[:, 5] / H).astype(np.float32),
                 conf=rows[:, 6].astype(np.float32), det=rows[:, 7].astype(np.int32),
                 ranges=raw["ranges"].astype(np.int32))
    return rows


def _teams(track: np.ndarray, det: np.ndarray, feats: np.ndarray) -> dict:
    """Kit clusters of tracklets (kit.cluster_teams), as the arrays saved with them."""
    from soccer_clips.track import kit

    ids, means, n = kit.track_kits(track.astype(int), det.astype(int), feats)
    cluster, centres = kit.cluster_teams(means, n)
    return {"team_track": ids.astype(np.int32), "team_cluster": cluster.astype(np.int8),
            "team_centres": centres.astype(np.float32)}


def recluster(store: Store, game: Game, model: str = DEFAULT_MODEL) -> int:
    """Redo the kit clusters of the saved tracklets (no tracking); returns clusters found."""
    from soccer_clips.track import kit

    feats = kit.load(store, game, model)
    p = path(store, game, model)
    if feats is None or not p.exists():
        raise ValueError(f"{game.id}: needs saved tracklets and kit colours")
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    if "det" not in arrays:
        raise ValueError(f"{game.id}: tracked before kit colours were kept; run people track again")
    arrays.update(_teams(arrays["track"], arrays["det"], feats))
    save_npz(p, **arrays)
    return len(arrays["team_centres"])


def _track_one(args):
    settings, game_id, model, p = args
    store = Store(settings)
    rows = track_game(store, store.load_game(game_id), model, p, log=lambda m: print(m, flush=True))
    return game_id, len(rows), len(np.unique(rows[:, 1]))


def track_games(store: Store, game_ids: list[str], model: str = DEFAULT_MODEL, p: Params | None = None,
                workers: int = 1):
    """track_game (saved) for several games, `workers` at a time; yields (game id, boxes, tracklets)."""
    import multiprocessing as mp

    jobs = [(store.settings, g, model, p) for g in game_ids]
    if workers <= 1:
        yield from map(_track_one, jobs)
        return
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        yield from pool.imap_unordered(_track_one, jobs)


def load_tracks(store: Store, game: Game, model: str = DEFAULT_MODEL) -> tuple[np.ndarray, np.ndarray] | None:
    """Saved tracklets as (rows of game frame, track, x1, y1, x2, y2, conf in pixels, ranges), or None."""
    p = path(store, game, model)
    if not p.exists():
        return None
    with np.load(p) as z:
        W, H = game.width, game.height
        rows = np.column_stack([z["frame"], z["track"], z["x1"] * W, z["y1"] * H, z["x2"] * W, z["y2"] * H,
                                z["conf"]]).astype(np.float64)
        return rows, z["ranges"]


# ---- following a player (Phase 3.3) ----
# Clicks while watching say "this is #38" (kind "is") or "the marker is on the wrong person
# from here" (kind "not"). They're kept as clicks (frame and position), not track ids, in
# tracks/people_picks.json, so they still apply after the game is tracked again: each time,
# a click is matched to the tracklet whose box it falls in. A tracklet's frames are split at
# the clicks on it; each piece takes the label of the click that starts it, and the piece
# before the first click takes that click's label too.

class Tracks:
    """A game's saved tracklets in memory, sorted by frame (positions as fractions of the frame)."""

    def __init__(self, frame: np.ndarray, track: np.ndarray, box: np.ndarray, ranges: list,
                 team: dict[int, int] | None = None):
        self.team = team or {}  # track id -> kit cluster (kit.cluster_teams)
        order = np.argsort(frame, kind="stable")
        self.frame, self.track = frame[order].astype(np.int32), track[order].astype(np.int32)
        self.box = box[order].astype(np.float32)
        self.ranges = [list(map(int, r)) for r in ranges]
        ids, first = np.unique(self.track, return_index=True)  # rows are in frame order: first = start
        last = len(self.track) - 1 - np.unique(self.track[::-1], return_index=True)[1]
        self.first = dict(zip(ids.tolist(), self.frame[first].tolist()))
        self.last = dict(zip(ids.tolist(), self.frame[last].tolist()))

    @classmethod
    def load(cls, p: Path) -> "Tracks":
        with np.load(p) as z:
            team = dict(zip(z["team_track"].tolist(), z["team_cluster"].tolist())) if "team_track" in z.files else None
            return cls(z["frame"], z["track"], np.column_stack([z["x1"], z["y1"], z["x2"], z["y2"]]), z["ranges"], team)

    @classmethod
    def from_rows(cls, rows: np.ndarray, game: Game, ranges) -> "Tracks":
        """From track_game's rows (pixels)."""
        W, H = game.width, game.height
        return cls(rows[:, 0], rows[:, 1], rows[:, 2:6] / [W, H, W, H], ranges)

    def box_of(self, tid: int, frame: int) -> np.ndarray | None:
        a, b = self.window(frame, frame + 1)
        i = np.flatnonzero(self.track[a:b] == tid)
        return self.box[a + i[0]] if len(i) else None

    def window(self, start: int, end: int) -> tuple[int, int]:
        """Row slice for frames [start, end)."""
        return int(np.searchsorted(self.frame, start)), int(np.searchsorted(self.frame, end))

    def at(self, frame: int, x: float, y: float, prefer: int | None = None, slack: float = 0.01) -> int | None:
        """
        Track id of the box containing (x, y) at `frame`, or None. Of several: `prefer` if it's one
        of them (the track a click was recorded on, while the game hasn't been re-tracked), else the
        smallest box.
        """
        a, b = self.window(frame, frame + 1)
        bx = self.box[a:b]
        inside = ((bx[:, 0] - slack <= x) & (x <= bx[:, 2] + slack) & (bx[:, 1] - slack <= y) & (y <= bx[:, 3] + slack))
        if not inside.any():
            return None
        i = np.flatnonzero(inside)
        if prefer is not None and prefer in self.track[a + i]:
            return prefer
        area = (bx[i, 2] - bx[i, 0]) * (bx[i, 3] - bx[i, 1])
        return int(self.track[a + i[np.argmin(area)]])


_cache: dict = {}


def tracks(store: Store, game: Game, model: str = DEFAULT_MODEL) -> Tracks | None:
    """The game's saved tracklets, kept in memory (one game at a time; reloaded when re-tracked)."""
    p = path(store, game, model)
    if not p.exists():
        return None
    key = (str(p), p.stat().st_mtime)
    if key not in _cache:
        _cache.clear()
        _cache[key] = Tracks.load(p)
    return _cache[key]


def _picks_path(store: Store, game: Game) -> Path:
    return tracks_dir(store, game) / "people_picks.json"


def load_picks(store: Store, game: Game) -> list[dict]:
    import json

    p = _picks_path(store, game)
    return json.loads(p.read_text())["picks"] if p.exists() else []


def _save_picks(store: Store, game: Game, picks: list[dict]) -> None:
    import json

    from soccer_clips.store import atomic_write_text

    atomic_write_text(_picks_path(store, game), json.dumps({"picks": picks}, indent=1) + "\n")


def segments(t: Tracks, picks: list[dict]) -> list[dict]:
    """Labelled stretches of tracklets from the picks: {track, start, end (exclusive), player}."""
    by_track: dict[int, list[dict]] = {}
    for p in picks:
        tid = t.at(p["frame"], p["x"], p["y"], prefer=p.get("track"))
        if tid is not None:
            by_track.setdefault(tid, []).append(p)
    out = []
    for tid, ps in by_track.items():
        ps = sorted(ps, key=lambda p: p["frame"])  # a later click on the same frame wins (stable sort)
        edges = [t.first[tid]] + [p["frame"] for p in ps[1:]] + [t.last[tid] + 1]
        for (a, b), p in zip(zip(edges, edges[1:]), ps):
            if b > a and p["kind"] == "is":
                out.append({"track": tid, "start": a, "end": b, "player": p["player"]})
    return sorted(out, key=lambda s: s["start"])


def follows(t: Tracks, picks: list[dict], player: int, frame: int) -> int | None:
    """Track id labelled as `player` at `frame` (the marker), or None."""
    for s in segments(t, picks):
        if s["player"] == player and s["start"] <= frame < s["end"]:
            a, b = t.window(frame, frame + 1)
            if s["track"] in t.track[a:b]:
                return s["track"]
    return None


def add_pick(store: Store, game: Game, frame: int, player: int, kind: str, x: float | None = None,
             y: float | None = None, model: str = DEFAULT_MODEL) -> dict:
    """
    Record a click while watching. "is": the player under (x, y) is `player`. "not": the
    marker for `player` is on the wrong person from this frame on; without (x, y) the
    marker's own box is used. Raises ValueError if there's no tracked box there.
    """
    t = tracks(store, game, model)
    if t is None:
        raise ValueError("this game hasn't been through player tracking")
    picks = load_picks(store, game)
    if x is None or y is None:
        if kind != "not":
            raise ValueError("click on the player")
        tid = follows(t, picks, player, frame)
        if tid is None:
            raise ValueError(f"no marker for #{player} here")
        a, b = t.window(frame, frame + 1)
        x1, y1, x2, y2 = t.box[a + int(np.flatnonzero(t.track[a:b] == tid)[0])]
        x, y = float((x1 + x2) / 2), float((y1 + y2) / 2)
        tid = t.at(frame, x, y, prefer=tid)
    else:
        tid = t.at(frame, x, y)
    if tid is None:
        raise ValueError("no tracked player there")
    pick = {"frame": frame, "x": round(x, 4), "y": round(y, 4), "player": player, "kind": kind, "track": tid}
    _save_picks(store, game, picks + [pick])
    return pick


def undo_pick(store: Store, game: Game) -> dict | None:
    """Remove the last pick; returns it."""
    picks = load_picks(store, game)
    if not picks:
        return None
    _save_picks(store, game, picks[:-1])
    return picks[-1]


OURS_SHARE = 0.15


def team_roles(t: Tracks, picks: list[dict], game: Game, player: int | None) -> dict[int, str]:
    """
    Name the kit clusters: "ours" = every cluster with at least OURS_SHARE of `player`'s known
    points (picks and callouts; one kit often splits into a light and a shaded cluster),
    "opponents" = the biggest of the rest (by box count), the others "other" (refs, keepers,
    spectators). Without known points, the two biggest are "team" and the rest "other".
    """
    if not t.team:
        return {}
    counts: dict[int, int] = {}
    ids, n = np.unique(t.track, return_counts=True)
    for tid, c in zip(ids.tolist(), n.tolist()):
        k = t.team.get(tid, -1)
        if k >= 0:
            counts[k] = counts.get(k, 0) + c
    votes: dict[int, int] = {}
    if player is not None:
        pts = [(p["frame"], p["x"], p["y"]) for p in picks if p["player"] == player and p["kind"] == "is"]
        pts += [(game.frame_at(c.callout.t_s), c.callout.x, c.callout.y) for c in game.clips
                if c.callout and c.players == [player]]
        for f, x, y in pts:
            k = t.team.get(t.at(f, x, y), -1) if t.at(f, x, y) is not None else -1
            if k >= 0:
                votes[k] = votes.get(k, 0) + 1
    by_size = sorted(counts, key=counts.get, reverse=True)
    roles = {k: "other" for k in counts}
    if votes:
        total = sum(votes.values())
        ours = {k for k, v in votes.items() if v >= max(2, OURS_SHARE * total)} or {max(votes, key=votes.get)}
        for k in ours:
            roles[k] = "ours"
        rest = [k for k in by_size if k not in ours]
        if rest:
            roles[rest[0]] = "opponents"
    else:
        for k in by_size[:2]:
            roles[k] = "team"
    return roles


# ---- yardstick from your labels ----

def label_points(store: Store, game: Game, player: int) -> tuple[list[tuple], list[tuple]]:
    """Where `player` is known to be: (frame, x, y) of "is" picks and of callouts on clips tagged
    with only them; and (frame, x, y) of "not" picks (the marker had moved to someone else)."""
    picks = [p for p in load_picks(store, game) if p["player"] == player]
    known = [(p["frame"], p["x"], p["y"]) for p in picks if p["kind"] == "is"]
    known += [(game.frame_at(c.callout.t_s), c.callout.x, c.callout.y) for c in game.clips
              if c.callout and c.players == [player]]
    return sorted(known), sorted((p["frame"], p["x"], p["y"]) for p in picks if p["kind"] == "not")


def check_labels(t: Tracks, known: list[tuple], wrong: list[tuple], fps: float) -> dict:
    """
    Against points where the player is known to be: hold = how long the tracklet under each
    lasts; conflicts = points that a tracklet under another point passes through but doesn't
    cover (it wandered onto someone else, or was joined to someone else); swaps left = "not"
    picks still on a tracklet that's under an earlier known point.
    """
    under = [(f, x, y, t.at(f, x, y)) for f, x, y in known if any(a <= f < b for a, b in t.ranges)]
    holds = [(t.last[k] - t.first[k] + 1) / fps for *_, k in under if k is not None]
    conflicts = 0
    for k in {k for *_, k in under if k is not None}:
        for f, x, y, _ in under:
            if t.first[k] <= f <= t.last[k]:
                b = t.box_of(k, f)
                if b is not None and not (b[0] - 0.01 <= x <= b[2] + 0.01 and b[1] - 0.01 <= y <= b[3] + 0.01):
                    conflicts += 1
    swaps = 0
    for f, x, y in wrong:
        k = t.at(f, x, y)
        swaps += k is not None and any(kk == k and ff < f for ff, _, _, kk in under)
    return {"points": len(under), "found": sum(k is not None for *_, k in under), "holds": holds,
            "conflicts": conflicts, "wrong": len(wrong), "swaps_left": swaps}


def evaluate_labels(store: Store, model: str = DEFAULT_MODEL, p: Params | None = None, player: int = 38,
                    saved: bool = False, games: list[str] | None = None, log=print) -> dict:
    """check_labels on every game (of `games`) with labels for `player`: tracked afresh with `p`, or the saved tracklets."""
    per = {}
    for game in store.list_games():
        if games and game.id not in games:
            continue
        known, wrong = label_points(store, game, player)
        if not known:
            continue
        if saved:
            t = tracks(store, game, model)
        else:
            raw = load_raw(store, game, model)
            t = Tracks.from_rows(track_game(store, game, model, p, save=False, log=log), game, raw["ranges"].tolist())
        if t is not None:
            per[game.id] = check_labels(t, known, wrong, game.fps_num / game.fps_den)
    holds = [h for r in per.values() for h in r["holds"]]
    tot = {k: sum(r[k] for r in per.values()) for k in ("points", "found", "conflicts", "wrong", "swaps_left")}
    return {"games": per, **tot, "hold_median_s": round(median(holds), 1) if holds else None,
            "hold_mean_s": round(sum(holds) / len(holds), 1) if holds else None}


# ---- yardstick ----

GAP_S = 1.0  # a run around the callout can have gaps up to this long


def callout_check(rows: np.ndarray, game: Game, clip: Clip) -> dict:
    """
    For a clip with a callout, given tracklet rows (game frame, track, x1, y1, x2, y2, ...): is
    there a box under the callout, and how long does its track last? The click can land just
    beside the box (between the legs, on the shadow), so failing a box containing it, the
    nearest box centre within the callout ring counts. Overlapping boxes: the smallest.
    """
    f = game.frame_at(clip.callout.t_s)
    x, y = clip.callout.x * game.width, clip.callout.y * game.height
    at = rows[rows[:, 0] == f]
    inside = at[(at[:, 2] <= x) & (x <= at[:, 4]) & (at[:, 3] <= y) & (y <= at[:, 5])]
    if not len(inside) and len(at):
        d = np.hypot((at[:, 2] + at[:, 4]) / 2 - x, (at[:, 3] + at[:, 5]) / 2 - y)
        if d.min() <= clip.callout.r * game.height:
            inside = at[[d.argmin()]]
    if not len(inside):
        return {"clip": clip.id, "found": False}
    box = inside[np.argmin((inside[:, 4] - inside[:, 2]) * (inside[:, 5] - inside[:, 3]))]
    a, b = game.frame_at(clip.start_s), game.frame_at(clip.end_s)
    present = np.unique(rows[rows[:, 1] == box[1], 0]).astype(int)
    in_clip = present[(present >= a) & (present < b)]
    # the run around the callout with no gap longer than GAP_S
    gap = round(GAP_S * game.fps_num / game.fps_den)
    k = int(np.searchsorted(present, f))
    lo = hi = k
    while lo > 0 and present[lo] - present[lo - 1] <= gap:
        lo -= 1
    while hi + 1 < len(present) and present[hi + 1] - present[hi] <= gap:
        hi += 1
    fps = game.fps_num / game.fps_den
    return {"clip": clip.id, "found": True, "track": int(box[1]), "box_h_px": round(float(box[5] - box[3]), 1),
            "coverage": round(len(in_clip) / max(1, b - a), 3),
            "run_s": round((present[hi] - present[lo] + 1) / fps, 1)}


def evaluate(store: Store, model: str = DEFAULT_MODEL, p: Params | None = None, players: set[int] | None = None,
             saved: bool = False, log=print) -> dict:
    """
    Track every game with saved boxes for `model` (not saved), or with `saved` use the saved
    tracklets as they are, and check each callout on a clip tagged with exactly one player (of
    `players`, if given) inside the covered ranges. Returns per-callout results and a summary.
    """
    checks, frames, tracks, boxes = [], 0, 0, 0
    for game in store.list_games():
        if saved:
            got = load_tracks(store, game, model)
            if got is None:
                continue
            rows, ranges = got[0], got[1].tolist()
        else:
            raw = load_raw(store, game, model)
            if not len(raw["ranges"]):
                continue
            rows = track_game(store, game, model, p, save=False, log=log)
            ranges = raw["ranges"].tolist()
        frames += sum(e - s for s, e in ranges)
        tracks += len(np.unique(rows[:, 1]))
        boxes += len(rows)
        for c in game.clips:
            if not c.callout or len(c.players) != 1 or (players and c.players[0] not in players):
                continue
            if not any(s <= game.frame_at(c.start_s) and game.frame_at(c.end_s) <= e for s, e in ranges):
                continue
            checks.append({"game": game.id, "player": c.players[0], **callout_check(rows, game, c)})
    found = [c for c in checks if c["found"]]
    minutes = frames / (30000 / 1001) / 60
    summary = {
        "callouts": len(checks),
        "found": round(len(found) / len(checks), 3) if checks else None,
        "coverage_median": round(median(c["coverage"] for c in found), 3) if found else None,
        "coverage_mean": round(sum(c["coverage"] for c in found) / len(checks), 3) if checks else None,
        "run_s_median": round(median(c["run_s"] for c in found), 1) if found else None,
        "tracks_per_min": round(tracks / minutes, 1) if frames else None,
        "people_per_frame": round(boxes / frames, 1) if frames else None,
    }
    return {"summary": summary, "callouts": checks}
