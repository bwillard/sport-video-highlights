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
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker, multi_gmc
    from ultralytics.utils import IterableSimpleNamespace

    args = IterableSimpleNamespace(**asdict(p), fuse_score=True, gmc_method="none", appearance_thresh=0.8,
                                   with_reid=False, model="auto")
    if p.tracker == "bytetrack":
        return BYTETracker(args)
    if p.tracker != "botsort":
        raise ValueError(f"unknown tracker {p.tracker!r}")

    class CameraBOTSORT(BOTSORT):
        warp = None  # set before each update: our camera step into this frame

        def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
            if self.warp is not None:
                multi_gmc(strack_pool, self.warp)
                multi_gmc(unconfirmed, self.warp)

    return CameraBOTSORT(args)


def track_boxes(dets: np.ndarray, steps: np.ndarray, p: Params | None = None) -> np.ndarray:
    """
    Tracklets for one stretch of frames. dets: rows of frame (0 = first frame of the stretch),
    x1, y1, x2, y2, conf; steps: (n, 2, 3) camera step into each frame from the one before.
    Returns rows of frame, track (from 1), x1, y1, x2, y2, conf.
    """
    p = p or Params()
    tracker = _tracker(p)
    order = np.argsort(dets[:, 0], kind="stable")
    dets = dets[order]
    starts = np.searchsorted(dets[:, 0], np.arange(len(steps) + 1))
    out = []
    for f in range(len(steps)):
        d = dets[starts[f]:starts[f + 1]]
        tracker.warp = steps[f] if f else None
        for x1, y1, x2, y2, tid, score, *_ in tracker.update(_Dets(d[:, 1:5], d[:, 5])):
            out.append((f, tid, x1, y1, x2, y2, score))
    return np.array(out, np.float64).reshape(-1, 7)


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
    raw = load_raw(store, game, model)
    mframes, motion = _motion(store, game)
    out, next_id = [], 1
    for s, e in raw["ranges"].tolist():
        steps = np.repeat(np.eye(2, 3)[None], e - s, 0)
        i = np.searchsorted(mframes, np.arange(s, e))
        ok = (i < len(mframes)) & (mframes[np.minimum(i, len(mframes) - 1)] == np.arange(s, e)) if len(mframes) else \
            np.zeros(e - s, bool)
        steps[ok] = motion[i[ok]]
        if not ok.all():
            log(f"{game.id}: no camera motion for {int((~ok).sum())} of frames {s}-{e}; taken as still")
        d = raw["dets"][(raw["dets"][:, 0] >= s) & (raw["dets"][:, 0] < e)].astype(np.float64)
        d[:, 0] -= s
        t = track_boxes(d, steps, p)
        if len(t):
            t[:, 0] += s
            ids, t[:, 1] = np.unique(t[:, 1], return_inverse=True)
            t[:, 1] += next_id
            next_id += len(ids)
        out.append(t)
    rows = np.vstack(out) if out else np.empty((0, 7))
    if save:
        W, H = game.width, game.height
        save_npz(path(store, game, model), frame=rows[:, 0].astype(np.int32), track=rows[:, 1].astype(np.int32),
                 x1=(rows[:, 2] / W).astype(np.float32), y1=(rows[:, 3] / H).astype(np.float32),
                 x2=(rows[:, 4] / W).astype(np.float32), y2=(rows[:, 5] / H).astype(np.float32),
                 conf=rows[:, 6].astype(np.float32), ranges=raw["ranges"].astype(np.int32))
    return rows


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
