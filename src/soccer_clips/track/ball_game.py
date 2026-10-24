"""
Game-ball tracks for whole games, or the parts of them that matter (Phase 3.2 / 4.1).

For each frame range: frames are read from the original video at exact positions
(ffmpeg, so frame numbers match the rest of the app), the fine-tuned detector runs
on the centre crop, camera motion is estimated, and ball_track turns the
detections into one game-ball path.

Per game, in games/<year>/<game_id>/tracks/:
  ball_raw.npz  the expensive part, kept so the path can be redone with new
                settings without detecting again: ranges (k, 2) [start, end) game
                frames; dets (m, 6) frame, x1, y1, x2, y2, conf; motion_frames (n,)
                and motion (n, 2, 3) camera steps (step into each frame from the one before)
  ball.npz      the path: frame, x, y, r (fractions: x of width, y and r of height),
                conf, kind (1 detected, 2 filled), plus the ranges it covers
  ball_fixes.json  corrections made while watching (see `fix`): balls placed by hand
                (they win their frame) and suppressed detections (a wrong ring's run);
                applied every time the path is rebuilt
Frames inside a range with no entry had no ball.

To run somewhere the full game videos aren't (a GPU pod), `prepare_segments` cuts just
the ranges still to do into <data root>/ballsrc/<game_id>/<start>-<end>.mp4 (re-encoded at crf 18, the quality the bench clips have;
re-encode, exact frames); `read_frames` uses such a segment when one matches the range.
"""

import subprocess
from pathlib import Path

import numpy as np

from soccer_clips import ffmpeg
from soccer_clips.models import Game
from soccer_clips.store import Store
from soccer_clips.track import ball_track, camera
from soccer_clips.track.benchmark import BALL, models_dir

CROP = (320, 180, 1280, 720)  # centre crop, as the ball model was trained
CLIP_PAD_S = 2.0  # track a little either side of each tagged clip
MIN_CONF = 0.05
BATCH = 8


def tracks_dir(store: Store, game: Game) -> Path:
    return store.game_dir(game.year, game.id) / "tracks"


def merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def subtract(want: list[tuple[int, int]], have: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Parts of `want` not covered by `have` (both merged)."""
    out = []
    for s, e in want:
        cur = s
        for hs, he in have:
            if he <= cur or hs >= e:
                continue
            if hs > cur:
                out.append((cur, hs))
            cur = max(cur, he)
        if cur < e:
            out.append((cur, e))
    return out


def ranges_for(game: Game, mode: str) -> list[tuple[int, int]]:
    """'clips': every tagged clip plus a little either side. 'all': kickoff (if sane) to the end."""
    last = game.frame_at(game.duration_s) - 1  # the stated duration can round one frame past the real end
    if mode == "all":
        k1, k2 = game.kickoff_1h_s, game.kickoff_2h_s
        start = k1 - 60 if k1 is not None and (k2 is None or k1 < k2) else 0.0
        return [(max(0, game.frame_at(start)), last)]
    return merge([(max(0, game.frame_at(c.start_s - CLIP_PAD_S)), min(last, game.frame_at(c.end_s + CLIP_PAD_S)))
                  for c in game.clips])


def segment_path(store: Store, game: Game, start: int, end: int) -> Path:
    return store.settings.games_dir.parent / "ballsrc" / game.id / f"{start}-{end}.mp4"


def prepare_segments(store: Store, game: Game, mode: str, log=print) -> list[Path]:
    """Cut the ranges this game still needs into segment files; returns their paths."""
    have = merge([tuple(r) for r in load_raw(store, game)["ranges"].tolist()])
    fps = game.fps_num / game.fps_den
    out = []
    for s, e in subtract(ranges_for(game, mode), have):
        path = segment_path(store, game, s, e)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            seek_s, duration_s = ffmpeg.cut_window(game, s / fps, e / fps)
            tmp = path.with_name(path.stem + ".part.mp4")
            subprocess.run([store.settings.ffmpeg, "-v", "error", "-y", "-ss", f"{seek_s:.6f}",
                            "-i", str(store.original_path(game)), "-frames:v", str(e - s), "-map", "0:v:0",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                            str(tmp)], check=True)
            tmp.replace(path)
        out.append(path)
    log(f"{game.id}: {len(out)} segment(s) ready")
    return out


def read_frames(store: Store, game: Game, start: int, end: int):
    """Yield frames start..end-1 as (h, w, 3) BGR arrays, from a prepared segment or the original."""
    w, h = game.width, game.height
    segment = segment_path(store, game, start, end)
    if segment.exists():
        cmd = [store.settings.ffmpeg, "-v", "error", "-i", str(segment)]
    else:
        fps = game.fps_num / game.fps_den
        seek_s, duration_s = ffmpeg.cut_window(game, start / fps, end / fps)
        cmd = [store.settings.ffmpeg, "-v", "error", "-ss", f"{seek_s:.6f}", "-i", str(store.original_path(game)),
               "-frames:v", str(end - start)]
    proc = subprocess.Popen(cmd + ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        for _ in range(end - start):
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def detect_range(store: Store, game: Game, model, start: int, end: int, log=print):
    """Detections (rows: game frame, x1, y1, x2, y2, conf) and camera steps for [start, end)."""
    ox, oy, cw, ch = CROP
    cam = camera.Tracker()
    rows, steps, batch, frames = [], [], [], []

    def flush():
        res = model.predict([f[oy:oy + ch, ox:ox + cw] for f in batch], imgsz=cw, conf=MIN_CONF,
                            classes=[BALL], verbose=False)  # full precision, by choice (2026-10-22)
        for fno, r in zip(frames, res):
            b = r.boxes.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(b.xyxy, b.conf):
                rows.append((fno, x1 + ox, y1 + oy, x2 + ox, y2 + oy, c))
        batch.clear()
        frames.clear()

    n = 0
    for n, frame in enumerate(read_frames(store, game, start, end), 1):
        steps.append(cam.step(frame))
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
    steps_arr = np.array(steps).reshape(-1, 2, 3)
    if len(steps_arr):
        steps_arr[0] = np.eye(2, 3)  # nothing before the range
    return np.array(rows, np.float64).reshape(-1, 6), np.arange(start, start + n), steps_arr


def load_raw(store: Store, game: Game) -> dict:
    path = tracks_dir(store, game) / "ball_raw.npz"
    if not path.exists():
        return {"ranges": np.empty((0, 2), int), "dets": np.empty((0, 6)), "motion_frames": np.empty(0, int),
                "motion": np.empty((0, 2, 3))}
    with np.load(path) as z:
        return {k: z[k] for k in ("ranges", "dets", "motion_frames", "motion")}


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


MIN_CHUNK = 3600  # frames (2 min): smallest piece worth its own worker
GPU_GB_PER_WORKER = 6.0  # YOLO26x at 1280, batch 8, full precision (~4.7 GB), with room to spare


def _on_gpu() -> bool:
    import torch

    return torch.cuda.is_available()


def fit_workers(wanted: int, log=print) -> int:
    """As many workers as the GPU's free memory allows (6 x ~4.7 GB ran a 24 GB 4090 out of memory)."""
    if wanted <= 1 or not _on_gpu():
        return wanted
    import torch

    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    n = max(1, min(wanted, int(free_gb // GPU_GB_PER_WORKER)))
    if n < wanted:
        log(f"{free_gb:.0f} GB of GPU memory free: {n} worker(s) instead of {wanted}")
    return n


def chunks(ranges: list[tuple[int, int]], workers: int) -> list[tuple[int, int]]:
    """Split ranges into about `workers` pieces each (none shorter than MIN_CHUNK), in order."""
    out = []
    for s, e in ranges:
        n = max(1, min(workers, (e - s) // MIN_CHUNK))
        edges = np.linspace(s, e, n + 1).round().astype(int)
        out += [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if b > a]
    return out


def _detect_chunk(args):
    """Worker: detect one chunk (runs in its own process, with its own copy of the model)."""
    settings, game_id, model_name, s, e = args
    from ultralytics import YOLO

    store = Store(settings)
    game = store.load_game(game_id)
    model = YOLO(models_dir(store) / f"{model_name}.pt")
    return s, e, *detect_range(store, game, model, s, e, log=lambda m: print(m, flush=True))


def track_game(store: Store, game_id: str, mode: str = "clips", model_name: str = "ball_yolo26x",
               workers: int = 1, log=print) -> int:
    """
    Detect what isn't done yet for this game's ranges, then redo the path. Returns frames
    detected. With workers > 1, long ranges are split into chunks detected in parallel
    processes (one CPU core each does decoding and camera motion; they share the GPU). The
    first frame of each chunk has no camera step from the frame before, which the path
    filter doesn't notice.
    """
    import multiprocessing as mp

    game = store.load_game(game_id)
    raw = load_raw(store, game)
    have = merge([tuple(r) for r in raw["ranges"].tolist()])
    todo = chunks(subtract(ranges_for(game, mode), have), workers)
    detected = 0

    def add(s, e, dets, mframes, steps):
        nonlocal have, detected
        raw["dets"] = np.vstack([raw["dets"], dets])
        raw["motion_frames"] = np.concatenate([raw["motion_frames"], mframes])
        raw["motion"] = np.concatenate([raw["motion"], steps])
        have = merge(have + [(s, s + len(mframes))])
        raw["ranges"] = np.array(have, int).reshape(-1, 2)
        save_npz(tracks_dir(store, game) / "ball_raw.npz", **raw)  # after each piece: resumable
        detected += len(mframes)
        log(f"{game.id}: detected frames {s}-{e}")

    workers = fit_workers(workers, log) if todo else workers
    failed = []
    if todo and workers > 1:
        # spawn, not fork: each worker starts its own CUDA context
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
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
        from ultralytics import YOLO

        model = YOLO(models_dir(store) / f"{model_name}.pt")
        for s, e in failed:
            add(s, e, *detect_range(store, game, model, s, e, log))
    retrack(store, game)
    return detected


FIX_CONF = 3.0  # a hand-placed ball outweighs any detection
FIX_R = 0.006  # its radius when not given (fraction of frame height, ~6 px)
SUPPRESS_PX = 15  # detections this close to a suppressed point are dropped
WRONG_RUN = 45  # a "wrong ring" fix suppresses the path's detections up to this many frames either side
FIX_WINDOW = 3600  # a fix redoes the path this many frames either side (2 min)


def _fixes_path(store: Store, game: Game) -> Path:
    return tracks_dir(store, game) / "ball_fixes.json"


def load_fixes(store: Store, game: Game) -> dict:
    import json

    path = _fixes_path(store, game)
    return json.loads(path.read_text()) if path.exists() else {"balls": [], "suppress": []}


def _apply_fixes(d: np.ndarray, fixes: dict, game: Game, s: int, e: int) -> np.ndarray:
    """Detection rows (frame, x1, y1, x2, y2, conf) for [s, e) with hand fixes applied."""
    W, H = game.width, game.height
    cx, cy = (d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2
    drop = np.zeros(len(d), bool)
    for p in fixes["suppress"]:
        if s <= p["frame"] < e:
            drop |= (d[:, 0] == p["frame"]) & (np.hypot(cx - p["x"] * W, cy - p["y"] * H) <= SUPPRESS_PX)
    added = []
    for b in fixes["balls"]:
        if s <= b["frame"] < e:
            drop |= d[:, 0] == b["frame"]  # the hand-placed ball is the only one that frame
            r = (b.get("r") or FIX_R) * H
            added.append((b["frame"], b["x"] * W - r, b["y"] * H - r, b["x"] * W + r, b["y"] * H + r, FIX_CONF))
    return np.vstack([d[~drop], np.array(added, float).reshape(-1, 6)])


def usable_dets(store: Store, game: Game, raw: dict) -> np.ndarray:
    """The game's ball detections, less those on fixed objects (world.find_fixed), if marked."""
    from soccer_clips.track import world

    blocked = world.load_blocked(store, game, len(raw["dets"]))
    return raw["dets"] if blocked is None else raw["dets"][~blocked]


def retrack(store: Store, game: Game, p: ball_track.Params | None = None,
            window: tuple[int, int] | None = None) -> int:
    """
    Rebuild ball.npz from the saved detections, camera motion and hand fixes. With `window`
    (game frames [a, b)), only that part is redone and the rest of the saved path is kept.
    Returns frames with a ball.
    """
    raw = load_raw(store, game)
    fixes = load_fixes(store, game)
    order = np.argsort(raw["motion_frames"])
    mframes, motion = raw["motion_frames"][order], raw["motion"][order]
    out, done = [], []
    dets = usable_dets(store, game, raw)
    for s, e in raw["ranges"].tolist():
        if window:
            s, e = max(s, window[0]), min(e, window[1])
            if s >= e:
                continue
        sel = (mframes >= s) & (mframes < e)
        steps = motion[sel].copy()
        if not len(steps):
            continue
        steps[0] = np.eye(2, 3)
        d = dets[(dets[:, 0] >= s) & (dets[:, 0] < e)]
        d = _apply_fixes(d, fixes, game, s, e)
        boxes = np.column_stack([d[:, 0] - s, np.full(len(d), BALL), np.full(len(d), -1), d[:, 1:6]])
        path = ball_track.track(boxes, steps, p)
        keep = path[:, 4] > 0
        f = np.nonzero(keep)[0]
        out.append(np.column_stack([f + s, path[keep]]))
        done.append((s, e))
    rows = np.vstack(out) if out else np.empty((0, 6))
    rows = np.column_stack([rows[:, 0], rows[:, 1] / game.width, rows[:, 2] / game.height, rows[:, 3] / game.height,
                            rows[:, 4], rows[:, 5]])
    old = load_path(store, game) if window else None
    if old is not None:  # keep the saved path outside the redone stretch
        keep = np.ones(len(old["frame"]), bool)
        for s, e in done:
            keep &= ~((old["frame"] >= s) & (old["frame"] < e))
        kept = np.column_stack([old[k][keep] for k in ("frame", "x", "y", "r", "conf", "kind")])
        rows = np.vstack([kept, rows])
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
    save_npz(tracks_dir(store, game) / "ball.npz",
             frame=rows[:, 0].astype(np.int32), x=rows[:, 1].astype(np.float32), y=rows[:, 2].astype(np.float32),
             r=rows[:, 3].astype(np.float32), conf=rows[:, 4].astype(np.float32),
             kind=rows[:, 5].astype(np.uint8), ranges=raw["ranges"].astype(np.int32))
    return len(rows)


def evaluate(store: Store, p: ball_track.Params | None = None, fixes: bool = False) -> dict:
    """
    Score whole-game ball paths at every labelled frame in the tracked games (the train and
    bench labels come from them): visible labels found / wrong / missed, and none labels with
    a ball claimed. Paths are rebuilt from the saved detections, by default without hand
    fixes (many fixed frames are labels too, so with them the score would be circular).
    Bench clips are labelled densely, so one moment can move the score by several points.
    """
    from soccer_clips.track import ball_labels

    labels = ball_labels.load_labels(store.settings)
    out = {}
    for game in store.list_games():
        if not (tracks_dir(store, game) / "ball_raw.npz").exists():
            continue
        raw = load_raw(store, game)
        mine = {f: l for (gid, f), l in labels.items() if gid == game.id and l.state in ("visible", "none")}
        n = dict.fromkeys(["visible", "found", "wrong", "missed", "none", "claimed"], 0)
        order = np.argsort(raw["motion_frames"])
        mframes, motion = raw["motion_frames"][order], raw["motion"][order]
        hand = load_fixes(store, game) if fixes else {"balls": [], "suppress": []}
        dets = usable_dets(store, game, raw)
        for s, e in raw["ranges"].tolist():
            todo = [f for f in mine if s <= f < e]
            if not todo:
                continue
            steps = motion[(mframes >= s) & (mframes < e)].copy()
            steps[0] = np.eye(2, 3)
            d = _apply_fixes(dets[(dets[:, 0] >= s) & (dets[:, 0] < e)], hand, game, s, e)
            boxes = np.column_stack([d[:, 0] - s, np.full(len(d), BALL), np.full(len(d), -1), d[:, 1:6]])
            path = ball_track.track(boxes, steps, p)
            for f in todo:
                x, y, _, _, kind = path[f - s]
                if mine[f].state == "none":
                    n["none"] += 1
                    n["claimed"] += int(kind != ball_track.NONE)
                    continue
                n["visible"] += 1
                if kind == ball_track.NONE:
                    n["missed"] += 1
                else:
                    n["found" if ball_labels.on_ball(mine[f], game, np.array([x]), np.array([y]))[0] else "wrong"] += 1
        out[game.id] = n
    return out


def fix(store: Store, game: Game, frame: int, action: str, x: float | None = None, y: float | None = None,
        r: float | None = None, label_set: str | None = None) -> str | None:
    """
    A correction while watching: action "ball" (it's at x, y) or "none" (the ring is wrong and
    there's no ball to see). Suppresses the wrong ring's run of detections nearby, records the
    fix, redoes the path around it, and optionally adds the frame as a labelled ball frame to
    a label set: "train", "bench", or "auto" (ball_labels.auto_set picks). Returns the set used.
    """
    import json

    from soccer_clips.store import atomic_write_text

    raw = load_raw(store, game)
    if not any(s <= frame < e for s, e in raw["ranges"].tolist()):
        raise ValueError(f"frame {frame} hasn't been tracked")
    fixes = load_fixes(store, game)
    path = load_path(store, game)
    W, H = game.width, game.height
    at = np.nonzero(path["frame"] == frame)[0] if path is not None else []
    if len(at):
        i = int(at[0])
        wrong = action == "none" or np.hypot((path["x"][i] - x) * W, (path["y"][i] - y) * H) > 40
        if wrong:  # suppress the run of detections the wrong ring came from, around this frame
            lo = hi = i
            while lo > 0 and path["frame"][lo - 1] == path["frame"][lo] - 1 and frame - path["frame"][lo - 1] <= WRONG_RUN:
                lo -= 1
            while (hi + 1 < len(path["frame"]) and path["frame"][hi + 1] == path["frame"][hi] + 1
                   and path["frame"][hi + 1] - frame <= WRONG_RUN):
                hi += 1
            for k in range(lo, hi + 1):
                if path["kind"][k] == ball_track.DETECTED:
                    fixes["suppress"].append({"frame": int(path["frame"][k]), "x": float(path["x"][k]),
                                              "y": float(path["y"][k])})
    if action == "ball":
        fixes["balls"] = [b for b in fixes["balls"] if b["frame"] != frame]
        fixes["balls"].append({"frame": frame, "x": x, "y": y, "r": r})
    atomic_write_text(_fixes_path(store, game), json.dumps(fixes, indent=1) + "\n")
    retrack(store, game, window=(frame - FIX_WINDOW, frame + FIX_WINDOW))
    if action == "ball":
        return _label_frame(store, game, frame, label_set, "visible", x, y, r)
    return _label_frame(store, game, frame, label_set, "none")


def _label_frame(store: Store, game: Game, frame: int, label_set: str | None, state: str,
                 x: float | None = None, y: float | None = None, r: float | None = None) -> str | None:
    """Add the frame to a label set ("auto": ball_labels.auto_set picks) with this game-ball label."""
    from soccer_clips.track import ball_labels

    if label_set == "auto":
        label_set = ball_labels.auto_set(store, game.id, frame)
    if label_set:
        ball_labels.frame_image(store, game, frame)  # so the labelling page can show it
        ball_labels.add_tasks(store, label_set, [ball_labels.BallTask(game_id=game.id, frame=frame)])
        ball_labels.add_label(store.settings, ball_labels.BallLabel(
            game_id=game.id, frame=frame, state=state, x=x, y=y, r=r))
    return label_set


def other_ball(store: Store, game: Game, frame: int, x: float, y: float, r: float | None = None,
               label_set: str | None = "auto") -> tuple[int, str | None]:
    """
    Mark another real ball (a spare, the next field) in a frame while watching, for training:
    the exporter boxes it, and scoring doesn't count detections on it as mistakes. Only
    labelled frames are used, so an unlabelled frame is labelled first, taking the ring as
    the game ball (it must be a detected or placed one; a filled-in guess isn't trusted).
    The path is unchanged. Returns (index of the entry, label set the frame was added to).
    """
    from soccer_clips.track import ball_labels, ball_others

    used = None
    if (game.id, frame) not in ball_labels.load_labels(store.settings):
        if not label_set:
            raise ValueError("labelling is switched off, so other balls wouldn't be used")
        path = load_path(store, game)
        at = np.nonzero(path["frame"] == frame)[0] if path is not None else []
        if not len(at) or path["kind"][at[0]] != ball_track.DETECTED:
            raise ValueError("mark the game ball in this frame first (K, or Shift+K for none)")
        i = int(at[0])
        used = _label_frame(store, game, frame, label_set, "visible", float(path["x"][i]), float(path["y"][i]),
                            min(0.1, float(path["r"][i])) or None)
    k = ball_others.key(game.id, frame)
    others = ball_others.edit(store.settings, lambda o: o.frames.setdefault(k, []).append(
        ball_others.OtherBall(x=x, y=y, r=r, verdict="ball")))
    return len(others.frames[k]) - 1, used


def load_path(store: Store, game: Game) -> dict | None:
    path = tracks_dir(store, game) / "ball.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
