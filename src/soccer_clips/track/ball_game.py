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
                            classes=[BALL], verbose=False)
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

    if todo and workers > 1:
        # spawn, not fork: each worker starts its own CUDA context
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            for result in pool.imap_unordered(_detect_chunk, [(store.settings, game.id, model_name, s, e)
                                                              for s, e in todo]):
                add(*result)
    elif todo:
        from ultralytics import YOLO

        model = YOLO(models_dir(store) / f"{model_name}.pt")
        for s, e in todo:
            add(s, e, *detect_range(store, game, model, s, e, log))
    retrack(store, game)
    return detected


def retrack(store: Store, game: Game, p: ball_track.Params | None = None) -> int:
    """Rebuild ball.npz from the saved detections and camera motion. Returns frames with a ball."""
    raw = load_raw(store, game)
    order = np.argsort(raw["motion_frames"])
    mframes, motion = raw["motion_frames"][order], raw["motion"][order]
    out = []
    for s, e in raw["ranges"].tolist():
        sel = (mframes >= s) & (mframes < e)
        steps = motion[sel].copy()
        if not len(steps):
            continue
        steps[0] = np.eye(2, 3)
        d = raw["dets"][(raw["dets"][:, 0] >= s) & (raw["dets"][:, 0] < e)]
        boxes = np.column_stack([d[:, 0] - s, np.full(len(d), BALL), np.full(len(d), -1), d[:, 1:6]])
        path = ball_track.track(boxes, steps, p)
        keep = path[:, 4] > 0
        f = np.nonzero(keep)[0]
        out.append(np.column_stack([f + s, path[keep]]))
    rows = np.vstack(out) if out else np.empty((0, 6))
    save_npz(tracks_dir(store, game) / "ball.npz",
             frame=rows[:, 0].astype(np.int32),
             x=(rows[:, 1] / game.width).astype(np.float32), y=(rows[:, 2] / game.height).astype(np.float32),
             r=(rows[:, 3] / game.height).astype(np.float32), conf=rows[:, 4].astype(np.float32),
             kind=rows[:, 5].astype(np.uint8), ranges=raw["ranges"].astype(np.int32))
    return len(rows)


def load_path(store: Store, game: Game) -> dict | None:
    path = tracks_dir(store, game) / "ball.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
