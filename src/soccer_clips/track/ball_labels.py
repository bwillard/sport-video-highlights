"""
Hand-labelled ball positions (Phase 4.1), for measuring ball detection and,
later, fine-tuning a ball model.

Frames are grouped in sets, kept apart so a model is never scored on frames
it was trained on:
  bench  frames spread over the bench clips; for scoring only
  train  random frames from the games, away from the bench clips

<labels_dir>/ball_tasks.json  the frames to label, per set
<labels_dir>/ball.jsonl       one line per answer, latest wins (append-only)
<labels_dir>/ball_guesses.json  a model's best guess per frame, shown as a
                              suggestion when labelling (never for bench)
<labels_dir>/frames/          cached frame images
Frames are numbered in game frames, so labels don't depend on clip files.
"""

import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from soccer_clips.config import Settings
from soccer_clips.models import Game
from soccer_clips.store import Store, _lock_for, atomic_write_text
from soccer_clips.track.bench import load_set

SetName = Literal["bench", "train"]
# visible: the ball is at x, y. hidden: it's behind someone/something, roughly at x, y.
# none: no ball in the frame. skip: can't tell.
State = Literal["visible", "hidden", "none", "skip"]


class BallTask(BaseModel):
    game_id: str
    frame: int  # game frame
    clip: str | None = None  # bench clip it comes from


class BallLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    frame: int
    state: State | Literal["cleared"]
    x: float | None = Field(default=None, ge=0, le=1)  # fraction of frame width
    y: float | None = Field(default=None, ge=0, le=1)  # fraction of frame height
    r: float | None = Field(default=None, gt=0, le=0.1)  # radius as a fraction of frame height; None = not sized
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Tasks(BaseModel):
    sets: dict[str, list[BallTask]] = {}


def labels_dir(settings: Settings) -> Path:
    return settings.games_dir.parent / "labels"


def _tasks_path(settings: Settings) -> Path:
    return labels_dir(settings) / "ball_tasks.json"


def _labels_path(settings: Settings) -> Path:
    return labels_dir(settings) / "ball.jsonl"


def load_tasks(settings: Settings) -> Tasks:
    path = _tasks_path(settings)
    return Tasks.model_validate_json(path.read_text()) if path.exists() else Tasks()


def save_tasks(settings: Settings, tasks: Tasks) -> None:
    atomic_write_text(_tasks_path(settings), tasks.model_dump_json(indent=1) + "\n")


def load_labels(settings: Settings) -> dict[tuple[str, int], BallLabel]:
    """Latest answer per (game, frame); cleared ones dropped."""
    path = _labels_path(settings)
    latest: dict[tuple[str, int], BallLabel] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                label = BallLabel.model_validate_json(line)
                latest[(label.game_id, label.frame)] = label
    return {k: v for k, v in latest.items() if v.state != "cleared"}


def add_label(settings: Settings, label: BallLabel) -> None:
    path = _labels_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path), open(path, "a") as fh:
        fh.write(label.model_dump_json() + "\n")


def sample_bench(store: Store, n: int = 150) -> list[BallTask]:
    """n frames spread evenly over the bench clips, in proportion to their length."""
    clips = load_set(store.settings).clips
    total_s = sum(c.end_s - c.start_s for c in clips)
    tasks = []
    for c in clips:
        game = store.load_game(c.game_id)
        first, last = game.frame_at(c.start_s), game.frame_at(c.end_s)
        k = max(1, round(n * (c.end_s - c.start_s) / total_s))
        step = (last - first) / k
        tasks += [BallTask(game_id=c.game_id, frame=first + int(step * (i + 0.5)), clip=c.name) for i in range(k)]
    return tasks


def sample_train(store: Store, n: int, seed: int | None = None) -> list[BallTask]:
    """
    n random frames: half from tagged clips (where the action is), half from
    anywhere in play (kickoff to the end). Frames within 10s of a bench clip
    are never picked.
    """
    rng = random.Random(seed)
    games = store.list_games()
    bench = [(c.game_id, c.start_s - 10, c.end_s + 10) for c in load_set(store.settings).clips]

    def near_bench(game: Game, t: float) -> bool:
        return any(g == game.id and lo <= t <= hi for g, lo, hi in bench)

    tasks, tries = [], 0
    while len(tasks) < n and tries < n * 50:
        tries += 1
        game = rng.choice(games)
        if game.clips and rng.random() < 0.5:
            c = rng.choice(game.clips)
            t = rng.uniform(c.start_s, c.end_s)
        else:
            t = rng.uniform(game.kickoff_1h_s or 0, game.duration_s - 1)
        if not near_bench(game, t):
            tasks.append(BallTask(game_id=game.id, frame=game.frame_at(t)))
    return tasks


def add_tasks(store: Store, set_name: SetName, new: list[BallTask]) -> int:
    """Append tasks to a set, skipping frames already in any set; returns how many were added."""
    tasks = load_tasks(store.settings)
    seen = {(t.game_id, t.frame) for ts in tasks.sets.values() for t in ts}
    fresh = []
    for t in new:  # also drops repeats within `new`
        if (t.game_id, t.frame) not in seen:
            seen.add((t.game_id, t.frame))
            fresh.append(t)
    tasks.sets.setdefault(set_name, []).extend(fresh)
    save_tasks(store.settings, tasks)
    return len(fresh)


BENCH_SHARE = 5  # one moment in this many goes to the benchmark
MOMENT_S = 30.0  # frames of the same play this close together go to the same set
NEAR_BENCH_S = 10.0  # frames this close to a bench clip always go to the benchmark


def auto_set(store: Store, game_id: str, frame: int) -> str:
    """
    Which label set a frame labelled while watching (a ball fix) should join. Fixes are the
    model's mistakes: the best training material, so most go to "train". One 30-second moment
    in five goes to "bench", so the benchmark also covers hard cases. A whole moment goes to
    one set (frames of the same play in both would flatter the benchmark), and anything
    near a bench clip goes to "bench" for the same reason. A frame already in a set stays there.
    """
    import hashlib

    for name, ts in load_tasks(store.settings).sets.items():
        if any(t.game_id == game_id and t.frame == frame for t in ts):
            return name
    game = store.load_game(game_id)
    for c in load_set(store.settings).clips:
        if c.game_id == game_id and game.frame_at(c.start_s - NEAR_BENCH_S) <= frame <= game.frame_at(c.end_s + NEAR_BENCH_S):
            return "bench"
    moment = frame // game.frame_at(MOMENT_S)
    h = int(hashlib.sha1(f"{game_id}:{moment}".encode()).hexdigest()[:8], 16)
    return "bench" if h % BENCH_SHARE == 0 else "train"


def frame_image(store: Store, game: Game, frame: int) -> Path:
    """One frame of the original as a high-quality JPEG, cached."""
    path = labels_dir(store.settings) / "frames" / game.id / f"{frame}.jpg"
    with _lock_for(path):  # the page prefetches, so the same frame can be asked for twice at once
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".part.jpg")
        seek = max(0.0, (frame - 0.5) * game.fps_den / game.fps_num)
        subprocess.run([store.settings.ffmpeg, "-v", "error", "-y", "-ss", f"{seek:.6f}",
                        "-i", str(store.original_path(game)), "-frames:v", "1", "-q:v", "2", str(tmp)],
                       check=True)
        tmp.replace(path)
    return path


# ---- scoring detections against the bench labels ----

MIN_TOL_PX = 10  # a detection counts if its centre is within this (or 2 ball radii) of the label


def on_ball(label: BallLabel, game: Game, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Which detection centres (pixels) lie on the labelled ball, visible or hidden."""
    if label.state not in ("visible", "hidden"):
        return np.zeros(len(cx), bool)
    tol = max(MIN_TOL_PX, 2 * (label.r or 0) * game.height)
    return np.hypot(cx - label.x * game.width, cy - label.y * game.height) <= tol


def new_tallies(confs) -> dict:
    keys = ("frames", "visible", "found", "hidden", "hidden_found", "dets", "on_game", "on_other", "junk")
    return {c: dict.fromkeys(keys, 0) for c in confs}


def tally(tallies: dict, label: BallLabel, conf: np.ndarray, on_game: np.ndarray,
          on_other: np.ndarray | None = None, on_unsure: np.ndarray | None = None) -> None:
    """
    Add one labelled frame's detections: their confidences, and which lie on the game ball,
    on another real ball (fine: the temporal filter drops those) or on a candidate marked
    unsure (left out). Everything else is junk.
    """
    none = np.zeros(len(conf), bool)
    on_other = none if on_other is None else on_other
    on_unsure = none if on_unsure is None else on_unsure
    for c, t in tallies.items():
        keep = (conf >= c) & ~on_unsure
        t["frames"] += 1
        t["dets"] += int(keep.sum())
        t["on_game"] += int((keep & on_game).sum())
        t["on_other"] += int((keep & on_other).sum())
        t["junk"] += int((keep & ~on_game & ~on_other).sum())
        if label.state == "visible":
            t["visible"] += 1
            t["found"] += int((keep & on_game).any())
        elif label.state == "hidden":
            t["hidden"] += 1
            t["hidden_found"] += int((keep & on_game).any())


def summarise(tallies: dict) -> dict:
    """recall: share of visible game balls found. precision: share of detections that are real balls
    (game or other). junk_per_frame: detections that aren't balls, per labelled frame."""
    ratio = lambda a, b: round(a / b, 3) if b else None
    return {str(c): {"recall": ratio(t["found"], t["visible"]),
                     "precision": ratio(t["on_game"] + t["on_other"], t["dets"]),
                     "junk_per_frame": ratio(t["junk"], t["frames"]),
                     "hidden_found": ratio(t["hidden_found"], t["hidden"]),
                     "visible": t["visible"], "hidden": t["hidden"], "dets": t["dets"], "other": t["on_other"]}
            for c, t in tallies.items()}


def score(store: Store, results: Path, confs=(0.1, 0.25, 0.4)) -> dict:
    """
    Per variant folder under `results`, against the bench labels: the share of visible
    game balls with a detection at >= conf on them (recall), the share of detections on a
    real ball, game or other (precision), and non-ball detections per frame (junk).
    """
    from soccer_clips.track import ball_others

    settings = store.settings
    labels = load_labels(settings)
    others = ball_others.load(settings)
    bench = {(t.game_id, t.frame): t for t in load_tasks(settings).sets.get("bench", [])}
    clips = {c.name: c for c in load_set(settings).clips}
    games: dict[str, Game] = {}
    out = {}
    for vdir in sorted(p for p in results.iterdir() if p.is_dir()):
        tallies = new_tallies(confs)
        for (gid, frame), task in bench.items():
            label = labels.get((gid, frame))
            if label is None or label.state == "skip" or task.clip not in clips:
                continue
            npz = vdir / f"{task.clip}.npz"
            if not npz.exists():
                continue
            game = games.setdefault(gid, store.load_game(gid))
            local = frame - game.frame_at(clips[task.clip].start_s)
            arr = np.load(npz)["boxes"]
            balls = arr[(arr[:, 1] == 32) & (arr[:, 0] == local)]
            masks = ball_others.classify(others, label, game, (balls[:, 3] + balls[:, 5]) / 2,
                                         (balls[:, 4] + balls[:, 6]) / 2)
            tally(tallies, label, balls[:, 7], *masks)
        out[vdir.name] = summarise(tallies)
    return out


def score_table(scores: dict) -> str:
    rows = ["variant              conf  found  real  junk/frame  hidden  (visible/hidden/dets/other balls)"]
    f = lambda v: "   –" if v is None else f"{v:4.0%}"
    for variant, by_conf in scores.items():
        for c, s in by_conf.items():
            junk = "   –" if s.get("junk_per_frame") is None else f"{s['junk_per_frame']:4.2f}"
            rows.append(f"{variant:20} {c:>4}  {f(s['recall'])}  {f(s['precision'])}        {junk}    {f(s['hidden_found'])}"
                        f"  ({s['visible']}/{s['hidden']}/{s['dets']}/{s.get('other', 0)})")
    return "\n".join(rows)


def summary(store: Store) -> list[dict]:
    labels = load_labels(store.settings)
    return [{"name": name, "total": len(ts), "labelled": sum((t.game_id, t.frame) in labels for t in ts)}
            for name, ts in load_tasks(store.settings).sets.items()]


def tasks_with_labels(store: Store, set_name: str) -> list[dict]:
    labels = load_labels(store.settings)
    from soccer_clips.track import ball_others

    # Bench labels score the models, so they must not be nudged by a model's guess.
    guesses = {} if set_name == "bench" else load_guesses(store.settings).frames
    others = ball_others.load(store.settings).frames
    return [{**t.model_dump(),
             "label": labels[(t.game_id, t.frame)].model_dump(mode="json") if (t.game_id, t.frame) in labels else None,
             "guess": g.model_dump() if (g := guesses.get(f"{t.game_id}:{t.frame}")) else None,
             # confirmed other balls, with their index for removing one
             "others": [{"index": i, **b.model_dump()} for i, b in enumerate(others.get(f"{t.game_id}:{t.frame}", []))
                        if b.verdict == "ball"]}
            for t in load_tasks(store.settings).sets.get(set_name, [])]


# ---- model guesses, to speed up labelling ----

GUESS_VARIANT = "m1280-centre"  # best stock ball variant on the bench (2026-10-19)
GUESS_MIN_CONF = 0.1


class Guess(BaseModel):
    x: float  # fractions, like BallLabel
    y: float
    r: float
    conf: float


class Guesses(BaseModel):
    model: str = ""
    # "game_id:frame" -> best guess, or None when the model found no ball
    frames: dict[str, Guess | None] = {}


def _guesses_path(settings: Settings) -> Path:
    return labels_dir(settings) / "ball_guesses.json"


def load_guesses(settings: Settings) -> Guesses:
    path = _guesses_path(settings)
    return Guesses.model_validate_json(path.read_text()) if path.exists() else Guesses()


def make_guesses(store: Store, set_name: str, log=print) -> int:
    """
    Run the guess model over every frame of a set that has no guess yet, saving
    as it goes so the labelling page can use them while this runs. Returns how
    many frames were guessed.
    """
    import cv2
    from ultralytics import YOLO

    from soccer_clips.track.benchmark import BALL, VARIANTS, models_dir

    if set_name == "bench":
        raise ValueError("no guesses for the bench set: its labels score the models")
    variant = VARIANTS[GUESS_VARIANT]
    settings = store.settings
    guesses = load_guesses(settings)
    if guesses.model and guesses.model != GUESS_VARIANT:
        guesses = Guesses()  # a different model's guesses: start over
    guesses.model = GUESS_VARIANT
    todo = [t for t in load_tasks(settings).sets.get(set_name, []) if f"{t.game_id}:{t.frame}" not in guesses.frames]
    if not todo:
        return 0
    models_dir(store).mkdir(parents=True, exist_ok=True)
    model = YOLO(models_dir(store) / variant.model)
    games: dict[str, Game] = {}
    for n, t in enumerate(todo, 1):
        game = games.setdefault(t.game_id, store.load_game(t.game_id))
        frame = cv2.imread(str(frame_image(store, game, t.frame)))
        ox, oy, w, h = variant.crop
        res = model.predict(frame[oy:oy + h, ox:ox + w], imgsz=variant.imgsz, conf=GUESS_MIN_CONF,
                            classes=[BALL], verbose=False)[0].boxes.cpu().numpy()
        best = None
        if len(res):
            (x1, y1, x2, y2), conf = res.xyxy[res.conf.argmax()], float(res.conf.max())
            best = Guess(x=float((x1 + x2) / 2 + ox) / game.width, y=float((y1 + y2) / 2 + oy) / game.height,
                         r=float(max(x2 - x1, y2 - y1) / 2) / game.height, conf=round(conf, 3))
        guesses.frames[f"{t.game_id}:{t.frame}"] = best
        if n % 10 == 0 or n == len(todo):
            atomic_write_text(_guesses_path(settings), guesses.model_dump_json() + "\n")
            log(f"guessed {n}/{len(todo)}")
    return len(todo)
