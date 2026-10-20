"""
Other real balls in labelled frames: spares by the sideline, balls on the next
field. They are not the game ball, but they are balls, so a detector should
find them; picking out the game ball is the temporal filter's job. Training
boxes them too, and scoring doesn't count detections on them as mistakes.

Candidates come from a model run over the labelled frames (`find`); each gets
a verdict in the review page (ball / not / unsure). Balls can also be added
by hand on the labelling page.

<labels_dir>/ball_others.json:
  {"frames": {"<game_id>:<frame>": [OtherBall, ...]}, "scanned": {"<model>": ["<game_id>:<frame>", ...]}}
"""

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from soccer_clips.config import Settings
from soccer_clips.models import Game
from soccer_clips.store import Store, _lock_for, atomic_write_text
from soccer_clips.track import ball_labels

Verdict = Literal["ball", "not", "unsure"]
FIND_CONF = 0.2  # weakest candidate shown for review


class OtherBall(BaseModel):
    x: float = Field(ge=0, le=1)  # fractions, like BallLabel
    y: float = Field(ge=0, le=1)
    r: float | None = Field(default=None, gt=0, le=0.1)
    conf: float | None = None  # model confidence; None when added by hand
    source: str = "manual"  # model name, or "manual"
    verdict: Verdict | None = None  # None = not reviewed yet; manual ones are "ball"


class Others(BaseModel):
    frames: dict[str, list[OtherBall]] = {}
    scanned: dict[str, list[str]] = {}  # model -> frames it has been run on


def _path(settings: Settings) -> Path:
    return ball_labels.labels_dir(settings) / "ball_others.json"


def load(settings: Settings) -> Others:
    path = _path(settings)
    return Others.model_validate_json(path.read_text()) if path.exists() else Others()


def save(settings: Settings, others: Others) -> None:
    atomic_write_text(_path(settings), others.model_dump_json() + "\n")


def key(game_id: str, frame: int) -> str:
    return f"{game_id}:{frame}"


def edit(settings: Settings, fn) -> Others:
    """Load, change with fn(others), save, under one lock (the scan and the page both write)."""
    with _lock_for(_path(settings)):
        others = load(settings)
        fn(others)
        save(settings, others)
        return others


def balls(others: Others, game_id: str, frame: int) -> list[OtherBall]:
    """Confirmed other balls in a frame."""
    return [b for b in others.frames.get(key(game_id, frame), []) if b.verdict == "ball"]


def _near(x: float, y: float, r: float | None, cx: np.ndarray, cy: np.ndarray, game: Game) -> np.ndarray:
    tol = max(ball_labels.MIN_TOL_PX, 2 * (r or 0) * game.height)
    return np.hypot(cx - x * game.width, cy - y * game.height) <= tol


def classify(others: Others, label: ball_labels.BallLabel, game: Game, cx: np.ndarray, cy: np.ndarray):
    """Masks over detection centres (pixels): on the game ball, on another real ball, on an unsure candidate."""
    on_game = ball_labels.on_ball(label, game, cx, cy)
    on_other = np.zeros(len(cx), bool)
    on_unsure = np.zeros(len(cx), bool)
    for b in others.frames.get(key(label.game_id, label.frame), []):
        if b.verdict == "ball":
            on_other |= _near(b.x, b.y, b.r, cx, cy, game)
        elif b.verdict == "unsure":
            on_unsure |= _near(b.x, b.y, b.r, cx, cy, game)
    return on_game, on_other & ~on_game, on_unsure & ~on_game & ~on_other


def find(store: Store, model_name: str = "ball_yolo26x", imgsz: int = 1920, conf: float = FIND_CONF,
         log=print) -> int:
    """
    Run a model over every labelled frame (bench first, then train; skipping frames it has
    already done) and add each ball detection that isn't on the game ball or an existing
    entry as a candidate to review. Saves after every frame, so review can start right away.
    """
    import cv2
    from ultralytics import YOLO

    from soccer_clips.track.benchmark import BALL, models_dir

    settings = store.settings
    labels = ball_labels.load_labels(settings)
    tasks = [t for name in ("bench", "train") for t in ball_labels.load_tasks(settings).sets.get(name, [])]
    done = set(load(settings).scanned.get(model_name, []))
    todo = [t for t in tasks if (t.game_id, t.frame) in labels and labels[(t.game_id, t.frame)].state != "skip"
            and key(t.game_id, t.frame) not in done]
    if not todo:
        return 0
    model = YOLO(models_dir(store) / f"{model_name}.pt")
    games: dict[str, Game] = {}
    added = 0
    for n, t in enumerate(todo, 1):
        game = games.setdefault(t.game_id, store.load_game(t.game_id))
        label = labels[(t.game_id, t.frame)]
        img = cv2.imread(str(ball_labels.frame_image(store, game, t.frame)))
        boxes = model.predict(img, imgsz=imgsz, conf=conf, classes=[BALL], verbose=False)[0].boxes.cpu().numpy()
        cx = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) / 2
        cy = (boxes.xyxy[:, 1] + boxes.xyxy[:, 3]) / 2
        rs = np.maximum(boxes.xyxy[:, 2] - boxes.xyxy[:, 0], boxes.xyxy[:, 3] - boxes.xyxy[:, 1]) / 2

        def add(others: Others) -> None:
            nonlocal added
            k = key(t.game_id, t.frame)
            entries = others.frames.setdefault(k, [])
            on_game = ball_labels.on_ball(label, game, cx, cy)
            for i in np.argsort(-boxes.conf):
                if on_game[i] or any(_near(e.x, e.y, e.r, cx[i:i + 1], cy[i:i + 1], game)[0] for e in entries):
                    continue
                entries.append(OtherBall(x=float(cx[i] / game.width), y=float(cy[i] / game.height),
                                         r=min(0.1, float(rs[i] / game.height)), conf=round(float(boxes.conf[i]), 3),
                                         source=model_name))
                added += 1
            if not entries:
                del others.frames[k]
            others.scanned.setdefault(model_name, []).append(k)

        edit(settings, add)
        if n % 10 == 0 or n == len(todo):
            log(f"scanned {n}/{len(todo)} frames, {added} candidate(s) so far")
    return added


def review_list(store: Store) -> list[dict]:
    """Every model candidate, in task order, with what the review page needs."""
    settings = store.settings
    labels = ball_labels.load_labels(settings)
    others = load(settings)
    out = []
    for name in ("bench", "train"):
        for t in ball_labels.load_tasks(settings).sets.get(name, []):
            label = labels.get((t.game_id, t.frame))
            for i, b in enumerate(others.frames.get(key(t.game_id, t.frame), [])):
                if b.source != "manual":
                    out.append({"set": name, "game_id": t.game_id, "frame": t.frame, "index": i, **b.model_dump(),
                                "label": label.model_dump(mode="json") if label else None})
    return out
