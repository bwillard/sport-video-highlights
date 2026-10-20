"""
Fine-tune a ball detector on our own labels (Phase 4.1) and score models on
the bench labels.

export  turns train-set labels into a YOLO dataset of 1280x720 pieces of the
        frame at native resolution (the size the detector sees at inference):
        the centre crop, plus one random crop around the ball. Visible balls
        become boxes; "none" frames are negatives; hidden and can't-tell frames
        are left out. Other real balls (spares, the next field; see ball_others)
        are boxed too, so the detector isn't taught that they aren't balls.
        The ball keeps its COCO class (32) in an 80-class dataset,
        so the pretrained head, which already knows "sports ball", carries over.
train   fine-tunes a YOLO base on that dataset (needs a GPU to be quick).
evaluate  runs a model on the bench frame images and scores it like `ball score`.
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from soccer_clips.store import Store
from soccer_clips.track import ball_labels, ball_others
from soccer_clips.track.benchmark import BALL, models_dir

CROP_W, CROP_H = 1280, 720
CENTRE = (320, 180)  # top-left of the centre crop in a 1920x1080 frame
COCO_NAMES = [  # the 80 COCO classes, in order; only "sports ball" (32) is labelled
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def dataset_dir(store: Store) -> Path:
    return ball_labels.labels_dir(store.settings) / "dataset"


def radius_model(labels: list[ball_labels.BallLabel]) -> np.ndarray:
    """Fit ball radius ~ a + b*y (fractions of frame height) on sized labels: farther = higher up = smaller."""
    sized = [(l.y, l.r) for l in labels if l.r and l.y is not None]
    if len(sized) < 10:
        return np.array([0.0, 0.008])  # ~9 px at 1080p
    y, r = np.array(sized).T
    b, a = np.polyfit(y, r, 1)
    return np.array([b, a])


def _crop_origin(x: float, y: float, rng: random.Random, w: int, h: int) -> tuple[int, int]:
    """Top-left of a random crop that contains (x, y) with a margin, inside the frame."""
    m = 40  # keep the ball this far inside the crop where possible; near the frame edge it can't be

    def pick(c: float, size: int, full: int) -> int:
        clamp = lambda v: min(max(v, 0), full - size)
        return rng.randint(clamp(int(c) + m - size), clamp(int(c) - m))

    return pick(x, CROP_W, w), pick(y, CROP_H, h)


def export(store: Store, val_frac: float = 0.2, seed: int = 0, log=print) -> Path:
    settings = store.settings
    labels = ball_labels.load_labels(settings)
    tasks = ball_labels.load_tasks(settings).sets.get("train", [])
    usable = [(t, labels[(t.game_id, t.frame)]) for t in tasks
              if (t.game_id, t.frame) in labels and labels[(t.game_id, t.frame)].state in ("visible", "none")]
    fit = radius_model([l for _, l in usable if l.state == "visible"])
    others = ball_others.load(settings)
    rng = random.Random(seed)
    rng.shuffle(usable)
    n_val = round(len(usable) * val_frac)
    out = dataset_dir(store)
    shutil.rmtree(out, ignore_errors=True)
    counts = {"train": [0, 0], "val": [0, 0]}  # images, boxes
    games = {}
    for k, (task, label) in enumerate(usable):
        split = "val" if k < n_val else "train"
        game = games.setdefault(task.game_id, store.load_game(task.game_id))
        img = cv2.imread(str(ball_labels.frame_image(store, game, task.frame)))
        h, w = img.shape[:2]
        origins = [CENTRE]
        if split == "train":  # one more crop at a random place, around the ball if there is one
            if label.state == "visible":
                origins.append(_crop_origin(label.x * w, label.y * h, rng, w, h))
            else:
                origins.append((rng.randint(0, w - CROP_W), rng.randint(0, h - CROP_H)))
        balls = [(b.x, b.y, b.r) for b in ball_others.balls(others, task.game_id, task.frame)]
        if label.state == "visible":
            balls.insert(0, (label.x, label.y, label.r))
        for j, (ox, oy) in enumerate(origins):
            lines = []
            for bx, by, br in balls:
                r = (br or float(np.polyval(fit, by))) * h
                cx, cy = bx * w - ox, by * h - oy
                if 0 <= cx < CROP_W and 0 <= cy < CROP_H:
                    side = max(4.0, 2 * r)
                    lines.append(f"{BALL} {cx / CROP_W:.6f} {cy / CROP_H:.6f} {side / CROP_W:.6f} {side / CROP_H:.6f}")
            stem = f"{task.game_id}_{task.frame}_{j}"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img[oy:oy + CROP_H, ox:ox + CROP_W],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[split][0] += 1
            counts[split][1] += len(lines)
    # Which train-set frames went to validation: checkpoints are chosen on these (never the bench).
    (out / "val_frames.json").write_text(json.dumps([[t.game_id, t.frame] for t, _ in usable[:n_val]]))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_NAMES))
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    log(f"dataset in {out}: train {counts['train'][0]} images / {counts['train'][1]} balls, "
        f"val {counts['val'][0]} / {counts['val'][1]} (radius fit r = {fit[1]:.4f} + {fit[0]:.4f}*y)")
    return out


SELECT_CONF = "0.25"  # the confidence a checkpoint is judged at


def selection_score(s: dict) -> float:
    """Harmonic mean of found and real at SELECT_CONF: finding the game ball, without junk."""
    r, p = s[SELECT_CONF]["recall"] or 0, s[SELECT_CONF]["precision"] or 0
    return 2 * r * p / (r + p) if r + p else 0.0


def train(store: Store, base: str, epochs: int = 60, batch: int = 8, imgsz: int = 1280, every: int = 5,
          log=print) -> Path:
    """
    Fine-tune `base` (e.g. yolo26x) on the exported dataset, saving a checkpoint every
    `every` epochs, then keep the one that scores best on the validation frames by our own
    measure (selection_score). Ultralytics' own pick goes by box fit (mostly mAP50-95) on
    the validation crops, which chose epoch 2 and 6 on our first run, and its early
    stopping is off for the same reason. The kept weights go to models/ball_<base>.pt and
    the comparison to labels/runs/<base>/selection.json.
    """
    from ultralytics import YOLO

    data = dataset_dir(store) / "data.yaml"
    if not data.exists():
        raise FileNotFoundError(f"no dataset at {data}; run `soccer-clips ball export` first")
    # The dataset may have been exported on another machine: point it at where it is now.
    lines = data.read_text().splitlines()
    data.write_text("\n".join([f"path: {dataset_dir(store)}"] + lines[1:]) + "\n")
    models_dir(store).mkdir(parents=True, exist_ok=True)
    model = YOLO(models_dir(store) / f"{base}.pt")
    runs = ball_labels.labels_dir(store.settings) / "runs"
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, patience=epochs + 1,
                save_period=every, project=str(runs), name=base, exist_ok=True, plots=False, verbose=False)
    weights = runs / base / "weights"
    val = [tuple(v) for v in json.loads((dataset_dir(store) / "val_frames.json").read_text())]
    candidates = sorted(weights.glob("epoch*.pt"), key=lambda p: int(p.stem[5:])) + [weights / "last.pt", weights / "best.pt"]
    results = {}
    for w in candidates:
        if w.exists():
            results[w.stem] = evaluate(store, w, frames=val, imgsz=imgsz)
            log(f"{base} {w.stem}: score {selection_score(results[w.stem]):.3f} "
                f"(found {results[w.stem][SELECT_CONF]['recall']}, real {results[w.stem][SELECT_CONF]['precision']})")
    pick = max(results, key=lambda k: selection_score(results[k]))
    (runs / base / "selection.json").write_text(json.dumps({"picked": pick, "scores": results}, indent=2))
    out = models_dir(store) / f"ball_{base}.pt"
    shutil.copy(weights / f"{pick}.pt", out)
    log(f"trained {base}: kept {pick} -> {out}")
    return out


def evaluate(store: Store, weights: Path, set_name: str = "bench", crop: bool = True, imgsz: int = 1280,
             confs=(0.1, 0.25, 0.4), frames: list[tuple[str, int]] | None = None) -> dict:
    """
    Found/real/junk for a model's ball detections on a label set's frame images (same rules
    as `score`), or on just `frames` (game_id, frame) when given.
    """
    from ultralytics import YOLO

    settings = store.settings
    model = YOLO(weights)
    labels = ball_labels.load_labels(settings)
    tallies = ball_labels.new_tallies(confs)
    others = ball_others.load(settings)
    games = {}
    tasks = (ball_labels.load_tasks(settings).sets.get(set_name, []) if frames is None
             else [ball_labels.BallTask(game_id=g, frame=f) for g, f in frames])
    for t in tasks:
        label = labels.get((t.game_id, t.frame))
        if label is None or label.state == "skip":
            continue
        game = games.setdefault(t.game_id, store.load_game(t.game_id))
        img = cv2.imread(str(ball_labels.frame_image(store, game, t.frame)))
        ox, oy = CENTRE if crop else (0, 0)
        if crop:
            img = img[oy:oy + CROP_H, ox:ox + CROP_W]
        boxes = model.predict(img, imgsz=imgsz, conf=min(confs), classes=[BALL], verbose=False)[0].boxes.cpu().numpy()
        cx = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) / 2 + ox
        cy = (boxes.xyxy[:, 1] + boxes.xyxy[:, 3]) / 2 + oy
        ball_labels.tally(tallies, label, boxes.conf, *ball_others.classify(others, label, game, cx, cy))
    return ball_labels.summarise(tallies)
