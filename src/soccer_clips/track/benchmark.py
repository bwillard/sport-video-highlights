"""
Detection + tracking benchmark on the bench clip set (Phase 3.2 / 4.1).

Each variant (model size, input size, optional centre crop) runs YOLO on every
frame of every bench clip. People go through BoT-SORT; balls are kept as raw
detections down to a low confidence, so we can see how often the ball is
nearly found rather than only what the tracker keeps. Results per variant go
to <bench_dir>/results/<variant>/: one .npz of boxes per clip and summary.json.
`draw` renders a clip's boxes onto the video for checking by eye.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from soccer_clips.store import Store, atomic_write_json
from soccer_clips.track.bench import BenchClip, bench_dir, load_set

PERSON, BALL = 0, 32  # COCO classes
MIN_CONF = 0.05  # keep weak detections; summaries apply their own thresholds
BALL_CONFS = (0.1, 0.25, 0.4)


@dataclass(frozen=True)
class Variant:
    name: str
    model: str  # ultralytics weights name
    imgsz: int
    crop: tuple[int, int, int, int] | None = None  # x, y, w, h in full-frame pixels


# Centre crop: the middle 2/3 of a 1920x1080 frame at native resolution. Veo's
# camera follows the play, so the ball is usually in there.
VARIANTS = {v.name: v for v in [
    Variant("n640", "yolo11n.pt", 640),
    Variant("s640", "yolo11s.pt", 640),
    Variant("n1280", "yolo11n.pt", 1280),
    Variant("s1280", "yolo11s.pt", 1280),
    Variant("s1280-centre", "yolo11s.pt", 1280, crop=(320, 180, 1280, 720)),
]}

# Columns of the per-clip box array.
COLS = ["frame", "cls", "track", "x1", "y1", "x2", "y2", "conf"]


def models_dir(store: Store) -> Path:
    """Downloaded model weights, kept next to the data instead of the current directory."""
    return store.settings.games_dir.parent / "models"


def results_dir(store: Store, variant: str) -> Path:
    return bench_dir(store.settings) / "results" / variant


def _tracker():
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    return BOTSORT(args=IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml"))))


def run_clip(model, variant: Variant, path: Path) -> tuple[np.ndarray, dict]:
    """Boxes (rows of COLS, full-frame pixels; track = -1 for balls) and timings for one clip."""
    tracker = _tracker()
    cap = cv2.VideoCapture(str(path))
    rows, n = [], 0
    t_decode = t_detect = t_track = 0.0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        ox, oy = 0, 0
        img = frame
        if variant.crop:
            ox, oy, w, h = variant.crop
            img = frame[oy:oy + h, ox:ox + w]
        res = model.predict(img, imgsz=variant.imgsz, conf=MIN_CONF, classes=[PERSON, BALL], verbose=False)[0]
        t2 = time.perf_counter()
        boxes = res.boxes.cpu().numpy()
        people = boxes[boxes.cls == PERSON]
        tracks = tracker.update(people, img)  # x1, y1, x2, y2, id, score, cls, idx
        t3 = time.perf_counter()
        for x1, y1, x2, y2, tid, score, *_ in tracks:
            rows.append((n, PERSON, tid, x1 + ox, y1 + oy, x2 + ox, y2 + oy, score))
        for (x1, y1, x2, y2), conf in zip(boxes.xyxy[boxes.cls == BALL], boxes.conf[boxes.cls == BALL]):
            rows.append((n, BALL, -1, x1 + ox, y1 + oy, x2 + ox, y2 + oy, conf))
        t_decode += t1 - t0
        t_detect += t2 - t1
        t_track += t3 - t2
        n += 1
    cap.release()
    arr = np.array(rows, dtype=np.float32).reshape(-1, len(COLS))
    return arr, {"frames": n, "decode_s": t_decode, "detect_s": t_detect, "track_s": t_track}


def _her_check(store: Store, clip: BenchClip, arr: np.ndarray, frames: int) -> dict | None:
    """Is the player under the callout tracked at that frame, and for how much of the clip?"""
    if clip.callout is None:
        return None
    game = store.load_game(clip.game_id)
    f = game.frame_at(clip.callout.t_s) - game.frame_at(clip.start_s)
    x, y = clip.callout.x * game.width, clip.callout.y * game.height
    people = arr[(arr[:, 1] == PERSON) & (arr[:, 0] == f)]
    inside = people[(people[:, 3] <= x) & (x <= people[:, 5]) & (people[:, 4] <= y) & (y <= people[:, 6])]
    if len(inside) == 0:
        # The click can land just beside the box (between the legs, on the shadow): take the
        # nearest box centre within the callout ring.
        if len(people):
            d = np.hypot((people[:, 3] + people[:, 5]) / 2 - x, (people[:, 4] + people[:, 6]) / 2 - y)
            if d.min() <= clip.callout.r * game.height:
                inside = people[[d.argmin()]]
    if len(inside) == 0:
        return {"player": clip.callout_player, "found": False}
    # Several boxes can contain the point (players overlapping); take the smallest.
    box = inside[np.argmin((inside[:, 5] - inside[:, 3]) * (inside[:, 6] - inside[:, 4]))]
    tid = box[2]
    present = np.unique(arr[(arr[:, 1] == PERSON) & (arr[:, 2] == tid), 0]).astype(int)
    # Run of frames around the callout with no gap longer than 1s.
    lo = hi = f
    s = set(present.tolist())
    while any(lo - k in s for k in range(1, 31)):
        lo = max(k for k in range(lo - 30, lo) if k in s)
    while any(hi + k in s for k in range(1, 31)):
        hi = min(k for k in range(hi + 1, hi + 31) if k in s)
    return {"player": clip.callout_player, "found": True, "track": int(tid),
            "box_h_px": round(float(box[6] - box[4]), 1),
            "track_coverage": round(len(present) / frames, 3), "run_around_callout_s": round((hi - lo + 1) / 29.97, 1)}


def run(store: Store, variant_names: list[str], log=print) -> None:
    from ultralytics import YOLO

    clips = load_set(store.settings).clips
    src = bench_dir(store.settings) / "clips"
    for name in variant_names:
        variant = VARIANTS[name]
        models_dir(store).mkdir(parents=True, exist_ok=True)
        model = YOLO(models_dir(store) / variant.model)
        out = results_dir(store, name)
        out.mkdir(parents=True, exist_ok=True)
        model.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=variant.imgsz, verbose=False)  # warm-up
        per_clip, wall0 = {}, time.perf_counter()
        for clip in clips:
            t0 = time.perf_counter()
            arr, timing = run_clip(model, variant, src / f"{clip.name}.mp4")
            wall = time.perf_counter() - t0
            np.savez_compressed(out / f"{clip.name}.npz", boxes=arr, cols=np.array(COLS))
            n = timing["frames"]
            balls = arr[arr[:, 1] == BALL]
            people = arr[arr[:, 1] == PERSON]
            per_clip[clip.name] = {
                **{k: round(v, 2) for k, v in timing.items()},
                "wall_s": round(wall, 2),
                "fps": round(n / wall, 2),
                "people_per_frame": round(len(people) / n, 1),
                "person_tracks": int(len(np.unique(people[:, 2]))),
                "ball_frames": {str(c): round(len(np.unique(balls[balls[:, 7] >= c, 0])) / n, 3) for c in BALL_CONFS},
                "ball_dets_per_frame": {str(c): round(float((balls[:, 7] >= c).sum()) / n, 2) for c in BALL_CONFS},
                "her": _her_check(store, clip, arr, n),
            }
            log(f"{name} {clip.name}: {n} frames, {n / wall:.1f} fps, "
                f"ball in {per_clip[clip.name]['ball_frames']['0.25']:.0%} of frames (conf ≥ 0.25), "
                f"her: {per_clip[clip.name]['her']}")
        frames = sum(c["frames"] for c in per_clip.values())
        wall = time.perf_counter() - wall0
        atomic_write_json(out / "summary.json", {
            "variant": {"model": variant.model, "imgsz": variant.imgsz, "crop": variant.crop},
            "frames": frames, "wall_s": round(wall, 1), "fps": round(frames / wall, 2),
            "detect_ms": round(1000 * sum(c["detect_s"] for c in per_clip.values()) / frames, 1),
            "track_ms": round(1000 * sum(c["track_s"] for c in per_clip.values()) / frames, 1),
            "decode_ms": round(1000 * sum(c["decode_s"] for c in per_clip.values()) / frames, 1),
            "clips": per_clip,
        })
        log(f"{name}: {frames} frames in {wall:.0f}s = {frames / wall:.1f} fps")


def _colour(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(tid))
    return tuple(int(c) for c in rng.integers(80, 256, 3))


def draw(store: Store, variant: str, clip_name: str, ball_conf: float = 0.1) -> Path:
    """Render the variant's boxes over the clip: people by track colour, her track thick yellow,
    balls as red circles with their confidence. Writes results/<variant>/<clip>.mp4."""
    clip = next(c for c in load_set(store.settings).clips if c.name == clip_name)
    out_dir = results_dir(store, variant)
    arr = np.load(out_dir / f"{clip_name}.npz")["boxes"]
    summary = json.loads((out_dir / "summary.json").read_text())
    her = (summary["clips"][clip_name].get("her") or {}).get("track")
    crop = summary["variant"]["crop"]
    cap = cv2.VideoCapture(str(bench_dir(store.settings) / "clips" / f"{clip_name}.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = out_dir / f"{clip_name}.mp4"
    enc = subprocess.Popen(
        [store.settings.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = arr[arr[:, 0] == f]
        if crop:
            x, y, cw, ch = crop
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (255, 255, 255), 1)
        for _, cls, tid, x1, y1, x2, y2, conf in rows:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            if cls == PERSON:
                is_her = her is not None and tid == her
                cv2.rectangle(frame, p1, p2, (0, 255, 255) if is_her else _colour(tid), 3 if is_her else 1)
                cv2.putText(frame, str(int(tid)), (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255) if is_her else _colour(tid), 1)
            elif conf >= ball_conf:
                c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, c, max(10, int(x2 - x1)), (0, 0, 255), 2)
                cv2.putText(frame, f"{conf:.2f}", (c[0] + 12, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"{variant}  frame {f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        enc.stdin.write(frame.tobytes())
        f += 1
    cap.release()
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"ffmpeg failed drawing {clip_name}")
    return out
