"""
Retraining the ball detector from the app (ball labels page), in two steps you start separately:

train    export the labels as a dataset, fine-tune YOLO26x on RunPod (gpu.ball_train_job), and score the new model
         and the one in use on today's bench, all of it and the bench frames labelled since the model in use was
         made (mostly fixes of balls it missed). The new model is kept unless it scores worse (WORSE_BY).
retrack  run the model over every game video on the network volume (gpu.ball_games_job), redoing every frame,
         then put the hand fixes back (ball_game.retrack). Hours, so it's a choice of its own.

Both run detached, like the recognition retrain (retrain.py: status, start, the pods' deadlines).
"""

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

from soccer_clips import gpu, retrain
from soccer_clips.store import Store, atomic_write_text

MODEL = "ball_yolo26x"  # the model in use, in models/
BASE = "yolo26x"  # what it's fine-tuned from
EPOCHS = 60
TRAIN_HOURS = 2.5  # hard limit on the training pod (runs so far: 35-39 min)
GAME_HOURS = 0.9  # per game on the re-tracking pod (runs so far: ~45 min a game), plus an hour
WORSE_BY = 0.01  # keep the model in use if the new one's found/real balance is this much lower on the bench
CONF = "0.25"  # the confidence models are compared at (ball_train.SELECT_CONF)
KIND = "ball"


def _models(store: Store) -> Path:
    from soccer_clips.track.benchmark import models_dir

    return models_dir(store)


def _info_path(store: Store) -> Path:
    return _models(store) / f"{MODEL}.json"


def _score(path: Path, prefix: str) -> dict | None:
    """{found, real, junk, f} at CONF for the first model in an eval JSON whose name starts with prefix."""
    from soccer_clips.track import ball_train

    if not path.exists():
        return None
    scores = json.loads(path.read_text())
    key = next((k for k in scores if k.startswith(prefix)), None)
    if key is None or CONF not in scores[key]:
        return None
    s = scores[key][CONF]
    return {"found": s.get("recall"), "real": s.get("precision"), "junk": s.get("junk_per_frame"),
            "f": round(ball_train.selection_score(scores[key]), 4), "frames": s.get("visible")}


def _run_scores(run: Path) -> dict:
    """A training run's scores: the new model and (if scored) the one in use then, on the bench and on the frames
    labelled since that model."""
    out = run / "results" / "out"
    got = {"new": _score(out / "finetuned.json", MODEL), "old": _score(out / "current.json", "ball_current"),
           "new_fixed": _score(out / "finetuned_subset.json", MODEL), "old_fixed": _score(out / "current_subset.json", "ball_current")}
    return {k: v for k, v in got.items() if v}


def model_info(store: Store) -> dict | None:
    """When the ball model in use was trained and how it scored; for one from before the app, from its run folder."""
    p = _info_path(store)
    if p.exists():
        return json.loads(p.read_text())
    model = _models(store) / f"{MODEL}.pt"
    if not model.exists():
        return None
    made = model.stat().st_mtime
    info = {"trained_at": datetime.fromtimestamp(made).isoformat(timespec="minutes")}
    for d in sorted(retrain._gpu_dir(store).glob("*-ball-train"), reverse=True):
        w = d / "results" / "models" / f"{MODEL}.pt"
        if w.exists() and abs(w.stat().st_mtime - made) < 3 * 86400 and _run_scores(d):
            info.update(run=d.name, scores=_run_scores(d).get("new"))
            break
    return info


def _labels(store: Store, since: str | None) -> dict:
    """Usable labels (ball visible, or no ball) in each set, and how many are newer than `since`; plus the bench
    frames labelled since (any answer but skip), to score both models on."""
    from soccer_clips.track import ball_labels

    labels = ball_labels.load_labels(store.settings)
    after = datetime.fromisoformat(since).astimezone() if since else None  # trained_at is local time; labels' at is UTC
    out = {"train": 0, "bench": 0, "train_new": 0, "bench_new": 0, "fixed_frames": []}
    for name, tasks in ball_labels.load_tasks(store.settings).sets.items():
        if name not in ("train", "bench"):
            continue
        for t in tasks:
            lab = labels.get((t.game_id, t.frame))
            if lab is None or lab.state == "skip":
                continue
            new = bool(after and lab.at and lab.at > after)
            if lab.state in ("visible", "none"):
                out[name] += 1
                out[f"{name}_new"] += new
            if name == "bench" and new:
                out["fixed_frames"].append([t.game_id, t.frame])
    return out


def _tracks(store: Store, info: dict | None) -> dict:
    """Games with a whole-game ball track, and how many of those were made before the model in use."""
    from soccer_clips.track.ball_game import tracks_dir

    made = datetime.fromisoformat(info["trained_at"]).timestamp() if info else 0
    games = store.list_games()
    raws = [tracks_dir(store, g) / "ball_raw.npz" for g in games]
    return {"games": len(games), "tracked": sum(r.exists() for r in raws),
            "older": sum(r.exists() and r.stat().st_mtime < made for r in raws)}


def estimate(store: Store) -> dict:
    """Time and cost of a training run (the last one, +10%) and of re-tracking every game (per game from the last
    whole-game runs, +10%), with the most the pods' hour limits allow."""
    train = retrain._last_run(store, "ball-train")
    tm, td = (retrain._run_cost(train) if train else None) or (40.0, 0.6)
    mins = dollars = n = 0.0
    for d in sorted(retrain._gpu_dir(store).glob("*-ball-games-all"), reverse=True)[:4]:
        cost = retrain._run_cost(d)
        log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
        games = (d / "uploads.txt").read_text().count("game.json") if (d / "uploads.txt").exists() else 0
        if cost and games and "FAILED" not in log:
            mins, dollars, n = mins + cost[0], dollars + cost[1], n + games
    per_m, per_d = (mins / n, dollars / n) if n else (45.0, 1.35)
    games = len(store.list_games())
    hours = retrack_hours(games)
    return {"train_minutes": round(tm * 1.1), "train_dollars": round(td * 1.1, 2), "train_max_dollars": round(TRAIN_HOURS * 1.8, 2),
            "retrack_minutes": round(per_m * games * 1.1), "retrack_dollars": round(per_d * games * 1.1, 2),
            "retrack_max_dollars": round(hours * 1.8, 2), "games": games}  # $1.79/h: the dearest so far


def retrack_hours(games: int) -> float:
    return math.ceil(GAME_HOURS * games) + 1.0


def history(store: Store, n: int = retrain.HISTORY) -> list[dict]:
    """The last n training runs that brought back scores, newest first."""
    use = (model_info(store) or {}).get("run")
    out = []
    for d in sorted(retrain._gpu_dir(store).glob("*-ball-train"), reverse=True):
        sc = _run_scores(d)
        if "new" not in sc:
            continue
        row = {"run": d.name, "date": datetime.strptime(d.name[:15], "%Y%m%d-%H%M%S").isoformat(timespec="minutes"),
               **sc, "in_use": d.name == use}
        for extra in ("verdict.json", "labels.json"):
            if (d / extra).exists():
                row.update(json.loads((d / extra).read_text()))
        cost = retrain._run_cost(d)
        if cost:
            row["dollars"] = cost[1]
        out.append(row)
        if len(out) >= n:
            break
    return out


def overview(store: Store) -> dict:
    """What the ball labels page shows: the model in use, labels then and now, tracks, costs, runs, the job."""
    info = model_info(store)
    labels = _labels(store, (info or {}).get("trained_at"))
    labels.pop("fixed_frames")
    return {"model": info, "labels": labels, "tracks": _tracks(store, info), "estimate": estimate(store),
            "history": history(store), "status": retrain.status(store, KIND), "runpod": bool(os.environ.get("RUNPOD_API_KEY"))}


def _better(new: dict | None, old: dict | None) -> tuple[bool, str]:
    if not new:
        return False, "the run brought back no scores"
    msg = f"finds {new['found']:.0%} of balls, {new['real']:.0%} of its finds real"
    if not old:
        return True, msg
    msg = f"finds {old['found']:.0%} → {new['found']:.0%} of balls, real {old['real']:.0%} → {new['real']:.0%}"
    return new["f"] >= old["f"] - WORSE_BY, msg


def run_train(store: Store, log=print, log_name: str | None = None) -> None:
    """Export, fine-tune on RunPod, compare with the model in use on the same bench, install if no worse."""
    from soccer_clips.track import ball_train

    save = lambda **kw: retrain._save_status(store, KIND, **kw)  # noqa: E731
    save(state="running", action="train", pid=os.getpid(), step="preparing the labels", log=log_name, message=None)
    root, models = gpu.data_root(store), _models(store)
    current = models / "ball_current.pt"
    spent = 0.0
    try:
        old = model_info(store)
        labels = _labels(store, (old or {}).get("trained_at"))
        ball_train.export(store, log=log)
        subset = None
        if labels["fixed_frames"]:
            subset = "labels/fixed_frames.json"
            atomic_write_text(root / subset, json.dumps(labels["fixed_frames"]) + "\n")
        if (models / f"{MODEL}.pt").exists():
            shutil.copy(models / f"{MODEL}.pt", current)
        job = gpu.ball_train_job(store, [], [BASE], EPOCHS, current="ball_current" if current.exists() else None, subset=subset)
        runner = gpu.Runner(store, job, gpu_types=gpu.GPU_TYPES, max_hours=TRAIN_HOURS, log=log)
        save(step="training", train_run=runner.run_id)
        atomic_write_text(runner.dir / "labels.json", json.dumps({"train": labels["train"], "bench": labels["bench"],
                                                                  "fixed": len(labels["fixed_frames"])}) + "\n")
        try:
            results = runner.run()
        finally:
            spent += (retrain._run_cost(runner.dir) or (0, 0))[1]
            save(dollars=round(spent, 2))
            current.unlink(missing_ok=True)
        sc = _run_scores(runner.dir)
        ok, why = _better(sc.get("new"), sc.get("old"))
        if sc.get("new_fixed") and sc.get("old_fixed"):
            why += f"; on the {sc['new_fixed']['frames']} balls you fixed: {sc['old_fixed']['found']:.0%} → {sc['new_fixed']['found']:.0%}"
        new_model = results / "models" / f"{MODEL}.pt"
        ok = ok and new_model.exists()
        atomic_write_text(runner.dir / "verdict.json", json.dumps({"installed": ok, "why": why}) + "\n")
        if not ok:
            save(state="done", step="kept the old model", message=f"kept the model in use: {why}",
                 finished=datetime.now().isoformat(timespec="seconds"))
            log(f"kept the old model: {why}")
            return
        dest = models / f"{MODEL}.pt"
        if dest.exists():  # one step back, by hand if need be
            shutil.copy(dest, dest.with_suffix(".prev.pt"))
            if old:
                atomic_write_text(_info_path(store).with_suffix(".prev.json"), json.dumps(old, indent=1) + "\n")
        shutil.copy(new_model, dest)
        info = {"trained_at": datetime.now().isoformat(timespec="minutes"), "run": runner.run_id, "scores": sc.get("new"),
                "labels": {"train": labels["train"], "bench": labels["bench"]}}
        atomic_write_text(_info_path(store), json.dumps(info, indent=1) + "\n")
        log(f"installed the new model: {why}")
        save(state="done", step="done", finished=datetime.now().isoformat(timespec="seconds"),
             message=f"new model in use ({why}). Re-track the games to see it in the tagger.")
    except BaseException as e:
        save(state="failed", dollars=round(spent, 2), message=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        raise


def run_retrack(store: Store, log=print, log_name: str | None = None) -> None:
    """Every game's ball track again with the model in use (from the videos on the volume), hand fixes put back."""
    from soccer_clips.track import ball_game

    save = lambda **kw: retrain._save_status(store, KIND, **kw)  # noqa: E731
    save(state="running", action="retrack", pid=os.getpid(), step="re-tracking every game", log=log_name, message=None)
    models = _models(store)
    spent = 0.0
    try:
        info = model_info(store) or {}
        # a name of its own for these weights: the volume only notices a changed file by its size
        name = f"{MODEL}-{''.join(c for c in info.get('trained_at', 'x') if c.isdigit())[:12]}"
        if not (models / f"{name}.pt").exists():
            shutil.copy(models / f"{MODEL}.pt", models / f"{name}.pt")
        ids = [g.id for g in store.list_games()]
        job = gpu.ball_games_job(store, ids, "all", model=name, fresh=True, log=log)
        runner = gpu.Runner(store, job, gpu_types=gpu.GPU_TYPES, max_hours=retrack_hours(len(ids)), log=log)
        save(retrack_run=runner.run_id)
        done: list[str] = []
        try:
            runner.run()
        finally:
            spent += (retrain._run_cost(runner.dir) or (0, 0))[1]
            results = runner.dir / "results"
            if results.exists():
                gpu.install_ball_tracks(store, results, log=log)
                for raw in sorted((results / "games").glob("*/*/tracks/ball_raw.npz")):
                    game = store.load_game(raw.parent.parent.name)
                    ball_game.retrack(store, game)  # the hand fixes, back on the new detections
                    done.append(game.id)
                    log(f"{game.id}: hand fixes put back")
        for old in models.glob(f"{MODEL}-*.pt"):  # earlier named copies
            if old.stem != name:
                old.unlink()
        missing = [g for g in ids if g not in done]
        save(state="done" if not missing else "failed", step="done", dollars=round(spent, 2),
             finished=datetime.now().isoformat(timespec="seconds"),
             message=f"re-tracked {len(done)} of {len(ids)} games with the model from {info.get('trained_at', '?')[:10]}"
                     + (f"; not done: {', '.join(missing)}" if missing else ""))
    except BaseException as e:
        save(state="failed", dollars=round(spent, 2), message=f"{type(e).__name__}: {e}",
             finished=datetime.now().isoformat(timespec="seconds"))
        raise
