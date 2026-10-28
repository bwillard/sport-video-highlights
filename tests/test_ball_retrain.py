import json
from datetime import datetime, timedelta, timezone

import pytest

from soccer_clips import ball_retrain, gpu, retrain
from soccer_clips.store import Store


def scores(found, real):
    return {"0.25": {"recall": found, "precision": real, "junk_per_frame": 0.1, "visible": 50}}


def test_keeps_the_new_model_unless_it_scores_worse():
    old = {"found": 0.77, "real": 0.88, "f": 0.82}
    assert ball_retrain._better({"found": 0.85, "real": 0.86, "f": 0.855}, old) == (True, "finds 77% → 85% of balls, real 88% → 86%")
    assert not ball_retrain._better({"found": 0.70, "real": 0.88, "f": 0.78}, old)[0]
    assert ball_retrain._better({"found": 0.5, "real": 0.5, "f": 0.5}, None)[0] and not ball_retrain._better(None, old)[0]


def test_labels_since_the_model_count_in_utc(settings):
    from soccer_clips.track import ball_labels

    store = Store(settings)
    old = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    tasks = ball_labels.Tasks(sets={"train": [ball_labels.BallTask(game_id="g", frame=f) for f in (1, 2)],
                                       "bench": [ball_labels.BallTask(game_id="g", frame=f) for f in (3, 4)]})
    ball_labels.save_tasks(settings, tasks)
    for f, at, state in ((1, old, "visible"), (2, old + timedelta(days=2), "none"), (3, old, "visible"), (4, old + timedelta(days=2), "hidden")):
        ball_labels.add_label(settings, ball_labels.BallLabel(game_id="g", frame=f, state=state, x=0.5 if state == "visible" else None,
                                                                 y=0.5 if state == "visible" else None, at=at))
    since = (old + timedelta(days=1)).astimezone().replace(tzinfo=None).isoformat(timespec="minutes")  # local, like trained_at
    got = ball_retrain._labels(store, since)
    assert got == {"train": 2, "bench": 1, "train_new": 1, "bench_new": 0, "fixed_frames": [["g", 4]]}


class FakeRunner:
    made = []

    def __init__(self, store, job, gpu_types, max_hours, log):
        self.run_id = f"20261028-0{len(FakeRunner.made)}0000-{job.name}"
        self.dir = gpu.data_root(store) / "gpu" / self.run_id
        self.dir.mkdir(parents=True)
        self.job, self.store = job, store
        FakeRunner.made.append((job, max_hours))

    def run(self):
        res = self.dir / "results"
        (self.dir / "log.txt").write_text("total 40 min, about $0.60\n")
        if self.job.name == "ball-train":
            (res / "out").mkdir(parents=True)
            (res / "models").mkdir()
            (res / "models" / "ball_yolo26x.pt").write_bytes(b"new")
            (res / "out" / "finetuned.json").write_text(json.dumps({"ball_yolo26x-centre": scores(0.85, 0.88)}))
            (res / "out" / "current.json").write_text(json.dumps({"ball_current-centre": scores(0.77, 0.88)}))
            (res / "out" / "finetuned_subset.json").write_text(json.dumps({"ball_yolo26x-centre": scores(0.6, 0.9)}))
            (res / "out" / "current_subset.json").write_text(json.dumps({"ball_current-centre": scores(0.2, 0.9)}))
        else:
            tracks = res / "games" / "2026" / "gu11-test-2026-10-10" / "tracks"
            tracks.mkdir(parents=True)
            (tracks / "ball_raw.npz").write_bytes(b"raw")
        return res


def test_train_then_retrack(settings, monkeypatch):
    from soccer_clips.track import ball_game, ball_train
    from soccer_clips.track.benchmark import models_dir

    store = Store(settings)
    models_dir(store).mkdir(parents=True, exist_ok=True)
    (models_dir(store) / "ball_yolo26x.pt").write_bytes(b"old")
    FakeRunner.made = []
    jobs = {}
    monkeypatch.setattr(gpu, "Runner", FakeRunner)
    monkeypatch.setattr(ball_train, "export", lambda store, log=print: None)
    monkeypatch.setattr(ball_retrain, "_labels", lambda store, since: {"train": 500, "bench": 200, "train_new": 160, "bench_new": 70,
                                                                       "fixed_frames": [["g", 4]]})

    def train_job(store, stock, bases, epochs, current=None, subset=None):
        jobs["train"] = dict(bases=bases, current=current, subset=subset,
                             had_current=(models_dir(store) / "ball_current.pt").exists())
        return gpu.Job("ball-train", [], "", [])

    monkeypatch.setattr(gpu, "ball_train_job", train_job)
    ball_retrain.run_train(store, log=lambda m: None)
    assert jobs["train"] == {"bases": ["yolo26x"], "current": "ball_current", "subset": "labels/fixed_frames.json", "had_current": True}
    assert not (models_dir(store) / "ball_current.pt").exists()  # the copy for scoring is gone
    assert (models_dir(store) / "ball_yolo26x.pt").read_bytes() == b"new"
    assert (models_dir(store) / "ball_yolo26x.prev.pt").read_bytes() == b"old"
    st = retrain.status(store, "ball")
    assert st["state"] == "done" and "77% → 85%" in st["message"] and "1 balls you fixed" not in st["message"]
    assert "on the 50 balls you fixed: 20% → 60%" in st["message"]
    info = ball_retrain.model_info(store)
    assert info["scores"]["found"] == 0.85 and info["labels"] == {"train": 500, "bench": 200}
    h = ball_retrain.history(store)[0]
    assert h["in_use"] and h["installed"] and h["old"]["found"] == 0.77 and h["new_fixed"]["found"] == 0.6

    # re-track: a named copy of the weights (the volume only notices a changed file by its size), every frame redone
    from conftest import make_video
    from soccer_clips.scan import scan

    make_video(settings, "2026/gu11-test-2026-10-10.mp4", duration_s=2, audio=False)
    scan(settings, store)

    def games_job(store, ids, mode, model, fresh, log):
        jobs["games"] = dict(ids=ids, mode=mode, model=model, fresh=fresh)
        return gpu.Job("ball-games-all", [], "", [])

    redone = []
    monkeypatch.setattr(gpu, "ball_games_job", games_job)
    monkeypatch.setattr(gpu, "install_ball_tracks", lambda store, results, log: 1)
    monkeypatch.setattr(ball_game, "retrack", lambda store, game: redone.append(game.id))
    ball_retrain.run_retrack(store, log=lambda m: None)
    name = jobs["games"]["model"]
    assert name.startswith("ball_yolo26x-") and (models_dir(store) / f"{name}.pt").read_bytes() == b"new"
    assert jobs["games"]["fresh"] and jobs["games"]["mode"] == "all" and redone == ["gu11-test-2026-10-10"]
    assert FakeRunner.made[-1][1] == ball_retrain.retrack_hours(1)
    st = retrain.status(store, "ball")
    assert st["state"] == "done" and st["action"] == "retrack" and "re-tracked 1 of 1 games" in st["message"]


def test_overview_and_estimate(settings):
    store = Store(settings)
    gdir = gpu.data_root(store) / "gpu"
    for name, log, uploads in (("20261021-143240-ball-train", "total 39 min, about $0.58\n", ""),
                               ("20261023-073558-ball-games-all", "total 227 min, about $6.77\n", "a/game.json\n" * 5)):
        (gdir / name).mkdir(parents=True)
        (gdir / name / "log.txt").write_text(log)
        (gdir / name / "uploads.txt").write_text(uploads)
    e = ball_retrain.estimate(store)
    assert e["train_minutes"] == 43 and e["train_dollars"] == pytest.approx(0.64)
    assert e["games"] == 0 and e["retrack_dollars"] == 0  # no games in this store
    o = ball_retrain.overview(store)
    assert o["model"] is None and o["history"] == [] and o["tracks"] == {"games": 0, "tracked": 0, "older": 0}
