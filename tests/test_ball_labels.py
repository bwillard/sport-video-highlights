import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.models import Clip
from soccer_clips.server.app import create_app
from soccer_clips.store import Store
from soccer_clips.track import ball_labels, ball_others, bench

GAME = "gu11-test-2026-10-10"


@pytest.fixture
def store(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=6)
    store = Store(settings)
    with TestClient(create_app(store)):  # startup registers the game
        pass
    with store.edit_game(GAME) as g:
        g.clips += [Clip(id="c1", start_s=1.0, end_s=2.0), Clip(id="c2", start_s=4.0, end_s=5.0)]
    bench.add_clip(store, GAME, "c1", "one")
    return store


@pytest.fixture
def client(store):
    with TestClient(create_app(store)) as c:
        yield c


def test_sampling_keeps_sets_apart(store):
    b = ball_labels.sample_bench(store, n=6)
    assert len(b) == 6 and all(30 <= t.frame < 60 and t.clip == "one" for t in b)
    assert ball_labels.add_tasks(store, "bench", b) == 6
    assert ball_labels.add_tasks(store, "bench", b) == 0  # no duplicates
    extra = [ball_labels.BallTask(game_id=GAME, frame=100)] * 2
    assert ball_labels.add_tasks(store, "train", extra) == 1  # repeats within one batch too
    train = ball_labels.sample_train(store, 20, seed=1)
    # never within 10s of a bench clip: this short game is all "near", so nothing can be picked
    assert train == []


def test_label_api_and_frames(store, client):
    ball_labels.add_tasks(store, "bench", ball_labels.sample_bench(store, n=3))
    assert client.get("/api/ball/sets").json() == [{"name": "bench", "total": 3, "labelled": 0}]
    tasks = client.get("/api/ball/sets/bench").json()
    t = tasks[0]
    assert t["label"] is None
    assert client.get("/api/ball/sets/nope").status_code == 404

    r = client.get(f"/ball/frame/{GAME}/{t['frame']}.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert client.get(f"/ball/frame/{GAME}/99999.jpg").status_code == 404

    base = {"game_id": GAME, "frame": t["frame"]}
    assert client.post("/api/ball/labels", json={**base, "state": "visible"}).status_code == 422  # needs x, y
    assert client.post("/api/ball/labels", json={**base, "state": "visible", "x": 0.5, "y": 0.5, "extra": 1}).status_code == 422
    assert client.post("/api/ball/labels", json={**base, "state": "visible", "x": 0.5, "y": 0.4, "r": 0.01}).status_code == 204
    got = client.get("/api/ball/sets/bench").json()[0]["label"]
    assert (got["state"], got["x"], got["y"], got["r"]) == ("visible", 0.5, 0.4, 0.01)
    # none drops coordinates; latest answer wins; cleared removes it
    client.post("/api/ball/labels", json={**base, "state": "none", "x": 0.1, "y": 0.1})
    got = client.get("/api/ball/sets/bench").json()[0]["label"]
    assert (got["state"], got["x"]) == ("none", None)
    client.post("/api/ball/labels", json={**base, "state": "cleared"})
    assert client.get("/api/ball/sets/bench").json()[0]["label"] is None
    assert client.get("/label-ball").status_code == 200


def test_score(store, settings):
    tasks = ball_labels.sample_bench(store, n=3)  # game frames 35, 45, 55 -> clip frames 5, 15, 25
    ball_labels.add_tasks(store, "bench", tasks)
    w, h = 320, 180
    labels = [("visible", 0.5, 0.5), ("hidden", 0.25, 0.5), ("none", None, None)]
    for t, (state, x, y) in zip(tasks, labels):
        ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=t.frame, state=state, x=x, y=y))
    local = [t.frame - 30 for t in tasks]

    def ball(f, cx, cy, conf):
        return [f, 32, -1, cx - 3, cy - 3, cx + 3, cy + 3, conf]

    # A spare ball confirmed in frame 3, and an unsure candidate in frame 1.
    def mark(o):
        o.frames[ball_others.key(GAME, tasks[2].frame)] = [ball_others.OtherBall(x=200 / w, y=50 / h, verdict="ball")]
        o.frames[ball_others.key(GAME, tasks[0].frame)] = [ball_others.OtherBall(x=300 / w, y=170 / h, verdict="unsure")]
    ball_others.edit(settings, mark)

    rows = [
        ball(local[0], 160 + 4, 90, 0.5),   # on the visible game ball
        ball(local[0], 20, 20, 0.15),       # junk, weak
        ball(local[0], 300, 170, 0.9),      # on the unsure candidate: left out
        ball(local[1], 80, 90, 0.3),        # on the hidden game ball
        ball(local[2], 100, 100, 0.6),      # junk: no ball there
        ball(local[2], 202, 50, 0.45),      # on the spare ball: a real ball, so fine
    ]
    vdir = bench.bench_dir(settings) / "results" / "v"
    vdir.mkdir(parents=True)
    np.savez_compressed(vdir / "one.npz", boxes=np.array(rows, np.float32))
    s = ball_labels.score(store, vdir.parent)["v"]
    assert s["0.1"] == {"recall": 1.0, "precision": 0.6, "junk_per_frame": round(2 / 3, 3), "hidden_found": 1.0,
                        "visible": 1, "hidden": 1, "dets": 5, "other": 1}
    assert s["0.25"]["precision"] == 0.75 and s["0.25"]["junk_per_frame"] == round(1 / 3, 3)
    assert s["0.4"] == {"recall": 1.0, "precision": round(2 / 3, 3), "junk_per_frame": round(1 / 3, 3),
                        "hidden_found": 0.0, "visible": 1, "hidden": 1, "dets": 3, "other": 1}
    assert "junk/frame" in ball_labels.score_table({"v": s})


def test_guesses_shown_for_train_never_bench(store, settings, client):
    b = ball_labels.sample_bench(store, n=2)
    ball_labels.add_tasks(store, "bench", b)
    ball_labels.add_tasks(store, "train", [ball_labels.BallTask(game_id=GAME, frame=150)])
    guess = ball_labels.Guess(x=0.5, y=0.5, r=0.01, conf=0.8)
    frames = {f"{GAME}:{t.frame}": guess for t in b} | {f"{GAME}:150": guess}
    path = ball_labels.labels_dir(settings) / "ball_guesses.json"
    path.write_text(ball_labels.Guesses(model="x", frames=frames).model_dump_json())
    assert all(t["guess"] is None for t in client.get("/api/ball/sets/bench").json())
    assert client.get("/api/ball/sets/train").json()[0]["guess"] == guess.model_dump()
    with pytest.raises(ValueError):
        ball_labels.make_guesses(store, "bench")


def test_crop_origin_contains_ball_and_stays_in_frame():
    import random

    from soccer_clips.track.ball_train import CROP_H, CROP_W, _crop_origin

    rng = random.Random(0)
    for x, y in [(10, 10), (1910, 1070), (960, 540), (300, 900)]:
        for _ in range(50):
            ox, oy = _crop_origin(x, y, rng, 1920, 1080)
            assert 0 <= ox <= 1920 - CROP_W and 0 <= oy <= 1080 - CROP_H
            assert ox <= x < ox + CROP_W and oy <= y < oy + CROP_H


def test_radius_model_fits_size_against_height():
    from soccer_clips.track.ball_train import radius_model

    labels = [ball_labels.BallLabel(game_id=GAME, frame=k, state="visible", x=0.5, y=y, r=0.002 + 0.02 * y)
              for k, y in enumerate(np.linspace(0.3, 0.9, 12))]
    b, a = radius_model(labels)
    assert abs(b - 0.02) < 1e-6 and abs(a - 0.002) < 1e-6
    assert list(radius_model(labels[:3])) == [0.0, 0.008]  # too few sized labels: a fixed guess


def test_other_balls_api(store, settings, client):
    ball_labels.add_tasks(store, "train", [ball_labels.BallTask(game_id=GAME, frame=100)])
    k = ball_others.key(GAME, 100)
    ball_others.edit(settings, lambda o: o.frames.setdefault(k, []).append(
        ball_others.OtherBall(x=0.2, y=0.3, r=0.01, conf=0.7, source="ball_yolo26x")))
    review = client.get("/api/ball/others/review").json()
    assert [(c["index"], c["verdict"], c["set"]) for c in review] == [(0, None, "train")]
    assert client.put(f"/api/ball/others/{GAME}/100/0", json={"verdict": "ball"}).status_code == 204
    assert client.put(f"/api/ball/others/{GAME}/100/5", json={"verdict": "ball"}).status_code == 404
    assert client.put(f"/api/ball/others/{GAME}/100/0", json={"verdict": "maybe"}).status_code == 422
    # a hand-added one is confirmed straight away and gets the next index
    assert client.post(f"/api/ball/others/{GAME}/100", json={"x": 0.6, "y": 0.6}).json() == {"index": 1}
    task = client.get("/api/ball/sets/train").json()[0]
    assert [(o["index"], o["x"]) for o in task["others"]] == [(0, 0.2), (1, 0.6)]
    # removing marks it "not": indexes stay put, and it drops out of the confirmed list
    client.put(f"/api/ball/others/{GAME}/100/0", json={"verdict": "not"})
    assert [o["index"] for o in client.get("/api/ball/sets/train").json()[0]["others"]] == [1]
    assert [c["index"] for c in client.get("/api/ball/others/review").json()] == [0]  # manual ones aren't reviewed
    assert client.get("/review-balls").status_code == 200


def test_export_boxes_other_balls(store, settings, monkeypatch):
    from soccer_clips.track import ball_train

    monkeypatch.setattr(ball_train, "CROP_W", 160)
    monkeypatch.setattr(ball_train, "CROP_H", 90)
    monkeypatch.setattr(ball_train, "CENTRE", (80, 45))
    tasks = [ball_labels.BallTask(game_id=GAME, frame=f) for f in (40, 50)]
    ball_labels.add_tasks(store, "train", tasks)
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=40, state="visible", x=0.5, y=0.5, r=0.02))
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=50, state="none"))
    ball_others.edit(settings, lambda o: o.frames.update({
        ball_others.key(GAME, 40): [ball_others.OtherBall(x=0.55, y=0.55, r=0.02, verdict="ball"),
                                     ball_others.OtherBall(x=0.45, y=0.45, verdict="not")],
        ball_others.key(GAME, 50): [ball_others.OtherBall(x=0.5, y=0.5, r=0.02, verdict="ball")]}))
    out = ball_train.export(store, val_frac=0.0)
    lines = {p.stem: p.read_text().split("\n")[:-1] for p in (out / "labels" / "train").glob("*.txt")}
    assert len(lines[f"{GAME}_40_0"]) == 2  # game ball + the confirmed spare (not the rejected one)
    assert len(lines[f"{GAME}_50_0"]) == 1  # no game ball, but a real ball all the same
    assert (out / "val_frames.json").exists()
