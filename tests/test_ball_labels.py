import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.models import Clip
from soccer_clips.server.app import create_app
from soccer_clips.store import Store
from soccer_clips.track import ball_labels, bench

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

    rows = [
        ball(local[0], 160 + 4, 90, 0.5),   # on the visible ball
        ball(local[0], 20, 20, 0.15),       # false hit, weak
        ball(local[1], 80, 90, 0.3),        # on the hidden ball
        ball(local[2], 100, 100, 0.6),      # ball where there is none
    ]
    vdir = bench.bench_dir(settings) / "results" / "v"
    vdir.mkdir(parents=True)
    np.savez_compressed(vdir / "one.npz", boxes=np.array(rows, np.float32))
    s = ball_labels.score(store, vdir.parent)["v"]
    assert s["0.1"] == {"recall": 1.0, "precision": 0.5, "hidden_found": 1.0, "visible": 1, "hidden": 1, "dets": 4}
    assert s["0.25"]["precision"] == round(2 / 3, 3)
    assert s["0.4"] == {"recall": 1.0, "precision": 0.5, "hidden_found": 0.0, "visible": 1, "hidden": 1, "dets": 2}


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
