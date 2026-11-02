import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_gallery import four_tracklets

from soccer_clips.server.app import create_app
from soccer_clips.track import people, status


def test_stages(settings):
    store, game = four_tracklets(settings)  # tracklets, a label and signatures; nothing else
    st = status.stages(store, game)
    assert st["track"]["state"] == "ok" and st["labels"]["state"] == "ok"
    assert st["detect"]["state"] == "missing" and st["detect"]["fix"].startswith("soccer-clips gpu run people-games")
    assert st["kit"]["state"] == st["crops"]["state"] == st["ball"]["state"] == st["field"]["state"] == "missing"
    assert st["sigs"]["state"] == "stale"  # made with "test", not the model in use
    assert st["gametime"]["state"] == "missing"
    with store.edit_game(game.id) as g:
        g.kickoff_1h_s, g.end_1h_s = 0.0, 3.0
    assert status.stages(store, store.load_game(game.id))["gametime"]["state"] == "partial"


def all_ours(monkeypatch, store, game):
    """The test game has no kit colours: take everyone for ours."""
    n = len(people.tracks(store, game).frame)
    monkeypatch.setattr(people, "roles_for", lambda store, game, model=None: ({}, np.zeros(n, np.int8)))


def test_recognition(settings, monkeypatch):
    # 1 is #38, labelled (frames 0-89); 3 looks like her and comes after (guessed); 2 and 4 nobody
    store, game = four_tracklets(settings)
    all_ours(monkeypatch, store, game)
    fps = game.fps_num / game.fps_den
    s = lambda frames: pytest.approx(frames / fps, abs=0.05)
    assert status.cached(store, game) is None
    r = status.recognition(store, game)
    assert r["current"] and not r["game_time"] and not r["field"]
    assert r["seconds"]["labelled"] == s(90) and r["seconds"]["appearance"] == s(90) and r["seconds"]["place"] == 0
    assert r["seconds"]["unknown"] == s(270)
    assert r["players"] == [{"player": 38, "labelled": s(90), "appearance": s(90), "place": 0}]
    assert r["per_frame"] == 2.5  # 3 on screen until frame 90 (1, 2, 4), then 2 (2, 3)
    assert [x["track"] for x in r["longest"]] == [2, 4] and r["longest"][0]["s"] == s(180)
    assert r["longest"][1]["looks"] == r["longest"][0]["looks"] == []  # she's labelled on another tracklet then
    assert sum(b["s"] for b in r["unknown_by_length"]) == pytest.approx(r["seconds"]["unknown"], abs=0.1)
    assert [b["s"] > 0 for b in r["unknown_by_length"]] == [False, True, False, False]  # 3 s and 6 s tracklets

    assert status.cached(store, game)["current"]
    people.add_pick(store, game, 100, 7, "is", 0.35, 0.4)  # tracklet 2 is #7
    assert not status.cached(store, game)["current"]  # kept, but out of date
    r = status.recognition(store, game)
    assert r["current"] and r["seconds"]["labelled"] == s(90 + 180)  # a tracklet's first label covers all of it

    # only game time counts
    with store.edit_game(game.id) as g:
        g.kickoff_1h_s, g.end_1h_s = 0.0, 90 / fps
    r = status.recognition(store, store.load_game(game.id))
    assert r["game_time"] and r["seconds"]["appearance"] == 0 and r["seconds"]["labelled"] == s(90 + 90)  # 1 and 2


def test_named_boxes_count_whatever_their_kit(settings):
    store, game = four_tracklets(settings)  # no kit: only the labelled tracklet reads as ours
    r = status.recognition(store, game)
    fps = game.fps_num / game.fps_den
    assert r["seconds"]["unknown"] == 0 and r["seconds"]["appearance"] == pytest.approx(90 / fps, abs=0.05)


def test_api(settings):
    store, game = four_tracklets(settings)
    client = TestClient(create_app(store))
    assert client.get("/games").status_code == 200
    got = client.get("/api/status").json()
    assert [g["id"] for g in got] == [game.id] and got[0]["recognition"] is None and got[0]["stages"]["track"]["state"] == "ok"
    r = client.get(f"/api/games/{game.id}/recognition").json()
    assert r["current"] and r["seconds"]["labelled"] > 0
    assert client.get("/api/status").json()[0]["recognition"]["current"]
    assert client.get("/api/games/nope/recognition").status_code == 404


def test_still_by_line():
    fps, W = 30.0, 46.0
    n = int(30 * fps / status.STEP)  # 30 s, sampled
    frame = np.arange(n) * status.STEP
    rng = np.random.default_rng(0)
    jitter = lambda: rng.normal(0, 0.3, (n, 2))
    people = {
        1: np.column_stack([np.full(n, 10.0), np.full(n, 21.0)]) + jitter(),  # parent 2 m inside the far line
        2: np.column_stack([np.full(n, 0.0), np.full(n, 0.0)]) + jitter(),  # keeper-still, mid-field
        3: np.column_stack([np.linspace(-30, 30, n), np.full(n, 22.0)]) + jitter(),  # winger running the line
        4: np.column_stack([np.full(n, 5.0), np.full(n, -25.0)]) + jitter(),  # bench, beyond the near line
    }
    track = np.concatenate([np.full(n, k) for k in people])
    xy = np.concatenate(list(people.values()))
    still = status.still_by_line(track, np.tile(frame, len(people)), xy, W, fps)
    assert [round(still[track == k].mean(), 2) for k in people] == [1.0, 0.0, 0.0, 1.0]
    short = status.still_by_line(np.ones(n // 6, int), frame[:n // 6], people[1][:n // 6], W, fps)
    assert not short.any()  # 5 s: too short to say
