import json

import numpy as np
from test_gallery import four_tracklets

from soccer_clips.track import positions


def test_smooth_within_each_tracklet():
    # two tracklets in frame order, interleaved; tracklet 1 has a one-frame jump (hidden feet) and an unplaced frame
    f = np.repeat(np.arange(20), 2)
    track = np.tile([1, 2], 20)
    xy = np.where(track[:, None] == 1, [[0.0, 0.0]], [[30.0, 5.0]]).astype(float)
    xy[(track == 1) & (f == 10)] = [0.0, 8.0]
    xy[(track == 1) & (f == 12)] = np.nan
    out = positions.smooth(track, f, xy, half=3)
    assert out.shape == xy.shape and out.dtype == np.float32
    assert np.allclose(out[track == 1], 0) and np.allclose(out[track == 2], [30, 5])  # jump gone, no mixing
    one = positions.smooth(np.array([1]), np.array([0]), np.array([[np.nan, np.nan]]))
    assert np.isnan(one).all()  # nothing to go by


def test_smooth_short_and_at_the_ends():
    f = np.arange(4)
    xy = np.column_stack([np.arange(4.0), np.zeros(4)])
    out = positions.smooth(np.ones(4, int), f, xy, half=7)  # the window is cut to the tracklet, not padded
    assert np.allclose(out[:, 0], 1.5)


def test_off_field():
    fps, L, W = 10.0, 60.0, 40.0
    n = 200  # 20 s at every frame
    f = np.arange(n)
    people = {
        1: np.column_stack([np.linspace(-20, 20, n), np.zeros(n)]),  # running through the middle
        2: np.column_stack([np.full(n, 31.5), np.zeros(n)]),  # 1.5 m beyond a goal line
        3: np.column_stack([np.full(n, 5.0), np.full(n, 18.5)]),  # parent standing 1.5 m inside the far line
    }
    track = np.concatenate([np.full(n, k) for k in people])
    xy = np.concatenate(list(people.values()))
    off = positions.off_field(track, np.tile(f, 3), xy, L, W, fps)
    assert [round(off[track == k].mean(), 2) for k in people] == [0.0, 1.0, 1.0]
    nan = positions.off_field(np.ones(3, int), np.arange(3), np.full((3, 2), np.nan), L, W, fps)
    assert not nan.any()  # not placed counts as on the field


def test_seam_jumps():
    f = np.arange(10)
    xy = np.column_stack([np.arange(10) * 0.1, np.full(10, -5.0)])  # near half, 0.1 m a frame
    xy[5:, 0] += 1.0  # a 1 m jump where the keyframe changes
    kf = np.where(f < 5, 3, 4)
    s = positions.seam_jumps(np.ones(10, int), f, xy, kf)
    assert s["near_seam_n"] == 1 and abs(s["near_seam_m"] - 1.1) < 1e-6
    assert s["near_step_n"] == 8 and abs(s["near_step_m"] - 0.1) < 1e-6
    assert s["far_seam_m"] is None and s["far_step_n"] == 0


def test_load_only_when_current(settings, monkeypatch):
    store, game = four_tracklets(settings)
    k = {"now": "a"}
    monkeypatch.setattr(positions, "key", lambda store, game: k["now"])
    from soccer_clips.track import ball_game

    ball_game.save_npz(positions.path(store, game), key=np.array("a"), xy=np.zeros((3, 2), np.float32),
                       off=np.zeros(3, bool), stats=np.array(json.dumps({"placed": 1.0})))
    got = positions.load(store, game)
    assert got is not None and got["stats"]["placed"] == 1.0 and got["xy"].shape == (3, 2)
    assert positions.saved_stats(store, game) == ({"placed": 1.0}, True)
    k["now"] = "b"  # re-tracked or re-calibrated since
    assert positions.load(store, game) is None and positions.saved_stats(store, game)[1] is False


def test_boxes_api_has_field_columns(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app
    from soccer_clips.track import people

    store, game = four_tracklets(settings)  # tracked, not calibrated: no positions yet
    client = TestClient(create_app(store))
    get = lambda: np.frombuffer(client.get(f"/api/games/{game.id}/people/boxes",
                                           params={"start": 0, "end": 450}).content, "<f4").reshape(-1, 11)
    rows = get()
    assert len(rows) and np.isnan(rows[:, 7:9]).all() and (rows[:, 9] == 0).all()

    n = len(people.tracks(store, game).frame)
    xy = np.column_stack([np.arange(n), np.zeros(n)]).astype(np.float32)
    off = np.arange(n) % 2 == 0
    monkeypatch.setattr(positions, "load", lambda store, g: {"xy": xy, "off": off})
    rows = get()
    assert np.allclose(rows[:, 7], np.arange(len(rows))) and (rows[:, 9] == (np.arange(len(rows)) % 2 == 0)).all()
