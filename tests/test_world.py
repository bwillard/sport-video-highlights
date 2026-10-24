import numpy as np
from fastapi.testclient import TestClient

from soccer_clips.server.app import create_app
from soccer_clips.track import ball_game, world
from test_ball_game import GAME, _game_with_wrong_path


def _one_view(n_frames, fps=30):
    """A world where every sampled frame matched keyframe 0 with no movement."""
    frames = np.arange(0, n_frames, world.SAMPLE, dtype=np.int32)
    return {"frames": frames, "key": np.zeros(len(frames), np.int32), "H": np.repeat(np.eye(3)[None], len(frames), 0),
            "links": np.empty((0, 2), np.int32), "link_H": np.empty((0, 3, 3))}


def test_fixed_object_vs_a_ball_passing_through():
    fps, n = 30, 30 * 300  # 5 minutes, the spot in view the whole time
    w = _one_view(n)
    f = np.arange(0, n, 3)
    still = (f, np.full(len(f), 900.0), np.full(len(f), 500.0))  # detected at one spot all along
    passes = np.arange(0, n, 30 * 20)  # a ball crossing another spot for 1 s every 20 s
    pf = np.concatenate([p + np.arange(30) for p in passes])
    moving = (pf, 400 + (pf % 30) * 2.0, np.full(len(pf), 300.0))
    ff = np.concatenate([still[0], moving[0]])
    xx, yy = np.concatenate([still[1], moving[1]]), np.concatenate([still[2], moving[2]])
    key, kx, ky = np.zeros(len(ff), np.int32), xx, yy
    detected, searched = world.spot_share(w, key, kx, ky, ff, fps)
    s = len(still[0])
    assert detected[0] >= 290 and abs(detected[0] - searched[0]) <= 2  # always seen when looked at
    assert (detected[s:] < world.FIXED_SECONDS).all()  # the ball is there 15 s in 5 minutes


def test_blocked_detections_skipped(settings):
    store, game = _game_with_wrong_path(settings)  # the only detections: junk the path follows
    raw = ball_game.load_raw(store, game)
    assert len(ball_game.load_path(store, game)["frame"])
    blocked = np.ones(len(raw["dets"]), bool)
    ball_game.save_npz(world.fixed_path(store, game), blocked=blocked, detected=np.zeros(len(blocked)),
                       searched=np.zeros(len(blocked)), n_dets=np.int64(len(blocked) + 1))
    assert world.load_blocked(store, game, len(raw["dets"])) is None  # made for other detections: ignored
    ball_game.save_npz(world.fixed_path(store, game), blocked=blocked, detected=np.zeros(len(blocked)),
                       searched=np.zeros(len(blocked)), n_dets=np.int64(len(blocked)))
    ball_game.retrack(store, game)
    assert not len(ball_game.load_path(store, game)["frame"])  # nothing left to be the ball
    with TestClient(create_app(store)) as client:
        assert len(client.get(f"/api/games/{GAME}/ball/raw").json()["fixed"]) == len(blocked)
