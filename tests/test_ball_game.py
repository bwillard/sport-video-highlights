import numpy as np
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.models import Clip
from soccer_clips.scan import scan
from soccer_clips.server.app import create_app
from soccer_clips.store import Store
from soccer_clips.track import ball_game

GAME = "gu11-test-2026-10-10"


def test_merge_and_subtract():
    assert ball_game.merge([(50, 60), (0, 10), (5, 20), (20, 25)]) == [(0, 25), (50, 60)]
    assert ball_game.subtract([(0, 100)], [(10, 20), (50, 60)]) == [(0, 10), (20, 50), (60, 100)]
    assert ball_game.subtract([(0, 10)], [(0, 10)]) == []


def test_ranges_and_exact_frames(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=6, size="320x180")
    store = Store(settings)
    scan(settings, store)
    with store.edit_game(GAME) as g:
        g.clips += [Clip(start_s=3.0, end_s=4.0), Clip(start_s=3.5, end_s=5.0)]
        g.kickoff_1h_s, g.kickoff_2h_s = 2.0, 1.0  # a 1st-half kickoff after the 2nd: ignored
    game = store.load_game(GAME)
    end = game.frame_at(game.duration_s) - 1
    assert ball_game.ranges_for(game, "clips") == [(game.frame_at(1.0), end)]
    assert ball_game.ranges_for(game, "all") == [(0, end)]
    frames = list(ball_game.read_frames(store, game, 30, 45))
    assert len(frames) == 15 and frames[0].shape == (180, 320, 3)
    # same pixels as reading from the start and skipping ahead: frame numbers line up
    from_start = list(ball_game.read_frames(store, game, 0, 45))
    assert np.abs(frames[0].astype(int) - from_start[30].astype(int)).mean() < 2


def test_ball_api_and_retrack(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=3, size="320x180")
    store = Store(settings)
    scan(settings, store)
    game = store.load_game(GAME)
    with TestClient(create_app(store)) as client:
        assert client.get(f"/api/games/{GAME}/ball").json()["frame"] == []
        # a ball rolling right through frames 10-39, as saved detections + a still camera
        dets = np.array([(f, 100 + 2 * f, 90, 106 + 2 * f, 96, 0.6) for f in range(10, 40)], float)
        ball_game.save_npz(ball_game.tracks_dir(store, game) / "ball_raw.npz", ranges=np.array([[0, 60]]),
                           dets=dets, motion_frames=np.arange(60), motion=np.repeat(np.eye(2, 3)[None], 60, 0))
        assert ball_game.retrack(store, game) == 30
        b = client.get(f"/api/games/{GAME}/ball").json()
        assert b["ranges"] == [[0, 60]] and b["frame"][0] == 10 and len(b["frame"]) == 30
        assert abs(b["x"][0] - (103 + 20) / 320) < 1e-3 and set(b["kind"]) == {1}


def test_segments_match_the_original(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=6, size="320x180")
    store = Store(settings)
    scan(settings, store)
    with store.edit_game(GAME) as g:
        g.clips.append(Clip(start_s=3.0, end_s=4.0))
    game = store.load_game(GAME)
    (s, e), = ball_game.ranges_for(game, "clips")
    original = list(ball_game.read_frames(store, game, s, e))
    paths = ball_game.prepare_segments(store, game, "clips", log=lambda m: None)
    assert paths == [ball_game.segment_path(store, game, s, e)] and paths[0].exists()
    from_segment = list(ball_game.read_frames(store, game, s, e))
    assert len(from_segment) == len(original) == e - s
    assert max(np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(original, from_segment)) < 2


def test_chunks():
    assert ball_game.chunks([(0, 100)], 4) == [(0, 100)]  # too short to split
    parts = ball_game.chunks([(0, 20000), (50000, 51000)], 4)
    assert parts[:4] == [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)] and parts[4] == (50000, 51000)
    assert ball_game.chunks([(0, 9000)], 6) == [(0, 4500), (4500, 9000)]  # 2 min minimum per chunk
