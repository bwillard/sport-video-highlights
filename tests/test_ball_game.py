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


def _game_with_wrong_path(settings):
    """A game whose only detections are a junk object moving 3 px/frame; the real ball is never seen."""
    make_video(settings, f"2026/{GAME}.mp4", duration_s=6, size="320x180")
    store = Store(settings)
    scan(settings, store)
    game = store.load_game(GAME)
    dets = np.array([(f, 40 + 3 * f, 50, 46 + 3 * f, 56, 0.5) for f in range(0, 60)], float)
    ball_game.save_npz(ball_game.tracks_dir(store, game) / "ball_raw.npz", ranges=np.array([[0, 170]]),
                       dets=dets, motion_frames=np.arange(170), motion=np.repeat(np.eye(2, 3)[None], 170, 0))
    ball_game.retrack(store, game)
    return store, game


def test_fix_ball_here_and_label(settings):
    from soccer_clips.track import ball_labels

    store, game = _game_with_wrong_path(settings)
    assert 30 in ball_game.load_path(store, game)["frame"].tolist()  # the junk is on the path
    with TestClient(create_app(store)) as client:
        r = client.post(f"/api/games/{GAME}/ball/fix", json={"frame": 30, "action": "ball", "x": 0.75, "y": 0.75,
                                                               "label_set": "train"})
        assert r.status_code == 200
        b = r.json()
        i = b["frame"].index(30)
        assert abs(b["x"][i] - 0.75) < 1e-3 and abs(b["y"][i] - 0.75) < 1e-3  # the ring is where we put it
        assert not any(abs(b["x"][k] - (43 + 3 * f) / 320) < 0.02 for k, f in enumerate(b["frame"]) if 0 <= f - 30 <= 10)
        assert client.post(f"/api/games/{GAME}/ball/fix", json={"frame": 30, "action": "ball"}).status_code == 422
        assert client.post(f"/api/games/{GAME}/ball/fix", json={"frame": 900, "action": "none"}).status_code == 422
    label = ball_labels.load_labels(settings)[(GAME, 30)]
    assert (label.state, label.x) == ("visible", 0.75)
    assert any(t.frame == 30 for t in ball_labels.load_tasks(settings).sets["train"])
    # fixes survive rebuilding the whole path
    ball_game.retrack(store, game)
    z = ball_game.load_path(store, game)
    assert abs(z["x"][z["frame"].tolist().index(30)] - 0.75) < 1e-3


def test_fix_wrong_ring(settings):
    from soccer_clips.track import ball_labels

    store, game = _game_with_wrong_path(settings)
    ball_game.fix(store, game, 30, "none", label_set="bench")
    frames = ball_game.load_path(store, game)["frame"].tolist()
    assert not any(abs(f - 30) <= 15 for f in frames)  # the junk run around the fix is gone
    assert ball_labels.load_labels(settings)[(GAME, 30)].state == "none"


def test_auto_set(settings):
    from soccer_clips.track import ball_labels, bench

    make_video(settings, f"2026/{GAME}.mp4", duration_s=6, size="320x180")
    store = Store(settings)
    scan(settings, store)
    with store.edit_game(GAME) as g:
        g.clips.append(Clip(id="c1", start_s=1.0, end_s=2.0))
    bench.add_clip(store, GAME, "c1", "one")
    game = store.load_game(GAME)
    assert ball_labels.auto_set(store, GAME, game.frame_at(5.0)) == "bench"  # within 10 s of a bench clip
    # far from bench clips: whole 30 s moments go to one set, about one in five to the benchmark
    frames = range(900, 900 * 400, 300)  # all well past the bench clip (frame numbers needn't be in the video)
    moments = {}
    for f in frames:
        moments.setdefault(f // game.frame_at(30.0), set()).add(ball_labels.auto_set(store, GAME, f))
    assert all(len(v) == 1 for v in moments.values())
    share = sum(v == {"bench"} for v in moments.values()) / len(moments)
    assert 0.1 < share < 0.3
    # a frame already in a set stays there
    ball_labels.add_tasks(store, "train", [ball_labels.BallTask(game_id=GAME, frame=150)])
    assert ball_labels.auto_set(store, GAME, 150) == "train"


def test_raw_detections_api(settings):
    store, game = _game_with_wrong_path(settings)
    ball_game.fix(store, game, 30, "ball", x=0.75, y=0.75)
    with TestClient(create_app(store)) as client:
        r = client.get(f"/api/games/{GAME}/ball/raw").json()
    assert r["frame"] == sorted(r["frame"]) and len(r["frame"]) == 60
    assert abs(r["x"][0] - 43 / 320) < 1e-3 and abs(r["y"][0] - 53 / 180) < 1e-3 and r["conf"][0] == 0.5
    assert r["fix_balls"][0]["frame"] == 30 and any(p["frame"] == 30 for p in r["suppressed"])


def test_other_balls_while_watching(settings):
    from soccer_clips.track import ball_labels, ball_others

    store, game = _game_with_wrong_path(settings)
    with TestClient(create_app(store)) as client:
        url = f"/api/games/{GAME}/ball/other"
        # an unlabelled frame is labelled first, with the (detected) ring as the game ball
        r = client.post(url, json={"frame": 20, "x": 0.5, "y": 0.5, "r": 0.02})
        assert r.status_code == 201 and r.json()["labelled_as"] in ("train", "bench")
        label = ball_labels.load_labels(settings)[(GAME, 20)]
        assert label.state == "visible" and abs(label.x - (43 + 60) / 320) < 0.01
        # a second ball on the same, now labelled frame: no new label
        r = client.post(url, json={"frame": 20, "x": 0.2, "y": 0.8})
        assert r.json() == {"index": 1, "labelled_as": None}
        # no ring to take as the game ball, or labelling off: refused
        assert client.post(url, json={"frame": 150, "x": 0.5, "y": 0.5}).status_code == 422
        assert client.post(url, json={"frame": 21, "x": 0.5, "y": 0.5, "label_set": None}).status_code == 422
        client.put(f"/api/ball/others/{GAME}/20/1", json={"verdict": "not"})
        others = client.get(f"/api/games/{GAME}/ball/raw").json()["others"]
    assert others == [{"frame": 20, "index": 0, "x": 0.5, "y": 0.5, "r": 0.02}]
    assert len(ball_others.balls(ball_others.load(settings), GAME, 20)) == 1


def test_evaluate_scores_whole_game_paths_at_labels(settings):
    from soccer_clips.track import ball_labels

    store, game = _game_with_wrong_path(settings)  # the path follows junk at (43 + 3f, 53) for frames 0-59
    for f, x in ((30, (43 + 90) / 320), (40, 0.9), (100, 0.9)):  # on the path, off it, where nothing was seen
        ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=f, state="visible", x=x, y=53 / 180))
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=150, state="none"))
    ball_game.fix(store, game, 100, "ball", x=0.9, y=53 / 180)  # a fix there: ignored unless asked for
    n = ball_game.evaluate(store)[GAME]
    assert (n["visible"], n["found"], n["wrong"], n["missed"], n["none"], n["claimed"]) == (3, 1, 1, 1, 1, 0)
    assert ball_game.evaluate(store, fixes=True)[GAME]["found"] == 2
