import numpy as np

from soccer_clips.track import touches as tc

FPS = 10.0


def test_nearest_two():
    ball = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan]])
    frame = np.array([0, 0, 0, 1, 2])
    xy = np.array([[1.0, 0], [0, 3.0], [np.nan, np.nan], [10.0, 0.5], [0.0, 0.0]])
    d1, row, d2 = tc.nearest_two(frame, xy, ball)
    assert d1[:2].tolist() == [1.0, 0.5] and row.tolist() == [0, 3, -1] and d2[0] == 3.0
    assert np.isinf(d2[1]) and np.isinf(d1[2])  # one player: no next nearest; no ball: nobody


def test_find_a_kick_not_a_runner_by():
    n = 60
    t = np.arange(n) / FPS
    ball = np.column_stack([np.where(t < 3, 2.0 * t, 6.0), np.where(t < 3, 0.0, -8.0 * (t - 3))])  # turned at 3 s
    d1 = np.full(n, 6.0)
    d1[25:36] = np.abs(np.arange(25, 36) - 30) * 0.3  # a player at the ball as it turns
    d1[10:21] = np.abs(np.arange(10, 21) - 15) * 0.3  # another running past it at 1.5 s: the ball goes on
    got = tc.find(ball, np.ones(n, bool), d1, FPS)
    assert got.tolist() == [30]
    assert tc.find(ball, np.r_[np.ones(20, bool), np.zeros(40, bool)], d1, FPS).tolist() == []  # out of play


def test_runs_and_describe():
    frames = np.array([10, 20, 60, 70])
    assert tc.runs(frames, np.array([5, 5, 5, 6]), FPS).tolist() == [0, 0, 1, 2]  # 4 s later: a new run
    ball = np.zeros((80, 2))
    d1, d2 = np.full(80, 0.5), np.full(80, np.inf)
    d2[20], d2[60] = 0.6, 4.5
    row = np.array([0] * 40 + [1] * 40)
    got = tc.describe(frames, ball, d1, row, d2, np.array([5, 6]), np.array([0, -1]), np.array([0, 0]), FPS)
    assert got[1]["ambiguous"] and got[1]["confidence"] == 0.05 and not got[0]["ambiguous"]
    assert got[0]["confidence"] == 1.0 and got[2]["team"] == -1 and got[2]["track"] == 6


def test_pair_and_compare():
    assert tc.pair([10, 12, 40], [11, 30, 41], 3) == [(0, 0), (2, 2)]  # 12 has nobody left near
    windows = [{"start": 0, "end": 100}]
    found = [{"frame": 10, "track": 1, "team": 0}, {"frame": 50, "track": 2, "team": 1},
             {"frame": 80, "track": 3, "team": 0}, {"frame": 150, "track": 4, "team": 0}]  # the last outside
    evs = [{"frame": 12, "kind": "pass", "track": 1, "team": 0, "keeper": 0},
           {"frame": 52, "kind": "touch", "track": 9, "team": 1, "keeper": 0},
           {"frame": 70, "kind": "goal", "track": None, "team": None, "keeper": 0},
           {"frame": 90, "kind": "shot", "track": 5, "team": 0, "keeper": 0}]
    got = tc.compare(found, evs, windows, FPS)
    assert got["labelled_n"] == 3 and got["found_n"] == 3 and got["found"] == 0.667 and got["labelled"] == 0.667
    assert got["player"] == 0.5 and got["team"] == 1.0 and got["kinds"] == {"pass": 1.0, "touch": 1.0, "shot": 0.0}


def test_touches_api_not_calibrated(settings):
    from fastapi.testclient import TestClient
    from test_gallery import four_tracklets

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)  # tracked, not calibrated
    got = TestClient(create_app(store)).get(f"/api/games/{game.id}/touches").json()
    assert "calibrated" in got["error"]


def test_correct_from_labels():
    ball = np.column_stack([np.arange(100, dtype=float), np.zeros(100)])
    found = [{"frame": 10, "x": 10.0, "y": 0.0, "track": 1, "team": 0, "keeper": 0, "margin": 0.1, "ambiguous": True,
              "confidence": 0.05, "run": 0},
             {"frame": 30, "x": 30.0, "y": 0.0, "track": 2, "team": 1, "keeper": 0, "margin": 2, "ambiguous": False,
              "confidence": 1.0, "run": 1},
             {"frame": 50, "x": 50.0, "y": 0.0, "track": 3, "team": 1, "keeper": 0, "margin": 2, "ambiguous": False,
              "confidence": 1.0, "run": 2}]
    def lab(f, kind, track, team):
        return {"frame": f, "kind": kind, "track": track, "team": team, "keeper": 0, "player": None, "guess": None,
                "at": [1.0, 2.0]}
    labs = [lab(12, "pass", 7, 0),  # the other player of the two: fixed, and a pass
            lab(31, "touch", 2, 1),  # the same player: its kind only
            lab(80, "shot", 9, 1),  # none found near: added, on the ball
            lab(90, "touch", None, None),  # on nobody: nothing
            lab(70, "goal", 5, 0)]  # not a touch
    got = tc.correct(found, labs, [lab(52, "none", None, None)], ball, FPS)
    assert [(t["frame"], t["track"], t.get("label"), bool(t.get("fixed")), bool(t.get("added"))) for t in got] == [
        (10, 7, "pass", True, False), (30, 2, "touch", False, False), (80, 9, "shot", False, True)]
    assert got[2]["x"] == 80.0 and got[0]["team"] == 0 and [t["run"] for t in got] == [0, 1, 2]
    assert found[0]["track"] == 1  # what was found is left as it was
    assert tc.label_team({"keeper": 2, "team": 0}) == 1 and tc.label_team({"team": 2, "player": 7}) == 0
