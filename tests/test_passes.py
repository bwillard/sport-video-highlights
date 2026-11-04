import numpy as np

from soccer_clips.track import passes as ps

FPS = 10.0


def touch(frame, track, team, keeper=0):
    return {"frame": frame, "track": track, "team": team, "keeper": keeper}


def line(n, pts):
    """A ball path through (frame, x) points along y = 0, straight between them."""
    f, x = zip(*pts)
    return np.column_stack([np.interp(np.arange(n), f, x), np.zeros(n)])


def test_find_outcomes():
    n = 200
    ball = line(n, [(0, 0), (10, 10), (20, 10), (22, 12), (40, 12), (50, 20), (90, 20), (100, 30), (199, 30)])
    in_play = np.ones(n, bool)
    poss = np.full(n, 2, np.int8)
    tc = [touch(0, 1, 0), touch(5, 1, 0),  # a dribble: not a pass
          touch(10, 1, 0), touch(20, 2, 0),  # the ball still: no pass
          touch(22, 2, 0), touch(40, 3, 1),  # not carried far enough: a tackle, not a pass
          touch(42, 3, 1), touch(52, 4, 1),  # 6.4 m: completed
          touch(55, 4, 1), touch(95, 5, 0),  # the next touch 4 s on: receiver unknown; theirs then (poss 2)
          touch(100, 5, 0), touch(110, 6, 1)]
    ball[100:110, 0] = np.linspace(30, 40, 10)
    got = ps.find(tc, ball, in_play, poss, FPS)
    assert [(p["frame"], p["outcome"]) for p in got] == [(42, "completed"), (55, "completed"), (100, "lost")]
    assert got[1]["to_track"] is None and got[1]["to_frame"] == 85 and got[0]["to_track"] == 4
    in_play[105:] = False
    got = ps.find(tc, ball, in_play, poss, FPS)
    assert got[-1]["outcome"] == "out" and got[-1]["to_frame"] == 105
    tc[11]["team"] = -1
    in_play[:] = True
    assert ps.find(tc, ball, in_play, poss, FPS)[-1]["outcome"] == "unknown"  # a kit read as neither


def test_into_danger():
    n = 100
    ball = line(n, [(0, 0), (10, 10), (99, 10)])
    tc = [touch(0, 1, 0), touch(10, 2, 0), touch(15, 2, 0), touch(20, 7, 1)]  # received, lost 1 s later
    got = ps.find(tc, ball, np.ones(n, bool), np.zeros(n, np.int8), FPS)
    assert got[0]["outcome"] == "completed" and got[0]["danger"]
    tc[3]["frame"] = 40  # kept 3 s
    assert not ps.find(tc, ball, np.ones(n, bool), np.zeros(n, np.int8), FPS)[0]["danger"]


def test_direction_and_describe():
    assert ps.direction(10, 2, 1) == "forward" and ps.direction(10, 2, -1) == "back"
    assert ps.direction(1, 5, 1) == "sideways" and ps.direction(1, 5, None) is None
    ball = line(50, [(0, -25), (10, 5), (49, 5)])
    air = np.zeros(50, bool)
    air[5] = True
    p = [{"frame": 0, "to_frame": 10}]
    ps.describe(p, ball, air, [(0, 50)], [1], 60.0)
    assert p[0]["length"] == 30.0 and p[0]["direction"] == "forward" and p[0]["air"]
    assert p[0]["from_third"] == 0 and p[0]["to_third"] == 1 and p[0]["to_x"] == 5.0


def test_chains_and_stats():
    in_play = np.ones(200, bool)
    tc = [touch(f, tr, tm) for f, tr, tm in
          [(0, 1, 0), (10, 2, 0), (20, 3, 0), (30, 4, 0), (40, 9, 1), (50, 8, 1), (60, 7, 1), (70, 1, 0)]]
    p = lambda f, to, team, oc: {"frame": f, "to_frame": to, "team": team, "outcome": oc,  # noqa: E731
                                 "danger": False, "direction": "forward"}
    passes = [p(0, 10, 0, "completed"), p(10, 20, 0, "completed"), p(20, 30, 0, "completed"),
              p(30, 40, 0, "lost"), p(50, 60, 1, "completed")]
    chs = ps.chains(passes, tc, in_play)
    assert [(c["team"], c["passes"], c["end"]) for c in chs] == [(0, 3, "lost"), (1, 1, "lost")]
    assert [x["chain"] for x in passes] == [0, 0, 0, None, 1]
    s = ps.team_stats(passes, chs, 0)
    assert s["passes"] == 4 and s["completion"] == 0.75 and s["chains"] == {"2": 0, "3": 1, "4+": 0, "longest": 3}
    in_play[15] = False  # out between the first pass's receiver and the next touch
    assert ps.chains([p(0, 10, 0, "completed")], tc[:3], in_play)[0]["end"] == "out"


def test_compare():
    windows = [{"start": 0, "end": 100}]
    found = [{"frame": 10, "track": 1, "to_track": 2, "outcome": "completed"},
             {"frame": 40, "track": 5, "to_track": 6, "outcome": "completed"},
             {"frame": 70, "track": 7, "to_track": None, "outcome": "lost"}]
    evs = [{"frame": 12, "kind": "pass", "track": 1, "team": 0, "keeper": 0},
           {"frame": 20, "kind": "touch", "track": 2, "team": 0, "keeper": 0},
           {"frame": 41, "kind": "pass", "track": 5, "team": 0, "keeper": 0},
           {"frame": 50, "kind": "touch", "track": 3, "team": 1, "keeper": 0},
           {"frame": 90, "kind": "pass", "track": 8, "team": 0, "keeper": 0}]
    got = ps.compare(found, evs, windows, np.ones(100, bool), FPS)
    assert got["labelled_n"] == 3 and got["found_n"] == 3 and got["found"] == 0.667 and got["labelled"] == 0.667
    assert got["kicker"] == 1.0 and got["receiver"] == 0.5 and got["outcome"] == 0.5 and got["outcome_n"] == 2


def test_passes_api_not_calibrated(settings):
    from fastapi.testclient import TestClient
    from test_gallery import four_tracklets

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)  # tracked, not calibrated
    got = TestClient(create_app(store)).get(f"/api/games/{game.id}/passes").json()
    assert "calibrated" in got["error"]
