import numpy as np

from soccer_clips.track import ball_state as bs

FPS = 10.0


def test_ball_per_frame():
    # the ball at 1 m a frame, a 3-frame gap (filled), then off the field (in the air: no place)
    f = np.r_[np.arange(10), np.arange(14, 20), [20]]
    xy = np.column_stack([f.astype(float), np.zeros(len(f))])
    off = np.r_[np.zeros(len(f) - 1, bool), True]
    out, speed, air = bs.ball_per_frame(f, xy, off, 25, FPS)
    assert np.allclose(out[10:14, 0], [10, 11, 12, 13])  # 0.4 s gap, filled in a straight line
    assert np.isnan(out[20:]).all() and air[20] and not air[:20].any()
    assert abs(speed[5] - 10.0) < 1e-6  # 1 m a frame at 10 fps


def test_nearest_and_control():
    ball = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan], [0.0, 20.0]])
    # rows: frame, team, track, xy
    frame = np.array([0, 0, 0, 1, 1, 3])
    team = np.array([0, 1, -1, 0, 1, 1])
    track = np.array([1, 2, 3, 1, 2, 2])
    xy = np.array([[1.0, 0], [5.0, 0], [0.1, 0], [11.0, 0], [11.0, 1.0], [0.0, 17.5]])
    dist, who = bs.nearest(frame, team, track, xy, ball)
    assert np.allclose(dist[0], [1, 5]) and who[0].tolist() == [1, 2]  # the referee (team -1) left out
    assert np.isinf(dist[2]).all() and who[3].tolist() == [-1, 2]
    ctl = bs.control(dist, ball, np.zeros(4, bool), width=40.0)
    # frame 3: 2.5 m away, but at the far touchline (reach 3 m)
    assert ctl.tolist() == [bs.OURS, bs.CONTESTED, bs.UNKNOWN, bs.THEIRS]


def test_bridge_possession_runs():
    O, T, L, U, C = bs.OURS, bs.THEIRS, bs.LOOSE, bs.UNKNOWN, bs.CONTESTED
    ctl = np.array([O, O, L, U, O, O, L, L, L, L, L, L, T, T, C, O], np.int8)
    br = bs.bridge(ctl, FPS)  # 2 frames between two of ours: ours; 6 loose before theirs: left
    assert br[:6].tolist() == [O] * 6 and br[6] == L
    in_play = np.ones(16, bool)
    in_play[15] = False
    known = br != U
    poss = bs.possession(br, in_play, known, FPS)
    assert poss.tolist() == [1] * 12 + [2, 2, 2, 0]  # loose stays with the run; contested doesn't change it
    assert bs.runs(poss) == [{"team": 1, "from": 0, "to": 12}, {"team": 2, "from": 12, "to": 15}]


def test_possession_forgets_after_a_long_gap():
    n = 100
    ctl = np.full(n, bs.UNKNOWN, np.int8)
    ctl[0] = bs.OURS
    known = ctl != bs.UNKNOWN
    poss = bs.possession(ctl, np.ones(n, bool), known, FPS)
    g = int(bs.GAP_S * FPS)
    assert (poss[:g + 1] == 1).all() and (poss[g + 1:] == 0).all()


def test_thirds_and_time():
    assert bs.thirds(np.array([-20.0, 0.0, 20.0, np.nan]), -1, 60.0).tolist() == [2, 1, 0, -1]  # attacking left
    poss = np.array([1] * 10 + [2] * 10 + [0] * 10, np.int8)
    in_play = np.ones(30, bool)
    xy = np.column_stack([np.r_[np.full(10, 25.0), np.full(20, np.nan)], np.zeros(30)])
    xy[10, 0] = -25.0  # theirs: in our defending third (we attack +x)
    got = bs.time_stats(poss, in_play, np.isfinite(xy[:, 0]), xy, [(0, 30)], [1], 60.0, FPS)
    g = got["game"]
    assert g["ours_s"] == 1.0 and g["theirs_s"] == 1.0 and g["unknown_s"] == 1.0 and g["ours_share"] == 0.5
    assert g["thirds"]["attacking"]["ours_s"] == 1.0 and g["thirds"]["defending"]["theirs_s"] == 1.0


def test_truth_and_compare():
    evs = [{"frame": 10, "kind": "touch", "team": 0, "keeper": 0},
           {"frame": 20, "kind": "pass", "team": 2, "keeper": 2},  # their goalie, in their own kit
           {"frame": 30, "kind": "touch", "team": 2, "keeper": 0},  # a kit read as neither: not known
           {"frame": 40, "kind": "touch", "team": 2, "keeper": 0, "player": 7},  # named: ours
           {"frame": 50, "kind": "goal", "team": 0, "keeper": 0}]
    tru = bs.truth(evs, [{"start": 5, "end": 60}], 70, FPS)
    assert tru[:10].tolist() == [0] * 10 and (tru[10:20] == 1).all() and (tru[20:30] == 2).all()
    assert (tru[30:40] == 0).all() and (tru[40:50] == 1).all() and (tru[50:] == 0).all()
    long = bs.truth(evs[:1], [{"start": 5, "end": 100}], 100, FPS)  # one touch, then nothing: a stoppage
    assert (long[10:10 + int(bs.TRUTH_GAP_S * FPS)] == 1).all() and not long[10 + int(bs.TRUTH_GAP_S * FPS):].any()
    poss = tru.copy()
    poss[25:30] = 1  # wrong for half a second
    ctl = np.where(poss == 1, bs.OURS, bs.THEIRS).astype(np.int8)
    got = bs.compare(poss, ctl, np.ones(70, bool), tru, evs, FPS)
    # labelled: 10..30 and 40..50 = 30 frames, less 0.5 s either side of the change at 20
    assert got["frames"] == 20 and got["covered"] == 1.0 and got["right"] == 0.75
    assert got["touches"] == 3 and got["touch_team"] == 1.0


def test_possession_api_not_calibrated(settings):
    from fastapi.testclient import TestClient
    from test_gallery import four_tracklets

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)  # tracked, not calibrated
    got = TestClient(create_app(store)).get(f"/api/games/{game.id}/possession").json()
    assert "calibrated" in got["error"]
