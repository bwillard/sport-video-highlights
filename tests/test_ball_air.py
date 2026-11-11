import numpy as np

from soccer_clips.track import ball_air

FPS = 30.0
MPP = 0.05  # metres per pixel across the picture where the map puts the ball


def _track():
    # the camera still (each frame's homography the same): the ball rolls across the picture for 2 s, is in the
    # air for 1 s (falling at 1.5 g at the map's scale), and rolls on
    f = np.arange(150)
    t = f / FPS
    x = 500 + 60 * t
    tau = t - 2.0
    a = 1.5 * ball_air.G / MPP  # px/s^2
    y = np.where((tau >= 0) & (tau <= 1.0), 500 - a / 2 * tau + a / 2 * tau ** 2, 500.0)
    M = np.repeat(np.eye(3)[None], len(f), axis=0)
    return f, np.column_stack([x, y]), M


def test_kappa():
    f, uv, M = _track()
    kap, rms = ball_air.kappa(f, uv, M, np.full(len(f), MPP), FPS)
    assert abs(kap[30]) < 0.05 and rms[30] < 0.1  # rolling: a straight line
    assert abs(kap[75] - 1.5) < 0.05 and rms[75] < 0.1  # in the air
    assert not (kap[60] >= ball_air.KAPPA_MIN and rms[60] <= ball_air.FIT_PX)  # across the take-off: no seed


def test_flights():
    f, uv, M = _track()
    kap, rms = ball_air.kappa(f, uv, M, np.full(len(f), MPP), FPS)
    fl = ball_air.flights(f, uv, M, kap, rms, FPS)
    assert len(fl) == 1
    a, b = fl[0]
    assert abs(a - 60) <= 2 and abs(b - 90) <= 2  # take-off and landing
    flat = uv.copy()
    flat[:, 1] = 500.0
    kap, rms = ball_air.kappa(f, flat, M, np.full(len(f), MPP), FPS)
    assert ball_air.flights(f, flat, M, kap, rms, FPS) == []


def test_place():
    # on the ground along x = frame / 10; a flight at frames 60-90 reading 20 m off that line (moved onto it),
    # one at 110-120 reading 2 m off (a ball popping up in a scramble: left as the map has it, still in the air)
    f = np.arange(150)
    seen = np.column_stack([f / 10.0, np.zeros(150)])
    seen[60:91, 1] = 20.0
    seen[110:121, 1] = 2.0
    xy, air, moved = ball_air.place(f, seen, np.ones(150, bool), [(60, 90), (110, 120)], FPS)
    assert air[60:91].all() and air[110:121].all() and not air[:60].any()
    assert moved[60:91].all() and not moved[110:121].any()
    assert np.allclose(xy[60:91], np.column_stack([f[60:91] / 10.0, np.zeros(31)]))
    assert np.allclose(xy[110:121], seen[110:121])
    # nothing on the ground near enough before it (more than ANCHOR_S) to measure it against: as the map has it
    xy, air, moved = ball_air.place(f, seen, (f < 20) | (f > 90), [(60, 90)], FPS)
    assert not moved.any() and air[60:91].all()
