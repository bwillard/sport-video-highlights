import numpy as np

from soccer_clips.track import ball_track, camera


def still_camera(n):
    return np.repeat(np.eye(2, 3)[None], n, axis=0)


def box(f, x, y, conf, r=5):
    return [f, 32, -1, x - r, y - r, x + r, y + r, conf]


def test_camera_reference_undoes_a_pan():
    steps = still_camera(4)
    steps[1:, 0, 2] = -10  # the camera pans right: the scene moves 10 px left each frame
    ref = camera.to_reference(steps)
    x, y = camera.apply(ref[3], np.array([500.0]), np.array([300.0]))
    assert np.allclose([x[0], y[0]], [530, 300])  # 3 frames later, 30 px further left on screen


def test_follows_the_moving_ball_ignores_a_spare_and_fills_gaps():
    n = 120
    rows = []
    for f in range(n):
        bx = 200 + 8 * f  # game ball rolling right
        if f % 10 not in (4, 5, 6):  # missed now and then
            rows.append(box(f, bx, 500, 0.5))
        rows.append(box(f, 1500, 900, 0.6))  # spare ball lying still, slightly more confident
        if f == 60:
            rows.append(box(f, 900, 200, 0.9))  # one-off junk far away
    path = ball_track.track(np.array(rows, np.float32), still_camera(n))
    kinds = path[:, 4]
    assert (kinds != ball_track.NONE).all()
    on_ball = np.hypot(path[:, 0] - (200 + 8 * np.arange(n)), path[:, 1] - 500) < 3
    assert on_ball.all()
    assert (kinds[[4, 5, 6, 64]] == ball_track.FILLED).all()
    assert (kinds[[0, 3, 7]] == ball_track.DETECTED).all()


def test_long_parked_ball_never_used_but_a_short_stop_is():
    n = 200
    rows = [box(f, 1500, 900, 0.6) for f in range(n)]  # spare: still for the whole clip
    path = ball_track.track(np.array(rows, np.float32), still_camera(n))
    assert (path[:, 4] == ball_track.NONE).all()  # no ball beats a spare ball
    rows = [box(f, 700, 500, 0.6) for f in range(60)]  # the game ball waiting 2 s for a goal kick...
    rows += [box(f, 700 + 10 * (f - 60), 500, 0.6) for f in range(60, 120)]  # ...then kicked
    path = ball_track.track(np.array(rows, np.float32), still_camera(120))
    assert (path[:, 4] == ball_track.DETECTED).all()


def test_long_gap_left_empty():
    n = 100
    rows = [box(f, 300 + 5 * f, 400, 0.5) for f in list(range(0, 20)) + list(range(70, 100))]
    path = ball_track.track(np.array(rows, np.float32), still_camera(n))
    assert (path[25:65, 4] == ball_track.NONE).all()  # 50 empty frames: longer than the 1 s fill
