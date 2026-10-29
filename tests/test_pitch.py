import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.scan import scan
from soccer_clips.server.app import create_app
from soccer_clips.store import Store
from soccer_clips.track import ball_game, pitch, world

TRUE = pitch.S0.copy()
TRUE[0], TRUE[1] = 72 * pitch.YD, 47 * pitch.YD  # not the 9v9 middle the solve starts from
CAM = np.array([0.0, -TRUE[1] / 2 - 5, 6.0])  # 5 m behind the near touchline, 6 m up


def _camera(pan_deg: float, tilt_deg: float, f: float) -> np.ndarray:
    """Pitch metres -> pixels for a pinhole at CAM, turned pan_deg right of straight across and tilted down."""
    th, ph = np.radians(pan_deg), np.radians(tilt_deg)
    fwd = np.array([np.sin(th) * np.cos(ph), np.cos(th) * np.cos(ph), -np.sin(ph)])
    right = np.array([np.cos(th), -np.sin(th), 0.0])
    R = np.array([right, np.cross(fwd, right), fwd])
    K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ CAM])


def _world(views):
    """A registration like world.py's: one sample per keyframe, links between neighbouring views
    (normalised like cv2.findHomography's, so some come out with the opposite sign)."""
    G = [_camera(*v) for v in views]
    links, link_H = [], []
    for a in range(len(G)):
        for b in (a + 1, a + 2):
            if b < len(G):
                m = G[b] @ np.linalg.inv(G[a])
                links.append((a, b))
                link_H.append(m / m[2, 2])
    n = len(G)
    w = {"frames": np.arange(n, dtype=np.int32) * 10, "key": np.arange(n, dtype=np.int32),
         "H": np.repeat(np.eye(3)[None], n, 0), "links": np.array(links, np.int32), "link_H": np.array(link_H)}
    return w, G


VIEWS = [(pan, 14 + abs(pan) * 0.05, 1500 + abs(pan) * 10) for pan in range(-60, 61, 8)]


def _in_view(G, pts):
    p = pitch.project(G, pts)
    return p, np.isfinite(p).all(axis=1) & (p[:, 0] > 20) & (p[:, 0] < 1900) & (p[:, 1] > 20) & (p[:, 1] < 1060)


def _clicks(G, rng, wrong=False):
    """Noisy clicks on whatever landmarks, lines and circle each view shows (a few views only)."""
    ck, px, tg = [], [], []
    for k in (0, 3, 7, 11, 15):
        for t in ["corner_ln", "corner_lf", "corner_rn", "corner_rf", "post_ln", "post_lf", "post_rn", "post_rf",
                  "centre", "half_f", "circle_f", "ga_lf", "pa_rf", "spot_l"]:
            p, ok = _in_view(G[k], pitch.landmark(t, TRUE)[None])
            if ok[0]:
                ck.append(k), px.append(p[0] + rng.normal(0, 1.5, 2)), tg.append(t)
        r = TRUE[pitch.SIZE_NAMES.index("circle")]
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            p, ok = _in_view(G[k], np.array([[r * np.cos(a), r * np.sin(a)]]))
            if ok[0]:
                ck.append(k), px.append(p[0] + rng.normal(0, 1.5, 2)), tg.append(pitch.CIRCLE)
        for x in np.linspace(-TRUE[0] / 2, TRUE[0] / 2, 9):
            for line, y in (("touch_f", TRUE[1] / 2), ("touch_n", -TRUE[1] / 2)):
                p, ok = _in_view(G[k], np.array([[x, y]]))
                if ok[0]:
                    ck.append(k), px.append(p[0] + rng.normal(0, 1.5, 2)), tg.append(line)
    if wrong:  # the far right corner, clicked on a second set of lines 4 m inside ours
        p, ok = _in_view(G[11], (pitch.landmark("corner_rf", TRUE) - [4, 4])[None])
        assert ok[0]
        ck.append(11), px.append(p[0]), tg.append("corner_rf")
    return np.array(ck), np.array(px), tg


def _errors(out, G):
    """Pitch error (m) of a grid of ground points in every view, mapped from pixels with the solve's P."""
    g = np.array([(x, y) for x in np.linspace(-33, 33, 23) for y in np.linspace(-21, 21, 15)])
    errs = []
    for k, Gk in enumerate(G):
        p, ok = _in_view(Gk, g)
        if ok.sum() < 4:
            continue
        P = np.repeat(out["P"][k][None], ok.sum(), 0)
        errs.append(np.linalg.norm(pitch.to_pitch(P, p[ok, 0], p[ok, 1]) - g[ok], axis=1))
    return np.concatenate(errs)


def test_landmarks_follow_the_sizes():
    s = pitch.S0
    L, W = s[0], s[1]
    assert np.allclose(pitch.landmark("corner_rf", s), [L / 2, W / 2])
    assert np.allclose(pitch.landmark("ga_ln", s), [-L / 2 + 5 * pitch.YD, -8 * pitch.YD])
    assert np.allclose(pitch.landmark("spot_r", s), [L / 2 - 10 * pitch.YD, 0])
    d = pitch.drawing()
    assert len(d["lines"]) == len(pitch.LINES) + 1 and len(d["posts"]) == 4


def test_solve_recovers_positions_and_field_size():
    w, G = _world(VIEWS)
    ck, px, tg = _clicks(G, np.random.default_rng(0))
    out = pitch.solve(w, ck, px, tg)
    assert np.isfinite(out["P"]).all()  # every view linked to the clicked ones
    e = _errors(out, G)
    assert np.median(e) < 0.3 and np.percentile(e, 95) < 1.0
    assert abs(out["sizes"][0] - TRUE[0]) < 1.0 and abs(out["sizes"][1] - TRUE[1]) < 1.0
    assert np.nanmax(out["click_px"]) < pitch.BAD_CLICK_PX


def test_click_on_the_wrong_lines_is_flagged():
    w, G = _world(VIEWS)
    ck, px, tg = _clicks(G, np.random.default_rng(1), wrong=True)
    out = pitch.solve(w, ck, px, tg)
    assert out["click_px"][-1] > pitch.BAD_CLICK_PX
    assert np.nanmax(out["click_px"][:-1]) < pitch.BAD_CLICK_PX
    assert np.median(_errors(out, G)) < 0.3


def test_unlinked_views_stay_uncalibrated():
    w, G = _world(VIEWS)
    keep = (w["links"] != 15).all(axis=1)  # the last view loses its links (its clicks are ignored)
    w["links"], w["link_H"] = w["links"][keep], w["link_H"][keep]
    ck, px, tg = _clicks(G, np.random.default_rng(2))
    out = pitch.solve(w, ck, px, tg)
    assert np.isnan(out["P"][15]).all() and np.isfinite(out["P"][:15]).all()
    assert np.isnan(out["click_px"][ck == 15]).all() and np.isfinite(out["click_px"][ck != 15]).all()


def test_too_few_clicks():
    w, G = _world(VIEWS)
    p = pitch.project(G[8], np.array([pitch.landmark(t, TRUE) for t in ("centre", "half_f", "circle_f")]))
    with pytest.raises(pitch.NotEnough):
        pitch.solve(w, np.full(3, 8), p, ["centre", "half_f", "circle_f"])


def test_start_from_the_centre_circle_alone():
    w, G = _world(VIEWS)
    k = 7  # the view straight across: centre spot, circle and halfway line
    r = TRUE[pitch.SIZE_NAMES.index("circle")]
    rng = np.random.default_rng(3)
    pts = [pitch.landmark(t, TRUE) for t in ("centre", "circle_n", "circle_f")]
    pts += [[r * np.cos(a), r * np.sin(a)] for a in np.radians([20, 70, 160, 200, 340])]
    px = pitch.project(G[k], np.array(pts)) + rng.normal(0, 1.0, (len(pts), 2))
    tg = ["centre", "circle_n", "circle_f"] + [pitch.CIRCLE] * 5
    ends = pitch._circle_ends(px[0], px[1], px[2], px[3:])
    assert np.abs(ends - pitch.project(G[k], np.array([[-r, 0], [r, 0]]))).max() < 4
    out = pitch.solve(w, np.full(len(tg), k), px, tg)
    g = np.array([(x, y) for x in np.linspace(-12, 12, 7) for y in np.linspace(-8, 8, 5)])  # around the circle
    p, ok = _in_view(G[k], g)
    e = np.linalg.norm(pitch.to_pitch(np.repeat(out["P"][k][None], ok.sum(), 0), p[ok, 0], p[ok, 1]) - g[ok], axis=1)
    assert np.median(e) < 0.3


def test_pitch_api(settings):
    game_id = "gu11-pitch-2026-10-10"
    make_video(settings, f"2026/{game_id}.mp4", duration_s=1, audio=False, size="1920x1080")
    store = Store(settings)
    scan(settings, store)
    game = store.load_game(game_id)
    w, G = _world(VIEWS)
    n = 10 * len(VIEWS)
    ball_game.save_npz(world.path(store, game), **w)
    ball_game.save_npz(ball_game.tracks_dir(store, game) / "ball_raw.npz", ranges=np.array([[0, n]]), dets=np.empty((0, 6)),
                       motion_frames=np.arange(n), motion=np.repeat(np.eye(2, 3)[None], n, 0))
    ck, px, tg = _clicks(G, np.random.default_rng(4))
    points = [{"frame": int(k) * 10, "x": float(x) / 1920, "y": float(y) / 1080, "target": t} for k, (x, y), t in zip(ck, px, tg)]
    with TestClient(create_app(store)) as client:
        s = client.get(f"/api/games/{game_id}/pitch").json()
        assert s["registered"] and not s["calibrated"] and not s["points"] and len(s["targets"]) > 40
        assert client.post(f"/api/games/{game_id}/pitch/points", json={**points[0], "target": "nowhere"}).status_code == 422
        pitch.save_points(store, game, points[:-2])
        assert client.post(f"/api/games/{game_id}/pitch/points", json={**points[-2], "solve": False}).json() == {"count": len(points) - 1}
        s = client.post(f"/api/games/{game_id}/pitch/points", json=points[-1]).json()
        assert s["calibrated"] and len(s["points"]) == len(points) and s["coverage"] == 1.0
        assert not any(p["bad"] for p in s["points"]) and s["points"][0]["pitch"] is not None
        assert abs(s["sizes"]["length"]["m"] - TRUE[0]) < 1 and s["sizes"]["length"]["sd"] < s["sizes"]["length"]["prior_sd"]
        rows = np.frombuffer(client.get(f"/api/games/{game_id}/pitch/frames?start=70&end=72").content, "<f4").reshape(-1, 10)
        assert rows[:, 0].tolist() == [70, 71]
        spot = pitch.landmark("centre", TRUE)[None]
        assert np.abs(pitch.project(rows[0, 1:].reshape(3, 3).astype(float), spot) - pitch.project(G[7], spot)).max() < 3
        s = client.delete(f"/api/games/{game_id}/pitch/points/{len(points) - 1}").json()
        assert s["calibrated"] and len(s["points"]) == len(points) - 1
        assert client.delete(f"/api/games/{game_id}/pitch/points/999").status_code == 404
        pitch.save_points(store, game, points[:2])
        s = client.get(f"/api/games/{game_id}/pitch").json()  # clicks changed by hand: solved again on the way
        assert not s["calibrated"] and "landmarks" in s["error"]



def test_box_sides_are_not_their_corners(tmp_path, monkeypatch):
    # a box's side line and its corner in the field once shared an id (ga_lf), and side clicks were solved as
    # the corner; saved clicks from then are read as the side, which holds whichever was meant
    assert not set(pitch.LINES) & set(pitch.LANDMARKS)
    assert {"ga_lf", "ga_lf_line"} <= set(pitch.LANDMARKS) and pitch.LINES["ga_lf_side"][:2] == ("ga_lf_line", "ga_lf")
    path = tmp_path / "pitch_points.json"
    monkeypatch.setattr(pitch, "points_path", lambda store, game: path)
    old = [{"frame": 1, "x": 0.5, "y": 0.5, "target": t} for t in ("ga_lf", "pa_rn", "ga_lf_line", "corner_ln", "goal_l")]
    path.write_text(json.dumps({"points": old}))
    assert [p["target"] for p in pitch.load_points(None, None)] == ["ga_lf_side", "pa_rn_side", "ga_lf_line", "corner_ln", "goal_l"]
    pitch.save_points(None, None, old)  # saved now: the ids mean what they say
    assert pitch.load_points(None, None) == old


def test_start_needs_four_landmarks_no_three_on_a_line():
    s = pitch.S0
    on = lambda *ts: np.array([pitch.landmark(t, s) for t in ts])
    assert pitch._general(on("corner_ln", "corner_lf", "corner_rn", "corner_rf"))
    assert not pitch._general(on("half_n", "circle_n", "circle_f", "ga_rn_line"))  # three on the halfway line
    assert not pitch._general(on("half_n", "circle_n", "circle_f", "half_f"))
    assert pitch._general(on("half_n", "circle_n", "circle_f", "ga_rn_line", "pa_rf"))
