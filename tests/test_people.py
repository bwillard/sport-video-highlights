from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_video
from soccer_clips import gpu
from soccer_clips.models import Callout, Clip
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import ball_game, people

GAME = "gu11-test-2026-10-10"


@pytest.fixture
def store(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=4, size="320x180")
    store = Store(settings)
    scan(settings, store)
    with store.edit_game(GAME) as g:
        g.clips += [Clip(id="c1", start_s=1.0, end_s=2.0, players=[38], callout=Callout(t_s=1.5, x=0.5, y=0.5)),
                    Clip(id="c2", start_s=3.0, end_s=3.5)]
    return store


def walkers(n: int, pan: float) -> tuple[np.ndarray, np.ndarray]:
    """Two players 30 px apart, still on the field, while the camera pans `pan` px a frame."""
    dets, steps = [], np.repeat(np.eye(2, 3)[None], n, 0)
    steps[1:, 0, 2] = -pan  # the picture slides left as the camera pans right
    for f in range(n):
        for x0 in (100, 130):
            x = x0 - pan * f
            dets.append((f, x, 50, x + 20, 110, 0.8))
    return np.array(dets, float), steps


def test_camera_motion_keeps_tracks_through_a_pan():
    dets, steps = walkers(40, pan=25.0)  # more than a box width a frame: no overlap from frame to frame
    with_cam = people.track_boxes(dets, steps, people.Params(tracker="botsort"))
    assert len(np.unique(with_cam[:, 1])) == 2
    assert len(with_cam) >= 2 * 38  # both tracked on (nearly) every frame
    without = people.track_boxes(dets, steps, people.Params(tracker="bytetrack"))
    assert len(np.unique(without[:, 1])) > 2 or len(without) < len(with_cam) / 2
    still_dets, still_steps = walkers(40, pan=0.0)
    for tracker in ("botsort", "bytetrack"):
        rows = people.track_boxes(still_dets, still_steps, people.Params(tracker=tracker))
        assert len(np.unique(rows[:, 1])) == 2


def test_ranges_for_callouts(store):
    game = store.load_game(GAME)
    a, b = game.frame_at(1.0 - people.CALLOUT_PAD_S), game.frame_at(2.0 + people.CALLOUT_PAD_S)
    last = game.frame_at(game.duration_s) - 1  # the 4 s video ends before the padding does
    assert people.ranges_for(game, "callouts") == [(max(0, a), min(last, b))]
    assert people.ranges_for(game, "all") == ball_game.ranges_for(game, "all")


def test_track_game_and_callout_yardstick(store):
    game = store.load_game(GAME)
    # her: a box under the callout (160, 90) from frame 20 to 50, leaving mid-clip; someone else throughout
    dets = [(f, 150, 60, 170, 120, 0.9) for f in range(20, 50)] + [(f, 40, 60, 60, 120, 0.9) for f in range(0, 100)]
    ball_game.save_npz(people.raw_path(store, game, "yolo11m"), ranges=np.array([[0, 100]]),
                       dets=np.array(dets, np.float32))
    ball_game.save_npz(ball_game.tracks_dir(store, game) / "ball_raw.npz", ranges=np.array([[0, 100]]),
                       dets=np.empty((0, 6)), motion_frames=np.arange(100), motion=np.repeat(np.eye(2, 3)[None], 100, 0))
    rows = people.track_game(store, game, "yolo11m")
    assert len(np.unique(rows[:, 1])) == 2
    with np.load(people.path(store, game, "yolo11m")) as z:
        assert z["x1"].max() <= 1 and z["ranges"].tolist() == [[0, 100]]
    r = people.evaluate(store, "yolo11m")
    assert people.evaluate(store, "yolo11m", saved=True) == r  # the saved tracklets score the same
    (c,) = r["callouts"]  # c2 has no callout
    clip_frames = game.frame_at(2.0) - game.frame_at(1.0)
    assert c["found"] and c["player"] == 38
    assert c["coverage"] == pytest.approx((50 - game.frame_at(1.0)) / clip_frames, abs=0.04)
    assert r["summary"]["found"] == 1.0 and r["summary"]["people_per_frame"] == pytest.approx(1.3, abs=0.05)


def test_detect_game_is_resumable(store, monkeypatch):
    calls = []

    class Model:
        def predict(self, batch, **kw):
            calls.append(len(batch))
            box = SimpleNamespace(xyxy=np.array([[10.0, 20, 30, 80]]), conf=np.array([0.7]))
            return [SimpleNamespace(boxes=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: box)))
                    for _ in batch]

    monkeypatch.setattr(people, "_load_model", lambda store, name: Model())
    game = store.load_game(GAME)
    (s, e), = people.ranges_for(game, "callouts")
    assert people.detect_game(store, GAME, mode="callouts") == e - s
    raw = people.load_raw(store, game, "yolo11m")
    assert raw["ranges"].tolist() == [[s, e]]
    assert raw["dets"][:, 0].tolist() == list(range(s, e))  # one box per frame, numbered in game frames
    assert people.detect_game(store, GAME, mode="callouts") == 0  # nothing left to do
    n = len(calls)
    assert people.detect_game(store, GAME, mode="all") == 0  # in a 4 s video the callout range is the whole game
    assert len(calls) == n


def test_people_job_uploads_only_what_it_needs(store):
    game = store.load_game(GAME)
    ball_game.save_npz(people.raw_path(store, game, "yolo26l"), ranges=np.array([[0, 10]]),
                       dets=np.empty((0, 6), np.float32))
    job = gpu.people_games_job(store, [GAME], mode="callouts", models=["yolo11m", "yolo26l"])
    assert sorted(job.uploads) == [f"games/2026/{GAME}/game.json", f"games/2026/{GAME}/tracks/people_raw_yolo26l.npz"]
    assert job.volume_files == [f"full/2026/{GAME}.mp4"]  # no weights here: the pod downloads them
    assert "--model $m" in job.script and "yolo11m yolo26l" in job.script and "--mode callouts" in job.script
    results = store.settings.games_dir.parent / "run" / "results"
    f = results / f"games/2026/{GAME}/tracks/people_raw_yolo11m.npz"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert gpu.install_people_raw(store, results, log=lambda m: None) == 1
    assert people.raw_path(store, game, "yolo11m").read_bytes() == b"x"


def two_players(store):
    """Tracklets for two players standing apart for 100 frames: left (x 0.1-0.2), right (x 0.6-0.7)."""
    game = store.load_game(GAME)
    dets = [(f, 32, 60, 64, 120, 0.9) for f in range(100)] + [(f, 192, 60, 224, 120, 0.9) for f in range(100)]
    ball_game.save_npz(people.raw_path(store, game, "yolo11m"), ranges=np.array([[0, 100]]),
                       dets=np.array(dets, np.float32))
    ball_game.save_npz(ball_game.tracks_dir(store, game) / "ball_raw.npz", ranges=np.array([[0, 100]]),
                       dets=np.empty((0, 6)), motion_frames=np.arange(100), motion=np.repeat(np.eye(2, 3)[None], 100, 0))
    people.track_game(store, game, "yolo11m")
    return game


def test_picks_label_tracklets_from_clicks(store):
    game = two_players(store)
    t = people.tracks(store, game)
    left = t.at(50, 0.15, 0.5)
    assert left is not None and t.at(50, 0.65, 0.5) not in (None, left) and t.at(50, 0.4, 0.5) is None
    with pytest.raises(ValueError, match="no tracked player"):
        people.add_pick(store, game, 50, 38, "is", 0.4, 0.5)
    people.add_pick(store, game, 50, 38, "is", 0.15, 0.5)
    (s,) = people.segments(t, people.load_picks(store, game))
    assert s == {"track": left, "start": t.first[left], "end": t.last[left] + 1, "player": 38}
    assert people.follows(t, people.load_picks(store, game), 38, 80) == left
    # the marker drifts onto someone else at 70: "not" from there, no click needed
    people.add_pick(store, game, 70, 38, "not")
    picks = people.load_picks(store, game)
    assert [(s["start"], s["end"]) for s in people.segments(t, picks)] == [(t.first[left], 70)]
    assert people.follows(t, picks, 38, 80) is None and people.follows(t, picks, 38, 60) == left
    with pytest.raises(ValueError, match="no marker"):
        people.add_pick(store, game, 80, 38, "not")
    assert people.undo_pick(store, game)["kind"] == "not"
    assert people.follows(t, people.load_picks(store, game), 38, 80) == left
    # clicks survive re-tracking: they're matched to whatever tracklet is under them now
    people.track_game(store, game, "yolo11m", people.Params(track_buffer=10))
    t2 = people.tracks(store, game)
    assert t2 is not t and len(people.segments(t2, people.load_picks(store, game))) == 1


def test_people_api(store):
    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app

    with TestClient(create_app(store)) as client:
        url = f"/api/games/{GAME}/people"
        assert client.get(url).json() == {"ranges": [], "teams": {}, "picks": [], "segments": []}
        assert client.post(f"{url}/picks", json={"frame": 5, "player": 38, "kind": "is", "x": 0.15, "y": 0.5}
                           ).status_code == 422  # not tracked yet
        two_players(store)
        assert client.get(url).json()["ranges"] == [[0, 100]]
        rows = np.frombuffer(client.get(f"{url}/boxes", params={"start": 10, "end": 20}).content, "<f4").reshape(-1, 7)
        assert len(rows) == 20 and set(rows[:, 0]) == set(range(10, 20)) and set(rows[:, 6]) == {-1}  # no kit yet
        assert np.allclose(rows[rows[:, 2] < 0.5][0, 2:6], [0.1, 60 / 180, 0.2, 120 / 180], atol=1e-4)
        assert client.get(f"{url}/boxes", params={"start": 0, "end": 5000}).status_code == 422
        r = client.post(f"{url}/picks", json={"frame": 50, "player": 38, "kind": "is", "x": 0.15, "y": 0.5}).json()
        assert r["pick"]["player"] == 38 and len(r["segments"]) == 1
        assert client.post(f"{url}/picks", json={"frame": 50, "player": 7, "kind": "is", "x": 0.4, "y": 0.5}
                           ).json()["detail"] == "no tracked player there"
        r = client.delete(f"{url}/picks/last").json()
        assert r["undone"]["frame"] == 50 and r["segments"] == []


def crossing(n: int = 60):
    """Two players run through each other: A left to right in red, B right to left in blue, crossing
    at frame n/2, where the detector loses B for a few frames. Returns dets, kit feats, still steps."""
    from soccer_clips.track import kit

    red, blue = np.zeros(kit.NBINS, np.uint8), np.zeros(kit.NBINS, np.uint8)
    red[0], blue[10] = 255, 255
    dets, feats = [], []
    for f in range(n):
        xa, xb = 100 + 4 * f, 100 + 4 * n - 4 * f
        dets.append((f, xa, 50, xa + 20, 110, 0.9))
        feats.append(red)
        if abs(f - n // 2) > 2:
            dets.append((f, xb, 52, xb + 20, 112, 0.9))
            feats.append(blue)
    return np.array(dets, float), np.array(feats), np.repeat(np.eye(2, 3)[None], n, 0)


def test_kit_gate_keeps_tracks_on_their_own_kit():
    dets, feats, steps = crossing()

    def kits_per_track(rows):
        return [set(feats[rows[rows[:, 1] == t, 7].astype(int)].argmax(1)) for t in np.unique(rows[:, 1])]

    gated = people.track_boxes(dets, steps, people.Params(kit_gate=0.5), feats)
    assert all(len(k) == 1 for k in kits_per_track(gated))  # nobody's track takes the other's box
    assert gated.shape[1] == 8 and np.array_equal(dets[gated[:, 7].astype(int), 0], gated[:, 0])


def test_split_on_kit_cuts_a_swapped_tracklet():
    from soccer_clips.track import kit

    red, blue = np.zeros(kit.NBINS, np.uint8), np.zeros(kit.NBINS, np.uint8)
    red[0], blue[10] = 255, 255
    feats = np.array([red] * 40 + [blue] * 40)
    rows = np.array([(f, 1, 0, 0, 10, 30, 0.9, f) for f in range(80)], float)  # one track: red, then blue
    out, nxt = people.split_on_kit(rows, feats, 0.5, 2)
    assert nxt == 3 and set(out[out[:, 0] < 40, 1]) == {1} and set(out[out[:, 0] >= 40, 1]) == {2}
    same = np.array([red] * 80)
    assert len(np.unique(people.split_on_kit(rows, same, 0.5, 2)[0][:, 1])) == 1


def test_link_joins_a_gap_in_the_same_kit_only():
    from soccer_clips.track import kit

    red, blue = np.zeros(kit.NBINS, np.uint8), np.zeros(kit.NBINS, np.uint8)
    red[0], blue[10] = 255, 255
    # track 1 runs right and stops at frame 29; track 2 starts at 40 where it would be by then (red);
    # track 3 starts at the same spot in blue
    rows = [(f, 1, 100 + 2 * f, 50, 120 + 2 * f, 110, 0.9, f) for f in range(30)]
    rows += [(f, 2, 100 + 2 * f, 50, 120 + 2 * f, 110, 0.9, f) for f in range(40, 70)]
    rows = np.array(rows, float)
    p = people.Params(link_gap_s=1.0)
    steps = np.repeat(np.eye(2, 3)[None], 70, 0)
    feats = np.array([red] * 70)
    assert len(np.unique(people.link_tracklets(rows, feats, steps, p, 30.0)[:, 1])) == 1
    feats[40:] = blue
    assert len(np.unique(people.link_tracklets(rows, feats, steps, p, 30.0)[:, 1])) == 2
    feats[40:] = red  # too far away: a 500 px jump in a third of a second
    far = rows.copy()
    far[30:, [2, 4]] += 500
    assert len(np.unique(people.link_tracklets(far, feats, steps, p, 30.0)[:, 1])) == 2


def test_label_yardstick(store):
    game = two_players(store)
    t = people.tracks(store, game)
    known = [(10, 0.15, 0.5), (80, 0.15, 0.5)]
    r = people.check_labels(t, known, [(60, 0.65, 0.5)], 30.0)
    assert r["found"] == 2 and r["conflicts"] == 0 and r["swaps_left"] == 0
    assert r["holds"][0] == pytest.approx(100 / 30, abs=0.1)
    r = people.check_labels(t, known + [(50, 0.65, 0.5)], [(60, 0.15, 0.5)], 30.0)  # a "known" point on the other player
    assert r["conflicts"] == 3 and r["swaps_left"] == 1  # each tracklet contradicts the points on the other


def test_team_clusters_and_roles(store):
    from soccer_clips.track import kit

    rng = np.random.default_rng(0)
    colours = np.eye(kit.NBINS, dtype=np.float32)[[0, 10, 4, 16]]  # red (us), blue (them), yellow (ref), dark
    means = np.vstack([colours[g] + 0.05 * rng.random(kit.NBINS) for g in [0] * 10 + [1] * 10 + [2] * 2 + [3] * 3])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    n = np.full(len(means), 100)
    n[-1] = 0  # no known kit
    cluster, centres = kit.cluster_teams(means, n, k=4)
    assert len(set(cluster[:10])) == 1 and len(set(cluster[10:20])) == 1 and cluster[0] != cluster[10]
    assert cluster[-1] == -1 and centres.shape == (4, kit.NBINS)
    # roles: the cluster under the labelled player is ours, the biggest other one opponents
    game = two_players(store)
    t = people.tracks(store, game)
    left, right = t.at(50, 0.15, 0.5), t.at(50, 0.65, 0.5)
    t.team = {left: 2, right: 5}
    picks = [{"frame": 50, "x": 0.15, "y": 0.5, "player": 38, "kind": "is"}]
    assert people.team_roles(t, picks, game, 38) == {2: "ours", 5: "opponents"}
    assert people.team_roles(t, [], game, 38) == {2: "team", 5: "team"}
