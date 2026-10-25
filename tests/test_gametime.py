import json
import subprocess

import numpy as np
import pytest

from soccer_clips import gametime
from soccer_clips.models import Callout, Clip, CutSegment
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import ball_game, ball_labels, kit, people

GAME = "gu11-test-2026-10-10"


def test_time_map():
    tm = gametime.TimeMap([CutSegment(orig_start=30, orig_end=120, new_start=0),
                           CutSegment(orig_start=180, orig_end=300, new_start=90)], 30, 1)
    new, ok = tm.frames([0, 30, 119, 120, 179, 180, 299, 300])
    assert ok.tolist() == [False, True, True, False, False, True, True, False]
    assert new[ok].tolist() == [0, 89, 90, 209]
    assert tm.time(7.0) == pytest.approx(7.0 - 90 / 30) and tm.time(5.0) is None
    assert tm.span(3.5, 6.5) == pytest.approx((2.5, 3.5), abs=1e-5)  # clipped at the end of the 1st segment
    assert tm.span(4.5, 5.5) is None  # all in half time
    assert tm.ranges([[0, 400]]).tolist() == [[0, 210]]  # both pieces, joined on the new timeline


@pytest.fixture
def store(settings, monkeypatch):
    monkeypatch.setattr(gametime, "PAD_S", 0.5)
    out = settings.full_dir / f"2026/{GAME}.mp4"
    out.parent.mkdir(parents=True)
    # 12 s, a keyframe every 15 frames (0.5 s) and no B-frames, like a Veo file
    subprocess.run([settings.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-c:v", "libx264", "-g", "15",
                    "-keyint_min", "15", "-sc_threshold", "0", "-bf", "0", "-c:a", "aac", "-shortest", str(out)],
                   check=True)
    store = Store(settings)
    scan(settings, store)
    return store


def test_cut_moves_everything_onto_game_time(store, settings):
    with store.edit_game(GAME) as g:
        g.kickoff_1h_s, g.end_1h_s, g.kickoff_2h_s, g.end_2h_s = 1.0, 4.0, 6.0, 10.0
        g.clips = [Clip(id="h2", start_s=7.0, end_s=8.0, players=[38], callout=Callout(t_s=7.5, x=0.5, y=0.5)),
                   Clip(id="ht", start_s=4.8, end_s=5.2)]  # in half time: dropped
    game = store.load_game(GAME)
    fps = game.fps
    d = ball_game.tracks_dir(store, game)
    n = game.frame_at(game.duration_s)
    frames = np.arange(0, n, 3)
    ball_game.save_npz(d / "ball_raw.npz", ranges=np.array([[0, n]]),
                       dets=np.column_stack([frames, np.full((len(frames), 4), 10.0), np.full(len(frames), 0.5)]),
                       motion_frames=np.arange(n), motion=np.repeat(np.eye(2, 3)[None], n, 0))
    dets = np.column_stack([frames, np.full(len(frames), 10.0), np.full(len(frames), 10.0),
                            np.full(len(frames), 40.0), np.full(len(frames), 90.0), np.full(len(frames), 0.9)])
    ball_game.save_npz(people.raw_path(store, game, "yolo11m"), ranges=np.array([[0, n]]), dets=dets.astype(np.float32))
    feats = np.zeros((len(frames), kit.NBINS), np.uint8)
    feats[:, 0] = np.arange(len(frames)) % 250  # a marker per row, to check rows stay in step
    ball_game.save_npz(kit.kit_path(store, game, "yolo11m"), feats=feats)
    ball_game.save_npz(people.path(store, game, "yolo11m"), frame=frames.astype(np.int32),
                       track=np.ones(len(frames), np.int32), x1=np.full(len(frames), 0.1, np.float32),
                       y1=np.full(len(frames), 0.1, np.float32), x2=np.full(len(frames), 0.2, np.float32),
                       y2=np.full(len(frames), 0.5, np.float32), conf=np.full(len(frames), 0.9, np.float32),
                       det=np.arange(len(frames), dtype=np.int32), ranges=np.array([[0, n]], np.int32))
    f2h, fht = game.frame_at(7.5), game.frame_at(5.0)
    (d / "people_picks.json").write_text(json.dumps({"picks": [
        {"frame": f2h, "x": 0.15, "y": 0.3, "player": 38, "kind": "is", "track": 1},
        {"frame": fht, "x": 0.15, "y": 0.3, "player": 38, "kind": "is", "track": 1}]}))
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=f2h, state="none"))
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=fht, state="none"))

    report = gametime.cut_game(store, GAME, log=lambda m: None)

    game2 = store.load_game(GAME)
    seg = game2.cut.segments
    assert len(seg) == 2 and seg[0].orig_start == 15 and seg[1].new_start == seg[0].orig_end - seg[0].orig_start
    shift = (seg[1].orig_start - seg[1].new_start) / fps  # seconds removed before the 2nd half
    assert store.original_path(game2) == settings.cut_root / f"2026/{GAME}.mp4"
    assert store.full_path(game2).exists()  # the original stays
    assert game2.duration_s == pytest.approx(sum(s.orig_end - s.orig_start for s in seg) / fps, abs=0.1)
    (c,) = game2.clips
    assert c.id == "h2" and c.start_s == pytest.approx(7.0 - shift) and c.callout.t_s == pytest.approx(7.5 - shift)
    assert report["clips dropped (outside game time)"] == ["ht"]
    assert game2.kickoff_2h_s == pytest.approx(6.0 - shift)
    # the same picture at the same game moment, in the cut video
    k = game.frame_at(8.0)
    a = next(ball_game.read_frames(store, game, k, k + 1)).astype(int)
    b = next(ball_game.read_frames(store, game2, k - round(shift * fps), k - round(shift * fps) + 1)).astype(int)
    assert np.abs(a - b).mean() < 1
    # player boxes, kit colours and tracklets dropped and moved together
    raw = people.load_raw(store, game2, "yolo11m")
    tm = gametime.TimeMap(seg, game.fps_num, game.fps_den)
    old_new, keep = tm.frames(frames)
    assert raw["dets"][:, 0].tolist() == old_new[keep].tolist()
    assert kit.load(store, game2, "yolo11m")[:, 0].tolist() == feats[keep, 0].tolist()
    t = people.tracks(store, game2, "yolo11m")
    assert len(t.frame) == keep.sum() and t.ranges == tm.ranges([[0, n]]).tolist()
    with np.load(people.path(store, game2, "yolo11m")) as z:
        assert (raw["dets"][z["det"], 0] == z["frame"]).all()
    picks = people.load_picks(store, game2)
    assert [p["frame"] for p in picks] == [tm.frame(f2h)] and report["player labels dropped"] == 1
    assert {(g, f) for (g, f) in ball_labels.load_labels(settings)} == {(GAME, tm.frame(f2h))}
    raw_ball = ball_game.load_raw(store, game2)
    assert raw_ball["ranges"].tolist() == tm.ranges([[0, n]]).tolist()
    # ...and the backup puts it all back
    with pytest.raises(ValueError, match="already been cut"):
        gametime.cut_game(store, GAME, log=lambda m: None)
    gametime.restore(store, GAME, log=lambda m: None)
    back = store.load_game(GAME)
    assert back.cut is None and len(back.clips) == 2 and back.clips[0].start_s == 7.0
    assert people.load_raw(store, back, "yolo11m")["dets"][:, 0].tolist() == frames.tolist()


def test_cut_needs_all_four_times(store):
    with store.edit_game(GAME) as g:
        g.kickoff_1h_s, g.kickoff_2h_s = 1.0, 6.0
    with pytest.raises(ValueError, match="set all four"):
        gametime.cut_game(store, GAME, log=lambda m: None)
    with store.edit_game(GAME) as g:
        g.end_1h_s, g.end_2h_s = 7.0, 10.0  # 1st half ends after the 2nd starts
    with pytest.raises(ValueError, match="in order"):
        gametime.cut_game(store, GAME, log=lambda m: None)


def test_gametime_api(store):
    import time

    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app

    with TestClient(create_app(store)) as client:
        url = f"/api/games/{GAME}"
        assert client.get(f"{url}/gametime/suggest").json()["notes"] == ["no player tracks yet"]
        assert client.post(f"{url}/gametime/cut").status_code == 422  # times not set
        g = client.patch(url, json={"kickoff_1h_s": 1.0, "end_1h_s": 4.0, "kickoff_2h_s": 6.0, "end_2h_s": 10.0}).json()
        assert g["end_1h_s"] == 4.0 and g["end_2h_s"] == 10.0
        job = client.post(f"{url}/gametime/cut").json()
        for _ in range(100):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert job["status"] == "done", job
        g = client.get(url).json()
        assert g["cut"]["file"] == f"2026/{GAME}.mp4" and g["kickoff_1h_s"] < 1.0
        assert client.get(f"/video/{GAME}").headers["content-length"] != "0"
        assert client.post(f"{url}/gametime/cut").status_code == 409
        assert client.post(f"{url}/gametime/restore").json()["cut"] is None
