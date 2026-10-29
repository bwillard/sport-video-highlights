from types import SimpleNamespace

import numpy as np
import pytest

from soccer_clips.track import bridge, people

GAME = SimpleNamespace(id="g", fps_num=30, fps_den=1, width=1920, height=1080)

# tracklet: (first frame, last frame, start xy, end xy, team); 30 fps
TRACKS = {
    1: (0, 100, (0, 0), (0, 0), 0),  # a plain link: 1 ends, 2 starts beside it a third of a second later
    2: (110, 300, (0.5, 0), (5, 0), 0),
    3: (0, 100, (10, 0), (10, 0), 0),  # a crowd: 3 and 4 end side by side, 5 starts between them
    4: (0, 100, (11, 0), (11, 0), 0),
    5: (110, 300, (10.5, 0), (10.5, 0), 0),
    6: (0, 100, (20, 0), (20, 0), 0),  # scraps: 6 ends, 7 (a scrap) starts there, then 8 a bit later
    7: (105, 110, (20.5, 0), (21, 0), 0),
    8: (150, 300, (21.5, 0), (22, 0), 0),
    9: (0, 100, (30, 0), (30, 0), 0),  # 10 starts 4 m on, 3 s later, but the way there was in plain view
    10: (190, 300, (34, 0), (34, 0), 0),
    11: (0, 100, (-20, 0), (-20, 0), 0),  # 11 -> 12 with 13 (kit misread) in the gap on the way, 14 an opponent
    12: (200, 300, (-14, 0), (-14, 0), 0),
    13: (120, 180, (-19, 0), (-15, 0), 2),
    14: (120, 180, (-18, 0), (-16, 0), 1),
}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    ids = sorted(TRACKS)
    ends = {"track": np.array(ids), "first": np.array([TRACKS[k][0] for k in ids]),
            "last": np.array([TRACKS[k][1] for k in ids]), "start": np.array([TRACKS[k][2] for k in ids], float),
            "end": np.array([TRACKS[k][3] for k in ids], float),
            "n": np.array([TRACKS[k][1] - TRACKS[k][0] + 1 for k in ids])}
    monkeypatch.setattr(bridge, "tracklet_ends", lambda store, game: ends)
    monkeypatch.setattr(bridge, "_roles", lambda store, game, tracks: np.array([TRACKS[k][4] for k in tracks.tolist()]))

    def unseen(store, game, E, ei, si, fps):  # 9 -> 10 crossed open grass in view; elsewhere they could hide
        return np.array([3.0 if (E["track"][e], E["track"][s]) == (9, 10) else 0.0 for e, s in zip(ei, si)])

    monkeypatch.setattr(bridge, "_unseen", unseen)
    monkeypatch.setattr(people, "_picks_path", lambda store, game: tmp_path / "none.json")
    monkeypatch.setattr(bridge, "_track_looks", lambda store, game: {})
    for c in (bridge._links_cache, bridge._unseen_memo):
        c.clear()
    yield
    for c in (bridge._links_cache, bridge._unseen_memo):
        c.clear()


def test_links(fake):
    L = bridge.links(None, GAME)
    assert L["next"][1] == 2
    assert 5 not in L["prev"] and 3 not in L["next"] and 4 not in L["next"]  # two could have gone on as 5
    assert L["next"][6] == 7 and L["next"][7] == 8  # 8 is 6 again later, not a rival to 7
    assert 9 not in L["next"]  # 4 m in 3 s fits, but not unseen in plain view
    assert L["next"][11] == 12


def seg(track, player, start=None, end=None):
    return {"track": track, "player": player, "start": TRACKS[track][0] if start is None else start,
            "end": TRACKS[track][1] + 1 if end is None else end}


def claims(out, player):
    return {c["track"] for c in out.get(player, [])}


def test_extend(fake):
    looks = {}
    out = bridge.extend(None, GAME, [seg(1, 7), seg(6, 9), seg(12, 5)], {}, looks=looks)
    assert claims(out, 7) == {2}
    assert claims(out, 9) == {7, 8}
    assert claims(out, 5) == {11, 13}  # backward from 12; the misread kit fills the gap, the opponent doesn't
    assert all(c["by"] == "place" for cs in out.values() for c in cs)
    # backward from 2 to 1, as a guess
    assert claims(bridge.extend(None, GAME, [], {7: [seg(2, 7)]}, looks=looks), 7) == {1}
    # a stretch that stops short of its tracklet's end doesn't go on
    assert not bridge.extend(None, GAME, [seg(1, 7, end=50)], {}, looks=looks)
    # stops at someone else's label, a tracklet turned down, or where the player is labelled elsewhere
    assert claims(bridge.extend(None, GAME, [seg(6, 9), seg(7, 4)], {}, looks=looks), 9) == set()
    assert claims(bridge.extend(None, GAME, [seg(6, 9)], {}, {9: {7}}, looks=looks), 9) == set()
    assert claims(bridge.extend(None, GAME, [seg(6, 9), seg(5, 9, 160, 200)], {}, looks=looks), 9) == {7}
    # the same tracklet reached for two players (forward from 6, back from 8): neither
    out = bridge.extend(None, GAME, [seg(6, 9), seg(8, 3)], {}, looks=looks)
    assert 7 not in claims(out, 9) and 7 not in claims(out, 3)


def test_extend_veto(fake, monkeypatch):
    # 2 looks like #8, not #7: a swap the field can't see
    a, b = np.eye(4)[0], np.eye(4)[1]
    monkeypatch.setattr(bridge, "_track_looks", lambda store, game: {2: b})
    assert not bridge.extend(None, GAME, [seg(1, 7)], {}, looks={7: a, 8: b})
    monkeypatch.setattr(bridge, "_track_looks", lambda store, game: {2: (a + b) / np.sqrt(2)})  # can't tell
    assert claims(bridge.extend(None, GAME, [seg(1, 7)], {}, looks={7: a, 8: b}), 7) == {2}


def test_unseen(monkeypatch):
    from soccer_clips.track import pitch

    # a camera looking straight down: 20 px a metre, the centre spot in the middle of the frame
    G = np.array([[20.0, 0, 960], [0, 20.0, 540], [0, 0, 1]])
    monkeypatch.setattr(pitch, "frame_matrices", lambda store, game, frames: np.repeat(G[None], len(frames), 0))
    # someone's box over the way at 0.5 s and 1 s (hidden behind them there), nobody after
    box = lambda x: [(960 + 20 * x - 10) / 1920, (540 - 60) / 1080, (960 + 20 * x + 10) / 1920, (540 + 5) / 1080]
    t = people.Tracks(np.array([15, 30]), np.array([50, 50]), np.array([box(5 / 3), box(10 / 3)]), [[0, 300]])
    monkeypatch.setattr(people, "tracks", lambda store, game: t)
    E = {"last": np.array([0, 0]), "first": np.array([90, 90]), "end": np.array([[0.0, 0], [0, 0]]),
         "start": np.array([[10.0, 0], [100, 0]])}
    # 0 -> 10 m over 3 s: checked at 0.5 s steps, hidden at 0.5 and 1 s, in plain view at 1.5, 2, 2.5 s;
    # 0 -> 100 m: in plain view at 0.5 and 1 s (16.7, 33.3 m), then past the frame's edge (50 m = 1960 px),
    # where being unseen means nothing
    got = bridge._unseen(None, GAME, E, np.array([0, 1]), np.array([0, 1]), 30.0)
    assert got[0] == pytest.approx(1.5)
    assert got[1] == pytest.approx(1.0)
