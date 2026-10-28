import numpy as np
import pytest

from conftest import make_video
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import ball_game, gallery, people, reid


def four_tracklets(settings):
    """1 = #38 (labelled, frames 0-89), 2 = a teammate who looks different (0-179), 3 = a look-alike later
    (90-179), 4 = a look-alike on screen while she's labelled (0-89): can't be her."""
    make_video(settings, "2026/gu11-test-2026-10-10.mp4", duration_s=6, audio=False)
    store = Store(settings)
    scan(settings, store)
    game = store.load_game("gu11-test-2026-10-10")
    spans = {1: (0, 90), 2: (0, 180), 3: (90, 180), 4: (0, 90)}
    xs = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7}
    rows = [(f, k) for k, (a, b) in spans.items() for f in range(a, b)]
    frame = np.array([f for f, _ in rows], np.int32)
    track = np.array([k for _, k in rows], np.int32)
    x1 = np.array([xs[k] for k in track], np.float32)
    n = len(rows)
    ball_game.save_npz(people.path(store, game, people.DEFAULT_MODEL), frame=frame, track=track, x1=x1, y1=np.full(n, 0.2, np.float32),
                       x2=x1 + 0.1, y2=np.full(n, 0.6, np.float32), conf=np.full(n, 0.9, np.float32),
                       det=np.arange(n, dtype=np.int32), ranges=np.array([[0, 180]], np.int32))
    people.add_pick(store, game, 10, 38, "is", 0.15, 0.4)
    rng = np.random.default_rng(0)
    her, other = rng.normal(size=512), rng.normal(size=512)
    looks = {1: her, 2: other, 3: her, 4: her}
    on_grid = frame % 10 == 0  # a crop every 10 frames: 9-18 per tracklet
    sig = np.array([looks[k] + 0.3 * rng.normal(size=512) for k in track[on_grid]])
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    ball_game.save_npz(reid.sigs_path(store, game), frame=frame[on_grid], track=track[on_grid],
                       sig=sig.astype(np.float16), model=np.array("test"))
    return store, game


def tracks_of(store, view, player=None):
    return sorted(it["track"] for it in gallery.pictures(store, view, player)["items"])


def test_views_assign_and_undo(settings):
    store, game = four_tracklets(settings)
    assert tracks_of(store, "player", 38) == [1]
    assert tracks_of(store, "unassigned") == [2, 3, 4]
    # the look-alike is probably her; the one on screen while she's labelled can't be; the teammate doesn't look it
    assert tracks_of(store, "probable", 38) == [3]
    items = {it["track"]: it for it in gallery.pictures(store, "unassigned")["items"]}
    assert items[3]["guesses"][0]["player"] == 38 and items[3]["guesses"][0]["score"] > 0.9
    assert items[4]["guesses"] == []  # busy: #38 is labelled elsewhere all the while
    s = gallery.summary(store)
    assert s["players"]["38"]["stretches"] == 1 and s["players"]["38"]["probable"] == 1 and s["unassigned"] == 3

    gallery.assign(store, [{**items[3], "label": None}], 38, "b1")
    assert tracks_of(store, "player", 38) == [1, 3] and tracks_of(store, "unassigned") == [2, 4]
    # the teammate is #5; then #38's look-alike moved from #38 to #5 (a wrong label): doubtful while it looks like her
    gallery.assign(store, [{**items[2], "label": None}], 5, "b2")
    three = next(it for it in gallery.pictures(store, "player", 38)["items"] if it["track"] == 3)
    gallery.assign(store, [three], 5, "b3")
    assert tracks_of(store, "player", 5) == [2, 3] and tracks_of(store, "player", 38) == [1]
    assert 3 in tracks_of(store, "unsure")  # (and #5's own: their look is now half her)
    assert gallery.undo(store, "b3") == 1 and tracks_of(store, "player", 38) == [1, 3]
    # not our team: unlabelled and out of the unassigned list, until taken back
    gallery.assign(store, [{**items[4], "label": None}], people.NOT_OURS, "b4")
    assert tracks_of(store, "unassigned") == []
    gallery.assign(store, [three], None, "b5")  # unlabel: back to unassigned
    assert tracks_of(store, "unassigned") == [3] and tracks_of(store, "player", 38) == [1]
    for b in ("b5", "b4", "b2", "b1"):
        gallery.undo(store, b)
    assert tracks_of(store, "unassigned") == [2, 3, 4]
    assert [p["kind"] for p in people.load_picks(store, game)] == ["is"]


def test_roster_api(settings):
    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)
    with TestClient(create_app(store)) as client:
        r = client.put("/api/roster", json={"players": [{"number": 38, "name": " A Player "}, {"number": 5}]})
        assert r.json()["roster"] == [{"number": 5}, {"number": 38, "name": "A Player"}]
        assert client.put("/api/roster", json={"players": [{"number": 5}, {"number": 5}]}).status_code == 422
        got = client.get("/api/roster").json()
        assert got["roster"][1]["name"] == "A Player" and got["players"]["38"]["stretches"] == 1
        three = next(it for it in client.get("/api/roster/pictures?view=probable&player=38").json()["items"])
        assert three["track"] == 3
        three = {k: three[k] for k in ("game", "track", "start", "end", "label")}
        assert client.post("/api/roster/assign", json={"player": 38, "batch": "x", "stretches": [three]}).json() == {"changed": 1}
        assert client.get("/api/roster/pictures?view=player&player=38").json()["total"] == 2
        assert client.delete("/api/roster/assign/x").json() == {"undone": 1}
        assert client.get("/players", follow_redirects=False).headers["location"] == "/roster"
        assert client.get("/roster").status_code == 200


def test_split_a_picture_of_two_girls(settings):
    store, game = four_tracklets(settings)
    spans = lambda view, player=None: sorted((it["track"], it["start"], it["end"]) for it in gallery.pictures(store, view, player)["items"])
    # the teammate's tracklet (0-179, unlabelled) is two girls from frame 90; #38's (0-89) from frame 40
    assert gallery.split(store, game.id, 2, [90], "s1") == 1
    assert (2, 0, 90) in spans("unassigned") and (2, 90, 180) in spans("unassigned")
    assert gallery.split(store, game.id, 1, [40, 0, 500], "s2") == 1  # its start, or past its end: nothing to split
    assert spans("player", 38) == [(1, 0, 40)] and (1, 40, 90) in spans("unassigned")
    second = next(it for it in gallery.pictures(store, "unassigned")["items"] if (it["track"], it["start"]) == (2, 90))
    gallery.assign(store, [second], 5, "a1")  # each part is assigned on its own
    assert spans("player", 5) == [(2, 90, 180)] and (2, 0, 90) in spans("unassigned")
    for b in ("a1", "s2", "s1"):
        gallery.undo(store, b)
    assert spans("player", 38) == [(1, 0, 90)] and (2, 0, 180) in spans("unassigned")
