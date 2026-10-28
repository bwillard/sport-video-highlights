from test_gallery import four_tracklets

from soccer_clips.track import reid, watch


def test_merge_and_intersect():
    assert watch.merge([(10, 20), (0, 5), (22, 30), (40, 41)], gap=2) == [[0, 5], [10, 30], [40, 41]]
    assert watch.intersect([[0, 10], [20, 30]], [[5, 25], [29, 40]]) == [[5, 10], [20, 25], [29, 30]]
    assert watch._minus([0, 100], [[10, 20], [90, 120]]) == [[0, 10], [20, 90]]


def test_runs_labelled_then_guessed(settings):
    store, game = four_tracklets(settings)  # #38 labelled on frames 0-89; a look-alike unlabelled on 90-179
    fps = game.fps
    only = watch.runs(store, game, 38, guesses=False)
    assert len(only) == 1 and only[0]["start_s"] == 0 and only[0]["labelled"] == 1.0
    assert abs(only[0]["end_s"] - (90 / fps + watch.PAD_S)) < 0.05  # a second of lead-out
    assert reid.auto_all(store, game).get(38)  # the look-alike is claimed for her...
    both = watch.runs(store, game, 38)
    assert len(both) == 1 and abs(both[0]["end_s"] - 180 / fps) < 0.05 and 0.4 < both[0]["labelled"] < 0.6  # ...half of it
    assert watch.runs(store, game, 38, min_score=1.0) == only  # nothing claimed at 1.0
    assert watch.runs(store, game, 7) == []


def test_watch_api(settings):
    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)
    client = TestClient(create_app(store))
    assert client.get("/watch").status_code == 200
    got = client.get("/api/watch", params={"player": 38, "guesses": False}).json()
    assert [r["game"] for r in got["runs"]] == [game.id] and got["games"][game.id]["fps"] == game.fps
    assert got["total_s"] == round(got["runs"][0]["end_s"] - got["runs"][0]["start_s"], 1)
    one = client.get("/api/watch", params={"player": 38, "game": game.id}).json()
    assert one["runs"][0]["labelled"] < 1


def test_boxes_for_the_highlight(settings):
    import numpy as np

    store, game = four_tracklets(settings)
    only = watch.boxes(store, game, 38, guesses=False)
    assert only.shape == (90, 6) and (only[:, 0] == np.arange(90)).all() and (only[:, 5] == 0).all()
    assert np.allclose(only[:, 1], 0.1) and np.allclose(only[:, 3], 0.2)  # tracklet 1 (smoothing keeps a still box still)
    both = watch.boxes(store, game, 38)
    guessed = both[both[:, 5] == 1]
    assert len(guessed) == 90 and guessed[0, 0] == 90 and np.allclose(guessed[:, 1], 0.5)  # the look-alike, after her
    assert (np.diff(both[:, 0]) >= 0).all()
    assert watch.boxes(store, game, 7).shape == (0, 6)


def test_boxes_api(settings):
    import numpy as np
    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)
    r = TestClient(create_app(store)).get("/api/watch/boxes", params={"player": 38, "game": game.id, "guesses": False})
    rows = np.frombuffer(r.content, "<f4").reshape(-1, 6)
    assert len(rows) == 90 and rows[0, 0] == 0
