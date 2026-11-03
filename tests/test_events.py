import pytest
from test_gallery import four_tracklets

from soccer_clips.track import events


def test_labels_add_remove(settings):
    store, game = four_tracklets(settings)  # #38 labelled on frames 0-89 (box x 0.1-0.2, y 0.2-0.6)
    ev = events.add_event(store, game, 30, "pass", 0.15, 0.4)
    events.add_event(store, game, 10, "touch")  # no player
    with pytest.raises(ValueError):
        events.add_event(store, game, 5, "dribble")
    with pytest.raises(ValueError):
        events.add_event(store, game, 5, "touch", x=0.1)
    got = events.resolved(store, game)
    assert [e["frame"] for e in got["events"]] == [10, 30]  # in time order
    p = got["events"][1]
    assert p["kind"] == "pass" and p["player"] == 38 and p["team"] == 0 and p["track"] is not None
    assert got["events"][0]["track"] is None and got["counts"]["pass"] == 1 and got["counts"]["touch"] == 1
    assert events.remove_event(store, game, ev["id"]) and not events.remove_event(store, game, ev["id"])
    assert [e["kind"] for e in events.resolved(store, game)["events"]] == ["touch"]


def test_windows_merge(settings):
    store, game = four_tracklets(settings)
    events.add_window(store, game, 100, 200)
    events.add_window(store, game, 300, 400)
    w = events.add_window(store, game, 380, 150)  # given backwards, overlapping both: one window
    assert [(x["start"], x["end"]) for x in events.load_labels(store, game)["windows"]] == [(100, 400)]
    with pytest.raises(ValueError):
        events.add_window(store, game, 50, 50)
    assert events.remove_window(store, game, w["id"]) and events.load_labels(store, game)["windows"] == []


def test_events_api(settings):
    from fastapi.testclient import TestClient

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)
    client = TestClient(create_app(store))
    url = f"/api/games/{game.id}/events"
    r = client.post(f"{url}/labels", json={"frame": 30, "kind": "shot", "x": 0.15, "y": 0.4}).json()
    assert r["event"]["kind"] == "shot" and r["events"][0]["player"] == 38
    assert client.post(f"{url}/labels", json={"frame": 30, "kind": "nope"}).status_code == 422
    assert client.post(f"{url}/windows", json={"start": 0, "end": 90}).json()["windows"][0]["end"] == 90
    assert client.delete(f"{url}/labels/{r['event']['id']}").json()["events"] == []
    assert client.delete(f"{url}/labels/{r['event']['id']}").status_code == 404
    assert client.get(f"{url}/labels").json()["kinds"] == list(events.KINDS)
