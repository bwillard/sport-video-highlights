from fastapi.testclient import TestClient
from test_gallery import four_tracklets

from soccer_clips.server.app import create_app
from soccer_clips.track import people, reid


def stretch_roles(store, game, track):
    t = people.tracks(store, game)
    return set(people.roles_for(store, game)[1][t.track == track].tolist())


def test_mark_unlabelled_and_undo(settings):
    # 1 is #38 (frames 0-89); 3 looks like her and comes after, so it's guessed as her
    store, game = four_tracklets(settings)
    assert {c["track"] for c in reid.auto_all(store, game)[38]} == {3}
    assert people.mark_not_ours(store, game, 3, 120, "b1") == 1
    t = people.tracks(store, game)
    assert people.not_ours(t, people.load_picks(store, game)) == [{"track": 3, "start": 90, "end": 180}]
    assert stretch_roles(store, game, 3) == {people.ROLE_CODES["other"]}
    assert 3 not in {c["track"] for c in reid.auto_all(store, game).get(38, [])}  # never guessed
    assert people.mark_not_ours(store, game, 3, 100, "b2") == 0  # already
    assert people.undo_answers(store, game, "b1") == 1
    assert people.not_ours(people.tracks(store, game), people.load_picks(store, game)) == []
    assert {c["track"] for c in reid.auto_all(store, game)[38]} == {3}


def test_mark_labelled(settings):
    store, game = four_tracklets(settings)
    assert people.mark_not_ours(store, game, 1, 50, "b1") == 1
    t, picks = people.tracks(store, game), people.load_picks(store, game)
    assert people.segments(t, picks) == []  # the label goes
    assert people.not_ours(t, picks) == [{"track": 1, "start": 0, "end": 90}]  # the whole stretch, from the tracklet's start
    assert people.undo_answers(store, game, "b1") == 1
    assert people.segments(people.tracks(store, game), people.load_picks(store, game))[0]["player"] == 38


def test_roster_not_ours_takes_the_whole_labelled_stretch(settings):
    store, game = four_tracklets(settings)  # #38's pick is at frame 10; her stretch starts at the tracklet's start
    people.assign_stretches(store, game, [{"track": 1, "start": 0, "end": 90, "label": 38}], people.NOT_OURS, "b1")
    t, picks = people.tracks(store, game), people.load_picks(store, game)
    assert people.segments(t, picks) == [] and people.not_ours(t, picks) == [{"track": 1, "start": 0, "end": 90}]


def test_api(settings):
    store, game = four_tracklets(settings)
    client = TestClient(create_app(store))
    url = f"/api/games/{game.id}/people"
    r = client.post(f"{url}/not-ours", json={"track": 1, "frame": 30, "batch": "b1"}).json()
    assert r["changed"] == 1 and r["segments"] == []
    assert client.post(f"{url}/not-ours", json={"track": 99, "frame": 30, "batch": "b2"}).status_code == 422
    r = client.delete(f"{url}/answers/b1").json()
    assert r["undone"] == 1 and r["segments"][0]["player"] == 38
