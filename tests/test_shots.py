import numpy as np

from soccer_clips.track import restarts as rs
from soccer_clips.track import shots as sh

FPS = 10.0
L, GOAL = 60.0, 5.0
HS = [(0, 1000)]
SIGNS = [1]  # we attack +x


def touch(frame, track, team, x, y=0.0, keeper=0):
    return {"frame": frame, "track": track, "team": team, "x": x, "y": y, "keeper": keeper}


def test_end_moments():
    ball = np.full((300, 2), np.nan)
    ball[85:101] = np.column_stack([np.linspace(20, 29, 16), np.zeros(16)])  # toward the +x goal, to their goalie
    ball[150:161, 0] = np.linspace(20, 21, 11)  # their goalie, the ball hardly coming: a back pass
    ball[150:161, 1] = 0.0
    ball[195:201] = [[29.0, 8.0]] * 6
    tc = [touch(80, 1, 0, 20), touch(100, 9, 1, 29, keeper=2), touch(150, 8, 1, 20), touch(160, 9, 1, 21, keeper=2)]
    out = [{"from": 200, "to": 260, "kind": "goal kick"}, {"from": 200, "to": 280, "kind": "kickoff"},
           {"from": 210, "to": 230, "kind": "throw-in"}]
    got = sh.end_moments(tc, out, [280], ball, HS, SIGNS, L, FPS)
    # the back pass isn't a shot's end; the goal kick and the kickoff at one moment: the stronger
    assert [(e["frame"], e["end"], e["kind"]) for e in got] == [(100, 1, "keeper"), (200, 1, "kickoff")]
    assert sh.end_moments(tc, out, [], ball, HS, SIGNS, L, FPS)[-1]["kind"] == "goal kick"  # no goal found


def test_find_shots():
    ball = np.full((300, 2), np.nan)
    ball[120:125] = np.column_stack([np.linspace(15, 19, 5), np.linspace(0, 0.4, 5)])  # heading in, 10 m/s
    tc = [touch(50, 3, 0, 18), touch(60, 1, 0, 25),  # our shot, saved
          touch(70, 9, 1, 29, keeper=2),
          touch(120, 4, 0, 15), touch(125, 7, 1, 19),  # blocked by a defender
          touch(200, 5, 1, -20)]  # theirs, far from the end that matters
    ends = [{"frame": 70, "end": 1, "kind": "keeper"}, {"frame": 180, "end": 1, "kind": "goal kick"},
            {"frame": 250, "end": -1, "kind": "kickoff"}]
    got = sh.find(tc, ends, ball, HS, SIGNS, L, GOAL, FPS)
    assert [(s["frame"], s["team"], s["outcome"]) for s in got] == [(60, 0, "saved"), (120, 0, "blocked"),
                                                                    (None, 1, "goal")]
    # over the line with no attacker in the 3 s before (at 180): left out
    assert got[0]["distance"] == 5.0 and got[0]["track"] == 1


def test_labelled_shots():
    ball = np.full((300, 2), np.nan)
    tc = [touch(60, 1, 0, 25), touch(70, 9, 1, 29, keeper=2),  # a cross the goalie caught: you labelled it a pass
          touch(150, 4, 0, 20), touch(155, 6, 1, 22)]  # you labelled a shot; a defender's next
    tc[0]["label"], tc[2]["label"] = "pass", "shot"
    ends = [{"frame": 70, "end": 1, "kind": "keeper"}]
    got = sh.find(tc, ends, ball, HS, SIGNS, L, GOAL, FPS)
    assert [(s["frame"], s["team"], s["outcome"]) for s in got] == [(150, 0, "blocked")]
    ends.append({"frame": 170, "end": 1, "kind": "goal kick"})
    assert sh.find(tc, ends, ball, HS, SIGNS, L, GOAL, FPS)[0]["outcome"] == "wide"  # where it got to wins


def test_goals_and_scorer():
    assert sh.kickoff_scorer(0, 0) == 0 and sh.kickoff_scorer(0, 1) is None and sh.kickoff_scorer(None, 1) == 1
    shots = [{"frame": 90, "at": 100, "outcome": "goal"}]
    kos = [{"kick": 400, "from": 100, "team": 0}, {"kick": 4000, "from": 3900, "team": None}]
    clips = [{"t": 9.0, "team": "theirs", "guessed": False},  # rated: decides the first
             {"t": 200.0, "team": "ours", "guessed": True}]  # no kickoff after: a goal from the clip alone
    got = sh.goals(shots, kos, clips, FPS)
    assert [(g["frame"], g["team"], g["by"], g["shot"]) for g in got] == [
        (100, 1, ["kickoff", "clip"], True), (2000, 0, ["clip"], False), (3900, None, ["kickoff"], False)]
    # two goals before one kickoff (the first's kickoff missed): the last is that kickoff's, the first its own; a
    # G label beside a clip of the same goal: one goal, the label deciding
    clips = [{"t": 10.0, "team": "ours", "guessed": False, "by": "clip"},
             {"t": 65.0, "team": "theirs", "guessed": False, "by": "clip"},
             {"t": 66.0, "team": "ours", "guessed": False, "by": "label"},
             {"t": 300.0, "team": "theirs", "guessed": True, "by": "clip"},
             {"t": 310.0, "team": "ours", "guessed": False, "by": "label"}]
    got2 = sh.goals([], [{"kick": 1000, "from": 980, "team": 1}], clips, FPS)
    assert [(g["frame"], g["team"], g["by"]) for g in got2] == [
        (100, 0, ["clip"]), (980, 0, ["kickoff", "clip", "label"]), (3000, 0, ["clip", "label"])]
    s = sh.team_stats([{"team": 0, "outcome": "saved"}, {"team": 0, "outcome": "goal"}], got, 0)
    assert s["shots"] == 2 and s["saved"] == 1 and s["goals"] == 1


def test_shots_compare():
    shots = [{"frame": 50, "at": 60, "outcome": "saved"}, {"frame": None, "at": 300, "outcome": "goal"}]
    evs = [{"frame": 51, "kind": "shot"}, {"frame": 280, "kind": "shot"}, {"frame": 500, "kind": "shot"},
           {"frame": 61, "kind": "save"}]
    got = sh.compare(shots, [{"team": 0}, {"team": None}], evs, [{"start": 0, "end": 400}], (1, 0), FPS)
    assert got["found"] == 0.667 and got["labelled"] == 1.0 and got["saves"] == 1.0
    assert got["goals"] == [1, 0] and got["score"] == [1, 0]


def test_restarts():
    restarts = [{"still": 90, "kick": 100, "x": 29.0, "y": 2.0, "kind": "goal kick", "half": 1},
                {"still": 140, "kick": 150, "x": 30.0, "y": 20.0, "kind": "corner", "half": 1},
                {"still": 190, "kick": 200, "x": 5.0, "y": 22.0, "kind": "throw-in", "half": 1},
                {"still": 250, "kick": 260, "x": 0.0, "y": 0.0, "kind": "kickoff", "half": 1},
                {"still": 290, "kick": 300, "x": 25.0, "y": 0.0, "kind": "goalie", "half": 1},
                {"still": 330, "kick": 340, "x": 10.0, "y": 0.0, "kind": "free kick", "half": 1}]
    out = [{"from": 80, "to": 95, "kind": "goal kick"},  # placed, then kicked: the last is the restart
           {"from": 80, "to": 100, "kind": "goal kick"},
           {"from": 130, "to": 150, "kind": "corner"}, {"from": 180, "to": 200, "kind": "throw-in"},
           {"from": 290, "to": 300, "kind": "goalie"}]
    # their goal kick (at the end we attack): theirs; a corner taken by them (the rules say ours)
    tc = [touch(100, 1, 1, 29), touch(150, 2, 1, 30),
          touch(175, 3, 1, 5), touch(200, 4, 0, 5),  # they touched it last: our throw-in
          touch(260, 5, 0, 0)]
    got = rs.listed(restarts, out, tc, SIGNS, FPS)
    assert [(r["frame"], r["kind"], r["team"], r["disagree"]) for r in got] == [
        (100, "goal kick", 1, False), (150, "corner", None, True), (200, "throw-in", 0, False),
        (260, "kickoff", 0, False)]
    st = rs.stats(got)
    assert st["corner"]["unknown"] == 1 and st["agree"] == 0.667 and st["agree_n"] == 3
    evs = [{"frame": 101, "kind": "restart", "team": 1, "keeper": 0},
           {"frame": 500, "kind": "restart", "team": 0, "keeper": 0}]
    assert rs.compare(got, evs, FPS) == {"labelled_n": 2, "found": 0.5, "team": 1.0, "team_n": 1}


def test_score_and_shots_api(settings):
    from fastapi.testclient import TestClient
    from test_gallery import four_tracklets

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)  # tracked, not calibrated
    client = TestClient(create_app(store))
    got = client.patch(f"/api/games/{game.id}", json={"score_ours": 3, "score_theirs": 1}).json()
    assert got["score_ours"] == 3 and store.load_game(game.id).score_theirs == 1
    assert client.patch(f"/api/games/{game.id}", json={"score_ours": -1}).status_code == 422
    assert "calibrated" in client.get(f"/api/games/{game.id}/shots").json()["error"]


def test_log_api_and_label_from_the_log(settings):
    from fastapi.testclient import TestClient
    from test_gallery import four_tracklets

    from soccer_clips.server.app import create_app

    store, game = four_tracklets(settings)  # tracked, not calibrated
    client = TestClient(create_app(store))
    assert "calibrated" in client.get(f"/api/games/{game.id}/log").json()["error"]
    # a label made from the game log: a moment and a tracklet, no click; pointed at the middle of its box
    got = client.post(f"/api/games/{game.id}/events/labels", json={"frame": 100, "kind": "none", "track": 3}).json()
    assert abs(got["event"]["x"] - 0.55) < 1e-4 and abs(got["event"]["y"] - 0.4) < 1e-4
    assert [e["kind"] for e in got["nots"]] == ["none"] and got["events"] == []
