import numpy as np

from soccer_clips.track import turnovers as tv

FPS = 10.0


def touch(frame, track, team, x=0.0):
    return {"frame": frame, "track": track, "team": team, "x": x, "y": 0.0}


def test_find_and_stats():
    tc = [touch(0, 1, 0, -5), touch(20, 7, 1, 5),  # our pass, theirs next: an interception (in their own half)
          touch(30, 8, -1), touch(40, 2, 0, 25),  # a kit read as neither passed over; won 2 s after theirs: recovery
          touch(45, 9, 1, 24),  # taken off us 0.5 s later, in their own third: a tackle
          touch(80, 3, 0, 20)]
    in_play = np.ones(100, bool)
    in_play[60:70] = False  # out of play between: not won, a restart
    got = tv.find(tc, {0}, in_play, [(0, 100)], [1], 60.0, FPS)
    assert [(t["frame"], t["team"], t["kind"], t["from_track"]) for t in got] == [
        (20, 1, "interception", 1), (40, 0, "recovery", 7), (45, 1, "tackle", 2)]
    assert [t["third"] for t in got] == [1, 2, 0] and [t["their_half"] for t in got] == [False, True, False]
    s = tv.team_stats(got, 0)
    assert s["won"] == 1 and s["lost"] == 2 and s["won_their_half"] == 1 and s["lost_by"]["tackle"] == 1
    assert s["lost_own_third"] == 0 and tv.team_stats(got, 1)["lost_own_third"] == 1
    # the touch at 40 a shot: their block at 45 ends the attack, not a ball lost
    assert [t["frame"] for t in tv.find(tc, {0}, in_play, [(0, 100)], [1], 60.0, FPS, {40})] == [20, 40]


def test_compare():
    evs = [{"frame": 10, "kind": "touch", "team": 0, "keeper": 0},
           {"frame": 30, "kind": "won", "team": 1, "keeper": 0},
           {"frame": 50, "kind": "touch", "team": 1, "keeper": 0}, {"frame": 70, "kind": "touch", "team": 0, "keeper": 0}]
    tos = [{"frame": 31, "team": 1}, {"frame": 90, "team": 0}]
    got = tv.compare(tos, evs, [{"start": 0, "end": 100}], np.ones(100, bool), FPS)
    assert got["labelled_n"] == 2 and got["found"] == 0.5 and got["labelled"] == 0.5 and got["team"] == 1.0
    assert got["won"] == 1.0
