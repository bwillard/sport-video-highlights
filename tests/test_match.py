import numpy as np

from soccer_clips.track import match

FPS = 30.0
SIZES = {"length": 60.0, "width": 40.0, "ga_depth": 4.6, "ga_width": 14.6, "pa_depth": 12.8, "pa_width": 32.9,
         "spot": 9.1}


def test_dead_balls():
    # rolling, then still at (10, 5) for 2 s, then kicked away; then still for half a second (not long enough)
    f = np.arange(300)
    x = np.where(f < 60, f * 0.2, 10.0)
    x = np.where(f >= 120, 10 + (f - 120) * 0.5, x)  # 15 m/s
    xy = np.column_stack([x, np.full(300, 5.0)])
    xy[200:215] = xy[199]  # half a second still: no restart
    xy[80:90] = np.nan  # a moment not seen (the kicker in the way) changes nothing
    got = match.dead_balls(f, xy, FPS)
    assert len(got) == 1
    r = got[0]
    assert 55 <= r["still"] <= 62 and r["kick"] == 122 and (r["x"], r["y"]) == (10.0, 5.0)


def test_restart_kind():
    k = lambda x, y: match.restart_kind(x, y, SIZES)  # noqa: E731
    assert k(0.5, -1) == "kickoff" and k(2.5, 4) == "kickoff"
    assert k(29.5, 19.5) == "corner" and k(-28, -18) == "corner"
    assert k(-27, 3) == "goal kick"
    assert k(30 - 9.1, 0.3) == "penalty"
    assert k(5, 19.5) == "throw-in" and k(5, -21) == "throw-in"
    assert k(-20, 10) == "goalie"  # in the penalty area, outside the goal area
    assert k(8, 3) == "free kick"


def test_formation_relative_to_the_ball():
    x = np.r_[np.full(20, 5.0), np.full(20, 15.0)]  # ours one side of a ball at 10, theirs the other
    role = np.r_[np.zeros(20, int), np.ones(20, int)]
    fm = match.formation(x, role, ball_x=10.0)
    assert fm == {"ours": -1.0, "theirs": 1.0, "clear": True, "split": True}
    mixed = match.formation(np.r_[x[:20] + np.tile([0, 10], 10), x[20:]], role, 10.0)
    assert not mixed["split"] and not mixed["clear"]  # ours on both sides: a free kick, not a kickoff
    few = match.formation(x[:20], role[:20], 10.0)
    assert few["clear"] and not few["split"] and few["theirs"] == 0  # too few of theirs to say


def test_direction():
    ko = lambda half, ours, clear=True: {"half": half, "ours": ours, "clear": clear}  # noqa: E731
    d = match.direction([ko(1, 0.9), ko(2, -0.8), ko(2, -1.0), ko(1, -1.0, clear=False)])
    assert d["found"] == "left" and d["sure"] and d["attack_1h"] == "left" and not d["set"]
    clash = match.direction([ko(1, 0.9), ko(2, 0.8)])  # both halves say the same side: something's wrong
    assert not clash["sure"]
    weak = match.direction([ko(2, 0.6)])
    assert weak["found"] == "right" and not weak["sure"]
    assert match.direction([], override="right")["attack_1h"] == "right"
    goals = [{"half": 1, "attack": 1}, {"half": 2, "attack": -1}, {"half": 1, "attack": 1}]  # rated goal clips
    assert match.direction([ko(1, 0.6)], goals=goals)["found"] == "right"  # outvote one weak kickoff
    info = {"direction": {"attack_1h": "right"}}
    assert match.attack_sign(info, 1) == 1 and match.attack_sign(info, 2) == -1


def test_keepers():
    # one half, we defend +x (the goal line at x = 30). Per frame: a goalie in other kit by our goal, a
    # goalie in their kit deep in their goal (-x), a defender of ours roaming in and out of our area, and an
    # attacker of theirs standing deep in our goal area (their kit: not our goalie).
    n = 120
    f = np.repeat(np.arange(n), 4)
    track = np.tile([1, 2, 3, 4], n)
    xy = np.zeros((len(f), 2))
    xy[track == 1] = [28.0, 1.0]
    xy[track == 2] = [-29.0, 0.0]
    xy[track == 3, 0] = np.where(np.arange(n) % 40 < 20, 25.0, 10.0)
    xy[track == 4] = [27.0, -2.0]
    role = np.select([track == 1, track == 2, track == 3, track == 4], [2, 1, 0, 1])
    k = match.keepers(track, f, xy, np.ones(len(f), bool), role, SIZES, [(0, n, 1)], fps=FPS)
    assert (k[track == 1] == 1).all() and (k[track == 2] == 2).all() and not k[(track == 3) | (track == 4)].any()


def test_keeper_seen():
    f = np.arange(100)
    keeper = (f < 50).astype(np.int8)  # our goalie found in the first 50 frames
    bxy = np.column_stack([np.full(100, 25.0), np.zeros(100)])  # the ball by the +x goal all along
    got = match.keeper_seen(keeper, f, f, bxy, 60.0, [(0, 100), (100, 200)], ours_end_1h=1, fps=FPS)
    first = {k["team"]: k for k in got if k["half"] == 1}
    assert first["ours"]["seen"] == 0.5 and first["theirs"]["seen"] is None


def test_out_of_play():
    f = np.arange(0, 600)
    xy = np.column_stack([np.zeros(600), np.zeros(600)])
    xy[100:150, 1] = 22  # over the touchline (width 40) at 100, picked up, thrown in at 160
    xy[150:160, 1] = 20.5
    restarts = [{"still": 150, "kick": 160, "kind": "throw-in"}, {"still": 300, "kick": 340, "kind": "free kick"},
                {"still": 400, "kick": 420, "kind": "goalie"}]
    got = match.out_of_play(restarts, f, xy, 60.0, 40.0, FPS, half_starts=[0])
    assert got == [{"from": 100, "to": 160, "kind": "throw-in"}, {"from": 300, "to": 340, "kind": "free kick"}]
    later = match.out_of_play(restarts[:1], f, xy, 60.0, 40.0, FPS, half_starts=[0, 120])
    assert later == [{"from": 120, "to": 160, "kind": "throw-in"}]  # not back past the half's start


def test_keepers_marked_and_their_kit():
    # we defend +x. Our goalie wears a kit (cluster 7) that reads as ours (role 0), so neither rule finds her:
    # not other kit, and not deep enough (8 m out). Marking tracklet 1 makes it ours, and her kit finds tracklet
    # 2 (her again, later, by our goal); tracklet 3 (cluster 7 too, but in midfield) isn't her
    n = 60
    track = np.repeat([1, 2, 3], n)
    frame = np.r_[np.arange(n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n)]
    xy = np.zeros((3 * n, 2))
    xy[track == 1] = [22.0, 1.0]
    xy[track == 2] = [21.0, -1.0]
    xy[track == 3] = [0.0, 5.0]
    role = np.zeros(3 * n, int)
    kit = np.full(3 * n, 7)
    on = np.ones(3 * n, bool)
    plain = match.keepers(track, frame, xy, on, role, SIZES, [(0, 3 * n, 1)], fps=FPS, kit=kit)
    assert not plain.any()
    got = match.keepers(track, frame, xy, on, role, SIZES, [(0, 3 * n, 1)], fps=FPS, kit=kit, marked={1: {1}})
    assert (got[track == 1] == 1).all() and (got[track == 2] == 1).all() and not got[track == 3].any()


def test_goalie_marks(settings):
    from test_gallery import four_tracklets

    store, game = four_tracklets(settings)  # #38's box: x 0.1-0.2, y 0.2-0.6, frames 0-89
    m = match.add_goalie(store, game, 30, 0.15, 0.4, "ours")
    match.add_goalie(store, game, 30, 0.95, 0.95, "theirs")  # nobody there
    from soccer_clips.track import people

    marks = match.goalie_marks(store, game, people.tracks(store, game))
    assert [(x["team"], x["track"] is not None) for x in marks] == [("ours", True), ("theirs", False)]
    import pytest

    with pytest.raises(ValueError):
        match.add_goalie(store, game, 30, 0.1, 0.1, "refs")
    assert match.remove_goalie(store, game, m["id"]) and not match.remove_goalie(store, game, m["id"])
    assert len(match.load_goalies(store, game)) == 1
