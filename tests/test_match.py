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
    # a kickoff: still for a second, then unseen for 2 s right up to the kick (the kicker over it), seen again where
    # it sat: the same restart; unseen longer than STILL_GAP_S, the stop ends at the gap and the kick isn't seen
    gap = xy.copy()
    gap[90:118] = np.nan
    assert [r["kick"] for r in match.dead_balls(f, gap, FPS)] == [122]
    gap[60:118] = np.nan
    gap[30:60] = gap[60 - 1] = (10.0, 5.0)
    assert match.dead_balls(f, gap, FPS, gap_s=1.0) == []


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



def test_keepers_not_goalie():
    # we defend +x; at their end (-x) two in other kit stand in the area: a parent behind the goal mouth, deepest
    # (1 m from the line), and their goalie (4 m out). Marked not a goalie, the parent gives way to the goalie
    n = 120
    f = np.repeat(np.arange(n), 2)
    track = np.tile([1, 2], n)
    xy = np.where(track[:, None] == 1, [[-29.0, 0.0]], [[-26.0, 1.0]])
    role = np.full(len(f), 2)
    on = np.ones(len(f), bool)
    k = match.keepers(track, f, xy, on, role, SIZES, [(0, n, 1)], fps=FPS)
    assert (k[track == 1] == 2).all() and not k[track == 2].any()
    k = match.keepers(track, f, xy, on, role, SIZES, [(0, n, 1)], fps=FPS, not_goalie={1})
    assert not k[track == 1].any() and (k[track == 2] == 2).all()


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
    restarts = [{"still": 150, "kick": 160, "kind": "throw-in"}, {"still": 300, "kick": 400, "kind": "free kick"},
                {"still": 450, "kick": 470, "kind": "goalie"}, {"still": 500, "kick": 540, "kind": "free kick"},
                {"still": 560, "kick": 580, "kind": "throw-in"}]
    got = match.out_of_play(restarts, f, xy, 60.0, 40.0, FPS, half_starts=[0])
    # the 1.3 s "free kick" and the throw-in with no crossing near it were a ball slowing in open play
    assert got == [{"from": 100, "to": 160, "kind": "throw-in"}, {"from": 300, "to": 400, "kind": "free kick"}]
    later = match.out_of_play(restarts[:1], f, xy, 60.0, 40.0, FPS, half_starts=[0, 120])
    assert later == [{"from": 120, "to": 160, "kind": "throw-in"}]  # not back past the half's start


def test_goal_after_a_ball_run_along_the_byline():
    # length 60: goal lines at +-30 (out past 30.5). A dribble along the byline reads just over it 10 m wide of
    # the goal at 100-104, play goes on (back in, 2 m into the field), the ball's lost at 140 among players and
    # goes in unseen; it's fetched out and the kickoff is at 400
    f = np.arange(0, 500)
    xy = np.column_stack([np.full(500, 29.0), np.full(500, 10.0)])
    xy[100:105, 0] = 30.8
    xy[105:140] = np.column_stack([np.linspace(30.0, 28.0, 35), np.linspace(10.0, 3.0, 35)])
    xy[140:300] = np.nan
    xy[300:400] = [0.0, 0.0]
    kick = [{"still": 380, "kick": 400, "kind": "kickoff"}]
    got = match.out_of_play(kick, f, xy, 60.0, 40.0, FPS, half_starts=[0], goal=5.0)
    assert got == [{"from": 140, "to": 400, "kind": "kickoff"}]  # from when the ball was last seen in play
    assert match.out_of_play(kick, f, xy, 60.0, 40.0, FPS, half_starts=[0])[0]["from"] == 100  # as before
    # a real goal's ball, back as soon, stays by the line (fetched, or a net read just inside): it was the goal
    near = xy.copy()
    near[105:140] = [29.8, 3.0]
    assert match.out_of_play(kick, f, near, 60.0, 40.0, FPS, half_starts=[0], goal=5.0)[0]["from"] == 100
    # so is one between the posts
    mouth = xy.copy()
    mouth[100:105, 1] = 2.0
    assert match.out_of_play(kick, f, mouth, 60.0, 40.0, FPS, half_starts=[0], goal=5.0)[0]["from"] == 100


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
    assert match.add_goalie(store, game, 40, 0.15, 0.4, "none")["team"] == "none"  # not a goalie
    assert len(match.load_goalies(store, game)) == 3
    match.remove_goalie(store, game, match.load_goalies(store, game)[-1]["id"])
    assert match.remove_goalie(store, game, m["id"]) and not match.remove_goalie(store, game, m["id"])
    assert len(match.load_goalies(store, game)) == 1


def test_kickoff_from_when_the_ball_went_in():
    # in at the +x goal (100), fetched out and over the line again (160; the last of it reads as over the touchline),
    # carried to the middle and kicked off: the goal is from the first crossing
    f = np.arange(400)
    xy = np.zeros((400, 2))
    xy[:100, 0] = np.linspace(20, 29, 100)
    xy[100:130] = (31.5, 0.0)
    xy[130:160] = (27.0, 3.0)  # back in the field by the goal: being fetched
    xy[160:170] = (31.0, 5.0)
    xy[170:175] = (25.0, 21.0)
    xy[175:260, 0] = np.linspace(25, 0, 85)
    ko = [{"kind": "kickoff", "still": 260, "kick": 331}]
    got = match.out_of_play(ko, f, xy, SIZES["length"], SIZES["width"], FPS, [0])
    assert [(o["from"], o["to"]) for o in got] == [(100, 331)]
    xy[130:160] = (15.0, 3.0)  # back in deep in the field: in play again, so the later crossing is the goal
    assert match.out_of_play(ko, f, xy, SIZES["length"], SIZES["width"], FPS, [0])[0]["from"] == 160
    xy[100:130] = (31.5, -18.0)  # kept in by the corner flag, reading as just over; crossed in, and the goal (160)
    xy[130:160] = (27.0, -3.0)
    assert match.out_of_play(ko, f, xy, SIZES["length"], SIZES["width"], FPS, [0])[0]["from"] == 100
    assert match.out_of_play(ko, f, xy, SIZES["length"], SIZES["width"], FPS, [0], SIZES["pa_width"])[0]["from"] == 160
    xy[100:130] = (33.0, -24.0)  # a cross in the air, reading over the goal line and the touchline: not the goal
    assert match.out_of_play(ko, f, xy, SIZES["length"], SIZES["width"], FPS, [0], SIZES["pa_width"])[0]["from"] == 160


def test_goal_in():
    # a kickoff's stretch: the ball in the +x goal at its start (a goal), or over a touchline by halfway (not one)
    f = np.arange(400)
    xy = np.zeros((400, 2))
    xy[100:130] = (31.5, 0.0)
    assert match.goal_in({"from": 100, "to": 331}, f, xy, SIZES["length"], FPS)
    xy[100:128] = (25.0, 0.0)  # over the line for 2 frames: a ball in the air
    assert not match.goal_in({"from": 100, "to": 331}, f, xy, SIZES["length"], FPS)
    xy[100:130] = (-3.0, 21.0)
    xy[300:310] = (31.5, 0.0)  # later than GOAL_IN_S: another moment
    assert not match.goal_in({"from": 100, "to": 331}, f, xy, SIZES["length"], FPS)
    assert not match.goal_in(None, f, xy, SIZES["length"], FPS)
    xy[100:130] = (31.5, -18.0)  # by the corner flag: not in the goal (further out than the touchline: in the air)
    assert not match.goal_in({"from": 100, "to": 331}, f, xy, SIZES["length"], FPS, SIZES["width"], SIZES["pa_width"])
    xy[100:130] = (31.5, -25.0)
    assert match.goal_in({"from": 100, "to": 331}, f, xy, SIZES["length"], FPS, SIZES["width"], SIZES["pa_width"])
