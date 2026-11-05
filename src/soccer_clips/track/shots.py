"""
Shots, goals and saves (Phase 6.5), from the touches (touches.py), the ball state and the game's shape
(match.py: restarts, kickoffs, goalies, out of play).

A shot is hard to see as it's struck (a fast ball, often in the air, at the far end of the field: the ball
path is there about half the time then), and easy to see by where it ends. So shots are found back from
their ends, and the shooter is the attacking team's last touch before:
- **Where a shot ends** at an end of the field: in the goal (a kickoff follows, away from the half starts:
  match.py's goals), over the goal line (a goal kick or a corner follows: from the ball going out), or in
  the defending goalie's hands (a touch by that end's goalie that isn't still theirs from before, with the
  ball APPROACH_M or more nearer the goal in the APPROACH_S before it). One per moment, the strongest.
- **The shooter:** the last touch within BACK_S before that, by the team attacking that end (or a kit read
  as neither), not a goalie, within SHOT_M of the goal. None found: the shooter isn't known, and the shot
  is kept only if it went in or the goalie got it.
- **Blocked:** a touch by the attacking team within SHOT_M of the goal, with the ball heading between the
  posts (MOUTH_PAD_M either side) at BLOCK_MS or more, and the next touch within BLOCK_S a defender's (not
  the goalie's).
- **Outcome:** goal, saved (the goalie), corner (saved, deflected or blocked out), wide (goal kick), blocked.
- **Your labels** (on the touches, touches.correct): a touch labelled P, T, W, V or K isn't a shot, even if a
  shot's end was found after it; one labelled S is, ending as the first end found within LABEL_SHOT_S after it
  (else the next touch: their goalie saved, a defender blocked; else missed). G labels are goals like your
  clips (and decide the team before them).
- **Goals:** each kickoff after one (who scored: the end the ball went out at and the team kicking off, which
  conceded, when they agree; kickoff_scorer),
  with your goal clips (ball_state.goal_moments): a clip within GOAL_NEAR_S before a kickoff is that goal;
  clips with no kickoff found and kickoffs with no clip are goals too, marked by what found them.

Against your labels (2026-11-04): see `compare` and the plan's Findings. Labelled shots are few (22, 13 in
full windows), so the numbers are rough.

<game>/tracks/shots.json: {key, shots: [{frame (the shooter's touch, or None), at (where it ended), end
(-1 / +1: the goal's side), team (0 ours, 1 theirs: attacking that end), track, x, y, distance, outcome}],
goals: [{frame, team, by: [kickoff / clip], kickoff_team, guessed}], stats}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store, atomic_write_text

VERSION = 3  # 2: your labels; 3: a G label dates its goal
BACK_S = 3.0  # the shooter's touch: this long before the shot's end, or less
SHOT_M = 30.0  # ...this close to the goal (centre of the goal line)
APPROACH_S = 1.5  # the goalie's touch ends a shot: the ball came...
APPROACH_M = 3.0  # ...this much nearer the goal in the time before
SAME_S = 1.0  # ends of shots this close (at one end) are one: the strongest
MOUTH_PAD_M = 3.0  # a blocked shot heads between the posts, or this close outside them
BLOCK_MS = 5.0  # ...this fast (m/s; over the BLOCK_LOOK_S after the touch)
BLOCK_LOOK_S = 0.4
BLOCK_S = 1.0  # ...and the next touch, a defender's, this soon
GOAL_NEAR_S = 120.0  # a goal clip ending this long before a kickoff is that kickoff's goal
SAME_GOAL_S = 30.0  # goal clips and labels with no kickoff after, this close: the same goal
TAKER_S = 0.7  # the kickoff's taker: a touch this close to the kick
TOL_S = 1.0  # measuring against labels: a shooter's touch this close to a labelled shot is it
LABEL_SHOT_S = 4.0  # a labelled shot: how it ended, from where the ball got to this soon after
NOT_SHOT = ("pass", "touch", "won", "save", "restart")  # a touch labelled one of these isn't a shot
ENDS = {"kickoff": "goal", "corner": "corner", "goal kick": "wide", "keeper": "saved"}  # how a shot ended
OUTCOMES = ("goal", "saved", "corner", "wide", "blocked", "missed")  # missed: a labelled shot, nothing found after
STRENGTH = ("kickoff", "corner", "goal kick", "keeper")  # one end per moment: the first of these
TEAMS = ("ours", "theirs")


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "shots.json"


def key(store: Store, game: Game) -> str | None:
    from soccer_clips.track import touches

    k = touches.key(store, game)
    if k is None:
        return None
    goals = sorted((c.start_s, c.end_s, c.rating) for c in game.clips if "goal" in c.tags)  # unrated ones too
    return f"s{VERSION} {k} {goals}"


# ---- the pieces (pure, for testing) ----

def half_of(f: int, hs: list[tuple[int, int]]) -> int | None:
    return next((k for k, (a, b) in enumerate(hs) if a <= f < b), None)


def attacking(end: int, sign: int) -> int:
    """The team (0 ours, 1 theirs) attacking the goal at `end` (-1 / +1) in a half we attack `sign`."""
    return 0 if end == sign else 1


def end_moments(touches: list[dict], out: list[dict], kickoff_goals: list[int], ball: np.ndarray,
                hs: list[tuple[int, int]], signs: list[int], length: float, fps: float) -> list[dict]:
    """Where shots can end: [{frame, end, kind (kickoff / corner / goal kick / keeper)}], one per moment."""
    got = []
    for i, t in enumerate(touches):
        h = half_of(t["frame"], hs)
        if t["keeper"] not in (1, 2) or h is None or not signs[h]:
            continue
        prev = touches[i - 1] if i else None
        if prev and prev["keeper"] == t["keeper"] and t["frame"] - prev["frame"] < BACK_S * fps:
            continue  # still the goalie's ball
        end = -signs[h] if t["keeper"] == 1 else signs[h]  # the goal that goalie defends
        f = t["frame"]
        seg = ball[max(0, f - round(APPROACH_S * fps)):f + 1]
        seg = seg[np.isfinite(seg[:, 0])]
        if len(seg) < 2:
            continue
        d = np.linalg.norm(seg - [end * length / 2, 0.0], axis=1)
        if d.max() - d[-1] >= APPROACH_M:
            got.append({"frame": f, "end": end, "kind": "keeper"})
    for o in out:
        if o["kind"] not in ("corner", "goal kick", "kickoff") or (o["kind"] == "kickoff"
                                                                    and o["to"] not in kickoff_goals):
            continue
        f = o["from"]
        x = ball[max(0, f - round(fps)):f + 1, 0]
        x = x[np.isfinite(x)]
        if len(x) and half_of(f, hs) is not None:
            got.append({"frame": int(f), "end": int(np.sign(x[-1])) or 1, "kind": o["kind"]})
    got.sort(key=lambda a: (STRENGTH.index(a["kind"]), a["frame"]))
    keep: list[dict] = []
    for a in got:
        if not any(a["end"] == b["end"] and abs(a["frame"] - b["frame"]) <= SAME_S * fps for b in keep):
            keep.append(a)
    return sorted(keep, key=lambda a: a["frame"])


def shooter(touches: list[dict], frames: np.ndarray, f: int, end: int, team: int, length: float,
            fps: float) -> dict | None:
    """The attacking team's last touch (or a kit read as neither) within BACK_S before frame f, not a goalie,
    within SHOT_M of the goal at `end`."""
    i = int(np.searchsorted(frames, f)) - 1
    while i >= 0 and frames[i] >= f - BACK_S * fps:
        t = touches[i]
        if t["keeper"] == 0 and t["team"] in (team, -1) and np.hypot(t["x"] - end * length / 2, t["y"]) <= SHOT_M:
            return t
        i -= 1
    return None


def heading_in(ball: np.ndarray, f: int, end: int, length: float, goal: float, fps: float) -> bool:
    """The ball, over BLOCK_LOOK_S after frame f, moving at BLOCK_MS or more toward the goal at `end`, on a line
    that crosses the goal line between the posts (MOUTH_PAD_M either side)."""
    k = round(BLOCK_LOOK_S * fps)
    seg = ball[f:f + k + 1]
    ok = np.flatnonzero(np.isfinite(seg[:, 0]))
    if len(ok) < 2 or ok[-1] == ok[0]:
        return False
    a, b = seg[ok[0]], seg[ok[-1]]
    v = (b - a) * fps / (ok[-1] - ok[0])
    if np.linalg.norm(v) < BLOCK_MS or v[0] * end <= 0:
        return False
    y = a[1] + v[1] * (end * length / 2 - a[0]) / v[0]
    return abs(y) <= goal / 2 + MOUTH_PAD_M


def find(touches: list[dict], ends: list[dict], ball: np.ndarray, hs: list[tuple[int, int]], signs: list[int],
         length: float, goal: float, fps: float) -> list[dict]:
    """The shots: back from where they ended (end_moments), and blocked ones; in time order."""
    frames = np.array([t["frame"] for t in touches], np.int64)
    out, used = [], set()
    for e in ends:
        h = half_of(e["frame"], hs)
        if h is None or not signs[h]:
            continue
        team = attacking(e["end"], signs[h])
        t = shooter(touches, frames, e["frame"], e["end"], team, length, fps)
        if t is not None and t.get("label") in NOT_SHOT:
            if e["kind"] != "kickoff":
                continue  # you said the kick before wasn't a shot
            t = None
        if t is None and e["kind"] not in ("kickoff", "keeper"):
            continue  # over the line with no attacker near: a clearance or a cross, not known as a shot
        if t is not None:
            used.add(t["frame"])
        out.append(_shot(t, e["frame"], e["end"], team, ENDS[e["kind"]], length))
    for i, t in enumerate(touches[:-1]):
        h = half_of(t["frame"], hs)
        if t["frame"] in used or t["keeper"] or h is None or not signs[h] or t.get("label") in NOT_SHOT + ("shot",):
            continue
        nx = touches[i + 1]
        for end in (-1, 1):
            team = attacking(end, signs[h])
            if (t["team"] in (team, -1) and np.hypot(t["x"] - end * length / 2, t["y"]) <= SHOT_M
                    and nx["team"] == 1 - team and not nx["keeper"] and nx["frame"] - t["frame"] <= BLOCK_S * fps
                    and heading_in(ball, t["frame"], end, length, goal, fps)):
                out.append(_shot(t, nx["frame"], end, team, "blocked", length))
    for i, t in enumerate(touches):  # labelled shots not found as one
        h = half_of(t["frame"], hs)
        if t.get("label") != "shot" or t["frame"] in used or h is None or not signs[h]:
            continue
        team = t["team"]
        end = (signs[h] if team == 0 else -signs[h]) if team >= 0 else int(np.sign(t["x"])) or 1
        team = attacking(end, signs[h])
        e = next((e for e in ends if e["end"] == end and 0 < e["frame"] - t["frame"] <= LABEL_SHOT_S * fps), None)
        nx = touches[i + 1] if i + 1 < len(touches) else None
        oc = (ENDS[e["kind"]] if e else "saved" if nx and nx["keeper"] and nx["team"] == 1 - team
              else "blocked" if nx and nx["team"] == 1 - team and nx["frame"] - t["frame"] <= BLOCK_S * fps
              else "missed")
        out.append(_shot(t, e["frame"] if e else t["frame"], end, team, oc, length))
    return sorted(out, key=lambda s: s["at"])


def _shot(t: dict | None, at: int, end: int, team: int, outcome: str, length: float) -> dict:
    r = lambda v: round(float(v), 2)  # noqa: E731
    return {"frame": t["frame"] if t else None, "at": int(at), "end": end, "team": team,
            "track": t["track"] if t else None, "x": r(t["x"]) if t else None, "y": r(t["y"]) if t else None,
            "distance": r(np.hypot(t["x"] - end * length / 2, t["y"])) if t else None, "outcome": outcome}


def kickoff_scorer(by_end: int | None, by_taker: int | None) -> int | None:
    """Who scored before a kickoff: the end the ball went out at and the team kicking off (who conceded) should
    agree (19 of 25 where both are known, 2026-11-04; some that don't look like kickoffs found where there was no
    goal); when they don't, not known. Either alone stands."""
    if by_end is not None and by_taker is not None:
        return by_end if by_end == by_taker else None
    return by_end if by_end is not None else by_taker


def goals(shots: list[dict], kickoffs: list[dict], marks: list[dict], fps: float) -> list[dict]:
    """The goals: kickoffs after one ([{kick, from, team (who scored: 0 / 1, or None)}]) and your marks of them
    ([{t, team ('ours' / 'theirs' / None), guessed, by ('clip': ball_state.goal_moments; 'label': a G)}]), the
    last marks within GOAL_NEAR_S before a kickoff being that goal (the surest decides the team: a label, then a rated
    clip, then the kickoff, then an unrated clip), marks with no kickoff within SAME_GOAL_S of each other one goal.
    [{frame, team, by, guessed, shot}]: team 0 ours, 1 theirs, None not known; frame your G label's moment, else
    the ball going out (kickoff; where the ball wasn't seen going in, that's only when it stopped for the
    kickoff), else the mark's."""
    code = {"ours": 0, "theirs": 1}
    rank = lambda c: (c.get("by") != "label", c["guessed"], c["team"] is None)  # noqa: E731

    def best(ms: list[dict]) -> dict | None:
        ms = [m for m in ms if m["team"] is not None]
        return min(ms, key=rank) if ms else None

    out = []
    left = sorted(marks, key=lambda c: c["t"])
    for k in kickoffs:
        near = [c for c in left if k["kick"] - GOAL_NEAR_S * fps <= c["t"] * fps <= k["kick"]]
        near = [c for c in near if near[-1]["t"] - c["t"] <= SAME_GOAL_S]  # the last goal before it; earlier: others
        left = [c for c in left if c not in near]
        s = next((s for s in shots if s["outcome"] == "goal" and s["at"] == k["from"]), None)
        b = best(near)
        team = k["team"]
        if b is not None and (team is None or not b["guessed"]):
            team = code[b["team"]]
        by = ["kickoff"] + sorted({c.get("by", "clip") for c in near})
        lab = next((c for c in near if c.get("by") == "label"), None)  # you said when
        out.append({"frame": round(lab["t"] * fps) if lab else int(k["from"]), "team": team, "by": by,
                    "guessed": b is None or b["guessed"],  # not settled by a label or a rated clip
                    "shot": s is not None})
    while left:  # marks with no kickoff found: one goal per bunch
        bunch = [c for c in left if c["t"] - left[0]["t"] <= SAME_GOAL_S]
        left = left[len(bunch):]
        b = best(bunch)
        out.append({"frame": round(bunch[0]["t"] * fps), "team": code[b["team"]] if b else None,
                    "by": sorted({c.get("by", "clip") for c in bunch}), "guessed": b is None or b["guessed"],
                    "shot": False})
    return sorted(out, key=lambda g: g["frame"])


def team_stats(shots: list[dict], gs: list[dict], team: int) -> dict:
    ps = [s for s in shots if s["team"] == team]
    return {"shots": len(ps), **{o: sum(s["outcome"] == o for s in ps) for o in OUTCOMES},
            "goals": sum(g["team"] == team for g in gs)}


# ---- making it ----

def build(store: Store, game: Game, log=None) -> dict:
    """Find and save the game's shots and goals; returns what's saved. ValueError when the touches can't be
    found yet."""
    import time

    from soccer_clips.track import ball_state, events, touches

    t0 = time.time()
    tc = touches.get(store, game)
    if tc is None:
        touches.build(store, game)  # raises why
        tc = touches.load(store, game)
    k = key(store, game)
    lab = events.resolved(store, game, guesses=False)
    marks = [{"t": e["frame"] / game.fps, "team": {1: "ours", 2: "theirs"}.get(ball_state.label_team(e)),
              "guessed": False, "by": "label"} for e in lab["events"] if e["kind"] == "goal"]
    shots, gs = work_out(store, game, tc["touches"], marks)
    stats = {name: team_stats(shots, gs, i) for i, name in enumerate(TEAMS)}
    stats["goals_not_known"] = sum(g["team"] is None for g in gs)
    stats["seconds"] = round(time.time() - t0, 1)
    out = {"key": k, "shots": shots, "goals": gs, "stats": stats}
    atomic_write_text(path(store, game), json.dumps(out) + "\n")
    _cache.pop(game.id, None)
    if log:
        def line(s: dict) -> str:
            return (f"{s['goals']} goals, {s['shots']} shots ({s['goal']} in, {s['saved']} saved, "
                    f"{s['corner']} corner, {s['wide']} wide, {s['blocked']} blocked, {s['missed']} missed)")

        log(f"{game.id}: ours {line(stats['ours'])}; theirs {line(stats['theirs'])}; goals not put down to a team "
            f"{stats['goals_not_known']}; {stats['seconds']} s")
    return out


def work_out(store: Store, game: Game, touch_list: list[dict], marks: list[dict]) -> tuple[list[dict], list[dict]]:
    """The shots and goals from these touches (the corrected ones, or as found for measuring) and goal marks
    beside your clips (G labels)."""
    from soccer_clips.track import ball_state, match, pitch

    st = ball_state.load(store, game)
    mi = match.get(store, game)
    sizes = match._sizes(pitch.load_cal(store, game))
    hs = match.halves(game)
    signs = [match.attack_sign(mi, h) for h in (1, 2)]
    ball = st["xy"].astype(np.float64)
    ends = end_moments(touch_list, mi["out_of_play"], mi["goals_by_kickoff"], ball, hs, signs, sizes["length"],
                       game.fps)
    shots = find(touch_list, ends, ball, hs, signs, sizes["length"], sizes["goal"], game.fps)
    kos = []
    for kick in mi["goals_by_kickoff"]:
        o = next((o for o in mi["out_of_play"] if o["to"] == kick and o["kind"] == "kickoff"), None)
        e = next((e for e in ends if o and e["kind"] == "kickoff" and e["frame"] == o["from"]), None)
        h = half_of(kick, hs)
        by_end = attacking(e["end"], signs[h]) if e and h is not None and signs[h] else None
        taker = next((t for t in touch_list if abs(t["frame"] - kick) <= TAKER_S * game.fps and t["team"] >= 0),
                     None)
        kos.append({"kick": int(kick), "from": int(o["from"]) if o else int(kick),
                    "team": kickoff_scorer(by_end, 1 - taker["team"] if taker else None)})
    clip_goals = ball_state.goal_moments(game, st["xy"], signs if mi["direction"]["sure"] or mi["direction"]["set"]
                                         else None, sizes["length"])
    gs = goals(shots, kos, [{**c, "by": "clip"} for c in clip_goals] + marks, game.fps)
    return shots, gs


_cache: dict[str, tuple[str, dict]] = {}


def load(store: Store, game: Game) -> dict | None:
    """The saved shots if up to date, else None."""
    k = key(store, game)
    if k is None:
        return None
    hit = _cache.get(game.id)
    if hit and hit[0] == k:
        return hit[1]
    p = path(store, game)
    if not p.exists():
        return None
    got = json.loads(p.read_text())
    if got.get("key") != k:
        return None
    _cache.clear()
    _cache[game.id] = (k, got)
    return got


def get(store: Store, game: Game, log=None) -> dict | None:
    """The shots, found first if missing or out of date; None if they can't be yet."""
    got = load(store, game)
    if got is not None:
        return got
    try:
        return build(store, game, log=log)
    except ValueError:
        return None


# ---- measured against the labels ----

def matches(s: dict, f: int, fps: float) -> bool:
    """A found shot is a labelled one at frame f: its shooter's touch within TOL_S, or (shooter not known) f
    between BACK_S before its end and TOL_S after."""
    if s["frame"] is not None:
        return abs(s["frame"] - f) <= TOL_S * fps
    return s["at"] - BACK_S * fps <= f <= s["at"] + TOL_S * fps


def compare(shots: list[dict], gs: list[dict], evs: list[dict], windows: list[dict], score: tuple | None,
            fps: float) -> dict:
    """Against your labels: labelled shots (S, anywhere) found (found, of labelled_n); in full windows, found
    ones labelled a shot or goal (labelled, of found_n); labelled saves (V) found as a saved shot's end within
    TOL_S (saves, of saves_n); and the goals per team against the final score (ours, theirs), if entered."""
    inside = lambda f: any(w["start"] <= f < w["end"] for w in windows)  # noqa: E731
    lab = sorted({e["frame"] for e in evs if e["kind"] == "shot"})
    found = sum(any(matches(s, f, fps) for s in shots) for f in lab)
    sw = [s for s in shots if inside(s["frame"] if s["frame"] is not None else s["at"])]
    lw = [e["frame"] for e in evs if e["kind"] in ("shot", "goal") and inside(e["frame"])]
    real = sum(any(matches(s, f, fps) for f in lw) for s in sw)
    sv = [e["frame"] for e in evs if e["kind"] == "save"]
    saves = sum(any(s["outcome"] == "saved" and abs(s["at"] - f) <= TOL_S * fps for s in shots) for f in sv)
    share = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"labelled_n": len(lab), "found": share(found, len(lab)), "found_n": len(sw),
            "labelled": share(real, len(sw)), "saves_n": len(sv), "saves": share(saves, len(sv)),
            "goals": [sum(g["team"] == 0 for g in gs), sum(g["team"] == 1 for g in gs)],
            "score": list(score) if score else None}


def evaluate(store: Store, game: Game) -> dict | None:
    """Shots as found (from the touches before your labels) against your labels, and the goals (with your labels)
    against the final score (compare); None if there are no shots yet."""
    from soccer_clips.track import events, touches

    got = get(store, game)
    if got is None:
        return None
    raw, _ = work_out(store, game, touches.load(store, game)["found"], [])
    lab = events.resolved(store, game)
    score = ((game.score_ours, game.score_theirs) if game.score_ours is not None and game.score_theirs is not None
             else None)
    return compare(raw, got["goals"], lab["events"], lab["windows"], score, game.fps)
