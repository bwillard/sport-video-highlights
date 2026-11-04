"""
Restarts for the game log (Phase 6.5a): kickoffs, throw-ins, goal kicks, corners, penalties and free kicks, with
the team that took each.

match.py (6.0) finds them: the ball parked, then kicked, classed by where it sat. Listed here are the ones that
restart play after a stoppage (the ball out of play up to the kick: match.py's out-of-play stretches), plus
every kickoff and penalty; the goalie's ball isn't a restart for the log. Where two stretches start at the same
moment (the ball placed, then kicked), the last kick is the restart.

**Which team**, from two signals that should agree:
- **who took it:** the touch (touches.py) at the kick, within TAKER_S, in a team's kit;
- **the rules:** a goal kick is the defending team's (whose goal line it's on), a corner and a penalty the
  attacking team's, a throw-in the team that didn't touch the ball last before it went out (within LAST_S).
  Kickoffs and free kicks have none here (who kicks off after a goal is shots.py's check on the scorer).
Both known and the same, or one known: that team. Both known and different: not known (`disagree`).

Worked out when asked (well under a second), not saved.
"""
from __future__ import annotations

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

TAKER_S = 0.7  # the taker: a touch this close to the kick
LAST_S = 10.0  # a throw-in: the last touch before the ball went out, this long before at most
KINDS = ("kickoff", "throw-in", "goal kick", "corner", "penalty", "free kick")  # for the log; not "goalie"
ALWAYS = ("kickoff", "penalty")  # listed even without an out-of-play stretch before them
TOL_S = 1.0  # measuring against labels: a restart this close to a labelled one is it


def stoppages(restarts: list[dict], out: list[dict]) -> list[dict]:
    """The restarts for the log: [{kick, still, from (ball out, or None), x, y, kind, half}], in time order."""
    last = {}
    for o in out:
        last[o["from"]] = max(last.get(o["from"], -1), o["to"])  # the same stoppage: the last kick
    ends = {to: f for f, to in last.items()}
    got = []
    for r in restarts:
        if r["kind"] not in KINDS or (r["kick"] not in ends and r["kind"] not in ALWAYS):
            continue
        got.append({**r, "from": ends.get(r["kick"])})
    return sorted(got, key=lambda r: r["kick"])


def by_rule(r: dict, touches: list[dict], frames: np.ndarray, sign: int, fps: float) -> int | None:
    """The team (0 ours, 1 theirs) the rules give the restart; None if they don't say. `sign`: the goal we
    attack in its half (+1 / -1, 0 not known)."""
    end = int(np.sign(r["x"])) or 1  # the end of the field it's at
    if r["kind"] in ("goal kick", "corner", "penalty"):
        if not sign:
            return None
        att = 0 if end == sign else 1
        return 1 - att if r["kind"] == "goal kick" else att
    if r["kind"] == "throw-in" and r["from"] is not None:
        i = int(np.searchsorted(frames, r["from"])) - 1
        while i >= 0 and frames[i] >= r["from"] - LAST_S * fps:
            if touches[i]["team"] >= 0:
                return 1 - touches[i]["team"]
            i -= 1
    return None


def taker(r: dict, touches: list[dict], frames: np.ndarray, fps: float) -> dict | None:
    """The touch at the restart's kick (the nearest within TAKER_S), or None."""
    a, b = np.searchsorted(frames, [r["kick"] - TAKER_S * fps, r["kick"] + TAKER_S * fps + 1])
    near = [touches[i] for i in range(a, b)]
    return min(near, key=lambda t: abs(t["frame"] - r["kick"])) if near else None


def listed(restarts: list[dict], out: list[dict], touches: list[dict], signs: list[int], fps: float) -> list[dict]:
    """The restarts with who took them: [{frame (the kick), kind, x, y, team (0 / 1 / None), track (the taker's
    tracklet, or None), by_taker, by_rule (teams, or None), disagree}]."""
    frames = np.array([t["frame"] for t in touches], np.int64)
    out_ = []
    for r in stoppages(restarts, out):
        sign = signs[r["half"] - 1] if 1 <= r["half"] <= len(signs) else 0
        t = taker(r, touches, frames, fps)
        a = t["team"] if t is not None and t["team"] >= 0 else None
        b = by_rule(r, touches, frames, sign, fps)
        team = a if b is None else b if a is None else a if a == b else None
        out_.append({"frame": int(r["kick"]), "kind": r["kind"], "x": r["x"], "y": r["y"], "team": team,
                     "track": t["track"] if t is not None else None, "by_taker": a, "by_rule": b,
                     "disagree": a is not None and b is not None and a != b})
    return out_


def stats(rs: list[dict]) -> dict:
    """Per kind: ours, theirs, not known; and how often the two signals agree where both are known."""
    out = {k: {"ours": 0, "theirs": 0, "unknown": 0} for k in KINDS}
    for r in rs:
        out[r["kind"]][{0: "ours", 1: "theirs"}.get(r["team"], "unknown")] += 1
    both = [r for r in rs if r["by_taker"] is not None and r["by_rule"] is not None]
    out["agree"] = round(sum(not r["disagree"] for r in both) / len(both), 3) if both else None
    out["agree_n"] = len(both)
    return out


def get(store: Store, game: Game) -> dict | None:
    """The game's restarts ({restarts, stats}), or None when the touches can't be found yet."""
    from soccer_clips.track import match, touches

    tc = touches.get(store, game)
    mi = match.load(store, game)
    if tc is None or mi is None:
        return None
    signs = [match.attack_sign(mi, h) for h in (1, 2)]
    rs = listed(mi["restarts"], mi["out_of_play"], tc["touches"], signs, game.fps)
    return {"restarts": rs, "stats": stats(rs)}


def compare(rs: list[dict], evs: list[dict], fps: float) -> dict:
    """Against restart labels (K): the share found (a restart within TOL_S), and of those, the team right where
    both are known."""
    from soccer_clips.track.ball_state import label_team

    lab = [e for e in evs if e["kind"] == "restart"]
    hit, team = 0, []
    for e in lab:
        r = min(rs, key=lambda r: abs(r["frame"] - e["frame"]), default=None)
        if r is None or abs(r["frame"] - e["frame"]) > TOL_S * fps:
            continue
        hit += 1
        lt = label_team(e)
        if lt and r["team"] is not None:
            team.append(r["team"] + 1 == lt)
    share = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"labelled_n": len(lab), "found": share(hit, len(lab)), "team": share(sum(team), len(team)),
            "team_n": len(team)}
