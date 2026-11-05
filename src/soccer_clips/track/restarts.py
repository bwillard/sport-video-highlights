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

**Punts:** the goalie putting the ball back in play from their hands. A goalie's touch (touches.py) with the
ball held at hand height in their box (the top HELD_TOP of it, on the video) for HELD_MIN frames in the HELD_S
before, then sent PUNT_M or more within PUNT_S: kicked out of the hands, or (at U11, where they can't punt, what
the videos show) put down and kicked, or thrown. One per PUNT_SAME_S; the goalie's team.

Worked out when asked (well under a second), not saved.
"""
from __future__ import annotations

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

TAKER_S = 0.7  # the taker: a touch this close to the kick
LAST_S = 10.0  # a throw-in: the last touch before the ball went out, this long before at most
KINDS = ("kickoff", "throw-in", "goal kick", "corner", "penalty", "free kick", "punt")  # for the log; not "goalie"
ALWAYS = ("kickoff", "penalty")  # listed even without an out-of-play stretch before them
TOL_S = 1.0  # measuring against labels: a restart this close to a labelled one is it
HELD_S = 3.0  # a punt: the ball in the goalie's hands within this before their touch...
HELD_MIN = 5  # ...for at least this many frames...
HELD_TOP = 0.7  # ...in the top this share of their box (hands, not feet)...
PUNT_M = 10.0  # ...then this far from where it was played...
PUNT_S = 3.0  # ...within this long
PUNT_SAME_S = 5.0  # punts this close together are one (the longest)


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


def punts(touches: list[dict], frame: np.ndarray, bx: np.ndarray, by: np.ndarray, xy: np.ndarray, t, fps: float) -> list[dict]:
    """The goalies' punts (HELD_*, PUNT_*), as restarts [{frame, kind "punt", x, y, team, track, by_taker, by_rule
    None, disagree False}]. frame, bx, by: the ball on the video (fractions of the frame, sorted by frame); xy:
    the ball on the field per frame; t: the tracklets (people.Tracks)."""
    got = []
    for x in touches:
        if not x["keeper"]:
            continue
        f = x["frame"]
        a, b = np.searchsorted(frame, [f - HELD_S * fps, f + 1])
        held = 0
        for i in range(a, b):
            lo, hi = t.window(int(frame[i]), int(frame[i]) + 1)
            k = next((j for j in range(lo, hi) if t.track[j] == x["track"]), None)
            if k is None:
                continue
            x1, y1, x2, y2 = t.box[k]
            pad = 0.3 * (x2 - x1)
            held += x1 - pad <= bx[i] <= x2 + pad and y1 <= by[i] <= y1 + HELD_TOP * (y2 - y1)
        seg = xy[f:f + int(PUNT_S * fps)]
        seg = seg[np.isfinite(seg[:, 0])]
        far = float(np.linalg.norm(seg - seg[0], axis=1).max()) if len(seg) > 1 else 0.0
        if held >= HELD_MIN and far >= PUNT_M:
            got.append(({"frame": f, "kind": "punt", "x": x["x"], "y": x["y"], "team": x["keeper"] - 1,
                         "track": x["track"], "by_taker": x["keeper"] - 1, "by_rule": None, "disagree": False}, far))
    out = []
    for r, far in got:  # one per hold: the longest
        if out and r["frame"] - out[-1][0]["frame"] <= PUNT_SAME_S * fps:
            if far > out[-1][1]:
                out[-1] = (r, far)
            continue
        out.append((r, far))
    return [r for r, _ in out]


def stats(rs: list[dict]) -> dict:
    """Per kind: ours, theirs, not known; and how often the two signals agree where both are known."""
    out = {k: {"ours": 0, "theirs": 0, "unknown": 0} for k in KINDS}
    for r in rs:
        out[r["kind"]][{0: "ours", 1: "theirs"}.get(r["team"], "unknown")] += 1
    both = [r for r in rs if r["by_taker"] is not None and r["by_rule"] is not None]
    out["agree"] = round(sum(not r["disagree"] for r in both) / len(both), 3) if both else None
    out["agree_n"] = len(both)
    return out


def get(store: Store, game: Game, found: bool = False) -> dict | None:
    """The game's restarts ({restarts, stats}), or None when the touches can't be found yet. found: from the
    touches as found, before your labels (for measuring against them)."""
    from soccer_clips.track import ball_game, ball_state, match, people, touches

    tc = touches.get(store, game)
    mi = match.load(store, game)
    if tc is None or mi is None:
        return None
    signs = [match.attack_sign(mi, h) for h in (1, 2)]
    tl = tc["found" if found else "touches"]
    rs = listed(mi["restarts"], mi["out_of_play"], tl, signs, game.fps)
    bp, st, t = ball_game.load_path(store, game), ball_state.load(store, game), people.tracks(store, game)
    if bp is not None and st is not None and t is not None:
        rs = sorted(rs + punts(tl, bp["frame"], bp["x"], bp["y"], st["xy"], t, game.fps), key=lambda r: r["frame"])
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
