"""
Passes, bad passes and chains (Phase 6.4), from the touches (touches.py).

- **A pass** is a touch followed by someone else's (another tracklet), with the ball carried PASS_M or more from
  where it was kicked on the way. Shorter, the ball was taken off them (a tackle) or handed on in a scramble:
  on your windows the touches you labelled T, not P, are mostly those (the player then lost the ball close by).
- **Outcome:** `out` if the ball goes out of play before the next touch; else `completed` when the next toucher
  is on the kicker's team, `lost` when on the other; `unknown` when either team isn't known (a kit read as
  neither). **Receiver unknown** when the next touch is more than RECV_S later (the ball path dropped out, or
  the receiver's touch was missed): the outcome is then whoever has the ball then (ball_state possession).
- **Into danger:** completed, but the receiver's ball goes to the other team within DANGER_S (not by a pass of
  theirs, which is their own bad pass).
- **Properties:** length (where kicked to where received, or the ball last seen), direction from the half's
  attacking direction (forward / back within 45°, else sideways), the third it starts and ends in, in the air
  (ball_state's airborne anywhere on the way).
- **Chains:** consecutive completed passes by one team, ended by a pass not completed, a touch by the other team
  or the ball going out.

Against your full windows (2026-11-04, 88 labelled passes in three games, one to one within touches.TOL_S): 57%
of labelled passes found (touches find 66% of them at all), 38% of passes found are labelled P or S (most of
the rest are labelled touches that went to someone else, or touches nobody labelled); the kicker the tracklet you
pointed at 61%, the receiver 50% (where one is found); the outcome agrees with your next label 63% (19 of 30). Both ends
of a pass have to be read right, so it is bounded by the touches: most wrong outcomes are the kicker or the
receiver read as the wrong player (two players together), or the receiver's touch missed. Other distances
(3-8 m), skipping ambiguous touches, and the same named player across tracklets were no better.

<game>/tracks/passes.json: {key, passes: [{frame, to_frame, track, team, keeper, to_track, to_team, outcome,
danger, x, y, to_x, to_y, length, direction, from_third, to_third, air, chain}], chains: [{team, passes, from,
to, end}], stats}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store, atomic_write_text

VERSION = 1
PASS_M = 5.0  # the ball carried this far from the kick before someone else's touch: a pass
RECV_S = 3.0  # the next touch later than this: receiver unknown
DANGER_S = 2.0  # completed, but the ball to the other team within this of receiving it: into danger
TEAMS = ("ours", "theirs")


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "passes.json"


def key(store: Store, game: Game) -> str | None:
    from soccer_clips.track import touches

    k = touches.key(store, game)
    return None if k is None else f"p{VERSION} {k}"


# ---- the pieces (pure, for testing) ----

def travel(ball: np.ndarray, f: int, g: int) -> float:
    """How far (m) the ball gets from where it was at frame f, up to frame g; NaN if it isn't seen at f."""
    seg = ball[f:g + 1]
    ok = np.isfinite(seg[:, 0])
    if not np.isfinite(ball[f, 0]) or not ok.any():
        return float("nan")
    return float(np.linalg.norm(seg[ok] - ball[f], axis=1).max())


def direction(dx: float, dy: float, sign: int | None) -> str | None:
    """forward / back (within 45° of the way we attack, `sign` along x) or sideways; None if not known."""
    if not sign or not np.isfinite(dx) or not np.isfinite(dy):
        return None
    a = dx * sign
    return "forward" if a > abs(dy) else "back" if -a > abs(dy) else "sideways"


def find(touches: list[dict], ball: np.ndarray, in_play: np.ndarray, poss: np.ndarray, fps: float) -> list[dict]:
    """The passes in the touches (in frame order): who to whom and how it went (no properties or chains yet)."""
    out = []
    recv = round(RECV_S * fps)
    for i in range(len(touches) - 1):
        t, nx = touches[i], touches[i + 1]
        if nx["track"] == t["track"]:
            continue  # the same player again: a dribble
        f, g = t["frame"], nx["frame"]
        gone = np.flatnonzero(~in_play[f:g])
        end = f + int(gone[0]) if len(gone) else g  # the ball out of play: the pass ends there
        if not travel(ball, f, end) >= PASS_M:
            continue
        p = {"frame": f, "to_frame": None, "track": t["track"], "team": t["team"], "keeper": t["keeper"],
             "to_track": None, "to_team": None, "outcome": "unknown", "danger": False, "next": i + 1}
        if len(gone):
            p["outcome"], p["to_frame"] = "out", end
        elif g - f > recv:
            p["to_frame"] = f + recv
            team = int(poss[f + recv]) - 1  # possession 1 ours, 2 theirs: as team codes
            if t["team"] >= 0 and team >= 0:
                p["outcome"] = "completed" if team == t["team"] else "lost"
        else:
            p.update(to_frame=g, to_track=nx["track"], to_team=nx["team"])
            if t["team"] >= 0 and nx["team"] >= 0:
                p["outcome"] = "completed" if nx["team"] == t["team"] else "lost"
        out.append(p)
    kicks = {p["frame"] for p in out}
    for p in out:  # into danger: the receiver's ball to the other team soon after, not by a pass of theirs
        if p["outcome"] != "completed" or p["to_track"] is None:
            continue
        j = p["next"]
        while j + 1 < len(touches) and touches[j + 1]["track"] == p["to_track"]:
            j += 1
        if j + 1 < len(touches) and touches[j]["frame"] not in kicks:
            k = touches[j + 1]
            lost = k["team"] >= 0 and k["team"] != p["team"] and in_play[p["to_frame"]:k["frame"]].all()
            p["danger"] = bool(lost and k["frame"] - p["to_frame"] < DANGER_S * fps)
    for p in out:
        del p["next"]
    return out


def describe(passes: list[dict], ball: np.ndarray, air: np.ndarray, hs: list[tuple[int, int]],
             signs: list[int | None], length: float | None) -> None:
    """Adds where, how long, which way, the thirds and in the air to each pass (in place)."""
    from soccer_clips.track.ball_state import thirds

    for p in passes:
        f, g = p["frame"], p["to_frame"]
        seg = ball[f:g + 1]
        seen = np.flatnonzero(np.isfinite(seg[:, 0]))
        a = ball[f]
        b = ball[g] if np.isfinite(ball[g, 0]) else seg[seen[-1]] if len(seen) else np.full(2, np.nan)
        h = next((k for k, (h0, h1) in enumerate(hs) if h0 <= f < h1), None)
        sign = signs[h] if h is not None else None
        r = lambda v: round(float(v), 2) if np.isfinite(v) else None  # noqa: E731
        th = thirds(np.array([a[0], b[0]]), sign, length) if sign and length else np.array([-1, -1])
        p.update(x=r(a[0]), y=r(a[1]), to_x=r(b[0]), to_y=r(b[1]), length=r(np.linalg.norm(b - a)),
                 direction=direction(b[0] - a[0], b[1] - a[1], sign),
                 from_third=int(th[0]) if th[0] >= 0 else None, to_third=int(th[1]) if th[1] >= 0 else None,
                 air=bool(air[f:g + 1].any()))


def chains(passes: list[dict], touches: list[dict], in_play: np.ndarray) -> list[dict]:
    """Runs of completed passes by one team: [{team, passes, from, to (frames), end: lost / out / unknown /
    open (the touches end)}]; each pass gets `chain` (its index, or None if not completed)."""
    out, cur = [], None
    by = {p["frame"]: p for p in passes}

    def close(end: str) -> None:
        nonlocal cur
        if cur is not None:
            out.append({**cur, "end": end})
        cur = None

    last = None
    for t in touches:
        if last is not None and cur is not None and not in_play[last:t["frame"]].all():
            close("out")
        if cur is not None and t["team"] >= 0 and t["team"] != cur["team"]:
            close("lost")
        p = by.get(t["frame"])
        if p is None:
            last = t["frame"]
            continue
        if p["outcome"] == "completed":
            if cur is None or cur["team"] != p["team"]:
                close("lost")
                cur = {"team": p["team"], "passes": 0, "from": p["frame"], "to": p["to_frame"]}
            cur["passes"] += 1
            cur["to"] = p["to_frame"]
            p["chain"] = len(out)
        else:
            p["chain"] = None
            close(p["outcome"])
        last = t["frame"]
    close("open")
    for p in passes:
        p.setdefault("chain", None)
    return out


def team_stats(passes: list[dict], chs: list[dict], team: int) -> dict:
    """One team's numbers: passes by outcome, into danger, completion (completed of completed + lost + out),
    by direction, and chains of 2, 3, 4+ passes and the longest."""
    ps = [p for p in passes if p["team"] == team]
    n = {o: sum(p["outcome"] == o for p in ps) for o in ("completed", "lost", "out", "unknown")}
    done = n["completed"] + n["lost"] + n["out"]
    cs = [c["passes"] for c in chs if c["team"] == team]
    return {"passes": len(ps), **n, "danger": sum(p["danger"] for p in ps),
            "completion": round(n["completed"] / done, 3) if done else None,
            "direction": {d: sum(p["direction"] == d for p in ps) for d in ("forward", "sideways", "back")},
            "chains": {"2": cs.count(2), "3": cs.count(3), "4+": sum(c >= 4 for c in cs),
                       "longest": max(cs, default=0)}}


# ---- making it ----

def build(store: Store, game: Game, log=None) -> dict:
    """Find and save the game's passes; returns what's saved. ValueError when the touches can't be found yet."""
    import time

    from soccer_clips.track import ball_state, match, touches

    t0 = time.time()
    tc = touches.get(store, game)
    if tc is None:
        touches.build(store, game)  # raises why
        tc = touches.load(store, game)
    st = ball_state.load(store, game)
    mi = match.get(store, game)
    k = key(store, game)
    ball = st["xy"].astype(np.float64)
    ps = find(tc["touches"], ball, st["in_play"], st["possession"], game.fps)
    describe(ps, ball, st["air"], match.halves(game), [match.attack_sign(mi, h) for h in (1, 2)],
             (st.get("timeline") or {}).get("length"))
    chs = chains(ps, tc["touches"], st["in_play"])
    stats = {name: team_stats(ps, chs, i) for i, name in enumerate(TEAMS)}
    stats["neither"] = sum(p["team"] < 0 for p in ps)
    stats["seconds"] = round(time.time() - t0, 1)
    out = {"key": k, "passes": ps, "chains": chs, "stats": stats}
    atomic_write_text(path(store, game), json.dumps(out) + "\n")
    _cache.pop(game.id, None)
    if log:
        def line(s: dict) -> str:
            c = s["chains"]
            pct = "–" if s["completion"] is None else f"{s['completion']:.0%}"
            return (f"{s['passes']} passes, {pct} completed ({s['completed']} completed, {s['lost']} lost, {s['out']} "
                    f"out, {s['unknown']} unknown; {s['danger']} into danger), chains 2: {c['2']}, 3: {c['3']}, "
                    f"4+: {c['4+']}, longest {c['longest']}")

        log(f"{game.id}: ours {line(stats['ours'])}; theirs {line(stats['theirs'])}; neither kit "
            f"{stats['neither']}; {stats['seconds']} s")
    return out


_cache: dict[str, tuple[str, dict]] = {}


def load(store: Store, game: Game) -> dict | None:
    """The saved passes if up to date, else None."""
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
    """The passes, found first if missing or out of date; None if they can't be yet."""
    got = load(store, game)
    if got is not None:
        return got
    try:
        return build(store, game, log=log)
    except ValueError:
        return None


# ---- measured against the labels ----

def label_outcome(e: dict, nx: dict | None, in_play: np.ndarray) -> str | None:
    """A labelled pass's outcome from the next label: out (out of play before it), completed / lost (its team
    against the pass's), unknown (a team not known); None with no next label."""
    from soccer_clips.track.ball_state import label_team

    if nx is None:
        return None
    if not in_play[e["frame"]:nx["frame"]].all():
        return "out"
    a, b = label_team(e), label_team(nx)
    return "unknown" if not a or not b else "completed" if a == b else "lost"


def compare(passes: list[dict], evs: list[dict], windows: list[dict], in_play: np.ndarray, fps: float) -> dict:
    """Passes found against labelled ones (P) inside full windows: the share of labelled passes found (found), of
    found ones labelled a pass or shot (labelled), and of pairs, the kicker the tracklet you pointed at (kicker),
    the receiver the one at your next label (receiver), and the outcome your next label gives (outcome, where both
    are completed / lost / out); *_n: how many pairs each share is of."""
    from soccer_clips.track.touches import TOL_S, pair

    inside = lambda f: any(w["start"] <= f < w["end"] for w in windows)  # noqa: E731
    ps = [p for p in passes if inside(p["frame"])]
    lab = sorted((e for e in evs if e["kind"] != "goal" and inside(e["frame"])), key=lambda e: e["frame"])
    pj = [j for j, e in enumerate(lab) if e["kind"] == "pass"]
    tol = round(TOL_S * fps)
    pairs = pair([p["frame"] for p in ps], [lab[j]["frame"] for j in pj], tol)
    kicks = pair([p["frame"] for p in ps], [e["frame"] for e in lab if e["kind"] in ("pass", "shot")], tol)
    kicker, receiver, outcome = [], [], []
    known = ("completed", "lost", "out")
    for i, jj in pairs:
        j = pj[jj]
        e, nx = lab[j], lab[j + 1] if j + 1 < len(lab) else None
        if e["track"] is not None:
            kicker.append(ps[i]["track"] == e["track"])
        if nx is not None and nx["track"] is not None and ps[i]["to_track"] is not None:
            receiver.append(ps[i]["to_track"] == nx["track"])
        lo = label_outcome(e, nx, in_play)
        if lo in known and ps[i]["outcome"] in known:
            outcome.append(lo == ps[i]["outcome"])
    share = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"labelled_n": len(pj), "found_n": len(ps), "found": share(len(pairs), len(pj)),
            "labelled": share(len(kicks), len(ps)), "kicker": share(sum(kicker), len(kicker)),
            "receiver": share(sum(receiver), len(receiver)), "outcome": share(sum(outcome), len(outcome)),
            "kicker_n": len(kicker), "receiver_n": len(receiver), "outcome_n": len(outcome)}


def evaluate(store: Store, game: Game) -> dict | None:
    """Passes against the game's labelled full windows (compare); None if there are none or no passes."""
    from soccer_clips.track import ball_state, events

    lab = events.resolved(store, game)
    if not lab["windows"]:
        return None
    got = get(store, game)
    st = ball_state.load(store, game)
    if got is None or st is None:
        return None
    return compare(got["passes"], lab["events"], lab["windows"], st["in_play"], game.fps)
