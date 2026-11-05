"""
Balls won and lost (Phase 6.6), from the touches (touches.py, with your labels in), the passes and the shots.

Each time the ball goes from one team to the other in play, one touch to the next (touches in a kit read as
neither team are passed over: on your windows that finds more of your labelled changes for as many wrong), the
team touching it next won it, by:
- **interception:** the other team's pass (passes.py) ended with us: a bad pass, a long ball they collected;
- **tackle:** taken off them within TACKLE_S of their touch, not from a pass;
- **recovery:** anything else: a loose ball after a shot, a deflection or a 50-50.
Not after a shot: the other team's next touch (a block, the goalie, a clearance after it) ends the attack, not
a ball lost. Lost is the same seen from the other side. Where: the third, from the winner's attacking direction (0 their
own, 1 middle, 2 attacking), and whether it was in the winner's attacking half.

Against your full windows (2026-11-04, 54 changes of team between your labels, matched within TOL_S): 50% found,
31% of those found are changes in your labels, the team that won it right 78% of the time. Each touch read as
the wrong team makes two changes that aren't there, so it is bounded by the touches' team (83%). Requiring the
new team to touch it twice found fewer for no more right, and the possession runs (ball_state.py) found fewer and
more wrong ones.

Worked out when asked (milliseconds), not saved.
"""
from __future__ import annotations

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

TACKLE_S = 1.5  # the ball taken this soon after the other team's touch (not a pass): a tackle
TOL_S = 1.0  # measuring against labels: a change this close to a labelled one is it
KINDS = ("interception", "tackle", "recovery")
THIRDS = ("defending", "middle", "attacking")  # from the winner's attacking direction
TEAMS = ("ours", "theirs")


def find(touches: list[dict], pass_frames: set, in_play: np.ndarray, hs: list[tuple[int, int]],
         signs: list[int], length: float | None, fps: float, shot_frames: set = frozenset()) -> list[dict]:
    """The changes of team: [{frame (the winner's touch), team (who won it: 0 / 1), track, from_track (who lost
    it), kind, x, y, third (from the winner's attacking direction, or None), their_half}]; none from a shot (the
    touches at shot_frames)."""
    tk = [t for t in touches if t["team"] >= 0]
    out = []
    for a, b in zip(tk, tk[1:]):
        if a["team"] == b["team"] or not in_play[a["frame"]:b["frame"]].all() or a["frame"] in shot_frames:
            continue
        kind = ("interception" if a["frame"] in pass_frames
                else "tackle" if b["frame"] - a["frame"] <= TACKLE_S * fps else "recovery")
        h = next((k for k, (h0, h1) in enumerate(hs) if h0 <= b["frame"] < h1), None)
        sign = (signs[h] if b["team"] == 0 else -signs[h]) if h is not None and signs[h] else 0
        third = None
        if sign and length:
            third = int(np.digitize(sign * b["x"], [-length / 6, length / 6]))
        out.append({"frame": b["frame"], "team": b["team"], "track": b["track"], "from_track": a["track"],
                    "kind": kind, "x": b["x"], "y": b["y"], "third": third,
                    "their_half": bool(sign * b["x"] > 0) if sign else None})
    return out


def team_stats(tos: list[dict], team: int) -> dict:
    """One team's balls won and lost: by kind, won by third (theirs from their direction), won in the other
    team's half, lost in our own third (the two numbers coaches usually want)."""
    won = [t for t in tos if t["team"] == team]
    lost = [t for t in tos if t["team"] == 1 - team]
    return {"won": len(won), "lost": len(lost),
            "won_by": {k: sum(t["kind"] == k for t in won) for k in KINDS},
            "lost_by": {k: sum(t["kind"] == k for t in lost) for k in KINDS},
            "won_third": {n: sum(t["third"] == i for t in won) for i, n in enumerate(THIRDS)},
            "won_their_half": sum(bool(t["their_half"]) for t in won),
            "lost_own_third": sum(t["third"] == 2 for t in lost)}  # the winner's attacking third is our own


def work_out(store: Store, game: Game, touch_list: list[dict], pass_list: list[dict], shot_list: list[dict]) -> list[dict]:
    from soccer_clips.track import ball_state, match

    st = ball_state.load(store, game)
    mi = match.load(store, game)
    return find(touch_list, {p["frame"] for p in pass_list}, st["in_play"], match.halves(game),
                [match.attack_sign(mi, h) for h in (1, 2)], (st.get("timeline") or {}).get("length"), game.fps,
                {x["frame"] for x in shot_list if x["frame"] is not None})


def get(store: Store, game: Game) -> dict | None:
    """The game's balls won and lost ({turnovers, stats}), or None when the passes can't be found yet."""
    from soccer_clips.track import passes, shots, touches

    tc, ps, sh = touches.get(store, game), passes.get(store, game), shots.get(store, game)
    if tc is None or ps is None or sh is None:
        return None
    tos = work_out(store, game, tc["touches"], ps["passes"], sh["shots"])
    return {"turnovers": tos, "stats": {n: team_stats(tos, i) for i, n in enumerate(TEAMS)}}


# ---- measured against the labels ----

def label_changes(evs: list[dict], windows: list[dict], in_play: np.ndarray) -> list[tuple[int, int]]:
    """The changes of team between your labels in full windows: [(frame of the winner's label, team 0 / 1)]."""
    from soccer_clips.track.ball_state import label_team

    out = []
    for w in windows:
        lab = sorted((e for e in evs if e["kind"] != "goal" and w["start"] <= e["frame"] < w["end"]),
                     key=lambda e: e["frame"])
        for a, b in zip(lab, lab[1:]):
            ta, tb = label_team(a), label_team(b)
            if ta and tb and ta != tb and in_play[a["frame"]:b["frame"]].all():
                out.append((b["frame"], tb - 1))
    return out


def compare(tos: list[dict], evs: list[dict], windows: list[dict], in_play: np.ndarray, fps: float) -> dict:
    """Against the changes of team between your labels in full windows: the share found (found, of
    labelled_n), of those found the share that are changes in your labels (labelled, of found_n), and the team
    that won it right (team); W labels found as a change within TOL_S (won, of won_n)."""
    from soccer_clips.track.touches import pair

    inside = lambda f: any(w["start"] <= f < w["end"] for w in windows)  # noqa: E731
    lab = label_changes(evs, windows, in_play)
    fw = [t for t in tos if inside(t["frame"])]
    pr = pair([t["frame"] for t in fw], [f for f, _ in lab], round(TOL_S * fps))
    ws = [e["frame"] for e in evs if e["kind"] == "won"]
    won = sum(any(abs(t["frame"] - f) <= TOL_S * fps for t in tos) for f in ws)
    share = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"labelled_n": len(lab), "found": share(len(pr), len(lab)), "found_n": len(fw),
            "labelled": share(len(pr), len(fw)), "team": share(sum(fw[i]["team"] == lab[j][1] for i, j in pr), len(pr)),
            "won_n": len(ws), "won": share(won, len(ws))}


def evaluate(store: Store, game: Game) -> dict | None:
    """Balls won and lost as found (from the touches before your labels) against your full windows; None
    without any."""
    from soccer_clips.track import ball_state, events, passes, shots, touches

    lab = events.resolved(store, game)
    tc = touches.get(store, game)
    if not lab["windows"] or tc is None:
        return None
    found = tc["found"]
    raw_shots, _ = shots.work_out(store, game, found, [])
    raw_passes, _, _ = passes.work_out(store, game, found, raw_shots)
    tos = work_out(store, game, found, raw_passes, raw_shots)
    return compare(tos, lab["events"], lab["windows"], ball_state.load(store, game)["in_play"], game.fps)
