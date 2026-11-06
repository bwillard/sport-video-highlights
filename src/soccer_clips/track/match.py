"""
The shape of a game (Phase 6.0): restarts, which goal we attack in each half, the goalies, and when the
ball is out of play. What the game's events (Phase 6) are read against. Built from the positions on the
field (positions.py) and the teams (people.roles_for).

- **Restarts:** the ball still on the ground for STILL_S (within STILL_M), then kicked (KICK_M away within
  KICK_S). Unseen for up to STILL_GAP_S and seen again where it sat, it's still sitting there: a kicker standing
  over the ball at a kickoff hides it (Elite Strikers' 4:30 kickoff, after a goal, was missed with 1 s; 3 s finds
  it and SAA's 18:18 one, and a few more throw-ins, corners and goal kicks, the rest the same). Still on the
  centre spot, then unseen up to KICK_GAP_S and seen next already away, it was kicked off unseen (Elite
  Strikers' 0:37): kept only with the teams lined up either side of it (HIDDEN_SPLIT: that one, FC-1's 21:02 and
  FCU's 18:29 and 1:10:10, each a goal clip before it; `clear` let through open play by the centre circle, and on
  every spot it found many restarts that weren't). Where it sat says what it was: kickoff (the middle of the halfway line; SAA's calibration puts
  the centre spot 3-4 m off, so a box around it, and further along the length up to KICKOFF_NEAR_M when the
  teams stand split either side of the ball: FC-1's views put kickoffs up to 9 m off), corner, goal kick, penalty, throw-in (by a touchline), the goalie's ball
  (still in a penalty area: in their hands, or a free kick there), free kick (anywhere else). Kickoffs
  within KICKOFF_SAME_S of each other are one (a kickoff retaken, or the ball stopping again).
- **Kickoffs and direction:** at a kickoff every player is in their own half, so where our kit stands
  says which goal we defend. Goal clips you've rated vote too (😊 ours, 🙁 theirs: where the ball ends). A kickoff votes when the formation is clear (our boxes' mean side at least
  FORMATION, and theirs not as clearly on the same side); each votes for its half, and the second half
  is the first reversed. Kickoffs away from the half starts are goals: one needs a clear formation, or the
  ball over a goal line within GOAL_IN_S of its out-of-play stretch starting, for GOAL_IN_MIN_S (in the net;
  FC-1's 6:27 was our goalie's punt, the ball in the air reading over the line by the corner flag for 2 frames;
  real goals 5–150); else it's a free kick by the centre spot (Elite Strikers' 6:36: the ball over a touchline, then still by the spot with the teams mixed;
  across the games, every kickoff with neither had no goal clip or label but EBU's 33:47, whose clip is the
  goal anyway).
- **Goalies:** per half and end, tracklets that stay in that penalty area and either aren't in either
  team's kit (the goalie's own colours) or stand deep (median within KEEPER_DEEP_M of the goal line) and
  aren't in the attacking team's kit (a goalie's kit is often clustered in with a team's). At most one
  per end per frame: the deepest. You can mark a goalie in the tagger (Shift+G, or 🧤 in the box a click
  opens; tracks/goalies.json, clicks matched to tracklets like player labels): the marked tracklet is that
  team's goalie, and its kit colour (the kit cluster most of its boxes are in) marks the rest: a tracklet
  mostly in that kit and mostly in the penalty area that team defends is their goalie too. A "not a goalie"
  mark (team "none") rules a tracklet out: it's never a goalie, and the deepest of the rest is.
- **Out of play:** from when the ball crossed a line (looking back up to LOOKBACK_S from a restart on a
  line or a kickoff) or, for a free kick, from when it stopped, to the restart kick. A restart on a line
  needs the ball out within OUT_NEAR_S of it stopping (an earlier crossing is another moment, or a stray
  ball path off the field); one without, and a free kick, needs the ball still at least STOP_S: shorter is
  a ball slowing at someone's feet in open play (in the labelled windows, every one was).

<game>/tracks/match.npz: `keeper` per tracklet row (0 none, 1 ours, 2 theirs; people.Tracks order), and
`info` (JSON: restarts, kickoffs, direction, out-of-play stretches, how many goalie frames), with `key`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

VERSION = 7  # 6: a kickoff mid-half needs a clear formation or the ball in a goal; 7: in it for GOAL_IN_MIN_S
STILL_S = 1.0  # the ball still this long...
STILL_M = 1.0  # ...within this of where it sits
KICK_M = 3.0  # then this far away...
KICK_S = 1.0  # ...within this long: kicked
STILL_GAP_S = 3.0  # the ball unseen this long (a player standing over it) and seen again where it sat: still still
KICK_GAP_S = 5.0  # still on the centre spot, then unseen up to this long and seen again away: kicked off while hidden
KICKOFF_BOX = (3.0, 5.0)  # |x|, |y| (metres) of a kickoff spot: the middle of the halfway line
KICKOFF_NEAR_M = 10.0  # further along the length than KICKOFF_BOX, up to this, a kickoff needs a split formation
SPLIT = (0.8, 0.6)  # ...our boxes' mean side relative to the ball at least the first, theirs opposite the second
HIDDEN_SPLIT = (0.7, 0.25)  # a kickoff kicked unseen: our side at least the first, theirs opposite at least the second
CORNER_M = 3.0  # within this of a corner (each way)
THROW_M = 2.0  # within this of a touchline, or beyond it
AREA_PAD_M = 1.5  # goal area, penalty area and spot, this much bigger
FORMATION = 0.5  # a kickoff counts when the mean side (+1 / -1) of our boxes is at least this, and theirs opposite
FORMATION_MIN_BOXES = 15  # box rows a team needs at a kickoff to say which side it's on
SURE = 1.5  # direction votes needed (first half minus second) to be sure
KEEPER_S = 2.0  # a goalie's tracklet: at least this long on the field...
KEEPER_IN = 0.8  # ...this share of it in one penalty area (other kit)...
KEEPER_DEEP_IN = 0.85  # ...or this share, and
KEEPER_DEEP_M = 6.0  # a median this close to the goal line, not in the attacking team's kit
KEEPER_KIT_S = 1.0  # in a marked goalie's kit: at least this long on the field...
KEEPER_KIT_IN = 0.6  # ...and this share of it in the penalty area their team defends
TEAM_CODES = {"ours": 1, "theirs": 2}  # keeper codes
NOT_GOALIE = "none"  # a goalie mark's team: this tracklet isn't a goalie
LOOKBACK_S = 30.0  # a restart on a line: when the ball went out, looked for this far back
GOAL_LOOKBACK_S = 90.0  # a kickoff: when the goal went in
GOAL_IN_S = 5.0  # a kickoff after a goal: the ball over a goal line this soon after its stretch starts (or a clear formation)...
GOAL_IN_MIN_S = 0.3  # ...for this long (the ball path's samples, a frame each)
OUT_NEAR_S = {"throw-in": 8.0, "corner": 8.0, "goal kick": 20.0}  # the ball out this soon before it stopped
STOP_S = 3.0  # a stoppage with no crossing (free kick, penalty, a line restart unseen going out): still this long
KICKOFF_SAME_S = 30.0  # kickoffs this close are one; a kickoff this close to a half's start time is that half's
LINE_PAD_M = 0.5  # the ball beyond a line by this much is out
NET_M = 8.0  # after a goal: the ball back in the field no further than this from that goal line is still being fetched


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "match.npz"


# ---- restarts ----

def dead_balls(frame: np.ndarray, xy: np.ndarray, fps: float, gap_s: float = STILL_GAP_S,
               kick_gap_s: float = KICK_GAP_S) -> list[dict]:
    """The ball still for STILL_S then kicked: [{still (first still frame), kick (first frame it's moved
    off), x, y (where it sat), hidden (on the centre spot, kicked while unseen: seen next already away, within
    kick_gap_s; build keeps it only as a kickoff with the teams lined up)}]. frame sorted; xy NaN where not placed
    (skipped)."""
    ok = np.isfinite(xy[:, 0])
    f, p = frame[ok], xy[ok].astype(np.float64)
    n, hw, out, i = len(f), int(round(STILL_S * fps)), [], 0
    while i < n:
        j = int(np.searchsorted(f, f[i] + hw))
        if j >= n:
            break
        c = np.median(p[i:j], axis=0)
        if j - i >= 0.6 * hw and (np.linalg.norm(p[i:j] - c, axis=1) < STILL_M).all():
            k = j
            while k < n and np.linalg.norm(p[k] - c) < STILL_M and f[k] - f[k - 1] < gap_s * fps:
                k += 1
            spot = abs(c[0]) < KICKOFF_BOX[0] and abs(c[1]) < KICKOFF_BOX[1]
            hid = spot and k < n and KICK_S * fps < f[k] - f[k - 1] <= kick_gap_s * fps  # kicked off unseen
            m = int(np.searchsorted(f, (f[k] if hid else f[k - 1]) + int(KICK_S * fps), "right"))
            gone = np.flatnonzero(np.linalg.norm(p[k:m] - c, axis=1) > KICK_M) if k < n else []
            if len(gone):
                out.append({"still": int(f[i]), "kick": int(f[k]), "x": round(float(c[0]), 2), "y": round(float(c[1]), 2),
                            **({"hidden": True} if hid else {})})
            i = k
        else:
            i += 1
    return out


def restart_kind(x: float, y: float, sizes: dict) -> str:
    """What a restart at (x, y) is, from the field's sizes ({length, width, ga_depth, ga_width, pa_depth,
    pa_width, spot})."""
    L, W, ax, ay = sizes["length"] / 2, sizes["width"] / 2, abs(x), abs(y)
    if ax < KICKOFF_BOX[0] and ay < KICKOFF_BOX[1]:
        return "kickoff"
    if ax > L - CORNER_M and ay > W - CORNER_M:
        return "corner"
    if ax > L - sizes["ga_depth"] - AREA_PAD_M and ay < sizes["ga_width"] / 2 + AREA_PAD_M:
        return "goal kick"
    if np.hypot(ax - (L - sizes["spot"]), ay) < 1.0:
        return "penalty"
    if ay > W - THROW_M:
        return "throw-in"
    if ax > L - sizes["pa_depth"] - AREA_PAD_M and ay < sizes["pa_width"] / 2 + AREA_PAD_M:
        return "goalie"
    return "free kick"


# ---- direction ----

def formation(x: np.ndarray, role: np.ndarray, ball_x: float = 0.0) -> dict:
    """Which side of the ball each team stands on along the length (mean sign of x - ball_x; 0 with too few
    boxes: relative to the ball, not the halfway line, so a calibration a few metres out along the length
    doesn't matter), whether it's a clear kickoff formation (ours on one side, and theirs not clearly on the
    same one: their kit is the less reliable, at SAA mostly the crowd), and whether the teams are cleanly
    split (SPLIT), which a free kick near the middle doesn't do."""
    side = lambda m: float(np.mean(np.sign(x[m] - ball_x))) if m.sum() >= FORMATION_MIN_BOXES else 0.0  # noqa: E731
    ours, theirs = side(role == 0), side(role == 1)
    clear = abs(ours) >= FORMATION and not (ours * theirs > 0 and abs(theirs) >= FORMATION)
    split = abs(ours) >= SPLIT[0] and ours * theirs < 0 and abs(theirs) >= SPLIT[1]
    return {"ours": round(ours, 2), "theirs": round(theirs, 2), "clear": bool(clear), "split": bool(split)}


def direction(kickoffs: list[dict], override: str | None = None, goals: list[dict] | None = None) -> dict:
    """Which goal we attack in the first half ("left" = toward -x), from the clear kickoffs ({half, ours}):
    our side at a first-half kickoff votes for it, at a second-half one against; and from rated goal clips
    ({half, attack}: +1 / -1 the end we must attack for it, goal_clip_votes), each a vote for our side being
    the other one. `sure` when the votes are SURE or more and the halves don't contradict each other; an
    override (game.attack_1h) wins."""
    goals = goals or []
    v = [sum(k["ours"] for k in kickoffs if k["clear"] and k["half"] == h)
         - sum(g["attack"] for g in goals if g["half"] == h) for h in (1, 2)]
    score = v[0] - v[1]  # > 0: we stand in the +x half in the first half, so attack -x (left)
    found = None if score == 0 else "left" if score > 0 else "right"
    sure = abs(score) >= SURE and not (v[0] * v[1] > 0 and min(abs(v[0]), abs(v[1])) >= FORMATION)
    out = {"found": found, "votes": [round(v[0], 2), round(v[1], 2)], "sure": bool(sure), "goal_clips": len(goals)}
    out["attack_1h"] = override or found
    out["set"] = override is not None
    return out


def goal_clip_votes(game: Game, bframe: np.ndarray, bxy: np.ndarray, kickoff_2h: int) -> list[dict]:
    """The goal clips you've rated: 😊 is our goal, 🙁 theirs (checked on the seven games: 25 of 25 agree with
    the kickoffs where those are sure). The end the ball got nearest in the clip, flipped for theirs, is the
    goal we attack in that half: [{half, attack: +1 / -1}]."""
    out = []
    for c in game.clips:
        if "goal" not in c.tags or c.rating is None:
            continue
        a, b = np.searchsorted(bframe, [game.frame_at(c.start_s), game.frame_at(c.end_s)])
        x = bxy[a:b, 0]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        end = float(np.sign(x[np.argmax(np.abs(x))]))
        out.append({"half": 2 if game.frame_at(c.start_s) >= kickoff_2h else 1,
                    "attack": end if c.rating == "happy" else -end})
    return out


def attack_sign(info: dict, half: int) -> int:
    """+1 if we attack the +x (right-hand) goal in this half, -1 left, 0 unknown."""
    a = info["direction"]["attack_1h"]
    if a is None:
        return 0
    s = 1 if a == "right" else -1
    return s if half == 1 else -s


# ---- goalies ----

def keepers(track: np.ndarray, frame: np.ndarray, xy: np.ndarray, on: np.ndarray, role: np.ndarray, sizes: dict,
            halves: list[tuple[int, int, int]], fps: float, kit: np.ndarray | None = None,
            marked: dict[int, set[int]] | None = None, not_goalie: set[int] | None = None) -> np.ndarray:
    """Per row: 1 our goalie, 2 theirs, 0 neither. halves: (first frame, end frame, +1 / -1 the end we defend).
    marked: {code: tracklet ids you've marked as that team's goalie}; with kit (the kit cluster per row), the
    kit each marked tracklet is mostly in finds the rest of that goalie (see the module docstring). Each mark
    adds its own: a goalie's kit can fall in different clusters in different light. not_goalie: tracklet ids
    you've marked as not a goalie, never one."""
    out = np.zeros(len(track), np.int8)
    marked = {c: ids for c, ids in (marked or {}).items() if ids}
    mrows = {c: np.isin(track, list(ids)) for c, ids in marked.items()}
    kits: dict[int, list[int]] = {}
    if kit is not None:
        for c, ids in marked.items():
            for tid in ids:
                k = kit[(track == tid) & (kit >= 0)]
                if len(k):
                    kits.setdefault(c, []).append(int(np.bincount(k).argmax()))
    L, pd, pw = sizes["length"] / 2, sizes["pa_depth"], sizes["pa_width"] / 2
    for a, b, ours_end in halves:
        lo, hi = np.searchsorted(frame, [a, b])
        r = np.arange(lo, hi)
        r = r[on[r]]
        if not_goalie:
            r = r[~np.isin(track[r], list(not_goalie))]
        for e in (-1, 1):
            attacker = 1 if e == ours_end else 0  # at our end they attack
            depth = L - e * xy[r, 0]
            inpa = (depth < pd + 1) & (depth > -1) & (np.abs(xy[r, 1]) < pw + 1)
            ids, inv = np.unique(track[r], return_inverse=True)
            if not len(ids):
                continue
            n = np.bincount(inv)
            nin = np.bincount(inv, inpa)
            nother = np.bincount(inv, role[r] == 2)
            natt = np.bincount(inv, role[r] == attacker)
            order = np.lexsort((depth, inv))
            med = depth[order][np.searchsorted(inv[order], np.arange(len(ids))) + n // 2]
            long_ = n >= KEEPER_S * fps
            own_kit = long_ & (nin >= KEEPER_IN * n) & (nother >= 0.5 * n)
            deep = long_ & (nin >= KEEPER_DEEP_IN * n) & (med <= KEEPER_DEEP_M) & (natt < 0.5 * n)
            code = 1 if e == ours_end else 2
            if code in kits:  # in the marked goalie's kit, around their goal
                nkit = np.bincount(inv, np.isin(kit[r], kits[code]), minlength=len(ids))
                own_kit |= (n >= KEEPER_KIT_S * fps) & (nkit >= 0.5 * n) & (nin >= KEEPER_KIT_IN * n)
            c = r[(own_kit | deep)[inv] & inpa]
            if not len(c):
                continue
            cd = L - e * xy[c, 0]
            c = c[np.lexsort((cd, frame[c]))]  # by frame, deepest first
            first = np.r_[True, frame[c][1:] != frame[c][:-1]]
            out[c[first]] = code
    for code, m in mrows.items():  # marked: that team's goalie wherever they are, and nobody else in those frames
        out[np.isin(frame, frame[m]) & (out == code)] = 0
        out[m] = code
    return out


NEAR_GOAL_M = 20.0  # the ball this close to a goal: its goalie should be in view


def keeper_seen(keeper: np.ndarray, frame: np.ndarray, bframe: np.ndarray, bxy: np.ndarray, length: float,
                hs: list[tuple[int, int]], ours_end_1h: int, fps: float) -> list[dict]:
    """Per half and team: of the frames with the ball within NEAR_GOAL_M of their goal, the share with their
    goalie found (None if the ball was never near), and how many minutes that is."""
    out = []
    for h, (a, b) in enumerate(hs[:2], 1):
        ours_end = ours_end_1h if h == 1 else -ours_end_1h
        m = (bframe >= a) & (bframe < b) & np.isfinite(bxy[:, 0])
        for team, code, e in (("ours", 1, ours_end), ("theirs", 2, -ours_end)):
            near = bframe[m][np.hypot(bxy[m, 0] - e * length / 2, bxy[m, 1]) < NEAR_GOAL_M]
            seen = np.isin(near, frame[keeper == code]).mean() if len(near) else None
            out.append({"half": h, "team": team, "seen": None if seen is None else round(float(seen), 3),
                        "near_min": round(len(near) / fps / 60, 1)})
    return out


# ---- goalies you've marked ----

def goalies_path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "goalies.json"


def load_goalies(store: Store, game: Game) -> list[dict]:
    p = goalies_path(store, game)
    return json.loads(p.read_text())["marks"] if p.exists() else []


def _save_goalies(store: Store, game: Game, marks: list[dict]) -> None:
    from soccer_clips.store import atomic_write_text

    atomic_write_text(goalies_path(store, game), json.dumps({"version": 1, "marks": sorted(marks, key=lambda m: m["frame"])},
                                                            indent=1) + "\n")


def add_goalie(store: Store, game: Game, frame: int, x: float, y: float, team: str, track: int | None = None) -> dict:
    """Mark the player at (x, y) (fractions of the frame) in `frame` as `team`'s goalie ("ours" / "theirs"),
    or as not a goalie ("none")."""
    import secrets

    if team not in (*TEAM_CODES, NOT_GOALIE):
        raise ValueError(f"team must be one of {', '.join((*TEAM_CODES, NOT_GOALIE))}")
    m = {"id": secrets.token_hex(4), "frame": int(frame), "x": round(float(x), 5), "y": round(float(y), 5),
         "team": team, "track": track}
    _save_goalies(store, game, load_goalies(store, game) + [m])
    return m


def remove_goalie(store: Store, game: Game, mark_id: str) -> bool:
    marks = load_goalies(store, game)
    keep = [m for m in marks if m["id"] != mark_id]
    if len(keep) == len(marks):
        return False
    _save_goalies(store, game, keep)
    return True


def goalie_marks(store: Store, game: Game, t) -> list[dict]:
    """The marks with the tracklet each is on now ({id, frame, team, track}; track None if no box is there)."""
    return [{"id": m["id"], "frame": m["frame"], "team": m["team"],
             "track": t.at(m["frame"], m["x"], m["y"], prefer=m.get("track"))} for m in load_goalies(store, game)]


# ---- out of play ----

def _into_net(last_out: int, lo: int, outside: np.ndarray, placed: np.ndarray, bxy: np.ndarray, length: float) -> int:
    """A kickoff's crossing: from the last one (an index), back to the first of a run over the same goal line
    with the ball no further than NET_M into the field in between. A crossing is over a goal line if any of it
    is (the ball fetched out of the net and kicked away can read as over a touchline as it goes)."""
    goal = np.abs(np.nan_to_num(bxy[:, 0])) > length / 2 + LINE_PAD_M

    def crossing(i: int) -> tuple[int | None, float]:
        # the crossing ending at i: the last sample in the field before it (None if none), and the goal line's end
        ins = np.flatnonzero(~outside[lo:i] & placed[lo:i])
        a = lo + ins[-1] if len(ins) else None
        g = np.flatnonzero(goal[(lo if a is None else a + 1):i + 1])
        return a, float(np.sign(bxy[(lo if a is None else a + 1) + g[0], 0])) if len(g) else 0.0

    k, end = crossing(last_out)
    i = last_out
    while end and k is not None:
        o = np.flatnonzero(outside[lo:k] & placed[lo:k])
        if not len(o):
            break
        p = lo + o[-1]  # the crossing before
        between = bxy[p + 1:k + 1][placed[p + 1:k + 1]]
        k2, e2 = crossing(p)
        if e2 != end or (np.sign(between[:, 0]) != end).any() or (np.abs(between[:, 0]) < length / 2 - NET_M).any():
            break
        i, k = p, k2
    return i


def goal_in(o: dict | None, bframe: np.ndarray, bxy: np.ndarray, length: float, fps: float) -> bool:
    """Whether the ball was in a goal within GOAL_IN_S of the start of a kickoff's out-of-play stretch (o: {from,
    to}, or None: none): over a goal line for GOAL_IN_MIN_S. A ball in the air (a punt) reads metres off on the field
    and can seem to cross for a moment."""
    if o is None:
        return False
    a, b = np.searchsorted(bframe, [o["from"], min(o["from"] + GOAL_IN_S * fps, o["to"])])
    return int((np.abs(np.nan_to_num(bxy[a:b, 0])) > length / 2 + LINE_PAD_M).sum()) >= GOAL_IN_MIN_S * fps


def out_of_play(restarts: list[dict], bframe: np.ndarray, bxy: np.ndarray, length: float, width: float,
                fps: float, half_starts: list[int]) -> list[dict]:
    """[{from, to, kind}]: the ball out of play before each restart (not the goalie's ball, which is in play):
    from when it last went over a line (for a restart on a line, or a kickoff after a goal; looking back no
    further than the half's start; for a kickoff, back over earlier crossings of the same goal line with the ball
    only near it in between, NET_M: fetched out of the net, it goes over again), else from when it stopped, to
    the kick. A stoppage with no crossing
    (none for a free kick; none within OUT_NEAR_S of the ball stopping, on a line) shorter than STOP_S isn't
    one: the ball was in play."""
    outside = (np.abs(bxy[:, 0]) > length / 2 + LINE_PAD_M) | (np.abs(bxy[:, 1]) > width / 2 + LINE_PAD_M)
    placed = np.isfinite(bxy[:, 0])
    out = []
    for r in restarts:
        if r["kind"] == "goalie":
            continue
        start = r["still"]
        back = {"throw-in": LOOKBACK_S, "corner": LOOKBACK_S, "goal kick": LOOKBACK_S, "kickoff": GOAL_LOOKBACK_S}.get(r["kind"])
        if back:  # not back past the start of the half
            since = max([h for h in half_starts if h <= r["still"]], default=r["still"] - back * fps)
            lo, hi = np.searchsorted(bframe, [max(r["still"] - back * fps, since), r["still"]])
            o = np.flatnonzero(outside[lo:hi] & placed[lo:hi])
            last_out = lo + o[-1] if len(o) else None
            near = OUT_NEAR_S.get(r["kind"])
            if last_out is not None and near and r["still"] - bframe[last_out] > near * fps:
                last_out = None  # the ball came back in long before: not this restart's crossing
            if last_out is not None and r["kind"] == "kickoff":
                last_out = _into_net(last_out, lo, outside, placed, bxy, length)
            if last_out is not None:
                inside = np.flatnonzero(~outside[lo:last_out] & placed[lo:last_out])
                start = int(bframe[lo + inside[-1] + 1]) if len(inside) else int(bframe[lo + o[0]])
        if start == r["still"] and r["kick"] - r["still"] < STOP_S * fps:
            continue  # no crossing, and too short a stop for a free kick: in play
        out.append({"from": int(start), "to": r["kick"], "kind": r["kind"]})
    return out


# ---- making it ----

def _sizes(cal: dict) -> dict:
    from soccer_clips.track import pitch

    return {k: float(v) for k, v in zip(pitch.SIZE_NAMES, cal["sizes"])}


def halves(game: Game) -> list[tuple[int, int]]:
    """Each half's [first frame, end frame) from the game times (the ones set)."""
    out = []
    for a, b in ((game.kickoff_1h_s, game.end_1h_s), (game.kickoff_2h_s, game.end_2h_s)):
        if a is not None and b is not None:
            out.append((game.frame_at(a), game.frame_at(b)))
    return out


def key(store: Store, game: Game) -> str | None:
    from soccer_clips.track import people, positions

    k = positions.key(store, game)
    if k is None:
        return None
    pp, gp = people._picks_path(store, game), goalies_path(store, game)
    goals = sorted((c.start_s, c.end_s, c.rating) for c in game.clips if "goal" in c.tags and c.rating)
    return (f"m{VERSION} {k} {pp.stat().st_mtime_ns if pp.exists() else '-'} "
            f"{gp.stat().st_mtime_ns if gp.exists() else '-'} {game.attack_1h} "
            f"{game.kickoff_1h_s} {game.end_1h_s} {game.kickoff_2h_s} {game.end_2h_s} {goals}")


def build(store: Store, game: Game, log=None) -> dict:
    """Work out and save the game's restarts, direction, goalies and out-of-play stretches; returns the info.
    ValueError when the game isn't placed on the field or its halves aren't set."""
    import time

    from soccer_clips.track import ball_game, people, pitch, positions

    t0 = time.time()
    pos = positions.get(store, game)
    hs = halves(game)
    if pos is None:
        raise ValueError("needs the players placed on the field (soccer-clips pitch place)")
    if len(hs) < 2:
        raise ValueError("needs both halves' times (tagger H / J)")
    k = key(store, game)
    cal = pitch.load_cal(store, game)
    sizes = _sizes(cal)
    t = people.tracks(store, game)
    role = people.roles_for(store, game)[1]
    fps = game.fps
    on = ~pos["off"] & np.isfinite(pos["xy"][:, 0])

    bf = pos["ball_frame"]
    bxy = np.where(pos["ball_off"][:, None], np.nan, pos["ball_xy"].astype(np.float64))
    in_game = lambda f: any(a <= f < b for a, b in hs)  # noqa: E731
    restarts = []
    for r in dead_balls(bf, bxy, fps):
        if not in_game(r["kick"]):
            continue
        r["kind"] = restart_kind(r["x"], r["y"], sizes)
        r["half"] = 1 if r["kick"] < hs[1][0] else 2
        if abs(r["x"]) < KICKOFF_NEAR_M and abs(r["y"]) < KICKOFF_BOX[1]:
            lo, hi = t.window(r["still"], r["kick"])
            m = on[lo:hi]
            fm = formation(pos["xy"][lo:hi, 0][m], role[lo:hi][m], r["x"])
            if r["kind"] == "kickoff" or fm["split"]:
                r.update(fm, kind="kickoff")
            if r.pop("hidden", False) and not (abs(fm["ours"]) >= HIDDEN_SPLIT[0] and fm["ours"] * fm["theirs"] < 0
                                               and abs(fm["theirs"]) >= HIDDEN_SPLIT[1]):
                continue  # kicked while unseen: only a kickoff with the teams lined up either side
        restarts.append(r)
    for r in restarts:  # a second kickoff soon after one is the same kickoff (retaken, or stopped again)
        if r["kind"] == "kickoff" and any(q["kind"] == "kickoff" and 0 < r["kick"] - q["kick"] < KICKOFF_SAME_S * fps
                                          for q in restarts):
            r["kind"] = "kickoff again"
    kickoffs = [r for r in restarts if r["kind"] == "kickoff"]
    d = direction(kickoffs, game.attack_1h, goal_clip_votes(game, bf, bxy, hs[1][0]))
    info = {"restarts": restarts, "direction": d}

    keeper = np.zeros(len(t.frame), np.int8)
    marks = goalie_marks(store, game, t)
    marked = {TEAM_CODES[tm]: {m["track"] for m in marks if m["team"] == tm and m["track"] is not None} for tm in TEAM_CODES}
    not_goalie = {m["track"] for m in marks if m["team"] == NOT_GOALIE and m["track"] is not None}
    info["goalie_marks"] = marks
    if d["attack_1h"]:
        ours_end = -attack_sign(info, 1)  # we defend the end opposite the one we attack
        keeper = keepers(t.track, t.frame, pos["xy"], on, role, sizes,
                         [(hs[0][0], hs[0][1], ours_end), (hs[1][0], hs[1][1], -ours_end)], fps,
                         kit=t.box_team, marked=marked, not_goalie=not_goalie)
    starts = [a for a, _ in hs]
    goals = [r for r in kickoffs if not any(abs(r["kick"] - h) < KICKOFF_SAME_S * fps for h in starts)]
    oop = out_of_play(restarts, bf, bxy, sizes["length"], sizes["width"], fps, starts)
    ends = {o["to"]: o for o in oop}
    for r in goals:  # after a goal: the teams lined up, or the ball in a goal; neither, a free kick by the centre spot
        if not r.get("clear") and not goal_in(ends.get(r["kick"]), bf, bxy, sizes["length"], fps):
            r["kind"] = "free kick"
    if any(r["kind"] != "kickoff" for r in goals):
        goals = [r for r in goals if r["kind"] == "kickoff"]
        oop = out_of_play(restarts, bf, bxy, sizes["length"], sizes["width"], fps, starts)
    info.update({
        "out_of_play": oop,
        "goals_by_kickoff": [r["kick"] for r in goals],
        "keepers": keeper_seen(keeper, t.frame, bf, bxy, sizes["length"], hs, -attack_sign(info, 1), fps)
        if d["attack_1h"] else [],
        "seconds": round(time.time() - t0, 1),
    })
    ball_game.save_npz(path(store, game), key=np.array(k), keeper=keeper, info=np.array(json.dumps(info)))
    _cache.pop(game.id, None)
    if log:
        from collections import Counter

        kinds = Counter(r["kind"] for r in restarts)
        log(f"{game.id}: {len(restarts)} restarts ({', '.join(f'{v} {k}' for k, v in kinds.most_common())}); "
            f"we attack {d['attack_1h'] or '?'} in the 1st half ({'set' if d['set'] else 'sure' if d['sure'] else 'not sure'}, "
            f"votes {d['votes']}); {len(goals)} goals by kickoffs; goalies seen "
            + ", ".join(f"{k['half']}H {k['team']} {k['seen']:.0%}" for k in info["keepers"] if k["seen"] is not None)
            + f" of the time the ball's near their goal; {info['seconds']} s")
    return info


_cache: dict[str, tuple[str, dict]] = {}


def load(store: Store, game: Game) -> dict | None:
    """The saved match info and goalie rows ({info..., keeper}) if up to date, else None."""
    k = key(store, game)
    if k is None:
        return None
    hit = _cache.get(game.id)
    if hit and hit[0] == k:
        return hit[1]
    p = path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        if str(z["key"]) != k:
            return None
        out = {**json.loads(str(z["info"])), "keeper": z["keeper"]}
    _cache.clear()
    _cache[game.id] = (k, out)
    return out


def get(store: Store, game: Game, log=None) -> dict | None:
    """The match info, worked out first if missing or out of date; None if it can't be yet."""
    got = load(store, game)
    if got is not None:
        return got
    try:
        build(store, game, log=log)
    except ValueError:
        return None
    return load(store, game)
