"""
Ball state and control by team (Phase 6.2): who has the ball, frame by frame, and possession time.

Built from the positions on the field (positions.py) and the game's shape (match.py):
- **Ball per frame:** the ball path in metres; gaps up to BRIDGE_S filled in a straight line. Speed over
  SPEED_HALF frames each way. **Airborne** (rough): projected more than positions.BALL_OFF_M outside the lines
  (no place kept), or faster than AIR_MS: with one camera a ball in the air reads further away and faster.
- **In play:** inside a half and not out of play (match.py: from crossing a line to the restart kick).
- **Nearest of each team:** the closest on-field player of ours and theirs (goalies with their team, refs and
  others left out) and how far. Within reach (REACH_M, plus up to REACH_FAR_M more at the far touchline,
  where a pixel is a metre or more) a team **controls** the ball: ours, theirs, contested (both), loose
  (neither, or the ball faster than CONTROL_MS: going past, not kept), unknown (no ball, or in the air). A short stretch (BRIDGE_S) that isn't one team's, between
  two of the same team's, is theirs.
- **Possession:** the last team to control the ball alone has it until the other team does or it goes out
  of play. Loose and contested moments belong to the run in progress; after GAP_S without a ball, nobody
  is known to have it until someone controls it again.
- **Phase of play:** each frame in play is ours or theirs (possession), or nobody's: **contested** (both teams
  within CONTEST_M for CONTEST_S or more: a duel), **loose** (the ball seen with nobody within LOOSE_M, or in
  the air, for LOOSE_S or more: a long ball, a clearance, a ball rolling into space), or **not known** (no ball
  for longer than GAP_S). Possession still runs on through these (who has the ball next is for passes); the
  phase is what the time and the timeline show.
- **Time:** per phase and half, and per third (from our attacking direction: where the ball is, last seen).
  Frames nobody is known to have are counted as not known, not shared out. A per-second timeline of the phase
  and where the ball is along the field, for drawing.

`evaluate` checks possession against the labelled full windows (events.py): between labelled touches, the
team of the last one has the ball.

<game>/tracks/ball_state.npz, per frame from 0 to the end of the 2nd half: `xy` (metres, NaN unknown),
`speed` (m/s), `air`, `in_play`, `near` (n, 2: metres to our nearest, theirs), `near_track`, `control`
(CONTROL codes), `possession` (0 unknown, 1 ours, 2 theirs), `phase` (PHASES); `info` (JSON: runs, stats,
timeline) and `key`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

VERSION = 3
BRIDGE_S = 0.5  # gaps in the ball path filled in a straight line up to this; also control bridged over this
SPEED_HALF = 3  # frames each way the ball's speed is measured over
AIR_MS = 30.0  # faster than this (m/s, projected) the ball's in the air: a youth kick on the ground is slower
CONTROL_MS = 7.0  # a ball faster than this (m/s) isn't controlled, whoever is near: a pass going past a
# defender. Possession right 70 / 67 / 74% -> 83 / 75 / 76% on the labelled windows; 5-8 m/s about the same
REACH_M = 1.0  # a player this close to the ball (near touchline) controls it...
REACH_FAR_M = 1.5  # ...plus up to this much more at the far touchline
GAP_S = 6.0  # no ball for longer than this: who has it isn't known until someone controls it
LOOSE, OURS, THEIRS, CONTESTED, UNKNOWN = 0, 1, 2, 3, -1  # control codes
TEAMS = {1: "ours", 2: "theirs"}
OUT, CONTESTED_P, LOOSE_P, NOT_KNOWN = 0, 3, 4, 5  # phase codes, with 1 ours and 2 theirs
PHASES = {OUT: "out of play", 1: "ours", 2: "theirs", CONTESTED_P: "contested", LOOSE_P: "loose", NOT_KNOWN: "not known"}
CONTEST_M = 2.0  # both teams this close to the ball...
CONTEST_S = 0.5  # ...for this long: contested (3-6% of play)
LOOSE_M = 3.0  # the ball seen with nobody this close (or in the air)...
LOOSE_S = 1.0  # ...for this long: loose (7-11% of play; nobody within 1 m is half of it, a dribble included)
TIMELINE_S = 1.0  # the timeline's step
THIRDS = ("defending", "middle", "attacking")  # from our attacking direction
TOL_S = 0.5  # measuring against labels: the ball changing teams this close to a labelled touch counts
TRUTH_GAP_S = 5.0  # ...and this long after a labelled touch with none since, who has it isn't known: in open
# play touches are 1.5-2.5 s apart, so a longer gap is a stoppage (spare balls by the bench, players standing about)


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "ball_state.npz"


def key(store: Store, game: Game) -> str | None:
    from soccer_clips.track import match

    k = match.key(store, game)
    return None if k is None else f"b{VERSION} {k}"


# ---- the pieces (pure, for testing) ----

def ball_per_frame(bframe: np.ndarray, bxy: np.ndarray, boff: np.ndarray, n: int,
                   fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xy (n, 2) metres with NaN where unknown, speed (n,) m/s with NaN, air (n,) bool) for frames 0..n-1
    from the ball path (frames, metres, off = too far outside the lines: in the air)."""
    xy = np.full((n, 2), np.nan)
    air = np.zeros(n, bool)
    m = (bframe >= 0) & (bframe < n)
    bframe, bxy, boff = bframe[m], bxy[m], boff[m]
    air[bframe[boff]] = True
    ok = ~boff & np.isfinite(bxy[:, 0])
    xy[bframe[ok]] = bxy[ok]
    known = np.flatnonzero(np.isfinite(xy[:, 0]))
    if len(known) > 1:  # short gaps: a straight line
        gap = np.diff(known)
        for i in np.flatnonzero((gap > 1) & (gap <= BRIDGE_S * fps + 1)):
            a, b = known[i], known[i + 1]
            w = (np.arange(a + 1, b) - a) / (b - a)
            xy[a + 1:b] = xy[a] + w[:, None] * (xy[b] - xy[a])
    speed = np.full(n, np.nan)
    k = SPEED_HALF
    if n > 2 * k:
        d = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=1) / (2 * k / fps)
        speed[k:n - k] = d
    with np.errstate(invalid="ignore"):
        air |= speed > AIR_MS
    return xy, speed, air


def nearest(frame: np.ndarray, team: np.ndarray, track: np.ndarray, xy: np.ndarray,
            ball: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each frame of `ball` (n, 2): (metres to the nearest player of ours and of theirs (n, 2), inf where
    none; their tracklet ids (n, 2), -1). Rows: frame, team (0 ours, 1 theirs, else left out), track, xy."""
    n = len(ball)
    dist = np.full((n, 2), np.inf)
    who = np.full((n, 2), -1, np.int64)
    m = (team >= 0) & (team <= 1) & (frame < n) & np.isfinite(xy[:, 0])
    m[m] &= np.isfinite(ball[frame[m], 0])
    f, tm, tr = frame[m], team[m], track[m]
    d = np.linalg.norm(xy[m] - ball[f], axis=1)
    for c in (0, 1):
        s = tm == c
        o = np.lexsort((d[s], f[s]))  # by frame, nearest first
        fs, ds, ts = f[s][o], d[s][o], tr[s][o]
        first = np.r_[True, fs[1:] != fs[:-1]] if len(fs) else np.zeros(0, bool)
        dist[fs[first], c] = ds[first]
        who[fs[first], c] = ts[first]
    return dist, who


def reach(ball_y: np.ndarray, width: float) -> np.ndarray:
    """How close a player must be to control the ball, by how far across the field it is (+y: far side)."""
    return REACH_M + REACH_FAR_M * np.clip((ball_y + width / 2) / width, 0, 1)


def control(dist: np.ndarray, ball: np.ndarray, air: np.ndarray, width: float,
            speed: np.ndarray | None = None) -> np.ndarray:
    """Per frame: OURS / THEIRS / CONTESTED / LOOSE (nobody within reach, or the ball faster than CONTROL_MS),
    UNKNOWN where the ball isn't placed or is in the air."""
    out = np.full(len(ball), UNKNOWN, np.int8)
    ok = np.isfinite(ball[:, 0]) & ~air
    r = reach(ball[ok, 1], width)
    a, b = dist[ok, 0] <= r, dist[ok, 1] <= r
    out[ok] = np.select([a & b, a, b], [CONTESTED, OURS, THEIRS], LOOSE)
    if speed is not None:
        with np.errstate(invalid="ignore"):
            out[ok & (speed > CONTROL_MS)] = LOOSE
    return out


def bridge(ctl: np.ndarray, fps: float) -> np.ndarray:
    """Stretches up to BRIDGE_S that aren't one team's alone, between two of the same team's: that team's."""
    out = ctl.copy()
    sole = np.flatnonzero((ctl == OURS) | (ctl == THEIRS))
    if len(sole) < 2:
        return out
    gap = np.diff(sole)
    for i in np.flatnonzero((gap > 1) & (gap <= BRIDGE_S * fps + 1)):
        a, b = sole[i], sole[i + 1]
        if ctl[a] == ctl[b]:
            out[a + 1:b] = ctl[a]
    return out


def possession(ctl: np.ndarray, in_play: np.ndarray, known: np.ndarray, fps: float) -> np.ndarray:
    """Per frame: 1 ours, 2 theirs, 0 not known (out of play, after GAP_S with no ball, before anyone has it)."""
    out = np.zeros(len(ctl), np.int8)
    cur, last = 0, -(10 ** 9)
    gap = GAP_S * fps
    for f in range(len(ctl)):
        if not in_play[f]:
            cur = 0
            continue
        if known[f]:
            last = f
        elif f - last > gap:
            cur = 0
        if ctl[f] in (OURS, THEIRS):
            cur = int(ctl[f])
        out[f] = cur
    return out


def _lasting(m: np.ndarray, k: float) -> np.ndarray:
    """m, kept only in stretches at least k frames long."""
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    out = np.zeros(len(m), bool)
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= k:
            out[a:b] = True
    return out


def phase_of_play(poss: np.ndarray, in_play: np.ndarray, near: np.ndarray, ball: np.ndarray, air: np.ndarray,
                  fps: float) -> np.ndarray:
    """Per frame: OUT, 1 ours, 2 theirs (possession), CONTESTED_P, LOOSE_P, NOT_KNOWN (PHASES)."""
    out = np.where(in_play, np.where(poss > 0, poss, NOT_KNOWN), OUT).astype(np.int8)
    with np.errstate(invalid="ignore"):
        free = in_play & ((np.isfinite(ball[:, 0]) & ~air & (near.min(1) > LOOSE_M)) | air)
        duel = in_play & (near.max(1) <= CONTEST_M)
    out[_lasting(free, LOOSE_S * fps)] = LOOSE_P
    out[_lasting(duel, CONTEST_S * fps)] = CONTESTED_P
    return out


def runs(poss: np.ndarray) -> list[dict]:
    """[{team (1 / 2, or a phase code), from, to (frames, end exclusive)}]: unbroken stretches of one code, 0
    left out."""
    edges = np.flatnonzero(np.diff(np.r_[0, poss, 0]))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if poss[a]:
            out.append({"team": int(poss[a]), "from": int(a), "to": int(b)})
    return out


def thirds(x: np.ndarray, sign: int, length: float) -> np.ndarray:
    """0 defending, 1 middle, 2 attacking third for each ball x (our attacking direction `sign`); -1 unknown."""
    a = sign * x
    out = np.full(len(x), -1, np.int8)
    ok = np.isfinite(a)
    out[ok] = np.digitize(a[ok], [-length / 6, length / 6])
    return out


def time_stats(phase: np.ndarray, known: np.ndarray, xy: np.ndarray,
               hs: list[tuple[int, int]], signs: list[int], length: float, fps: float) -> dict:
    """Seconds per phase (phase_of_play) and our share of possession per half and for the game, overall and
    per third (the ball where it was last seen)."""
    last_x = xy[:, 0].copy()  # forward-filled: a run's frames without a ball count where it was last seen
    idx = np.where(np.isfinite(last_x), np.arange(len(last_x)), 0)
    np.maximum.accumulate(idx, out=idx)
    last_x = last_x[idx]

    def block(sl: np.ndarray, sign: int) -> dict:
        p = phase[sl]
        ip = p != OUT
        s = lambda m: round(float(m.sum()) / fps, 1)  # noqa: E731
        ours, theirs = s(p == 1), s(p == 2)
        b = {"in_play_s": s(ip), "out_s": s(~ip), "ball_seen": round(float(known[sl][ip].mean()), 3) if ip.any() else None,
             "ours_s": ours, "theirs_s": theirs, "contested_s": s(p == CONTESTED_P), "loose_s": s(p == LOOSE_P),
             "unknown_s": s(p == NOT_KNOWN), "ours_share": round(ours / (ours + theirs), 3) if ours + theirs else None}
        th = thirds(last_x[sl], sign, length)
        b["thirds"] = {}
        for i, name in enumerate(THIRDS):
            o, t = s((p == 1) & (th == i)), s((p == 2) & (th == i))
            b["thirds"][name] = {"ours_s": o, "theirs_s": t}
        return b

    out = {"halves": []}
    for (a, b), sign in zip(hs, signs):
        out["halves"].append(block(np.arange(a, b), sign))
    # the game: both halves added up (each third already from our attacking direction in its half)
    g = {k: round(sum(h[k] for h in out["halves"]), 1)
         for k in ("in_play_s", "out_s", "ours_s", "theirs_s", "contested_s", "loose_s", "unknown_s")}
    g["ours_share"] = round(g["ours_s"] / (g["ours_s"] + g["theirs_s"]), 3) if g["ours_s"] + g["theirs_s"] else None
    g["thirds"] = {n: {k: round(sum(h["thirds"][n][k] for h in out["halves"]), 1) for k in ("ours_s", "theirs_s")}
                   for n in THIRDS}
    out["game"] = g
    return out


def timeline(phase: np.ndarray, xy: np.ndarray, hs: list[tuple[int, int]], signs: list[int], fps: float) -> dict:
    """Per TIMELINE_S step of each half, for drawing: the most common phase, and where the ball is along the
    field (mean x of where it's seen, metres, + toward the goal we attack; None if not seen).
    {step_s, halves: [{from (frame), phase: [...], x: [...]}]}."""
    k = max(1, round(TIMELINE_S * fps))
    out = {"step_s": k / fps, "halves": []}
    for (a, b), sign in zip(hs, signs):
        if b <= a:
            continue
        i = np.arange(b - a) // k
        m = i[-1] + 1
        counts = np.bincount(i * 6 + phase[a:b], minlength=m * 6).reshape(m, 6)
        x = sign * xy[a:b, 0]
        ok = np.isfinite(x)
        n = np.bincount(i[ok], minlength=m)
        tot = np.bincount(i[ok], weights=x[ok], minlength=m)
        xs = [round(float(t / c), 1) if c else None for t, c in zip(tot, n)]
        out["halves"].append({"from": int(a), "phase": counts.argmax(1).tolist(), "x": xs})
    return out


def goal_moments(game: Game, xy: np.ndarray) -> list[dict]:
    """The goals, from the goal clips (😊 ours, 🙁 theirs, unrated: team None), in the halves: [{t (seconds), team}]
    by time, t where the ball gets nearest an end in the clip (the clip's last 2 s if it isn't seen)."""
    from soccer_clips.track import match

    hs = match.halves(game)
    out = []
    for c in game.clips:
        if "goal" not in c.tags:
            continue
        a, b = game.frame_at(c.start_s), min(game.frame_at(c.end_s), len(xy))
        if not any(h0 <= a < h1 for h0, h1 in hs):
            continue
        x = np.abs(xy[a:b, 0]) if b > a else np.zeros(0)
        t = (a + int(np.nanargmax(x))) / game.fps if np.isfinite(x).any() else max(c.start_s, c.end_s - 2.0)
        out.append({"t": round(t, 2), "team": {"happy": "ours", "sad": "theirs"}.get(c.rating)})
    return sorted(out, key=lambda g: g["t"])


# ---- making it ----

def _row_teams(role: np.ndarray, keeper: np.ndarray | None) -> np.ndarray:
    """Per tracklet row: 0 ours, 1 theirs, -1 neither; goalies with their team whatever their kit reads as."""
    team = np.where((role == 0) | (role == 1), role, -1)
    if keeper is not None and len(keeper) == len(role):
        team = np.where(keeper == 1, 0, np.where(keeper == 2, 1, team))
    return team


def build(store: Store, game: Game, log=None) -> dict:
    """Work out and save who had the ball, frame by frame; returns the info. ValueError when the game's
    shape (match.py) can't be worked out yet or which goal we attack isn't known."""
    import time

    from soccer_clips.track import ball_game, match, people, pitch, positions

    t0 = time.time()
    mi = match.get(store, game)
    if mi is None:
        raise ValueError("needs the game's shape first (soccer-clips pitch match: placed players, both halves' times)")
    if not mi["direction"]["attack_1h"]:
        raise ValueError("which goal we attack isn't known: set it in the tagger's Game time box")
    k = key(store, game)
    pos = positions.get(store, game)
    sizes = match._sizes(pitch.load_cal(store, game))
    t = people.tracks(store, game)
    role = people.roles_for(store, game)[1]
    fps = game.fps
    hs = match.halves(game)
    n = hs[-1][1]

    xy, speed, air = ball_per_frame(pos["ball_frame"].astype(np.int64), pos["ball_xy"].astype(np.float64),
                                    pos["ball_off"], n, fps)
    in_play = np.zeros(n, bool)
    for a, b in hs:
        in_play[a:b] = True
    for o in mi["out_of_play"]:
        in_play[o["from"]:o["to"]] = False
    on = ~pos["off"] & np.isfinite(pos["xy"][:, 0])
    team = np.where(on, _row_teams(role, mi["keeper"]), -1)
    near, near_track = nearest(t.frame, team, t.track, pos["xy"].astype(np.float64), xy)
    ctl = bridge(control(near, xy, air, sizes["width"], speed), fps)
    known = np.isfinite(xy[:, 0]) & ~air
    poss = possession(ctl, in_play, known, fps)
    signs = [match.attack_sign(mi, h) for h in (1, 2)]
    phase = phase_of_play(poss, in_play, near, xy, air, fps)
    stats = time_stats(phase, known, xy, hs, signs, sizes["length"], fps)
    ip = in_play & known
    stats["control"] = {name: round(float((ctl[ip] == c).mean()), 3) if ip.any() else None
                        for name, c in (("ours", OURS), ("theirs", THEIRS), ("contested", CONTESTED), ("loose", LOOSE))}
    rs = runs(np.where(phase == NOT_KNOWN, 0, phase))
    info = {"runs": rs, "stats": stats, "timeline": {**timeline(phase, xy, hs, signs, fps), "length": sizes["length"]},
            "seconds": round(time.time() - t0, 1)}
    ball_game.save_npz(path(store, game), key=np.array(k), xy=xy.astype(np.float32), speed=speed.astype(np.float32),
                       air=air, in_play=in_play, near=near.astype(np.float32), near_track=near_track.astype(np.int32),
                       control=ctl, possession=poss, phase=phase, info=np.array(json.dumps(info)))
    _cache.pop(game.id, None)
    if log:
        g = stats["game"]
        th = g["thirds"]
        ip = g["in_play_s"] or 1
        log(f"{game.id}: ours {_pct(g['ours_share'])} of possession (1H {_pct(stats['halves'][0]['ours_share'])}, "
            f"2H {_pct(stats['halves'][1]['ours_share'])}); of time in play: ours {_pct(g['ours_s'] / ip)}, theirs "
            f"{_pct(g['theirs_s'] / ip)}, contested {_pct(g['contested_s'] / ip)}, loose {_pct(g['loose_s'] / ip)}, "
            f"not known {_pct(g['unknown_s'] / ip)}; "
            f"thirds (ours / theirs, min): " + ", ".join(f"{nm} {th[nm]['ours_s'] / 60:.1f} / {th[nm]['theirs_s'] / 60:.1f}"
                                                        for nm in THIRDS)
            + f"; {len(rs)} runs; {info['seconds']} s")
    return info


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"



_cache: dict[str, tuple[str, dict]] = {}


def load(store: Store, game: Game) -> dict | None:
    """The saved ball state ({info..., per-frame arrays}) if up to date, else None."""
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
        out = {**json.loads(str(z["info"])), **{n: z[n] for n in z.files if n not in ("key", "info")}}
    _cache.clear()
    _cache[game.id] = (k, out)
    return out


def get(store: Store, game: Game, log=None) -> dict | None:
    """The ball state, worked out first if missing or out of date; None if it can't be yet."""
    got = load(store, game)
    if got is not None:
        return got
    try:
        build(store, game, log=log)
    except ValueError:
        return None
    return load(store, game)


# ---- measured against the labels ----

def label_team(e: dict) -> int:
    """A resolved event label's team as a possession code: 1 ours, 2 theirs (goalies with their team; a named
    or guessed player is ours), 0 not known."""
    if e.get("keeper") in (1, 2):
        return int(e["keeper"])
    if e.get("player") is not None or e.get("guess") is not None:
        return 1
    return {0: 1, 1: 2}.get(e.get("team"), 0)


def truth(evs: list[dict], windows: list[dict], n: int, fps: float) -> np.ndarray:
    """Per frame inside full windows: the team of the last labelled touch (1 / 2), for up to TRUTH_GAP_S; 0
    after a touch whose team isn't known or a goal (until the next touch), before the window's first touch,
    and outside windows. evs: resolved labels (every kind but a goal is a touch)."""
    out = np.zeros(n, np.int8)
    gap = round(TRUTH_GAP_S * fps)
    for w in windows:
        a, b = max(0, w["start"]), min(n, w["end"])
        cur, f0 = 0, a
        for e in sorted((e for e in evs if a <= e["frame"] < b), key=lambda e: e["frame"]):
            out[f0:min(e["frame"], f0 + gap)] = cur
            cur = 0 if e["kind"] == "goal" else label_team(e)
            f0 = e["frame"]
        out[f0:min(b, f0 + gap)] = cur
    return out


def compare(poss: np.ndarray, ctl: np.ndarray, in_play: np.ndarray, tru: np.ndarray, evs: list[dict],
            fps: float) -> dict:
    """How possession agrees with the labels: of in-play frames in windows with a labelled team (leaving out
    TOL_S either side of the labelled team changing: the ball on its way between teams), the share we have a
    team for (covered) and the share of those that's right; and at each labelled touch with a team, whether
    that team controls the ball within TOL_S (or anybody does, or the ball is seen). A kick is a fast ball, so
    a touch can be missed here and still change possession right."""
    tol = round(TOL_S * fps)
    near_change = np.zeros(len(tru), bool)
    for f in np.flatnonzero(np.diff(tru) != 0) + 1:
        if tru[f - 1] and tru[f] and tru[f - 1] != tru[f]:
            near_change[max(0, f - tol):f + tol] = True
    m = in_play[:len(tru)] & (tru > 0) & ~near_change
    p = poss[:len(tru)][m]
    have = p > 0
    out = {"frames": int(m.sum()), "covered": round(float(have.mean()), 3) if m.any() else None,
           "right": round(float((p[have] == tru[m][have]).mean()), 3) if have.any() else None,
           "touches": 0, "touch_team": None, "touch_anyone": None, "touch_ball": None}
    team_hit = anyone = seen = n = 0
    for e in evs:
        c = label_team(e)
        if not c or e["kind"] == "goal" or e["frame"] >= len(ctl):
            continue
        w = ctl[max(0, e["frame"] - tol):e["frame"] + tol + 1]
        n += 1
        team_hit += bool((w == c).any() or (w == CONTESTED).any())
        anyone += bool(np.isin(w, [OURS, THEIRS, CONTESTED]).any())
        seen += bool((w != UNKNOWN).any())
    if n:
        out.update(touches=n, touch_team=round(team_hit / n, 3), touch_anyone=round(anyone / n, 3),
                   touch_ball=round(seen / n, 3))
    return out


def evaluate(store: Store, game: Game) -> dict | None:
    """Possession against the game's labelled full windows (compare); None if there are none or no ball state."""
    from soccer_clips.track import events

    lab = events.resolved(store, game)
    if not lab["windows"]:
        return None
    st = get(store, game)
    if st is None:
        return None
    n = len(st["possession"])
    inside = [e for e in lab["events"] if any(w["start"] <= e["frame"] < w["end"] for w in lab["windows"])]
    got = compare(st["possession"], st["control"], st["in_play"], truth(inside, lab["windows"], n, game.fps), inside, game.fps)
    got["windows"] = len(lab["windows"])
    got["window_min"] = round(sum(w["end"] - w["start"] for w in lab["windows"]) / game.fps / 60, 1)
    return got
