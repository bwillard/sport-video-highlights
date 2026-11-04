"""
Touches (Phase 6.3): who played the ball, and when.

A touch is a moment a player is at the ball and the ball's motion changes: a kick, a trap, a turn. From the
ball state (ball_state.py: the ball per frame in metres, in play or not) and the players on the field:
- **At the ball:** the nearest on-field player of anyone (goalies, and players in a kit read as neither team:
  leaving those out finds the toucher less often), and how far. A touch is a dip in that distance to within
  TOUCH_M, the lowest in TOUCH_GAP_S either way.
- **The ball's motion changes:** within KICK_WIN_S of the dip, the ball's velocity (over KICK_K frames each
  way, on the path smoothed over SMOOTH_HALF frames each way) changes by KICK_MS or more. A player running
  past the ball doesn't count.
- **Who:** that nearest player: tracklet, team (0 ours, 1 theirs, -1 neither kit), goalie. `margin`: how much
  further the next nearest player is; `ambiguous` when that's under AMBIG_M (a scramble). On your windows the
  toucher is the tracklet you pointed at 73% of the time with a margin over 2 m, 50% under 0.5 m, so
  `confidence` is the margin over SURE_M, up to 1.
- **Runs:** touches by the same tracklet with less than DRIBBLE_S between them are one run (a dribble).
- **Your labels correct them** (`correct`, 6.7): a labelled touch (any kind but a goal) within TOL_S of one found
  puts its player there and its kind on it (`label`; `fixed` when the player changed); one with none found near
  is added (`added`); a "none" label takes the one found there away. Passes, shots and the rest are built on
  the corrected touches; the yardsticks use what was found before (`found`).

Against your full windows (2026-11-04, three games, 265 labelled touches, matched one to one within TOL_S: your
labels come up to 3/4 s after the moment): 64% of labelled touches found (Elite Strikers 75%, FC-1 61%, EBU 60%),
48% of touches found are labelled (many of the rest are real: a dribble touch, a throw-in, a touch nobody
labelled), the tracklet you pointed at 65%, the team 83%. Missed ones are mostly upstream: the ball not tracked
then, or the player at it not placed on the field. Shorter gaps, other distances and kick sizes, and a score
mixing them (fitted on two games, tried on the third) were no better.

<game>/tracks/touches.json: {key, touches: [{frame, x, y (metres), track, team, keeper, margin, ambiguous,
confidence, run, and from labels: label, fixed, added}], found (the same before your labels), stats}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store, atomic_write_text

VERSION = 2  # 2: corrected by your labels
TOUCH_M = 2.0  # the nearest player within this of the ball...
TOUCH_GAP_S = 0.5  # ...at the lowest in this long either way
SMOOTH_HALF = 2  # frames each way the ball path is averaged over before its velocity is measured
KICK_K = 4  # frames each way the ball's velocity is measured over
KICK_MS = 4.0  # the velocity changing by this much (m/s)...
KICK_WIN_S = 0.4  # ...within this of the dip
AMBIG_M = 0.3  # the next nearest player this much further or less: ambiguous
SURE_M = 2.0  # a margin this big or more: confidence 1
DRIBBLE_S = 2.0  # the same tracklet's touches this close together: one run
TOL_S = 0.75  # measuring against labels: a touch this close to a labelled one is it


def path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "touches.json"


def key(store: Store, game: Game) -> str | None:
    from soccer_clips.track import ball_state, events

    k = ball_state.key(store, game)
    lp = events.labels_path(store, game)
    return None if k is None else f"t{VERSION} {k} {lp.stat().st_mtime_ns if lp.exists() else '-'}"


# ---- the pieces (pure, for testing) ----

def smooth(xy: np.ndarray, h: int) -> np.ndarray:
    """The path averaged over h frames each way; NaN where the frame has no ball or fewer than h around it do."""
    n = len(xy)
    ok = np.isfinite(xy[:, 0])
    c = np.r_[0, np.cumsum(ok)]
    s = np.vstack([[0.0, 0.0], np.cumsum(np.where(ok[:, None], xy, 0.0), 0)])
    i = np.arange(n)
    a, b = np.clip(i - h, 0, n), np.clip(i + h + 1, 0, n)
    cnt = c[b] - c[a]
    out = (s[b] - s[a]) / np.maximum(cnt, 1)[:, None]
    out[~ok | (cnt < h)] = np.nan
    return out


def velocity_change(xy: np.ndarray, k: int, fps: float) -> np.ndarray:
    """Per frame, m/s: how much the velocity over the k frames after differs from the k frames before."""
    n = len(xy)
    out = np.full(n, np.nan)
    if n > 2 * k:
        before = (xy[k:n - k] - xy[:n - 2 * k]) * fps / k
        after = (xy[2 * k:] - xy[k:n - k]) * fps / k
        out[k:n - k] = np.linalg.norm(after - before, axis=1)
    return out


def nearest_two(frame: np.ndarray, xy: np.ndarray, ball: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each frame of `ball` (n, 2): metres to the nearest player and its row (-1), and to the next nearest
    (inf). Rows: frame, xy (NaN: not placed); only the rows to count (on the field)."""
    n = len(ball)
    d1, d2 = np.full(n, np.inf), np.full(n, np.inf)
    row = np.full(n, -1, np.int64)
    idx = np.flatnonzero((frame >= 0) & (frame < n) & np.isfinite(xy[:, 0]))
    idx = idx[np.isfinite(ball[frame[idx], 0])]
    f = frame[idx]
    d = np.linalg.norm(xy[idx] - ball[f], axis=1)
    o = np.lexsort((d, f))
    f, d, idx = f[o], d[o], idx[o]
    first = np.r_[True, f[1:] != f[:-1]] if len(f) else np.zeros(0, bool)
    second = np.r_[False, first[:-1]] & ~first if len(f) else np.zeros(0, bool)
    d1[f[first]], row[f[first]] = d[first], idx[first]
    d2[f[second]] = d[second]
    return d1, row, d2


def dips(d: np.ndarray, within: float, gap: int) -> np.ndarray:
    """Frames where d is within `within` and the lowest for `gap` frames either way (the lowest first)."""
    s = np.where(np.isfinite(d), d, np.inf)
    idx = np.flatnonzero(s <= within)
    idx = idx[np.argsort(s[idx], kind="stable")]
    taken = np.zeros(len(s), bool)
    out = []
    for i in idx:
        if not taken[max(0, i - gap):i + gap + 1].any():
            out.append(i)
            taken[i] = True
    return np.sort(np.array(out, np.int64))


def find(ball: np.ndarray, in_play: np.ndarray, d1: np.ndarray, fps: float) -> np.ndarray:
    """The frames of touches: dips of the nearest player's distance (d1) in play, with the ball's velocity
    changing near them."""
    dv = velocity_change(smooth(ball, SMOOTH_HALF), KICK_K, fps)
    c = dips(np.where(in_play, d1, np.inf), TOUCH_M, round(TOUCH_GAP_S * fps))
    k = round(KICK_WIN_S * fps)
    dv = np.where(np.isfinite(dv), dv, 0.0)
    return np.array([i for i in c if dv[max(0, i - k):i + k + 1].max() >= KICK_MS], np.int64)


def runs(frames: np.ndarray, tracks: np.ndarray, fps: float) -> np.ndarray:
    """Run number per touch (in frame order): the same tracklet again within DRIBBLE_S is the same run."""
    out = np.zeros(len(frames), np.int64)
    for i in range(1, len(frames)):
        same = tracks[i] == tracks[i - 1] and frames[i] - frames[i - 1] <= DRIBBLE_S * fps
        out[i] = out[i - 1] + (not same)
    return out


def describe(frames: np.ndarray, ball: np.ndarray, d1: np.ndarray, row: np.ndarray, d2: np.ndarray,
             track: np.ndarray, team: np.ndarray, keeper: np.ndarray, fps: float) -> list[dict]:
    """The touches at `frames`, with who (rows: track, team, keeper per row of the game's tracklets)."""
    out = []
    r = row[frames]
    ru = runs(frames, track[r], fps)
    for f, i, run in zip(frames, r, ru):
        margin = float(d2[f] - d1[f]) if np.isfinite(d2[f]) else SURE_M
        out.append({"frame": int(f), "x": round(float(ball[f, 0]), 2), "y": round(float(ball[f, 1]), 2),
                    "track": int(track[i]), "team": int(team[i]), "keeper": int(keeper[i]),
                    "margin": round(margin, 2), "ambiguous": margin < AMBIG_M,
                    "confidence": round(min(1.0, margin / SURE_M), 2), "run": int(run)})
    return out


def label_team(e: dict) -> int:
    """A resolved label's team as a touch's: 0 ours, 1 theirs, -1 not known (goalies with their team, a labelled
    player ours)."""
    if e.get("keeper") in (1, 2):
        return e["keeper"] - 1
    if e.get("team") in (0, 1):
        return e["team"]
    return 0 if e.get("player") is not None else -1


def correct(found: list[dict], labs: list[dict], nots: list[dict], ball: np.ndarray, fps: float) -> list[dict]:
    """The touches with your labels put in (resolved labels: labs every kind, nots the "none" ones): see the
    module's notes. In frame order, runs worked out again."""
    tol = round(TOL_S * fps)
    gone = {i for i, _ in pair([t["frame"] for t in found], [e["frame"] for e in nots], tol)}
    out = [dict(t) for i, t in enumerate(found) if i not in gone]
    lt = [e for e in labs if e["kind"] != "goal"]
    hit = set()
    for i, j in pair([t["frame"] for t in out], [e["frame"] for e in lt], tol):
        e, t = lt[j], out[i]
        hit.add(j)
        t["label"] = e["kind"]
        if e["track"] is not None:
            team = label_team(e)
            t["fixed"] = e["track"] != t["track"]
            t.update(track=e["track"], keeper=e["keeper"], team=team if team >= 0 or t["fixed"] else t["team"])
    for j, e in enumerate(lt):
        if j in hit or e["track"] is None:
            continue  # a label on nobody: nothing to add
        f = e["frame"]
        at = ball[f] if f < len(ball) and np.isfinite(ball[f, 0]) else e["at"]
        if at is None:
            continue
        out.append({"frame": f, "x": round(float(at[0]), 2), "y": round(float(at[1]), 2), "track": e["track"],
                    "team": label_team(e), "keeper": e["keeper"], "margin": None, "ambiguous": False,
                    "confidence": 1.0, "run": 0, "label": e["kind"], "added": True})
    out.sort(key=lambda t: t["frame"])
    fr = np.array([t["frame"] for t in out], np.int64)
    for t, r in zip(out, runs(fr, np.array([t["track"] for t in out]), fps)):
        t["run"] = int(r)
    return out


# ---- making it ----

def build(store: Store, game: Game, log=None) -> dict:
    """Find and save the game's touches; returns what's saved. ValueError when the ball state can't be
    worked out yet (ball_state.build)."""
    import time

    from soccer_clips.track import ball_state, events, match, people, positions

    t0 = time.time()
    st = ball_state.get(store, game)
    if st is None:
        ball_state.build(store, game)  # raises why
        st = ball_state.load(store, game)
    k = key(store, game)
    pos = positions.get(store, game)
    t = people.tracks(store, game)
    role = people.roles_for(store, game)[1]
    keeper = match.get(store, game)["keeper"]
    fps = game.fps
    ball = st["xy"].astype(np.float64)
    on = ~pos["off"] & np.isfinite(pos["xy"][:, 0])
    rows = np.flatnonzero(on)
    d1, r, d2 = nearest_two(t.frame[rows], pos["xy"][rows].astype(np.float64), ball)
    row = np.where(r >= 0, rows[np.maximum(r, 0)], -1)
    team = ball_state._row_teams(role, keeper)  # goalies with their team
    frames = find(ball, st["in_play"], d1, fps)
    found = describe(frames, ball, d1, row, d2, t.track, team, keeper, fps)
    lab = events.resolved(store, game, guesses=False)
    got = correct(found, lab["events"], lab["nots"], ball, fps)
    play_min = float(st["in_play"].sum()) / fps / 60
    stats = {"touches": len(got), "per_min": round(len(got) / play_min, 1) if play_min else None,
             "ours": sum(g["team"] == 0 for g in got), "theirs": sum(g["team"] == 1 for g in got),
             "neither": sum(g["team"] == -1 for g in got), "ambiguous": sum(g["ambiguous"] for g in got),
             "runs": len({g["run"] for g in got}), "found": len(found),
             "from_labels": {"fixed": sum(bool(g.get("fixed")) for g in got),
                             "added": sum(bool(g.get("added")) for g in got),
                             "taken_away": len(found) - sum(not g.get("added") for g in got)},
             "seconds": round(time.time() - t0, 1)}
    out = {"key": k, "touches": got, "found": found, "stats": stats}
    atomic_write_text(path(store, game), json.dumps(out) + "\n")
    _cache.pop(game.id, None)
    if log:
        log(f"{game.id}: {stats['touches']} touches ({stats['per_min']} a minute of play; ours {stats['ours']}, "
            f"theirs {stats['theirs']}, neither kit {stats['neither']}; {stats['ambiguous']} ambiguous), "
            f"{stats['runs']} runs; {stats['seconds']} s")
    return out


_cache: dict[str, tuple[str, dict]] = {}


def load(store: Store, game: Game) -> dict | None:
    """The saved touches if up to date, else None."""
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
    """The touches, found first if missing or out of date; None if they can't be yet."""
    got = load(store, game)
    if got is not None:
        return got
    try:
        return build(store, game, log=log)
    except ValueError:
        return None


# ---- measured against the labels ----

def pair(found: list[int], labelled: list[int], tol: int) -> list[tuple[int, int]]:
    """One-to-one (found index, labelled index) pairs within tol frames, the closest first."""
    cand = sorted((abs(a - b), i, j) for i, a in enumerate(found) for j, b in enumerate(labelled) if abs(a - b) <= tol)
    used_f, used_l, out = set(), set(), []
    for _, i, j in cand:
        if i not in used_f and j not in used_l:
            used_f.add(i)
            used_l.add(j)
            out.append((i, j))
    return out


def compare(touches: list[dict], evs: list[dict], windows: list[dict], fps: float) -> dict:
    """Touches found against labelled ones (resolved labels; every kind but a goal is a touch) inside full
    windows: the share of labelled ones found (found), of found ones labelled (labelled), and of pairs, the same
    tracklet you pointed at (player) and the same team (team, where both are known); found per kind."""
    from soccer_clips.track.ball_state import label_team

    inside = lambda f: any(w["start"] <= f < w["end"] for w in windows)  # noqa: E731
    tf = [t for t in touches if inside(t["frame"])]
    lf = [e for e in evs if e["kind"] != "goal" and inside(e["frame"])]
    pairs = pair([t["frame"] for t in tf], [e["frame"] for e in lf], round(TOL_S * fps))
    same = [tf[i]["track"] == lf[j]["track"] for i, j in pairs if lf[j]["track"] is not None]
    teams = [({0: 1, 1: 2}[tf[i]["team"]], label_team(lf[j])) for i, j in pairs if tf[i]["team"] in (0, 1)]
    teams = [a == b for a, b in teams if b]
    hit = {j for _, j in pairs}
    kinds = {}
    for j, e in enumerate(lf):
        kinds.setdefault(e["kind"], [0, 0])
        kinds[e["kind"]][0] += j in hit
        kinds[e["kind"]][1] += 1
    share = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"labelled_n": len(lf), "found_n": len(tf), "found": share(len(pairs), len(lf)),
            "labelled": share(len(pairs), len(tf)), "player": share(sum(same), len(same)),
            "team": share(sum(teams), len(teams)), "kinds": {k: share(a, b) for k, (a, b) in kinds.items()}}


def evaluate(store: Store, game: Game) -> dict | None:
    """Touches against the game's labelled full windows (compare); None if there are none or no touches."""
    from soccer_clips.track import events

    lab = events.resolved(store, game)
    if not lab["windows"]:
        return None
    got = get(store, game)
    if got is None:
        return None
    return compare(got["found"], lab["events"], lab["windows"], game.fps)  # as found, before your labels
