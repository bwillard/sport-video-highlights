"""
The games page: where each game is in the pipeline, and how much of our team the tracking names.

- `stages`: which steps have been run on a game (detection, kit, tracking, crops, signatures, ball, field
  calibration, game time, labels), each ok / partial / stale (made before something it's built from was redone)
  / missing, with what to run to fix it. Only file dates and small arrays: fast enough for every game at once.
- `recognition`: of our team's time on the field (players on screen, within FIELD_PAD_M of the lines, in game
  time), how much is labelled, guessed by appearance, guessed by place on the field, or nobody yet; per player,
  the unknown time by how long its tracklets are (short scraps: the tracking; long ones: labels or the
  appearance model), and the longest unknown tracklets to go and label, with who each looks most like. Seconds
  a game (cached on disk in tracks/recognition.json until anything it reads changes).
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store, atomic_write_text

VERSION = 2  # bump when recognition's numbers change meaning, to redo every game's cache
STEP = 5  # every 5th frame (6 a second): plenty for totals, a fifth of the work
FIELD_PAD_M = 3.0  # further off the lines than this is the bench, warm-ups or parents, not the game
LENGTHS_S = (2, 10, 30)  # unknown time by tracklet length: under 2 s, 2-10 s, 10-30 s, 30 s and more
LONGEST = 15  # unknown tracklets listed
FULL = 0.98  # share of the video detected / tracked that counts as the whole game


def _mtime(p: Path) -> float | None:
    return p.stat().st_mtime if p.exists() else None


def _newer(a: Path, b: Path) -> bool:
    """a was made after b (both exist)."""
    ta, tb = _mtime(a), _mtime(b)
    return ta is not None and tb is not None and ta > tb + 1


def _stage(state: str, text: str, tip: str = "", fix: str | None = None) -> dict:
    return {"state": state, "text": text, "tip": tip, "fix": fix}


def _covered(ranges, total: int) -> float:
    return min(1.0, sum(int(b) - int(a) for a, b in ranges) / max(total, 1))


def game_spans(game: Game) -> list[tuple[int, int]] | None:
    """Game frames of the halves that have both a start and an end; None if neither has."""
    spans = [(game.frame_at(a), game.frame_at(b)) for a, b in
             [(game.kickoff_1h_s, game.end_1h_s), (game.kickoff_2h_s, game.end_2h_s)] if a is not None and b is not None]
    return spans or None


def stages(store: Store, game: Game) -> dict:
    from soccer_clips.track import ball_game, kit, people, pitch, reid, world

    d = ball_game.tracks_dir(store, game)
    total = game.frame_at(game.duration_s)
    out = {}

    times = [game.kickoff_1h_s, game.end_1h_s, game.kickoff_2h_s, game.end_2h_s]
    n = sum(v is not None for v in times)
    halves = [f"{(b - a) / 60:.0f}'" for a, b in [times[:2], times[2:]] if a is not None and b is not None]
    out["gametime"] = _stage("ok" if n == 4 else "partial" if n else "missing", " + ".join(halves) or f"{n}/4 set",
                             "When each half starts and ends: the scores below only count game time",
                             None if n == 4 else f"tagger H / J, or soccer-clips gametime suggest {game.id} --apply")

    raw = people.raw_path(store, game, people.DEFAULT_MODEL)
    if raw.exists():
        with np.load(raw) as z:
            cov = _covered(z["ranges"], total)
        out["detect"] = _stage("ok" if cov >= FULL else "partial", f"{cov:.0%}",
                               f"Person boxes ({people.DEFAULT_MODEL}) over {cov:.0%} of the video",
                               None if cov >= FULL else f"soccer-clips gpu run people-games --mode all --games {game.id}")
    else:
        out["detect"] = _stage("missing", "—", "No person boxes yet",
                               f"soccer-clips gpu run people-games --mode all --games {game.id}")

    kp = kit.kit_path(store, game, people.DEFAULT_MODEL)
    if not kp.exists():
        out["kit"] = _stage("missing", "—", "No kit colours: teams can't be told apart", f"soccer-clips people kit {game.id}")
    elif _newer(raw, kp):
        out["kit"] = _stage("stale", "older", "Detected again since the kit colours were worked out",
                            f"soccer-clips people kit {game.id}")
    else:
        out["kit"] = _stage("ok", "✓", "Kit colour of every box")

    tp = people.path(store, game, people.DEFAULT_MODEL)
    if not tp.exists():
        out["track"] = _stage("missing", "—", "No tracklets yet", f"soccer-clips people track {game.id}")
    else:
        with np.load(tp) as z:
            cov = _covered(z["ranges"], total)
        if _newer(raw, tp) or _newer(kp, tp):
            out["track"] = _stage("stale", "older", "Detected again or kit redone since tracking",
                                  f"soccer-clips people track {game.id}")
        else:
            out["track"] = _stage("ok" if cov >= FULL else "partial", f"{cov:.0%}", f"Tracklets over {cov:.0%} of the video")

    cp = reid.crops_path(store, game)
    if not cp.exists():
        out["crops"] = _stage("missing", "—", "No player crops", f"soccer-clips reid crops --games {game.id}")
    elif _newer(tp, cp):
        out["crops"] = _stage("stale", "older", "Tracked again since the crops were cut (tracklet ids changed)",
                              f"soccer-clips reid crops --games {game.id}")
    else:
        out["crops"] = _stage("ok", "✓", "Player crops for recognition")

    sp = reid.sigs_path(store, game)
    model = reid.model_path(store)
    fix = f"soccer-clips reid sigs --games {game.id} (or soccer-clips gpu run reid-sigs)"
    if not sp.exists():
        out["sigs"] = _stage("missing", "—", "No appearance signatures: no guesses by appearance", fix)
    else:
        with np.load(sp) as z:
            made_with = str(z["model"]) if "model" in z.files else "?"
        if _newer(cp, sp):
            out["sigs"] = _stage("stale", "older", "Crops cut again since the signatures were made", fix)
        elif made_with != model.name or _newer(model, sp):
            out["sigs"] = _stage("stale", "old model", f"Made with {made_with}; the model in use is {model.name}"
                                 + (" (retrained since)" if made_with == model.name else ""), fix)
        else:
            out["sigs"] = _stage("ok", "✓", f"Signatures made with {made_with}")

    br, bp = d / "ball_raw.npz", d / "ball.npz"
    fix = f"soccer-clips gpu run ball-games --mode all --games {game.id}"
    if not br.exists() or not bp.exists():
        out["ball"] = _stage("missing", "—", "No game-ball path", fix)
    else:
        with np.load(br) as z:
            cov = _covered(z["ranges"], total)
        with np.load(bp) as z:
            f = z["frame"]
        spans = game_spans(game) or [(0, total)]
        inside = sum(int(((f >= a) & (f < b)).sum()) for a, b in spans)
        found = inside / max(sum(b - a for a, b in spans), 1)
        tip = f"Ball searched over {cov:.0%} of the video; placed (seen or filled) in {found:.0%} of game time"
        if _newer(br, bp):
            out["ball"] = _stage("stale", "older", "Detected again since the path was made",
                                 f"soccer-clips ball track-game {game.id} --retrack")
        else:
            out["ball"] = _stage("ok" if cov >= FULL else "partial", f"{found:.0%}", tip, None if cov >= FULL else fix)

    points = pitch.load_points(store, game)
    if not world.path(store, game).exists() or not br.exists():
        out["field"] = _stage("missing", "—", "Frames not registered yet, so the field can't be calibrated",
                              f"soccer-clips ball fixed-objects {game.id}")
    elif not points:
        out["field"] = _stage("missing", "—", "Not calibrated: no map, no following by place",
                              "tagger: ✏️ calibrate (Q), click spots on the video then on the map")
    else:
        cal = pitch.load_cal(store, game)
        if cal is None or "P" not in cal:
            err = str(cal.get("error", "")) if cal is not None else ""
            out["field"] = _stage("stale", f"{len(points)} clicks", err or "Clicked but not solved yet",
                                  "open the game in the tagger to solve, or click more spots")
        else:
            px = np.asarray(cal["click_px"], float)
            bad = int((px > pitch.BAD_CLICK_PX).sum())
            cov = float(cal["coverage"])
            L, W = (float(v) for v in cal["sizes"][:2])
            tip = (f"{len(points)} clicks, {bad} off by more than {pitch.BAD_CLICK_PX:.0f} px; field {L:.0f} × {W:.0f} m; "
                   f"{cov:.0%} of the game placed on it")
            if "rejected" in cal:
                tip = f"The latest clicks' solve can't be a field ({cal['rejected']}), so the one before is kept. " + tip
            out["field"] = _stage("partial" if bad or cov < 0.9 or "rejected" in cal else "ok", f"{L:.0f}×{W:.0f} m", tip,
                                  "tagger: ✏️ calibrate (Q); fix or remove the clicks marked off"
                                  if bad or "rejected" in cal else
                                  "tagger: click spots in the views not placed yet" if cov < 0.9 else None)

    picks = people.load_picks(store, game)
    named = {p["player"] for p in picks if p["kind"] == "is"}
    n = sum(p["kind"] == "is" for p in picks)
    out["labels"] = _stage("ok" if n else "missing", f"{n} · {len(named)} players" if n else "—",
                           f"{n} labels (L clicks and roster answers) naming {len(named)} players",
                           None if n else "tagger: V to show players, L and click one")
    return out


# ---- how much of our team is named ----

_lock = threading.Lock()  # people.tracks keeps one game in memory: one game at a time


def _path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "recognition.json"


def _key(store: Store, game: Game) -> str:
    """Everything recognition reads: the game's tracklets, labels, signatures, field and registration, every
    game's labelled looks (the guesses use them), the halves and the default player (who decides our kit)."""
    from soccer_clips.track import ball_game, people, pitch, reid, world

    d = ball_game.tracks_dir(store, game)
    files = [people.path(store, game, people.DEFAULT_MODEL), people._picks_path(store, game), reid.sigs_path(store, game),
             pitch.cal_path(store, game), pitch.points_path(store, game), world.path(store, game), d / "ball_raw.npz",
             *(reid._looks_path(store, g) for g in store.list_games())]
    stamps = [f.stat().st_mtime_ns if f.exists() else 0 for f in files]
    return json.dumps([VERSION, STEP, FIELD_PAD_M, reid.AUTO_MIN, reid.AUTO_MARGIN, store.settings.default_player,
                       game_spans(game), stamps])


def cached(store: Store, game: Game) -> dict | None:
    """The last recognition worked out for the game (current: whether it's still up to date), or None."""
    p = _path(store, game)
    if not p.exists():
        return None
    try:
        got = json.loads(p.read_text())
    except ValueError:
        return None
    if got.get("version") != VERSION:
        return None
    current = got.pop("key", None) == _key(store, game)
    return {**got, "current": current}


def recognition(store: Store, game: Game) -> dict | None:
    """How much of our team's time on the field is named (see the module docstring), worked out again if
    anything it reads changed; None if the game isn't tracked."""
    from soccer_clips.track import people

    if not people.path(store, game, people.DEFAULT_MODEL).exists():
        return None
    with _lock:
        got = cached(store, game)
        if got and got["current"]:
            return got
        out = _work_out(store, game)
        key = _key(store, game)  # after: working it out can refresh the looks caches it reads (reid.player_sums)
        atomic_write_text(_path(store, game), json.dumps({**out, "key": key}) + "\n")
        return {**out, "current": True}


def _work_out(store: Store, game: Game) -> dict:
    from soccer_clips.track import people, pitch, reid

    t = people.tracks(store, game)
    fps = game.fps_num / game.fps_den
    roles = people.roles_for(store, game)[1]
    spans = game_spans(game)
    rows = np.flatnonzero(t.frame % STEP == 0)
    if spans:
        f = t.frame[rows]
        rows = rows[np.any([(f >= a) & (f < b) for a, b in spans], axis=0)]
    frames_seen = len(np.unique(t.frame[rows]))  # sampled frames with anyone tracked

    # who each sampled box is: 1 labelled, 2 guessed by appearance, 3 by place, 0 nobody (labels win)
    key = t.track[rows].astype(np.int64) << 32 | t.frame[rows].astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    how = np.zeros(len(rows), np.int8)
    who = np.full(len(rows), -1, np.int64)

    def mark(stretches, value):
        for k, a, b, p in stretches:
            lo, hi = np.searchsorted(ks, [k << 32 | a, k << 32 | b])
            how[order[lo:hi]], who[order[lo:hi]] = value, p

    auto = reid.auto_all(store, game)
    claims = [(c["track"], c["start"], c["end"], p, c.get("by")) for p, cs in auto.items() for c in cs]
    mark([c[:4] for c in claims if c[4] == "place"], 3)
    mark([c[:4] for c in claims if c[4] != "place"], 2)
    mark([(s["track"], s["start"], s["end"], s["player"]) for s in people.segments(t, people.load_picks(store, game))], 1)
    # our team: in our kit, or named (following by place goes through boxes whose kit was misread)
    ours = (roles[rows] == 0) | (how > 0)
    rows, how, who = rows[ours], how[ours], who[ours]

    # on the field: feet within FIELD_PAD_M of the lines (frames the field can't place count as on it)
    cal = pitch.load_cal(store, game)
    field = cal is not None and "P" in cal
    off = np.zeros(len(rows), bool)
    if field:
        b = t.box[rows]
        xy = pitch.ground(store, game, t.frame[rows], (b[:, 0] + b[:, 2]) / 2 * game.width, b[:, 3] * game.height)
        if xy is not None:
            L, W = (float(v) for v in cal["sizes"][:2])
            with np.errstate(invalid="ignore"):
                off = (np.abs(xy[:, 0]) > L / 2 + FIELD_PAD_M) | (np.abs(xy[:, 1]) > W / 2 + FIELD_PAD_M)
        else:
            field = False
    off_rows, rows, how, who = rows[off], rows[~off], how[~off], who[~off]

    secs = lambda n: round(float(n) * STEP / fps, 1)
    counts = np.bincount(how, minlength=4)
    players = []
    for p in sorted(set(who[who >= 0].tolist())):
        c = np.bincount(how[who == p], minlength=4)
        players.append({"player": int(p), "labelled": secs(c[1]), "appearance": secs(c[2]), "place": secs(c[3])})

    # the unknown, by how long the tracklet it's in is; and the tracklets with the most of it
    unknown = rows[how == 0]
    ids = np.array(sorted(t.first), np.int64)
    first = np.array([t.first[k] for k in ids.tolist()], np.int64)
    last = np.array([t.last[k] for k in ids.tolist()], np.int64)
    at = np.searchsorted(ids, t.track[unknown])
    length = (last[at] - first[at] + 1) / fps
    edges = [0, *LENGTHS_S, np.inf]
    by_length = [{"from": a, "to": None if b == np.inf else b, "s": secs(((length >= a) & (length < b)).sum())}
                 for a, b in zip(edges, edges[1:])]
    uk, first_row, n = np.unique(t.track[unknown], return_index=True, return_counts=True)
    top = np.argsort(-n, kind="stable")[:LONGEST]
    longest = []
    for i in top:
        k, f = int(uk[i]), int(t.frame[unknown[first_row[i]]])
        looks = reid.guess(store, game, k, f)["players"][:2]  # who it looks like, though not enough to guess
        longest.append({"track": k, "t": round(f / fps, 1), "s": secs(n[i]), "length": round(float(length[first_row[i]]), 1),
                        "looks": [{"player": x["player"], "score": x["score"]} for x in looks
                                  if not (x["busy"] or x["rejected"])]})

    return {"version": VERSION, "made_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "game_time": spans is not None, "field": field, "auto_min": reid.AUTO_MIN,
            "seconds": {"labelled": secs(counts[1]), "appearance": secs(counts[2]), "place": secs(counts[3]),
                        "unknown": secs(counts[0]), "off_field": secs(len(off_rows))},
            "per_frame": round(len(rows) / max(frames_seen, 1), 2),
            "players": players, "unknown_by_length": by_length, "longest": longest}


def overview(store: Store) -> list[dict]:
    """Every game with its stages and its last recognition (maybe out of date: see `current`), newest first."""
    games = sorted(store.list_games(), key=lambda g: (g.played_on is not None, g.played_on, g.id), reverse=True)
    out = []
    for g in games:
        spans = game_spans(g)
        out.append({"id": g.id, "played_on": g.played_on.isoformat() if g.played_on else None,
                    "description": g.description, "opponent": g.opponent, "duration_s": round(g.duration_s, 1),
                    "game_s": round(sum(b - a for a, b in spans) * g.fps_den / g.fps_num, 1) if spans else None,
                    "fps": g.fps_num / g.fps_den, "clips": len(g.clips), "stages": stages(store, g),
                    "recognition": cached(store, g)})
    return out
