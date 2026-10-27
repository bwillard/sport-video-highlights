"""
The roster page's pictures: every stretch of every tracklet with crops, in every game, each labelled as a
player or not (people.segments and the gaps between them), scored against every player's look (their
labelled crops in all games, reid.player_sums).

Views:
- "player": a player's labelled stretches, least like their other labels first.
- "probable": unlabelled stretches that look like the player (PROBABLE_MIN, PROBABLE_GAP), best first.
- "unassigned": unlabelled our-team stretches of at least MIN_CROPS crops.
- "unsure": labelled stretches that look at least as much like someone else (within UNSURE_GAP).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store
from soccer_clips.track import people, reid

MIN_CROPS = 3  # crops (~1 s apart) an unlabelled stretch needs to be shown: shorter ones are slivers
UNSURE_GAP = 0.02  # a label is unsure when someone else scores within this of (or above) its own player
PROBABLE_MIN = 0.9  # an unlabelled stretch is probably a player at this score (labelled ones: 90% >= 0.895)
PROBABLE_GAP = 0.05  # and this far ahead of the next player (labelled ones: next best ~0.83 median)
BUSY_SHARE = 0.2  # a player labelled elsewhere for this share of a stretch can't be it (one place at a time)

_tables: dict[str, tuple[tuple, dict]] = {}


def _files(store: Store, game: Game) -> list[Path]:
    return [reid.sigs_path(store, game), people.path(store, game, people.DEFAULT_MODEL), people._picks_path(store, game)]


def table(store: Store, game: Game) -> dict | None:
    """
    The game's stretches as arrays: track, start, end (exclusive), label (-1 none), n (crops), frame (a middle
    crop, else a box), sum (of the crops' signatures; zero if none), busy (per stretch, the players labelled on
    another tracklet for BUSY_SHARE of it). Unlabelled stretches only if ours (team split) with >= MIN_CROPS crops
    and not marked people.NOT_OURS. Kept in memory per game until the signatures, tracklets or picks change.
    None if the game hasn't been tracked and signed.
    """
    files = _files(store, game)
    if not files[0].exists() or not files[1].exists():
        return None
    key = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
    if _tables.get(game.id, (None,))[0] == key:
        return _tables[game.id][1]
    t = people.Tracks.load(files[1])  # not the shared cache: that's the game open in the tagger
    picks = people.load_picks(store, game)
    segs = people.segments(t, picks)
    _, roles = people.roles_of(t, picks, game, store.settings.default_player)
    ids, inv = np.unique(t.track, return_inverse=True)
    theirs = set(ids[np.bincount(inv, weights=roles >= 1, minlength=len(ids)) / np.bincount(inv, minlength=len(ids)) > 0.5].tolist())
    hidden = {(p["track"], p["frame"]) for p in picks if p["player"] == people.NOT_OURS and "track" in p}
    s = reid.load_sigs(store, game)
    order = np.lexsort((s["frame"], s["track"]))
    s_track, s_frame, s_sig = s["track"][order], s["frame"][order], s["sig"][order]
    by_track: dict[int, list[dict]] = {}
    for sg in segs:
        by_track.setdefault(sg["track"], []).append(sg)
    rows = []  # (track, start, end, label, crop indices)
    for k in set(s_track.tolist()) | set(by_track):
        a, b = np.searchsorted(s_track, [k, k + 1])
        fr = s_frame[a:b]
        parts, at = [], t.first[k]
        for sg in sorted(by_track.get(k, []), key=lambda x: x["start"]):
            if sg["start"] > at:
                parts.append((at, sg["start"], -1))
            parts.append((sg["start"], sg["end"], sg["player"]))
            at = sg["end"]
        if at <= t.last[k]:
            parts.append((at, t.last[k] + 1, -1))
        for lo, hi, label in parts:
            i, j = a + np.searchsorted(fr, lo), a + np.searchsorted(fr, hi)
            if label < 0 and (j - i < MIN_CROPS or k in theirs):
                continue
            rows.append((k, lo, hi, label, i, j))
    if hidden:  # a NOT_OURS pick sits at the first box of the stretch it hides
        hid = {}
        for k, f in hidden:
            hid.setdefault(k, []).append(f)
        rows = [r for r in rows if r[3] >= 0 or not any(r[1] <= f < r[2] for f in hid.get(r[0], []))]
    n = len(rows)
    out = {"track": np.array([r[0] for r in rows], np.int64), "start": np.array([r[1] for r in rows], np.int64),
           "end": np.array([r[2] for r in rows], np.int64), "label": np.array([r[3] for r in rows], np.int64),
           "n": np.array([r[5] - r[4] for r in rows], np.int64), "frame": np.zeros(n, np.int64),
           "sum": np.zeros((n, s_sig.shape[1] if len(s_sig) else 512), np.float32), "busy": [()] * n}
    for x, (k, lo, hi, label, i, j) in enumerate(rows):
        if j > i:
            out["sum"][x] = s_sig[i:j].sum(0)
            out["frame"][x] = s_frame[(i + j) // 2]
        else:  # labelled without crops: a frame the tracklet has a box in
            wa, wb = t.window(lo, hi)
            on = np.flatnonzero(t.track[wa:wb] == k)
            out["frame"][x] = t.frame[wa + on[len(on) // 2]] if len(on) else lo
    # one place at a time: who's labelled on another tracklet for much of each unlabelled stretch
    spans: dict[int, list[tuple[int, int, int]]] = {}
    for sg in segs:
        spans.setdefault(sg["player"], []).append((sg["start"], sg["end"], sg["track"]))
    for x in np.flatnonzero(out["label"] < 0):
        lo, hi, k = int(out["start"][x]), int(out["end"][x]), int(out["track"][x])
        out["busy"][x] = tuple(p for p, ss in spans.items()
                               if sum(max(0, min(hi, b) - max(lo, a)) for a, b, tr in ss if tr != k) > BUSY_SHARE * (hi - lo))
    _tables[game.id] = (key, out)
    return out


def _looks(store: Store) -> tuple[list[int], np.ndarray]:
    """Players with labelled crops anywhere, and the sum of those crops' signatures for each (P x 512)."""
    totals: dict[int, np.ndarray] = {}
    for g in store.list_games():
        for p, (total, _) in reid.player_sums(store, g).items():
            totals[p] = totals.get(p, 0) + total
    players = sorted(totals)
    return players, np.array([totals[p] for p in players], np.float32).reshape(-1, 512)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _scored(game: Game, tab: dict, players: list[int], sums: np.ndarray) -> list[dict]:
    """Every stretch of the game as a dict, with `score` (against its own label's look, leaving the stretch
    itself out) and `guesses` (the other players, best first; busy ones left out)."""
    if not len(tab["track"]):
        return []
    col = {p: i for i, p in enumerate(players)}
    has = tab["n"] > 0
    sim = _unit(tab["sum"]) @ _unit(sums).T if len(players) else np.zeros((len(tab["track"]), 0))
    fps = game.fps
    out = []
    for x in range(len(tab["track"])):
        label = int(tab["label"][x])
        own = None
        row = sim[x].copy()
        if has[x] and label in col:  # own look without this stretch
            rest = sums[col[label]] - tab["sum"][x]
            if np.linalg.norm(rest) > 1e-6:
                own = round(float(_unit(tab["sum"][x]) @ _unit(rest)), 3)
            row[col[label]] = -np.inf
        for p in tab["busy"][x]:
            if p in col:
                row[col[p]] = -np.inf
        guesses = []
        if has[x]:
            for i in np.argsort(-row)[:3]:
                if np.isfinite(row[i]):
                    guesses.append({"player": players[i], "score": round(float(row[i]), 3)})
        start, end = int(tab["start"][x]), int(tab["end"][x])
        out.append({"game": game.id, "track": int(tab["track"][x]), "start": start, "end": end,
                    "frame": int(tab["frame"][x]), "n": int(tab["n"][x]), "secs": round((end - start) / fps, 1),
                    "label": label if label >= 0 else None, "score": own, "guesses": guesses})
    return out


def _games(store: Store, game_id: str | None) -> list[Game]:
    games = [g for g in store.list_games() if game_id in (None, "", g.id)]
    return sorted(games, key=lambda g: (str(g.played_on or ""), g.id), reverse=True)


def _probable(it: dict) -> int | None:
    """The player an unlabelled stretch probably is (PROBABLE_MIN, PROBABLE_GAP), or None."""
    gs = it["guesses"]
    if it["label"] is not None or not gs or gs[0]["score"] < PROBABLE_MIN:
        return None
    return gs[0]["player"] if len(gs) < 2 or gs[0]["score"] - gs[1]["score"] >= PROBABLE_GAP else None


def _unsure(it: dict) -> bool:
    gs = it["guesses"]
    return it["label"] is not None and it["score"] is not None and bool(gs) and gs[0]["score"] > it["score"] - UNSURE_GAP


def _all_scored(store: Store, game_id: str | None) -> list[dict]:
    """Every game's (or one game's) stretches, scored; the games' tables built side by side the first time."""
    players, sums = _looks(store)
    games = _games(store, game_id)
    with ThreadPoolExecutor(4) as pool:
        tabs = list(pool.map(lambda g: table(store, g), games))
    return [it for g, tab in zip(games, tabs) if tab is not None for it in _scored(g, tab, players, sums)]


def pictures(store: Store, view: str, player: int | None = None, game_id: str | None = None, sort: str = "",
             offset: int = 0, limit: int = 60) -> dict:
    """A page of one view's stretches (see the module notes) across the games (or one): {items, total}."""
    keep = {
        "player": lambda it: it["label"] == player,
        "probable": lambda it: player is not None and _probable(it) == player,
        "unassigned": lambda it: it["label"] is None,
        "unsure": _unsure,
    }.get(view, lambda it: False)
    items = [it for it in _all_scored(store, game_id) if keep(it)]
    top = lambda it: it["guesses"][0]["score"] if it["guesses"] else -1.0
    order = {
        "time": None,  # games newest first, then by time in the game (as gathered)
        "least": lambda it: (it["score"] is None, it["score"] or 0),  # no crops: last
        "likely": lambda it: -top(it),
        "doubt": lambda it: top(it),
        "long": lambda it: -it["n"],
        "gap": lambda it: (it["score"] or 0) - top(it),
    }.get(sort or {"player": "least", "probable": "likely", "unassigned": "long", "unsure": "gap"}.get(view, "time"))
    if order is None:
        items.sort(key=lambda it: it["start"])
        rank = {g.id: i for i, g in enumerate(_games(store, None))}
        items.sort(key=lambda it: rank.get(it["game"], 0))
    else:
        items.sort(key=order)
    return {"items": items[offset:offset + limit], "total": len(items)}


def summary(store: Store) -> dict:
    """Per player (every label anywhere): stretches, seconds and games labelled, and unlabelled stretches whose
    best guess they are; plus counts of unassigned and unsure stretches."""
    out: dict[int, dict] = {}
    blank = lambda: {"stretches": 0, "seconds": 0.0, "games": set(), "probable": 0}
    unassigned = unsure = 0
    for it in _all_scored(store, None):
        if it["label"] is not None:
            st = out.setdefault(it["label"], blank())
            st["stretches"] += 1
            st["seconds"] += it["secs"]
            st["games"].add(it["game"])
            unsure += _unsure(it)
            continue
        unassigned += 1
        p = _probable(it)
        if p is not None:
            out.setdefault(p, blank())["probable"] += 1
    players_out = {str(p): {**v, "games": len(v["games"]), "seconds": round(v["seconds"])} for p, v in out.items()}
    return {"players": players_out, "unassigned": unassigned, "unsure": int(unsure)}


def assign(store: Store, stretches: list[dict], player: int | None, batch: str) -> int:
    """people.assign_stretches for stretches from any games ({game, track, start, end, label})."""
    n = 0
    by_game: dict[str, list[dict]] = {}
    for sg in stretches:
        by_game.setdefault(sg["game"], []).append(sg)
    for gid, sgs in by_game.items():
        n += people.assign_stretches(store, store.load_game(gid), sgs, player, batch)
    return n


def undo(store: Store, batch: str) -> int:
    """Take back an assignment in every game it touched."""
    n = 0
    for g in store.list_games():
        if any(p.get("batch") == batch for p in people.load_picks(store, g)):
            n += people.undo_answers(store, g, batch)
    return n
