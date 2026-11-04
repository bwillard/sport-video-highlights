"""
Game events (Phase 6). For now, the labels they'll be measured against (6.1).

Labels are made in the tagger (Shift+E, then a key per event): a moment, a kind, and where you pointed (the
player who did it). Like player labels they're kept as clicks, not track ids, and matched to the tracklet
under the click each time they're read, so they survive re-tracking. A pass, shot or save is a touch too;
"won" is the touch that won the ball back.

Full windows are stretches where every touch is labelled: they measure touches and possession completely
(found / wrong / missed). Elsewhere a label only says that this happened, not that nothing else did.

<game>/tracks/event_labels.json: {version, events: [{id, frame, kind, x, y (fractions of the frame; None:
not on a player), track (under the click when made)}], windows: [{id, start, end (frames)}]}.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store, atomic_write_text

KINDS = ("touch", "pass", "shot", "save", "goal", "won")
VERSION = 1


def labels_path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "event_labels.json"


def load_labels(store: Store, game: Game) -> dict:
    p = labels_path(store, game)
    if not p.exists():
        return {"version": VERSION, "events": [], "windows": []}
    return json.loads(p.read_text())


def _save(store: Store, game: Game, d: dict) -> None:
    d["events"].sort(key=lambda e: e["frame"])
    d["windows"].sort(key=lambda w: w["start"])
    atomic_write_text(labels_path(store, game), json.dumps({**d, "version": VERSION}, indent=1) + "\n")


def add_event(store: Store, game: Game, frame: int, kind: str, x: float | None = None, y: float | None = None,
              track: int | None = None) -> dict:
    """Label an event; returns it (with its id)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r} (one of {', '.join(KINDS)})")
    if (x is None) != (y is None):
        raise ValueError("give both x and y, or neither")
    ev = {"id": secrets.token_hex(4), "frame": int(frame), "kind": kind,
          "x": None if x is None else round(float(x), 5), "y": None if y is None else round(float(y), 5),
          "track": track}
    d = load_labels(store, game)
    d["events"].append(ev)
    _save(store, game, d)
    return ev


def remove_event(store: Store, game: Game, event_id: str) -> bool:
    d = load_labels(store, game)
    n = len(d["events"])
    d["events"] = [e for e in d["events"] if e["id"] != event_id]
    if len(d["events"]) == n:
        return False
    _save(store, game, d)
    return True


def add_window(store: Store, game: Game, start: int, end: int) -> dict:
    """A full window: every touch from frame `start` to `end` is (or will be) labelled. Overlapping ones merge."""
    start, end = sorted((int(start), int(end)))
    if end - start < 1:
        raise ValueError("a window needs some length")
    d = load_labels(store, game)
    keep, merged = [], {"start": start, "end": end}
    for w in d["windows"]:
        if w["end"] < merged["start"] or w["start"] > merged["end"]:
            keep.append(w)
        else:
            merged = {"start": min(w["start"], merged["start"]), "end": max(w["end"], merged["end"])}
    win = {"id": secrets.token_hex(4), **merged}
    d["windows"] = keep + [win]
    _save(store, game, d)
    return win


def remove_window(store: Store, game: Game, window_id: str) -> bool:
    d = load_labels(store, game)
    n = len(d["windows"])
    d["windows"] = [w for w in d["windows"] if w["id"] != window_id]
    if len(d["windows"]) == n:
        return False
    _save(store, game, d)
    return True


def namer(store: Store, game: Game, t=None):
    """A function (track, frame) -> (player labelled there, else None; player guessed by appearance or place,
    else None), as things stand."""
    from soccer_clips.track import people, reid

    t = t if t is not None else people.tracks(store, game)
    if t is None:
        return lambda track, f: (None, None)
    by: dict[str, dict[int, list]] = {"is": {}, "guess": {}}
    for s in people.segments(t, people.load_picks(store, game)):
        by["is"].setdefault(s["track"], []).append(s)
    for p, cs in reid.auto_all(store, game).items():
        for c in cs:
            by["guess"].setdefault(c["track"], []).append({**c, "player": p})

    def named(kind: str, track: int, f: int) -> int | None:
        return next((s["player"] for s in by[kind].get(track, ()) if s["start"] <= f < s["end"]), None)

    def name(track: int, f: int) -> tuple[int | None, int | None]:
        p = named("is", track, f)
        return p, None if p is not None else named("guess", track, f)

    return name


def resolved(store: Store, game: Game) -> dict:
    """The labels with who each event was, as things stand: {events: [{..., track, team (0 ours, 1 theirs,
    2 other, None), player (labelled), guess (by appearance or place), keeper (1 our goalie, 2 theirs, 0),
    at (pitch metres or None)}], windows, kinds, counts}."""
    from soccer_clips.track import match, people, positions

    d = load_labels(store, game)
    t = people.tracks(store, game)
    got = people.roles_for(store, game) if t is not None else None
    role = got[1] if got else None
    pos = positions.load(store, game) if t is not None else None
    mi = match.load(store, game) if t is not None else None
    name = namer(store, game, t)
    out = []
    for e in d["events"]:
        r = {**e, "team": None, "player": None, "guess": None, "keeper": 0, "at": None}
        tr = t.at(e["frame"], e["x"], e["y"], prefer=e["track"]) if t is not None and e["x"] is not None else None
        r["track"] = tr
        if tr is not None:
            a, b = t.window(e["frame"], e["frame"] + 1)
            i = a + int(np.flatnonzero(t.track[a:b] == tr)[0])
            r["team"] = int(role[i]) if role is not None and role[i] >= 0 else None
            r["player"], r["guess"] = name(tr, e["frame"])
            if mi is not None and len(mi["keeper"]) == len(t.frame):
                r["keeper"] = int(mi["keeper"][i])
            if pos is not None and len(pos["xy"]) == len(t.frame) and np.isfinite(pos["xy"][i, 0]):
                r["at"] = [round(float(v), 2) for v in pos["xy"][i]]
        out.append(r)
    counts = {k: sum(e["kind"] == k for e in d["events"]) for k in KINDS}
    return {"events": out, "windows": d["windows"], "kinds": list(KINDS), "counts": counts}
