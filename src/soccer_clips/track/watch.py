"""
The player view: one player's game, played straight from the game video with everything else skipped.

Nothing is rendered: the page gets a list of runs (stretches of the game video) and plays them back to back,
loading the next run in a second, hidden video while the current one plays, so each jump is a swap.

- `stretches`: where the player is on screen, in game frames: their labelled tracklets (people.segments),
  plus, if asked, the stretches the appearance model is sure are them (reid.auto_all, at the tagger's
  threshold).
- `runs`: those padded, merged across short gaps (the tracking briefly losing them, or the camera
  glancing away), and short leftovers dropped.

A later rule (e.g. only while the ball is near them) narrows `stretches` before `runs`: `intersect` it
with frame ranges from the ball track and the player's boxes.
"""

from soccer_clips.models import Game
from soccer_clips.store import Store

PAD_S = 1.0  # lead-in and lead-out around each stretch
GAP_S = 3.0  # stretches closer than this (after padding) play as one run
MIN_S = 2.0  # runs shorter than this are dropped (a glimpse)


def merge(spans: list[tuple[int, int]], gap: int = 0) -> list[list[int]]:
    """[start, end) spans joined where they overlap or are at most `gap` apart, sorted."""
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intersect(xs: list[list[int]], ys: list[list[int]]) -> list[list[int]]:
    """Where two sorted lists of disjoint [start, end) spans overlap."""
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if a < b:
            out.append([a, b])
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def stretches(store: Store, game: Game, player: int, guesses: bool = True,
              min_score: float | None = None) -> tuple[list[list[int]], list[list[int]]]:
    """Frames where `player` is on screen: (labelled, guessed), each merged; guessed leaves out the labelled."""
    from soccer_clips.track import people, reid

    labelled = merge([(s["start"], s["end"]) for s in _segments(store, game) if s["player"] == player])
    guessed: list[list[int]] = []
    if guesses:
        auto = reid.auto_all(store, game, reid.AUTO_MIN if min_score is None else min_score)
        spans = merge([(c["start"], c["end"]) for c in auto.get(player, [])])
        guessed = [g for s in spans for g in _minus(s, labelled)]  # what the guesses add
    return labelled, guessed


_segs: dict = {}


def _segments(store: Store, game: Game) -> list[dict]:
    """people.segments of the game, kept per game until it's re-tracked or its picks change (people.tracks
    keeps one game in memory, and loading one takes about a second)."""
    from soccer_clips.track import people

    files = [people.path(store, game, people.DEFAULT_MODEL), people._picks_path(store, game)]
    key = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
    if _segs.get(game.id, (None,))[0] != key:
        t = people.tracks(store, game)
        _segs[game.id] = (key, people.segments(t, people.load_picks(store, game)) if t else [])
    return _segs[game.id][1]


def _minus(span: list[int], holes: list[list[int]]) -> list[list[int]]:
    out, a = [], span[0]
    for h0, h1 in holes:
        if h1 <= a or h0 >= span[1]:
            continue
        if h0 > a:
            out.append([a, h0])
        a = max(a, h1)
    if a < span[1]:
        out.append([a, span[1]])
    return out


def runs(store: Store, game: Game, player: int, guesses: bool = True, min_score: float | None = None,
         pad_s: float = PAD_S, gap_s: float = GAP_S, min_s: float = MIN_S) -> list[dict]:
    """The runs to play: [{start_s, end_s, labelled}] in the game video's seconds; labelled = the share of the
    run's on-screen time that comes from labels (the rest is guesses)."""
    labelled, guessed = stretches(store, game, player, guesses, min_score)
    fps, last = game.fps, game.frame_at(game.duration_s)
    pad = round(pad_s * fps)
    out = []
    for a, b in merge([(max(0, a - pad), min(last, b + pad)) for a, b in labelled + guessed], round(gap_s * fps)):
        if b - a < min_s * fps:
            continue
        lab = sum(y - x for x, y in intersect([[a, b]], labelled))
        seen = lab + sum(y - x for x, y in intersect([[a, b]], guessed))
        out.append({"start_s": round(a / fps, 3), "end_s": round(b / fps, 3),
                    "labelled": round(lab / seen, 2) if seen else 0.0})
    return out


def playlist(store: Store, games: list[Game], player: int, guesses: bool = True,
             min_score: float | None = None) -> dict:
    """Runs across games, in game order: {runs: [{game, start_s, end_s, labelled}], games: {id: {...}}}."""
    out, info = [], {}
    for g in games:  # one at a time: people.tracks holds one game, so threads would keep reloading each other's
        rs = runs(store, g, player, guesses, min_score)
        if not rs:
            continue
        out += [{"game": g.id, **r} for r in rs]
        info[g.id] = {"description": g.description, "played_on": g.played_on.isoformat() if g.played_on else None,
                      "duration_s": g.duration_s, "fps": g.fps,
                      "kickoff_1h_s": g.kickoff_1h_s, "kickoff_2h_s": g.kickoff_2h_s}
    return {"runs": out, "games": info, "total_s": round(sum(r["end_s"] - r["start_s"] for r in out), 1)}
