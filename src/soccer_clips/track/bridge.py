"""
Following a player across holes in the tracking by where they are on the field.

A tracklet ends when its player is hidden for a moment (behind the referee, in a crowd) or their kit is
misread, and the appearance guesses can't carry a label over a second or two: there's no new look to go
by. On the field it's often plain who it is: one teammate's tracklet ends at a spot, another starts a few
metres away a moment later, and no other teammate was near. Then it's the same player.

Links (made once per game, from the calibrated field): the end of our-team tracklet k and the start of
our-team tracklet c are plausibly one player when c starts within R0_M + SPEED_MS x gap metres of where
k ended (cost = distance / that) and they could have stayed unseen in between: the straight way from one
to the other, every half second of the gap, is inside someone else's box (hidden behind them) or out of
the camera's view. Each half second of it in plain view with no box there, beyond the first FREE_S, adds
UNSEEN_COST: a player crossing open grass in view gets detected. A link costing under 1 is plausible.
Links are taken greedily, most plausible first, and only when clear: no other way on from k, and no other
way in to c, among the ends and starts not linked yet, is within RATIO of it, except ways that could be
the same player again later (k -> c, then c -> c2: c2 doesn't compete with c). Linking the plainest
first lets a player's scraps (short tracklets while she's half hidden) join up before the next stretch.
A tracklet whose kit was misread (other or unknown: in shade, or in the haze looking into the sun) can be
linked too, at KIT_COST more, but only after every link between our-kit tracklets is made as if it weren't
there, and only when nothing of any kit rivals it: FC-1's #28 came out from behind a defender after
scoring with every box read as another kit, and following stopped. Opponents, and tracklets you marked as
not on our team, never are. Across the games (2026-11-07) these links claim ~100 s, mostly scraps; the
longest four were all right (a red kit in shade, a close-up).

Labels (each time they change): from every stretch where a player is labelled or guessed that runs to
its tracklet's end, the links are followed forward, and from one that runs from its start, backward. Each
tracklet reached is guessed as that player, until one is labelled, turned down for them, or overlaps
them being somewhere else, or one that looks clearly more like another labelled teammate (by appearance,
when it's long enough to tell: a swap the field can't see). In a link's gap, a tracklet not linked to anything, between the two ends in
time and on the way in place, fills it: a scrap of ours, or one whose kit wasn't read as ours (a player
half hidden behind the yellow referee reads as "other").

Yardstick (evaluate): per player, hide every other labelled stretch and follow the links from the rest
alone; time claimed on a hidden stretch is right, on a stretch labelled as someone else wrong.
"""
from __future__ import annotations

import warnings

import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

R0_M = 2.0  # where a player's feet are on the field is this uncertain (far boxes, a frame or two of lag)
SPEED_MS = 3.0  # and they move about this fast (U11s jog 2-3 m/s; a sprint, ~5, still fits a short gap)
MAX_GAP_S = 5.0  # longer holes aren't bridged: anyone could have come and gone
OVERLAP_S = 0.5  # the next tracklet may start this much before the last ends (the tracker splitting one)
RATIO = 3.0  # a link is taken when no other is within this factor of it, either side
EDGE_BOXES = 5  # where a tracklet starts / ends: the median of its first / last few boxes
FILL_ROLES = (0, 2, -1)  # what fills a gap: ours not linked to anyone, other (a misread kit), unknown; not opponents
CLAIM_TOL = 30  # frames: a stretch ending this close to its tracklet's end runs to it (guesses go by ~1 s crops)
GRID = 15  # frames between the checks along a gap that the player could be hidden (~0.5 s)
FREE_S = 0.7  # this much of a gap in plain view goes free (a missed detection or two)
UNSEEN_COST = 0.25  # added per half second beyond it
HIDE_PAD = (0.5, 0.25)  # a point counts as behind a box within this share of its width / height around it
KIT_COST = 0.25  # added to a link to or from a tracklet whose kit wasn't read as ours (other, or unknown)
VETO = 0.05  # a tracklet that looks this much more like another labelled teammate (cosine) isn't followed into
VETO_CROPS = 3  # appearance crops a tracklet needs before its look counts (~3 s)

_ends_cache: dict = {}
_links_cache: dict = {}
_unseen_memo: dict = {}


def tracklet_ends(store: Store, game: Game) -> dict | None:
    """Every tracklet's start and end on the field: {track, first, last, start (n, 2), end (n, 2), n boxes}
    (metres, NaN where the view isn't placed). None if the game isn't tracked or calibrated. Cached (in
    tracks/bridge_ends.npz too: ~8 s a game) until either changes."""
    from soccer_clips.track import ball_game, people, pitch

    tp, cp = people.path(store, game, people.DEFAULT_MODEL), pitch.cal_path(store, game)
    if not tp.exists() or not cp.exists():
        return None
    key = f"{tp.stat().st_mtime_ns} {cp.stat().st_mtime_ns}"
    hit = _ends_cache.get(game.id)
    if hit and hit[0] == key:
        return hit[1]
    disk = ball_game.tracks_dir(store, game) / "bridge_ends.npz"
    if disk.exists():
        with np.load(disk) as z:
            if str(z["key"]) == key:
                out = {k: z[k] for k in ("track", "first", "last", "start", "end", "n")}
                _ends_cache[game.id] = (key, out)
                return out
    cal = pitch.load_cal(store, game)
    if cal is None or "P" not in cal:
        return None
    t = people.tracks(store, game)
    order = np.lexsort((t.frame, t.track))
    tr = t.track[order]
    starts = np.r_[0, np.flatnonzero(np.diff(tr)) + 1]
    stops = np.r_[starts[1:], len(tr)]
    n = stops - starts
    k = min(EDGE_BOXES, int(n.max()) if len(n) else 1)
    # the first and last k boxes of each tracklet (repeating the last one for short ones)
    off = np.arange(k)
    head = order[np.minimum(starts[:, None] + off, stops[:, None] - 1)]
    tail = order[np.maximum(stops[:, None] - 1 - off, starts[:, None])]
    rows = np.concatenate([head.ravel(), tail.ravel()])
    bx = t.box[rows]
    at = pitch.ground(store, game, t.frame[rows], (bx[:, 0] + bx[:, 2]) / 2 * game.width, bx[:, 3] * game.height)
    if at is None:
        return None
    at = at.reshape(2, len(starts), k, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all NaN: not placed
        start, end = np.nanmedian(at[0], axis=1), np.nanmedian(at[1], axis=1)
    out = {"track": tr[starts].astype(np.int64), "first": t.frame[order[starts]].astype(np.int64),
           "last": t.frame[order[stops - 1]].astype(np.int64), "start": start, "end": end, "n": n}
    ball_game.save_npz(disk, key=np.array(key), **out)
    _ends_cache[game.id] = (key, out)
    return out


def _roles(store: Store, game: Game, tracks: np.ndarray) -> np.ndarray:
    """Each tracklet's team, by most of its boxes (people.box_roles codes: 0 ours, 1 opponents, 2 other, -1)."""
    from soccer_clips.track import people

    t = people.tracks(store, game)
    box_role = people.roles_for(store, game)[1]
    ids, inv = np.unique(t.track, return_inverse=True)
    votes = np.zeros((len(ids), 4), np.int64)
    np.add.at(votes, (inv, box_role.astype(np.int64) + 1), 1)
    role = np.argmax(votes, axis=1) - 1
    return role[np.searchsorted(ids, tracks)]


def _costs(e_end: np.ndarray, e_last: np.ndarray, s_start: np.ndarray, s_first: np.ndarray, fps: float):
    """Plausible links from ends to starts: (end index, start index, cost) with cost < RATIO."""
    order = np.argsort(s_first, kind="stable")
    sf = s_first[order]
    lo = np.searchsorted(sf, e_last - OVERLAP_S * fps, side="left")
    hi = np.searchsorted(sf, e_last + MAX_GAP_S * fps, side="right")
    cnt = hi - lo
    ei = np.repeat(np.arange(len(e_last)), cnt)
    si = order[np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
    gap = np.maximum(s_first[si] - e_last[ei], 0) / fps
    with np.errstate(invalid="ignore"):
        cost = np.linalg.norm(s_start[si] - e_end[ei], axis=1) / (R0_M + SPEED_MS * gap)
    keep = np.isfinite(cost) & (cost < RATIO)
    return ei[keep], si[keep], cost[keep]


def _unseen(store: Store, game: Game, E: dict, ei: np.ndarray, si: np.ndarray, fps: float) -> np.ndarray:
    """Seconds of each link's gap (ends ei -> starts si) where the player would have been in plain view with
    nobody's box where they'd be (a straight walk between the two, checked every GRID frames)."""
    from soccer_clips.track import people, pitch

    t0, t1 = E["last"][ei], E["first"][si]
    g0 = t0 // GRID + 1
    n = np.maximum((t1 - 1) // GRID - g0 + 1, 0)  # grid frames strictly inside the gap
    out = np.zeros(len(ei))
    if not n.sum():
        return out
    row = np.repeat(np.arange(len(ei)), n)
    gf = (np.repeat(g0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))) * GRID
    alpha = ((gf - t0[row]) / np.maximum(t1[row] - t0[row], 1))[:, None]
    p = E["end"][ei][row] * (1 - alpha) + E["start"][si][row] * alpha
    uf, inv = np.unique(gf, return_inverse=True)
    G = pitch.frame_matrices(store, game, uf)
    q = pitch.project(G[inv], p) / [game.width, game.height]  # where they'd be in the frame (fractions)
    seen = np.isfinite(q).all(axis=1) & (q[:, 0] > 0) & (q[:, 0] < 1) & (q[:, 1] > 0) & (q[:, 1] < 1)
    t = people.tracks(store, game)
    hidden = np.zeros(len(gf), bool)
    rows = np.flatnonzero(seen)
    rows = rows[np.argsort(inv[rows], kind="stable")]  # grouped by grid frame
    cut = np.flatnonzero(np.diff(inv[rows])) + 1
    lo_hi = np.searchsorted(t.frame, np.stack([uf, uf + 1]).astype(t.frame.dtype))  # each grid frame's boxes
    for r in np.split(rows, cut) if len(rows) else []:
        j = inv[r[0]]
        a, b = lo_hi[0, j], lo_hi[1, j]
        if a == b:
            continue
        bx = t.box[a:b]
        w, h = bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]
        x, y = q[r, 0:1], q[r, 1:2]
        inside = ((x >= bx[:, 0] - HIDE_PAD[0] * w) & (x <= bx[:, 2] + HIDE_PAD[0] * w)
                  & (y >= bx[:, 1] - HIDE_PAD[1] * h) & (y <= bx[:, 3] + HIDE_PAD[1] * h))
        hidden[r] = inside.any(axis=1)
    np.add.at(out, row, (seen & ~hidden) * GRID / fps)
    return out


def links(store: Store, game: Game) -> dict | None:
    """The game's links between our-team tracklets (and misread kits): {next: {track: track}, prev: {track: track},
    cost: {(a, b): c}}, plus the ends table (tracklet_ends), each tracklet's team and whether it's marked not
    ours. None without a calibrated field."""
    from soccer_clips.track import people

    ends = tracklet_ends(store, game)
    if ends is None:
        return None
    pp = people._picks_path(store, game)  # the team split goes by the labels
    key = (id(ends), pp.stat().st_mtime_ns if pp.exists() else 0)
    hit = _links_cache.get(game.id)
    if hit and hit[0] == key:
        return hit[1]
    fps = game.fps_num / game.fps_den
    role = _roles(store, game, ends["track"])
    marked = np.isin(ends["track"], list(_not_ours(store, game)))
    nodes = np.flatnonzero(np.isin(role, FILL_ROLES) & ~marked)
    ei, si, cost = _costs(ends["end"][nodes], ends["last"][nodes], ends["start"][nodes], ends["first"][nodes], fps)
    ok = ei != si
    ei, si, cost = nodes[ei[ok]], nodes[si[ok]], cost[ok]
    # the unseen time doesn't change with the labels (only which tracklets are ours): kept per game
    memo = _unseen_memo.setdefault(game.id, {})
    if memo.get("ends") is not ends:
        memo.clear()
        memo["ends"] = ends
    pairs = ei * len(ends["track"]) + si
    known = np.array([q in memo for q in pairs.tolist()], bool)
    unseen = np.zeros(len(ei))
    unseen[known] = [memo[q] for q in pairs[known].tolist()]
    new = ~known
    if new.any():
        unseen[new] = _unseen(store, game, ends, ei[new], si[new], fps)
        memo.update(zip(pairs[new].tolist(), unseen[new].tolist()))
    cost = cost + UNSEEN_COST * np.maximum(unseen - FREE_S, 0) / (GRID / fps)
    misread = (role[ei] != 0) | (role[si] != 0)
    cost = cost + KIT_COST * misread
    keep = cost < RATIO
    ei, si, cost, misread = ei[keep], si[keep], cost[keep], misread[keep]
    used_end, used_start = set(), set()
    nxt, prv, costs = {}, {}, {}
    tr = ends["track"]
    # ours to ours first, as if nothing else were there; then through a misread kit, where nothing of any kit rivals it
    for among, take in ((~misread, ~misread), (np.ones(len(ei), bool), misread)):
        for e, s, c in _clear(ei[among], si[among], cost[among], take[among], used_end, used_start):
            a, b = int(tr[e]), int(tr[s])
            nxt[a], prv[b], costs[(a, b)] = b, a, round(float(c), 3)
    out = {"next": nxt, "prev": prv, "cost": costs, "ends": ends, "role": role, "marked": marked,
           "row": {int(k): i for i, k in enumerate(tr.tolist())}}
    _links_cache[game.id] = (key, out)
    return out


def _clear(ei: np.ndarray, si: np.ndarray, cost: np.ndarray, take: np.ndarray, used_end: set, used_start: set):
    """The links (end row, start row, cost) to make, most plausible first, from candidates ei -> si where `take`:
    under 1, and clear among these candidates (no other way on from the end, or in to the start, within RATIO,
    but ways that could be the same player again later). Marks the ends and starts it uses."""
    plausible = set(zip(ei[cost < 1].tolist(), si[cost < 1].tolist()))  # (a, b): b could be a again, later
    again = lambda a, b: (a, b) in plausible or (b, a) in plausible  # noqa: E731
    by_end: dict[int, list] = {}
    by_start: dict[int, list] = {}
    for e, s, c in zip(ei.tolist(), si.tolist(), cost.tolist()):
        by_end.setdefault(e, []).append((c, s))
        by_start.setdefault(s, []).append((c, e))
    for j in np.argsort(cost, kind="stable").tolist():
        c, e, s = cost[j], int(ei[j]), int(si[j])
        if c >= 1:
            break
        if not take[j] or e in used_end or s in used_start:
            continue
        rival = min([x for x, s2 in by_end[e] if s2 != s and s2 not in used_start and not again(s, s2)]
                    + [x for x, e2 in by_start[s] if e2 != e and e2 not in used_end and not again(e2, e)],
                    default=np.inf)
        if rival < RATIO * c:
            continue
        used_end.add(e)
        used_start.add(s)
        yield e, s, c


def _not_ours(store: Store, game: Game) -> set[int]:
    """Tracklets with a stretch marked as not anyone on our team: never followed onto."""
    from soccer_clips.track import people

    picks = people.load_picks(store, game)
    if not any(p["player"] == people.NOT_OURS for p in picks):
        return set()
    return {x["track"] for x in people.not_ours(people.tracks(store, game), picks)}


def _fillers(L: dict, a: int, b: int, fps: float, taken: set[int]) -> list[int]:
    """Tracklets not read as ours that fill the gap between a's end and b's start, on the way between them."""
    E, ia, ib = L["ends"], L["row"][a], L["row"][b]
    t0, t1 = E["last"][ia], E["first"][ib]
    if t1 - t0 <= 1:
        return []
    m = ((E["first"] >= t0 - OVERLAP_S * fps) & (E["last"] <= t1 + OVERLAP_S * fps) & np.isin(L["role"], FILL_ROLES)
         & ~L["marked"])
    out, busy = [], []
    for i in sorted(np.flatnonzero(m).tolist(), key=lambda i: -E["n"][i]):
        k = int(E["track"][i])
        if k in taken or k in (a, b) or k in L["next"] or k in L["prev"]:
            continue  # linked: it's someone's way on
        g_in = max(E["first"][i] - t0, 0) / fps
        g_out = max(t1 - E["last"][i], 0) / fps
        with np.errstate(invalid="ignore"):
            c_in = np.linalg.norm(E["start"][i] - E["end"][ia]) / (R0_M + SPEED_MS * g_in)
            c_out = np.linalg.norm(E["start"][ib] - E["end"][i]) / (R0_M + SPEED_MS * g_out)
        if not (c_in < 1 and c_out < 1):
            continue
        if any(E["first"][i] <= q1 and q0 <= E["last"][i] for q0, q1 in busy):
            continue  # two at once: one of them is someone else, and which isn't plain
        busy.append((E["first"][i], E["last"][i]))
        out.append(k)
    return out


_sigs_cache: dict = {}


def _track_looks(store: Store, game: Game) -> dict[int, np.ndarray]:
    """Each tracklet's unit mean appearance signature, for those with VETO_CROPS crops or more."""
    from soccer_clips.track import reid

    p = reid.sigs_path(store, game)
    if not p.exists():
        return {}
    key = (game.id, p.stat().st_mtime_ns)
    if _sigs_cache.get("key") != key:
        s = reid.load_sigs(store, game)
        ids, inv, n = np.unique(s["track"], return_inverse=True, return_counts=True)
        tot = np.zeros((len(ids), s["sig"].shape[1]), np.float32)
        np.add.at(tot, inv, s["sig"])
        ok = n >= VETO_CROPS
        tot = tot[ok] / np.maximum(np.linalg.norm(tot[ok], axis=1, keepdims=True), 1e-9)
        _sigs_cache.clear()
        _sigs_cache.update(key=key, looks=dict(zip(ids[ok].tolist(), tot)))
    return _sigs_cache["looks"]


def player_looks(store: Store, skip: str | None = None) -> dict[int, np.ndarray]:
    """Every labelled player's look (reid.player_look), from all games but `skip`."""
    from soccer_clips.track import reid

    players = sorted({p for g in store.list_games() for p in reid.player_sums(store, g)})
    out = {p: reid.player_look(store, p, skip)[0] for p in players}
    return {p: v for p, v in out.items() if v is not None}


def extend(store: Store, game: Game, segs: list[dict], guesses: dict[int, list[dict]],
           rejected: dict[int, set[int]] | None = None, looks: dict[int, np.ndarray] | None = None
           ) -> dict[int, list[dict]]:
    """
    New guesses from following the links: {player: [{track, start, end, frame, margin, by: "place"}]}. segs:
    people.segments (labelled, every player); guesses: the appearance guesses ({player: [{track, start, end}]});
    rejected: tracklets turned down per player; looks: each player's look (player_looks, the default). Nothing on
    a tracklet where anyone is labelled or guessed.
    """
    L = links(store, game)
    if L is None:
        return {}
    E, row = L["ends"], L["row"]
    fps = game.fps_num / game.fps_den
    rejected = rejected or {}
    looks = player_looks(store) if looks is None else looks
    tl = _track_looks(store, game)
    ids = list(looks)
    M = np.array([looks[p] for p in ids], np.float32) if ids else None
    first = lambda k: int(E["first"][row[k]])
    last = lambda k: int(E["last"][row[k]])
    known = {}  # track -> [(start, end, player)] labelled or guessed: not claimed again
    for x in segs:
        known.setdefault(x["track"], []).append((x["start"], x["end"], x["player"]))
    for p, cs in guesses.items():
        for c in cs:
            known.setdefault(c["track"], []).append((c["start"], c["end"], p))
    where = {}  # player -> [(start, end, track)] where they're labelled
    for x in segs:
        where.setdefault(x["player"], []).append((x["start"], x["end"], x["track"]))

    def free(k: int, a: int, b: int, player: int) -> bool:
        # k can be the player over [a, b): not labelled or guessed there, not turned down, and they aren't elsewhere
        if k in rejected.get(player, set()) or k not in row:
            return False
        if any(s < b and a < e for s, e, _ in known.get(k, [])):
            return False
        if any(s < b and a < e and k2 != k for s, e, k2 in where.get(player, [])):
            return False
        v = tl.get(k)
        if v is None or M is None or player not in looks:
            return True
        sim = M @ v  # looks more like another labelled teammate than them: a swap
        mine = sim[ids.index(player)]
        return not any(x - mine > VETO for p, x in zip(ids, sim.tolist()) if p != player)

    out: dict[int, dict[int, dict]] = {}

    def claim(player: int, k: int, cost: float | None, hops: int) -> None:
        a, b = first(k), last(k) + 1
        old = out.get(player, {}).get(k)
        if old and old["hops"] <= hops:
            return
        out.setdefault(player, {})[k] = {"track": k, "start": a, "end": b, "frame": (a + b) // 2, "by": "place",
                                         "margin": None if cost is None else round(1 - cost, 3), "hops": hops}

    starts = [(x["track"], x["start"], x["end"], x["player"]) for x in segs]
    starts += [(c["track"], c["start"], c["end"], p) for p, cs in guesses.items() for c in cs]
    for k0, a0, b0, player in starts:
        if k0 not in row:
            continue
        for step, edge, to_end in ((L["next"], lambda k: b0 >= last(k) + 1 - CLAIM_TOL, True),
                                   (L["prev"], lambda k: a0 <= first(k) + CLAIM_TOL, False)):
            k = k0
            if not edge(k):
                continue
            seen, hops = {k0}, 0
            while k in step and step[k] not in seen:
                n = step[k]
                hops += 1
                pair = (k, n) if to_end else (n, k)
                go = free(n, first(n), last(n) + 1, player)
                theirs = any(p == player for _, _, p in known.get(n, []))  # back to where they're known
                if go or theirs:
                    for f in _fillers(L, *pair, fps, set(known) | set(out.get(player, {}))):
                        if free(f, first(f), last(f) + 1, player):
                            claim(player, f, None, hops)
                if not go:
                    break  # labelled or guessed (them or someone else), turned down, or they're elsewhere then
                claim(player, n, L["cost"][pair], hops)
                seen.add(n)
                k = n
    # one tracklet claimed for two players: neither
    owner: dict[int, set[int]] = {}
    for p, cs in out.items():
        for k in cs:
            owner.setdefault(k, set()).add(p)
    return {p: sorted((c for k, c in cs.items() if len(owner[k]) == 1), key=lambda c: c["start"])
            for p, cs in out.items()}


def evaluate(store: Store, game_ids: list[str], log=print) -> dict:
    """
    Per game and summed: right / wrong / unknown seconds claimed by following the links from half of each
    player's labels (two rounds, so each labelled stretch is hidden once), and the hidden seconds there were.
    Others' labels don't stop a chain here, so a walk onto one is seen as the mistake it is.
    """
    from soccer_clips.track import people

    total = {"hidden_s": 0.0, "right_s": 0.0, "wrong_s": 0.0, "unknown_s": 0.0}
    per = {}
    for gid in game_ids:
        game = store.load_game(gid)
        if links(store, game) is None:
            continue
        fps = game.fps_num / game.fps_den
        t = people.tracks(store, game)
        segs = people.segments(t, people.load_picks(store, game))
        r = {k: 0.0 for k in total}
        looks = player_looks(store, skip=gid)  # their looks from this game would include the hidden stretches
        for player in sorted({x["player"] for x in segs}):
            mine = [x for x in segs if x["player"] == player]
            others = [x for x in segs if x["player"] != player]
            for half in (0, 1):
                hidden = mine[half::2]
                if not hidden:
                    continue
                shown = [x for x in mine if x not in hidden]
                r["hidden_s"] += sum(x["end"] - x["start"] for x in hidden) / fps
                for c in extend(store, game, shown, {}, looks=looks).get(player, []):
                    span = c["end"] - c["start"]
                    ov = lambda xs: sum(max(0, min(c["end"], x["end"]) - max(c["start"], x["start"]))
                                        for x in xs if x["track"] == c["track"])
                    right, wrong = ov(hidden), ov(others)
                    r["right_s"] += right / fps
                    r["wrong_s"] += wrong / fps
                    r["unknown_s"] += max(span - right - wrong, 0) / fps
        r = {k: round(v, 1) for k, v in r.items()}
        claimed = r["right_s"] + r["wrong_s"]
        r["precision"] = round(r["right_s"] / claimed, 3) if claimed else None
        r["coverage"] = round(r["right_s"] / r["hidden_s"], 3) if r["hidden_s"] else None
        per[gid] = r
        for k in total:
            total[k] += r[k]
        log(f"{gid}: {r}")
    claimed = total["right_s"] + total["wrong_s"]
    total = {k: round(v, 1) for k, v in total.items()}
    total["precision"] = round(total["right_s"] / claimed, 3) if claimed else None
    total["coverage"] = round(total["right_s"] / total["hidden_s"], 3) if total["hidden_s"] else None
    return {"games": per, "total": total}
