"""
Game-ball path from per-frame ball detections (Phase 4.1 temporal filter).

A detector finds every ball-like thing in every frame: the game ball (when it
can see it), spare balls, the odd head or cleat. This picks one path through
a clip, offline, so it can look both ways in time:

1. Camera motion (camera.py) maps every frame into the clip's first frame, so a
   ball lying still stays still even while the camera pans and zooms.
2. Detections are linked frame to frame into short tracklets (gated by how far
   a ball can move, allowing a few missed frames).
3. A chain of non-overlapping tracklets is chosen by dynamic programming: it
   gains for confident detections, pays for frames with no ball, pays for
   jumps a ball couldn't make in the time, and pays more for tracklets that sit
   still for seconds (spare balls).
4. Short gaps between chosen tracklets are filled by interpolating in the
   shared reference and mapping back; longer gaps stay empty.

Output per frame: x, y (frame pixels), r, conf, kind: 0 none, 1 detected, 2 filled.

Defaults were set on the bench labels (2026-10-21) with a coarse grid, checked on two
detectors: filling gaps up to 1 s and not discounting weak detections helped a few
points; the start cost made no difference. A ball still for 3 s+ is only used when the chain
leaves it with a kick or arrives at it as it comes to rest (set pieces). With the fine-tuned detector, which also finds
spare balls, the path fell back on a spare when the game ball went undetected (5 of 12
wrong positions); ignoring balls parked for 3 s halved the wrong positions (10% -> 5%)
without losing any found frames (2026-10-21).

Over whole games the camera estimate drifts during pans, so a still object far away seemed
to wander and was taken for the game ball (a flag behind a goal, 2026-10-23). Stillness is
now also judged over 1 s spans, set pieces are only credited when the camera kept the ball in
place on screen, and parked tracklets never fill gaps: on the labelled frames of the six
games (ball_game.evaluate), found 55% -> 67%, wrong 24% -> 13%; bench unchanged.
"""

from dataclasses import dataclass, field

import numpy as np

from soccer_clips.track import camera

BALL = 32
NONE, DETECTED, FILLED = 0, 1, 2


@dataclass
class Params:
    min_conf: float = 0.1  # weakest detection considered
    gate_px: float = 45.0  # max distance from prediction for the next frame (frame pixels)
    gate_per_gap_px: float = 15.0  # extra allowance per missed frame
    max_link_gap: int = 6  # frames a tracklet may go undetected and still continue
    conf_offset: float = 0.0  # each detection scores conf - offset
    miss_cost: float = 0.02  # per frame with no ball between chosen tracklets
    start_cost: float = 0.3  # starting a chain from nothing
    vmax_px: float = 40.0  # fastest plausible ball, reference pixels per frame
    jump_cost: float = 0.004  # per reference pixel of the jump between tracklets
    teleport_gap: int = 45  # after this many empty frames any jump is allowed (ball left the view)
    teleport_cost: float = 1.0
    parked_frames: int = 90  # a tracklet still for this long (3 s) looks like a spare ball...
    parked_spread_px: float = 12.0  # ...if it stays within this (reference pixels)
    parked_penalty: float = 1.0  # times its detection score: 1 = never used (it can't beat start_cost)
    # Stillness is also judged over short spans, since over a minute the camera estimate drifts
    # (a still object far away can seem to wander 100+ px across pans): a tracklet is parked
    # too if it moves at most still_px per still_lag frames for most of its life (a
    # still object's tracklet also has pan slips and the odd hop onto a passing ball). Whole
    # tracklets only: a ball held at a player's feet for a few seconds, then played on, stays
    # the game ball; tracklets are cut only at kicks (split_kicks).
    still_lag: int = 10
    still_px: float = 20.0  # moves at most this (or 2 radii) per still_lag. Camera slip grows with the span,
    # so a short one: over 1 s (30 px) a far-off chair during fast pans still looked like it moved
    # (PRSC 30:05); 25-40 px over 1 s and 20 px over 1/3 s scored alike on the game labels.
    still_share: float = 0.75
    fill_max_gap: int = 30  # interpolate gaps up to this many frames (1 s)
    # A parked ball the path leaves by being kicked (or arrives at as it comes to rest) is the
    # game ball after all: a free kick, corner or goal kick waiting to be taken.
    kick_gap: int = 15  # frames between the parked ball and the moving one (the kicker may hide it)
    rest_jump_px: float = 30.0  # coming to rest: the parked ball must start this close to where the path was
    moving_px: float = 40.0  # the kicked (or arriving) tracklet must travel at least this far (reference pixels)
    # The camera follows the game ball, so a set piece waiting to be taken stays put on screen
    # too (28 credited on the games moved at most 227 px); a still object off to the side
    # drifts across the screen as the camera follows play.
    set_piece_screen_px: float = 300.0
    kick_frames: int = 20  # split_kicks: moving off means getting moving_px away within this (slow drift doesn't)


@dataclass
class Tracklet:
    frames: list[int] = field(default_factory=list)
    x: list[float] = field(default_factory=list)  # frame pixels
    y: list[float] = field(default_factory=list)
    r: list[float] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    rx: list[float] = field(default_factory=list)  # reference pixels
    ry: list[float] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def add(self, f, x, y, r, c, rx, ry):
        for lst, v in zip((self.frames, self.x, self.y, self.r, self.conf, self.rx, self.ry), (f, x, y, r, c, rx, ry)):
            lst.append(v)

    def parked(self, p: Params) -> bool:
        if self.end - self.start < p.parked_frames:
            return False
        tol = max(p.parked_spread_px, 2 * float(np.median(self.r)))
        rx, ry = np.asarray(self.rx), np.asarray(self.ry)
        if max(np.ptp(rx), np.ptp(ry)) <= tol:
            return True
        f = np.asarray(self.frames)  # detections about still_lag frames apart
        a = np.clip(np.searchsorted(f, f - p.still_lag // 2), 0, len(f) - 1)
        b = np.clip(np.searchsorted(f, f + p.still_lag // 2, side="right") - 1, 0, len(f) - 1)
        tol = max(p.still_px, 2 * float(np.median(self.r)))
        return float(np.mean(np.hypot(rx[b] - rx[a], ry[b] - ry[a]) <= tol)) >= p.still_share

    def moving(self, p: Params) -> bool:
        """Travels somewhere: not just a short, still piece (a spare ball briefly hidden and seen again)."""
        return float(np.hypot(np.ptp(self.rx), np.ptp(self.ry))) >= p.moving_px

    def raw_score(self, p: Params) -> float:
        return float(np.sum(np.asarray(self.conf) - p.conf_offset))

    def score(self, p: Params) -> float:
        s = self.raw_score(p)
        if not self.parked(p):
            return s
        # never worth more than showing no ball for the frames it covers (it would fill gaps for free)
        return s * (1 - p.parked_penalty) - p.parked_penalty * p.miss_cost * (self.end - self.start + 1)


def watched(t: Tracklet, p: Params) -> bool:
    """The camera kept it in place on screen, as it does with a set piece about to be taken."""
    return float(np.ptp(t.x)) <= p.set_piece_screen_px


def kicked(ti: Tracklet, tj: Tracklet, p: Params) -> bool:
    """tj kicks the ball away from parked ti."""
    return watched(ti, p) and tj.moving(p)


def arrives(ti: Tracklet, tj: Tracklet, p: Params) -> bool:
    """Moving ti comes to rest as parked tj."""
    return watched(tj, p) and ti.moving(p)


def detections(boxes: np.ndarray, n_frames: int, min_conf: float) -> list[np.ndarray]:
    """Per frame: (k, 4) array of ball detections cx, cy, r, conf from a benchmark box array."""
    b = boxes[(boxes[:, 1] == BALL) & (boxes[:, 7] >= min_conf) & (boxes[:, 0] < n_frames)]
    b = b[np.argsort(b[:, 0], kind="stable")]
    rows = np.column_stack([(b[:, 3] + b[:, 5]) / 2, (b[:, 4] + b[:, 6]) / 2,
                            np.maximum(b[:, 5] - b[:, 3], b[:, 6] - b[:, 4]) / 2, b[:, 7]])
    out = [np.empty((0, 4)) for _ in range(n_frames)]
    frames, starts = np.unique(b[:, 0].astype(int), return_index=True)
    for f, d in zip(frames, np.split(rows, starts[1:])):  # one sort, not a scan per frame
        out[f] = d
    return out


def split_kicks(t: Tracklet, p: Params) -> list[Tracklet]:
    """
    Cut a tracklet where a ball still for parked_frames moves off (gets moving_px away within
    kick_frames), or where one arrives that fast and stays still that long, so the still part
    is judged as parked on its own and `choose` decides whether it was a set piece. Drift in
    the camera estimate is far slower, so a still object isn't cut into pieces too short to
    look parked (with no time limit, every 3 s of a far-off flag was "kicked" by the drift).
    """
    f = np.asarray(t.frames)
    if f[-1] - f[0] < p.parked_frames:
        return [t]
    rx, ry = np.asarray(t.rx), np.asarray(t.ry)
    tol = max(p.parked_spread_px, 2 * float(np.median(t.r)))
    n = len(f)

    def still(i, j):  # detections i..j-1 within tol of each other
        return j - i > 1 and max(np.ptp(rx[i:j]), np.ptp(ry[i:j])) <= tol

    def far(i, j, k):  # any of detections i..j-1 at least moving_px from detection k
        return j > i and float(np.max(np.hypot(rx[i:j] - rx[k], ry[i:j] - ry[k]))) >= p.moving_px

    cuts = []
    i = 0
    while i < n:
        s0 = cuts[-1] if cuts else 0  # the piece so far starts here
        lo = max(s0, int(np.searchsorted(f, f[i] - p.parked_frames)))  # still for parked_frames up to i, then moves off
        soon = int(np.searchsorted(f, f[i] + p.kick_frames, side="right"))
        if f[i] - f[s0] >= p.parked_frames and still(lo, i + 1) and far(i + 1, soon, i):
            j = i  # cut where it starts moving
            while j + 1 < n and np.hypot(rx[j + 1] - rx[i], ry[j + 1] - ry[i]) <= tol:
                j += 1
            cuts.append(j + 1)
            i = j + 1
            continue
        hi = np.searchsorted(f, f[i] + p.parked_frames, side="right")  # arrives, then still for parked_frames
        if (i > s0 and f[-1] - f[i] >= p.parked_frames and still(i, hi)
                and far(max(s0, int(np.searchsorted(f, f[i] - p.kick_frames))), i, i)):
            cuts.append(i)
            i = hi
            continue
        i += 1
    if not cuts:
        return [t]
    out = []
    for a, b in zip([0] + cuts, cuts + [n]):
        if b > a:
            piece = Tracklet()
            for lst, src in zip((piece.frames, piece.x, piece.y, piece.r, piece.conf, piece.rx, piece.ry),
                                (t.frames, t.x, t.y, t.r, t.conf, t.rx, t.ry)):
                lst.extend(src[a:b])
            out.append(piece)
    return out


def link(dets: list[np.ndarray], steps: np.ndarray, ref: np.ndarray, p: Params) -> list[Tracklet]:
    """Greedy frame-to-frame linking, nearest first, with constant-velocity prediction in frame pixels."""
    done: list[Tracklet] = []
    active: list[dict] = []  # {"t": Tracklet, "pos": (x, y) in this frame's pixels, "vel": (vx, vy), "last": f}
    for f, d in enumerate(dets):
        m = np.vstack([steps[f], [0, 0, 1]])
        for a in active:  # carry last positions into this frame's pixels
            (x,), (y,) = camera.apply(m, np.array([a["pos"][0]]), np.array([a["pos"][1]]))
            a["pos"] = (x, y)
        done += [a["t"] for a in active if f - a["last"] > p.max_link_gap]  # lost too long: finished
        active = [a for a in active if f - a["last"] <= p.max_link_gap]
        used = set()
        pairs = []
        for ai, a in enumerate(active):
            gap = f - a["last"]
            px, py = a["pos"][0] + a["vel"][0] * gap, a["pos"][1] + a["vel"][1] * gap
            gate = p.gate_px + p.gate_per_gap_px * (gap - 1)
            for di, (x, y, r, c) in enumerate(d):
                dist = np.hypot(x - px, y - py)
                if dist <= gate:
                    pairs.append((dist, ai, di))
        matched = set()
        rx, ry = camera.apply(ref[f], d[:, 0], d[:, 1]) if len(d) else (np.empty(0), np.empty(0))
        for dist, ai, di in sorted(pairs):
            if ai in matched or di in used:
                continue
            matched.add(ai)
            used.add(di)
            a = active[ai]
            x, y, r, c = d[di]
            gap = f - a["last"]
            a["vel"] = ((x - a["pos"][0]) / gap, (y - a["pos"][1]) / gap)
            a["pos"], a["last"] = (x, y), f
            a["t"].add(f, x, y, r, c, rx[di], ry[di])
        for di, (x, y, r, c) in enumerate(d):
            if di not in used:
                t = Tracklet()
                t.add(f, x, y, r, c, rx[di], ry[di])
                active.append({"t": t, "pos": (x, y), "vel": (0.0, 0.0), "last": f})
    return done + [a["t"] for a in active]


def choose(tracklets: list[Tracklet], p: Params) -> list[Tracklet]:
    """
    Best chain of tracklets, in time order (dynamic programming over tracklets sorted by end).
    The whole clip is decided at once, so a parked ball is credited when the chain later
    leaves it with a kick, or arrives at it as the ball comes to rest.

    Predecessors ending more than teleport_gap frames before tj starts can always be reached
    (at teleport cost plus miss_cost per empty frame), so for them only the best value of
    best[i] + miss_cost * end_i matters: a prefix maximum over tracklets in end order, looked
    up by binary search on tj's start (starts aren't in end order, so it can't be a single
    moving pointer). Only near ones are compared one by one, which keeps a whole game quick.
    """
    ts = sorted(tracklets, key=lambda t: t.end)
    if not ts:
        return []
    scores = [t.score(p) for t in ts]
    parked = [t.parked(p) for t in ts]
    moving = [t.moving(p) for t in ts]
    seen = [watched(t, p) for t in ts]  # see kicked / arrives
    ends = np.array([t.end for t in ts])
    best = np.empty(len(ts))
    prev = np.full(len(ts), -1)
    credited = np.zeros(len(ts), bool)  # a parked tracklet's own score already added along its best chain
    pm_val = np.full(len(ts), -np.inf)  # pm_val[k]: max of best[i] + miss_cost * end_i over i <= k
    pm_arg = np.full(len(ts), -1)
    for j, tj in enumerate(ts):
        best[j] = scores[j] - p.start_cost
        k = int(np.searchsorted(ends, tj.start - p.teleport_gap, side="left"))  # ts[:k] end long before tj
        k = min(k, j)
        if k > 0 and pm_arg[k - 1] >= 0:
            v = pm_val[k - 1] - p.miss_cost * tj.start - p.teleport_cost + scores[j]
            if v > best[j]:
                best[j], prev[j] = v, pm_arg[k - 1]
        for i in range(k, j):
            ti = ts[i]
            gap = tj.start - ti.end
            if gap <= 0:
                continue
            jump = np.hypot(tj.rx[0] - ti.rx[-1], tj.ry[0] - ti.ry[-1])
            if gap > p.teleport_gap:
                trans = p.teleport_cost + p.miss_cost * gap
            elif jump <= p.vmax_px * gap + 20:
                trans = p.miss_cost * (gap - 1) + p.jump_cost * jump
            else:
                continue  # too far, too fast
            bonus, credit = 0.0, False
            if gap <= p.kick_gap:
                if parked[i] and not credited[i] and seen[i] and moving[j]:
                    bonus = ts[i].raw_score(p)  # the ball was kicked from where it sat
                elif parked[j] and jump <= p.rest_jump_px and seen[j] and moving[i]:
                    bonus, credit = ts[j].raw_score(p), True  # the ball came to rest here
            v = best[i] + scores[j] - trans + bonus
            if v > best[j]:
                best[j], prev[j], credited[j] = v, i, credit
        here = best[j] + p.miss_cost * tj.end
        if j and pm_val[j - 1] >= here:
            pm_val[j], pm_arg[j] = pm_val[j - 1], pm_arg[j - 1]
        else:
            pm_val[j], pm_arg[j] = here, j
    if best.max() <= 0:
        return []  # nothing worth more than no ball at all
    j = int(np.argmax(best))
    chain = []
    while j >= 0:
        chain.append(ts[j])
        j = prev[j]
    return chain[::-1]


def chain_value(chain: list[Tracklet], p: Params) -> float:
    """What `choose` scores a chain at (for tests and debugging)."""
    v, credited = chain[0].score(p) - p.start_cost, False
    for ti, tj in zip(chain, chain[1:]):
        gap = tj.start - ti.end
        jump = np.hypot(tj.rx[0] - ti.rx[-1], tj.ry[0] - ti.ry[-1])
        trans = (p.teleport_cost + p.miss_cost * gap if gap > p.teleport_gap
                 else p.miss_cost * (gap - 1) + p.jump_cost * jump)
        bonus, credit = 0.0, False
        if gap <= p.kick_gap:
            if ti.parked(p) and not credited and kicked(ti, tj, p):
                bonus = ti.raw_score(p)
            elif tj.parked(p) and jump <= p.rest_jump_px and arrives(ti, tj, p):
                bonus, credit = tj.raw_score(p), True
        v += tj.score(p) - trans + bonus
        credited = credit
    return v


def path(chain: list[Tracklet], ref: np.ndarray, n_frames: int, p: Params) -> np.ndarray:
    """(n, 5): x, y, r, conf, kind per frame; gaps up to fill_max_gap interpolated in reference pixels."""
    out = np.zeros((n_frames, 5))
    pts = []  # (frame, rx, ry, r) of every chosen detection, in time order
    for t in chain:
        for f, x, y, r, c, rx, ry in zip(t.frames, t.x, t.y, t.r, t.conf, t.rx, t.ry):
            out[f] = (x, y, r, c, DETECTED)
            pts.append((f, rx, ry, r))
    for (f0, x0, y0, r0), (f1, x1, y1, r1) in zip(pts, pts[1:]):
        if 1 < f1 - f0 <= p.fill_max_gap + 1:
            for f in range(f0 + 1, f1):
                a = (f - f0) / (f1 - f0)
                inv = np.linalg.inv(ref[f])
                (x,), (y,) = camera.apply(inv, np.array([x0 + a * (x1 - x0)]), np.array([y0 + a * (y1 - y0)]))
                out[f] = (x, y, r0 + a * (r1 - r0), 0.0, FILLED)
    return out


def track(boxes: np.ndarray, steps: np.ndarray, p: Params | None = None) -> np.ndarray:
    """Game-ball path for one clip from its benchmark box array and camera steps."""
    p = p or Params()
    n = len(steps)
    ref = camera.to_reference(steps)
    dets = detections(boxes, n, p.min_conf)
    tracklets = [piece for t in link(dets, steps, ref, p) for piece in split_kicks(t, p)]
    return path(choose(tracklets, p), ref, n, p)


def top1(boxes: np.ndarray, n_frames: int, min_conf: float) -> np.ndarray:
    """Baseline: the most confident detection in each frame, same layout as `track`."""
    out = np.zeros((n_frames, 5))
    for f, d in enumerate(detections(boxes, n_frames, min_conf)):
        if len(d):
            x, y, r, c = d[np.argmax(d[:, 3])]
            out[f] = (x, y, r, c, DETECTED)
    return out


# ---- scoring against the bench labels ----

def evaluate(store, results_dir, variant: str, p: Params | None = None, method: str = "track") -> dict:
    """
    For each labelled bench frame, is the path on the game ball? visible labels: found (within
    max(10 px, 2 radii)), wrong (a position elsewhere) or missed (no position). hidden labels:
    near (within twice that, the label is approximate) or not. none labels: whether the path
    claims a ball anyway. Also writes each clip's path to <variant>/<clip>.<method>.npy.
    """
    from soccer_clips.track import ball_labels
    from soccer_clips.track.bench import bench_dir, load_set

    p = p or Params()
    settings = store.settings
    labels = ball_labels.load_labels(settings)
    clips = {c.name: c for c in load_set(settings).clips}
    tasks = [t for t in ball_labels.load_tasks(settings).sets.get("bench", []) if t.clip in clips]
    vdir = results_dir / variant
    n = dict.fromkeys(["visible", "found", "found_filled", "wrong", "missed", "hidden", "hidden_near",
                       "none", "none_claimed"], 0)
    paths = {}
    for name in sorted({t.clip for t in tasks}):
        boxes = np.load(vdir / f"{name}.npz")["boxes"]
        steps = camera.load_or_estimate(bench_dir(settings) / "clips" / f"{name}.mp4")
        paths[name] = track(boxes, steps, p) if method == "track" else top1(boxes, len(steps), p.min_conf)
        np.save(vdir / f"{name}.{method}.npy", paths[name])
    for t in tasks:
        label = labels.get((t.game_id, t.frame))
        if label is None or label.state == "skip":
            continue
        game = store.load_game(t.game_id)
        local = t.frame - game.frame_at(clips[t.clip].start_s)
        x, y, r, c, kind = paths[t.clip][local]
        if label.state == "none":
            n["none"] += 1
            n["none_claimed"] += int(kind != NONE)
            continue
        tol = max(ball_labels.MIN_TOL_PX, 2 * (label.r or 0) * game.height)
        dist = np.hypot(x - label.x * game.width, y - label.y * game.height)
        if label.state == "visible":
            n["visible"] += 1
            if kind == NONE:
                n["missed"] += 1
            elif dist <= tol:
                n["found"] += 1
                n["found_filled"] += int(kind == FILLED)
            else:
                n["wrong"] += 1
        else:
            n["hidden"] += 1
            n["hidden_near"] += int(kind != NONE and dist <= 2 * tol)
    pct = lambda a, b: round(n[a] / n[b], 3) if n[b] else None
    return {"found": pct("found", "visible"), "wrong": pct("wrong", "visible"), "missed": pct("missed", "visible"),
            "filled_share_of_found": pct("found_filled", "found"), "hidden_near": pct("hidden_near", "hidden"),
            "claimed_when_none": pct("none_claimed", "none"), "counts": n}


def draw(store, results_dir, variant: str, clip_name: str, method: str = "track"):
    """
    Review video: every ball detection as a small grey circle with its confidence, the chosen
    path as a green ring (detected) or yellow ring (filled), and on labelled bench frames the
    hand label as a white cross. Writes <variant>/<clip>.<method>.mp4.
    """
    import subprocess

    import cv2

    from soccer_clips.track import ball_labels
    from soccer_clips.track.bench import bench_dir, load_set

    settings = store.settings
    vdir = results_dir / variant
    boxes = np.load(vdir / f"{clip_name}.npz")["boxes"]
    path_ = np.load(vdir / f"{clip_name}.{method}.npy")
    clip = next(c for c in load_set(settings).clips if c.name == clip_name)
    game = store.load_game(clip.game_id)
    first = game.frame_at(clip.start_s)
    labels = {f - first: l for (g, f), l in ball_labels.load_labels(settings).items() if g == clip.game_id}
    cap = cv2.VideoCapture(str(bench_dir(settings) / "clips" / f"{clip_name}.mp4"))
    out = vdir / f"{clip_name}.{method}.mp4"
    enc = subprocess.Popen(
        [settings.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
         "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for _, _, _, x1, y1, x2, y2, conf in boxes[(boxes[:, 1] == BALL) & (boxes[:, 0] == f) & (boxes[:, 7] >= 0.1)]:
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(frame, c, 6, (160, 160, 160), 1)
            cv2.putText(frame, f"{conf:.2f}", (c[0] + 8, c[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        x, y, r, conf, kind = path_[f] if f < len(path_) else (0, 0, 0, 0, NONE)
        if kind != NONE:
            colour = (0, 220, 0) if kind == DETECTED else (0, 220, 255)
            cv2.circle(frame, (int(x), int(y)), int(max(14, 2.5 * r)), colour, 2)
        if f in labels and labels[f].state in ("visible", "hidden"):
            lx, ly = int(labels[f].x * 1920), int(labels[f].y * 1080)
            cv2.drawMarker(frame, (lx, ly), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
        state = {DETECTED: "detected", FILLED: "filled", NONE: "no ball"}[int(kind)]
        cv2.putText(frame, f"{variant} {method}  frame {f}  {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        enc.stdin.write(frame.tobytes())
        f += 1
    cap.release()
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"ffmpeg failed drawing {clip_name}")
    return out
