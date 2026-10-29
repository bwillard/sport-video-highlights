"""
Where things are on the pitch (Phase 5), in metres.

A Veo camera doesn't move, so world.py's keyframes are views of one scene, linked by exact
homographies. Tying each keyframe to the ground once per game places every pixel of every frame:
frame pixel -> keyframe pixel (world.place) -> pitch metres (inverse of the keyframe's P below).

Pitch coordinates: metres, origin at the centre spot, x along the length toward the goal on the
right of the camera's view, y across toward the far touchline (the camera is on the near side).

Calibration (`solve`): each keyframe k gets P[k], a homography from pitch metres to its pixels.
Solved together from (1) clicks, "this pixel is that landmark / somewhere on that line", (2) the
keyframe links, which must agree about where pitch points are in both views, and (3) the field's
sizes, which are unknowns too: 9v9 fields vary (70-80 x 45-55 yd) and their sizes aren't posted,
but the centre circle, goals and areas are a standard size whatever the field, so clicks on them
say how many metres the field is.

<game>/tracks/pitch_points.json: the clicks, {frame, x, y (fractions of the frame), target}.
<game>/tracks/pitch_cal.npz: P (keyframes, 3, 3; NaN where not calibrated), sizes and how sure of
each, per click its residual (pixels) and where it lands on the pitch, the share of the game placed,
and a signature of the clicks it was made from (or, when they aren't enough yet, just why).
"""

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from soccer_clips.models import Game
from soccer_clips.store import Store

YD, FT = 0.9144, 0.3048

# name: (starting value, how far it may drift without a click saying so), metres. US 9v9 guidance
# (secondary sources; the penalty area's size varies most between them).
SIZES = {
    "length": (75 * YD, 5 * YD),
    "width": (50 * YD, 5 * YD),
    "circle": (8 * YD, 0.5 * YD),  # centre circle radius
    "goal": (18.5 * FT, 0.3 * FT),  # between the posts
    "ga_depth": (5 * YD, 1 * YD),  # goal area
    "ga_width": (16 * YD, 1.5 * YD),
    "pa_depth": (14 * YD, 2 * YD),  # penalty area
    "pa_width": (36 * YD, 3 * YD),
    "spot": (10 * YD, 1 * YD),  # penalty spot from the goal line
}
SIZE_NAMES = list(SIZES)
S0 = np.array([v for v, _ in SIZES.values()])
S_SIGMA = np.array([s for _, s in SIZES.values()])


def _lin(x: dict | None = None, y: dict | None = None) -> np.ndarray:
    """A point whose coordinates are linear in the sizes: (2, n sizes) coefficients."""
    m = np.zeros((2, len(SIZE_NAMES)))
    for row, d in ((0, x or {}), (1, y or {})):
        for k, v in d.items():
            m[row, SIZE_NAMES.index(k)] = v
    return m


def _landmarks() -> dict[str, tuple[np.ndarray, str]]:
    """id: (coefficients, label). Ends l / r (left / right in the camera's view), sides n / f (near / far)."""
    out = {"centre": (_lin(), "centre spot"),
           "half_n": (_lin(y={"width": -0.5}), "halfway line, near touchline"),
           "half_f": (_lin(y={"width": 0.5}), "halfway line, far touchline"),
           "circle_n": (_lin(y={"circle": -1}), "centre circle meets halfway line, near"),
           "circle_f": (_lin(y={"circle": 1}), "centre circle meets halfway line, far")}
    for end, ex, word in (("l", -0.5, "left"), ("r", 0.5, "right")):
        sx = 1 if ex < 0 else -1  # into the field from this end's goal line
        out[f"spot_{end}"] = (_lin(x={"length": ex, "spot": sx}), f"{word} penalty spot")
        for side, sy, sw in (("n", -0.5, "near"), ("f", 0.5, "far")):
            out[f"corner_{end}{side}"] = (_lin(x={"length": ex}, y={"width": sy}), f"{word} {sw} corner")
            out[f"post_{end}{side}"] = (_lin(x={"length": ex}, y={"goal": sy}), f"{word} goal, {sw} post (foot)")
            for a, name in (("ga", "goal area"), ("pa", "penalty area")):
                out[f"{a}_{end}{side}_line"] = (_lin(x={"length": ex}, y={f"{a}_width": sy}),
                                                f"{word} {name}, {sw} side on the goal line")
                out[f"{a}_{end}{side}"] = (_lin(x={"length": ex, f"{a}_depth": sx}, y={f"{a}_width": sy}),
                                           f"{word} {name}, {sw} corner in the field")
    return out


LANDMARKS = _landmarks()


def _lines() -> dict[str, tuple[str, str, str]]:
    """id: (landmark at one end, landmark at the other, label); the centre circle is separate."""
    out = {"touch_n": ("corner_ln", "corner_rn", "near touchline"), "touch_f": ("corner_lf", "corner_rf", "far touchline"),
           "halfway": ("half_n", "half_f", "halfway line")}
    for end, word in (("l", "left"), ("r", "right")):
        out[f"goal_{end}"] = (f"corner_{end}n", f"corner_{end}f", f"{word} goal line")
        for a, name in (("ga", "goal area"), ("pa", "penalty area")):
            out[f"{a}_{end}"] = (f"{a}_{end}n", f"{a}_{end}f", f"{word} {name}, front edge")
            for side, sw in (("n", "near"), ("f", "far")):  # not just {a}_{end}{side}: that's the corner
                out[f"{a}_{end}{side}_side"] = (f"{a}_{end}{side}_line", f"{a}_{end}{side}", f"{word} {name}, {sw} side")
    return out


LINES = _lines()
assert not LINES.keys() & LANDMARKS.keys()
CIRCLE = "circle"  # click target: somewhere on the centre circle


def targets() -> list[dict]:
    """Every click target, for the page: {id, kind (point / line / circle), label}."""
    return ([{"id": k, "kind": "point", "label": v[1]} for k, v in LANDMARKS.items()]
            + [{"id": k, "kind": "line", "label": v[2]} for k, v in LINES.items()]
            + [{"id": CIRCLE, "kind": "circle", "label": "centre circle"}])


def landmark(target: str, sizes: np.ndarray) -> np.ndarray:
    return LANDMARKS[target][0] @ sizes


def drawing(sizes: np.ndarray | None = None) -> dict:
    """The field's markings in metres, for the map and the overlay: polylines and the goal posts."""
    s = S0 if sizes is None else np.asarray(sizes)
    pt = lambda k: landmark(k, s).tolist()
    lines = [[pt(a), pt(b)] for a, b, _ in LINES.values()]
    t = np.linspace(0, 2 * np.pi, 65)
    r = s[SIZE_NAMES.index("circle")]
    lines.append(np.column_stack([r * np.cos(t), r * np.sin(t)]).tolist())
    return {"lines": lines, "posts": [pt(f"post_{e}{d}") for e in "lr" for d in "nf"],
            "spots": [pt("centre"), pt("spot_l"), pt("spot_r")], "length": float(s[0]), "width": float(s[1]),
            "circle": float(r), "landmarks": {k: pt(k) for k in LANDMARKS},  # click targets, for snapping on the map
            "segments": {k: [pt(a), pt(b)] for k, (a, b, _) in LINES.items()}}


# ---- projective helpers ----

def _h(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)


def project(P: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points (..., 2) through 3x3 matrices P (broadcast); NaN where they land behind the view."""
    u = np.einsum("...ij,...j->...i", P, _h(np.asarray(pts, float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = u[..., :2] / u[..., 2:]
    out[u[..., 2] <= 0] = np.nan
    return out


# ---- solving ----

W, H = 1920, 1080  # keyframe pixels are full-size frame pixels
NI = np.array([[W / 2, 0, W / 2], [0, W / 2, H / 2], [0, 0, 1.0]])  # normalised pixels -> pixels
SM = np.diag([1 / 40, 1 / 40, 1.0])  # metres -> normalised pitch units
CLICK_PX = 3.0  # how far a careful click may be off
LINK_PX = 6.0  # links disagree on ground points by ~5 px (median; 23 px at the 90th percentile, PRSC): ORB
# matches are mostly background and a Veo view isn't an exact rotation. Each link counts about as much as
# one click (its points share the weight), so links carry the calibration into other views but don't
# outvote the clicks.
DAMP = 1e-3  # a faint pull back to the starting point, for keyframes little else pins down
MARGIN_M = 4.0  # links are checked on ground this far beyond the lines
LINK_POINTS = 8  # per link
MIN_PX_PER_M = 2.0  # pitch points further away than this (near the horizon) aren't used for links
BAD_CLICK_PX = 12.0  # a click this far off after the solve is flagged
INIT_HOPS = 2  # clicks this many links away from a keyframe help start it off
TOL = 1e-6  # stop when a step gains less than this share of the cost. Not looser: the robust reweighting
# creeps along a flat valley (PRSC: the circle's size against the links, 0.03% of the cost a step for 40
# steps), and stopping at 1e-3 left positions off by a median 1.6 m across the field


class NotEnough(ValueError):
    pass


def _conic(pts: np.ndarray) -> np.ndarray | None:
    """The conic (3x3 symmetric) through 5+ image points, least squares."""
    if len(pts) < 5:
        return None
    m, s = pts.mean(axis=0), pts.std() + 1e-9
    T = np.array([[1 / s, 0, -m[0] / s], [0, 1 / s, -m[1] / s], [0, 0, 1]])
    x, y = ((pts - m) / s).T
    _, _, vt = np.linalg.svd(np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)]))
    a, b, c, d, e, f = vt[-1]
    C = np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])
    return T.T @ C @ T


def _circle_ends(centre: np.ndarray, near: np.ndarray, far: np.ndarray, on_circle: np.ndarray) -> np.ndarray | None:
    """
    Where the centre circle's other diameter (the one along the field) ends, in the image, left
    then right: (2, 2). From the circle's image (a conic through the clicks) and its centre and
    halfway-line crossings: the centre is their midpoint on the pitch, which fixes the halfway
    line's vanishing point (the centre's harmonic conjugate), and that point's polar is the
    image of the other diameter.
    """
    C = _conic(np.vstack([on_circle, near, far]))
    if C is None:
        return None
    d = far - near
    L = np.linalg.norm(d)
    if L < 1:
        return None
    d /= L
    tc = float((centre - near) @ d)
    num, den = -tc * L, L - 2 * tc  # harmonic conjugate of tc w.r.t. 0 and L, as num / den
    v = np.array([near[0] * den + d[0] * num, near[1] * den + d[1] * num, den])
    line = C @ v
    n = line[:2] / np.linalg.norm(line[:2])
    p0 = centre - n * (float(line @ _h(centre)) / np.linalg.norm(line[:2]))  # the centre, onto the line
    u = np.array([-n[1], n[0]])
    P0, U = _h(p0), np.array([u[0], u[1], 0.0])
    qa, qb, qc = U @ C @ U, 2 * (P0 @ C @ U), P0 @ C @ P0
    disc = qb * qb - 4 * qa * qc
    if disc <= 0 or abs(qa) < 1e-12:
        return None
    ends = np.array([p0 + u * (-qb + sg * np.sqrt(disc)) / (2 * qa) for sg in (-1, 1)])
    return ends[np.argsort(ends[:, 0])]


def _general(src: np.ndarray, tol_m: float = 1.0) -> bool:
    """Whether 4 of the pitch points have no 3 on one line, which a homography needs: not true when all but one
    are on one line (the halfway line's three landmarks and one more)."""
    pts = np.unique(np.round(src, 2), axis=0)
    if len(pts) < 4:
        return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = pts[j] - pts[i]
            e = pts - pts[i]
            off = np.abs(d[0] * e[:, 1] - d[1] * e[:, 0]) / np.linalg.norm(d)
            if (off > tol_m).sum() < 2:
                return False
    return True


def _start(world: dict, ck: np.ndarray, cpx: np.ndarray, ctarget: list[str]) -> tuple[int, np.ndarray]:
    """
    (root keyframe, its pitch -> pixels homography) from the clicks it and its near neighbours
    show: 4+ landmarks not all on one line, or the centre spot, both places the circle crosses
    the halfway line and 3+ more points on the circle (often all a kickoff view shows clearly).
    """
    from soccer_clips.track import world as wmod

    views = wmod._views(world, INIT_HOPS)
    best = None
    for k in sorted(set(ck.tolist())):
        src, dst, seen, circ = [], [], {}, []
        for k2, H_k_k2 in views.get(k, [(k, np.eye(3))]):
            inv = np.linalg.inv(H_k_k2)
            for i in np.flatnonzero(ck == k2):
                t = ctarget[i]
                if t not in LANDMARKS and t != CIRCLE:
                    continue
                p = project(inv, cpx[i])
                if np.isnan(p).any() or not (-W < p[0] < 2 * W and -H < p[1] < 2 * H):
                    continue
                if t == CIRCLE:
                    circ.append(p)
                    continue
                src.append(landmark(t, S0))
                dst.append(p)
                seen.setdefault(t, p)
        if {"centre", "circle_n", "circle_f"} <= seen.keys() and len(circ) >= 3:
            ends = _circle_ends(seen["centre"], seen["circle_n"], seen["circle_f"], np.array(circ))
            if ends is not None:
                r = S0[SIZE_NAMES.index("circle")]
                src += [[-r, 0.0], [r, 0.0]]
                dst += list(ends)
                seen.update({"circle_l": ends[0], "circle_r": ends[1]})
        if len(seen) < 4:
            continue
        src, dst = np.array(src, float), np.array(dst, float)
        if not _general(src):
            continue
        score = (len(seen), len(src))
        if best is None or score > best[0]:
            Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(src) > 4 else 0, 20.0)
            if Hm is not None:
                if np.median((_h(src) @ Hm.T)[:, 2]) < 0:  # points in view come out with w > 0
                    Hm = -Hm
                best = (score, k, Hm)
    if best is None:
        raise NotEnough("click at least 4 landmarks, no 3 of them on one line (or the centre spot, both ends of the "
                        "circle on the halfway line and 3 more points on the circle), in one view or views next to each other")
    return best[1], best[2]


def _spread(world: dict, root: int, P_root: np.ndarray) -> dict[int, np.ndarray]:
    """P for every keyframe reachable from the root along links (breadth first)."""
    nbr: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (a, b), Hab in zip(world["links"].tolist(), world["link_H"]):
        nbr.setdefault(a, []).append((b, Hab))
        nbr.setdefault(b, []).append((a, np.linalg.inv(Hab)))
    P = {root: P_root / np.abs(P_root).max()}
    q = deque([root])
    while q:
        a = q.popleft()
        for b, Hab in nbr.get(a, []):
            if b not in P:
                m = Hab @ P[a]
                P[b] = m / np.abs(m).max()
                q.append(b)
    return P


PIX_GRID = np.array([(x, y) for x in np.linspace(80, W - 80, 10) for y in np.linspace(80, H - 80, 6)])


def _on_ground(P: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """(K, g) bool: which PIX_GRID points each keyframe shows on the field (or just beyond), not near the horizon."""
    q = project(np.linalg.inv(P)[:, None], PIX_GRID[None])  # (K, g, 2) on the pitch
    p2 = project(P[:, None], q + [0, 1.0])  # a metre further away
    with np.errstate(invalid="ignore"):
        return ((np.abs(q[..., 0]) <= sizes[0] / 2 + MARGIN_M) & (np.abs(q[..., 1]) <= sizes[1] / 2 + MARGIN_M)
                & (np.linalg.norm(p2 - PIX_GRID[None], axis=-1) >= MIN_PX_PER_M))


def _link_samples(world: dict, idx: dict[int, int], P: np.ndarray, sizes: np.ndarray):
    """
    (a, b, H_ab, pixel in a, weight) rows: a few ground pixels of each link's first view that the
    second view shows too. Pixels, not pitch points, so that the links don't care how many metres
    the field is (only the clicks and the standard sizes say that).
    """
    ok = _on_ground(P, sizes)
    A, B, Hs, X, Wt = [], [], [], [], []
    for (a, b), Hab in zip(world["links"].tolist(), world["link_H"]):
        if a not in idx or b not in idx:
            continue
        cand = PIX_GRID[ok[idx[a]]]
        pb = project(Hab, cand)
        with np.errstate(invalid="ignore"):
            both = np.flatnonzero((pb[:, 0] > 10) & (pb[:, 0] < W - 10) & (pb[:, 1] > 10) & (pb[:, 1] < H - 10))
        if len(both) < 2:
            continue
        pick = both[np.linspace(0, len(both) - 1, min(LINK_POINTS, len(both))).round().astype(int)]
        A += [idx[a]] * len(pick)
        B += [idx[b]] * len(pick)
        Hs += [Hab] * len(pick)
        X.append(cand[pick])
        Wt += [1 / (LINK_PX * np.sqrt(len(pick)))] * len(pick)
    if not A:
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3, 3)), np.empty((0, 2)), np.empty(0)
    return np.array(A), np.array(B), np.array(Hs), np.concatenate(X), np.array(Wt)


def _jproj(u: np.ndarray) -> np.ndarray:
    """d(u0/u2, u1/u2) / du: (n, 2, 3)."""
    z = 1 / u[:, 2]
    J = np.zeros((len(u), 2, 3))
    J[:, 0, 0] = z
    J[:, 1, 1] = z
    J[:, 0, 2] = -u[:, 0] * z * z
    J[:, 1, 2] = -u[:, 1] * z * z
    return J


def _click_spec(ctarget: list[str]) -> dict:
    """Per click: kind (0 point, 1 line, 2 circle) and size coefficients of its point / line ends (n, 2, sizes)."""
    n, nS = len(ctarget), len(SIZE_NAMES)
    kind, A, B = np.zeros(n, int), np.zeros((n, 2, nS)), np.zeros((n, 2, nS))
    for i, t in enumerate(ctarget):
        if t in LANDMARKS:
            A[i] = LANDMARKS[t][0]
        elif t in LINES:
            kind[i] = 1
            A[i], B[i] = LANDMARKS[LINES[t][0]][0], LANDMARKS[LINES[t][1]][0]
        elif t == CIRCLE:
            kind[i] = 2
        else:
            raise ValueError(f"unknown target {t!r}")
    return {"kind": kind, "A": A, "B": B}


def _perp(v: np.ndarray) -> np.ndarray:
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _click_residuals(P: np.ndarray, sizes: np.ndarray, ck: np.ndarray, cpx: np.ndarray, spec: dict) -> np.ndarray:
    """(n, 2) in pixels: point clicks, off by (dx, dy); line / circle clicks, (distance across the line, 0)."""
    Pk, kind = P[ck], spec["kind"]
    out = np.zeros((len(ck), 2))
    pt = kind == 0
    out[pt] = project(Pk[pt], spec["A"][pt] @ sizes) - cpx[pt]
    ln = ~pt
    if ln.any():
        Pl, c = Pk[ln], cpx[ln]
        q = project(np.linalg.inv(Pl), c)  # the click on the pitch
        e1, e2 = spec["A"][ln] @ sizes, spec["B"][ln] @ sizes
        with np.errstate(divide="ignore", invalid="ignore"):
            tan = (e2 - e1) / np.linalg.norm(e2 - e1, axis=1, keepdims=True)
            qs = e1 + tan * np.sum((q - e1) * tan, axis=1, keepdims=True)  # nearest point on the line
            circ = kind[ln] == 2
            r = sizes[SIZE_NAMES.index("circle")]
            qc = q[circ] * (r / np.linalg.norm(q[circ], axis=1, keepdims=True))
            qs[circ], tan[circ] = qc, _perp(qc) / r
            pa, pb = project(Pl, qs), project(Pl, qs + tan)
            nn = _perp(pb - pa)
            nn /= np.linalg.norm(nn, axis=1, keepdims=True)
            out[ln, 0] = np.sum(nn * (c - pa), axis=1)
    return out


def _lm(fun, jac, x: np.ndarray, n_robust: int, iters: int = 100) -> tuple[np.ndarray, float]:
    """
    Levenberg-Marquardt on sparse normal equations, with the first n_robust residuals robust
    (soft L1, by reweighting): a pixel a few off counts fully, a wrong click far less.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    def cost(r):
        z = r[:n_robust] ** 2
        return float(np.sum(2 * (np.sqrt(1 + z) - 1)) + np.sum(r[n_robust:] ** 2))

    r = fun(x)
    c = cost(r)
    lam = 1e-3
    for it in range(iters):
        J = jac(x)
        w = np.ones(len(r))
        w[:n_robust] = 1 / np.sqrt(1 + r[:n_robust] ** 2)
        Jw = diags(w) @ J
        A = (J.T @ Jw).tocsc()
        g = J.T @ (w * r)
        d = A.diagonal()
        while True:
            step = spsolve(A + diags(lam * (d + 1e-9)), -g)
            if -(2 * g @ step + step @ (A @ step)) < TOL * max(c, 1):  # the model expects next to nothing: done
                return x, c
            r2 = fun(x + step)
            c2 = cost(r2)
            if c2 < c:
                break
            lam *= 10
            if lam > 1e8:
                return x, c
        x, r, done = x + step, r2, (c - c2) < TOL * max(c, 1)
        c, lam = c2, max(lam / 10, 1e-9)
        if done:
            break
    return x, c


def solve(world: dict, ck: np.ndarray, cpx: np.ndarray, ctarget: list[str], log=None) -> dict:
    """
    Calibrate every keyframe linked to the clicked ones. Clicks: keyframe (ck), keyframe pixels
    (cpx, (n, 2)) and target ids. Returns P (keyframes, 3, 3) pitch metres -> keyframe pixels
    (NaN where not reached), sizes, and per click its residual in pixels.
    """
    from scipy.sparse import coo_matrix

    ck, cpx = np.asarray(ck, int), np.asarray(cpx, float).reshape(-1, 2)
    root, P_root = _start(world, ck, cpx, ctarget)
    start = _spread(world, root, P_root)
    keys = sorted(start)
    idx = {k: i for i, k in enumerate(keys)}
    K = len(keys)
    NIinv, SMinv = np.linalg.inv(NI), np.linalg.inv(SM)
    M0 = np.array([NIinv @ start[k] @ SMinv for k in keys])
    M0 /= np.abs(M0).reshape(K, 9).max(axis=1)[:, None, None]
    fixed = np.abs(M0.reshape(K, 9)).argmax(axis=1)  # each M keeps its largest entry, for scale
    free = np.ones((K, 9), bool)
    free[np.arange(K), fixed] = False
    nS = len(SIZE_NAMES)
    n_par = 8 * K + nS
    use = np.array([k in idx for k in ck.tolist()])
    cki = np.array([idx.get(k, -1) for k in ck.tolist()])[use]
    cpx_u, spec = cpx[use], _click_spec([t for t, u in zip(ctarget, use) if u])
    m0_free = M0.reshape(K, 9)[free].reshape(K, 8)

    def unpack(x):
        M = M0.reshape(K, 9).copy()
        M[free] = x[:8 * K]
        M = M.reshape(K, 3, 3)
        return M, NI @ M @ SM, x[8 * K:]

    def residuals(x, la, lb, lH, lX, lw):
        M, P, s = unpack(x)
        r_click = _click_residuals(P, s, cki, cpx_u, spec) / CLICK_PX
        r_link = (project(P[lb], project(np.linalg.inv(P[la]), lX)) - project(lH, lX)) * lw[:, None]
        r_size = (s - S0) / S_SIGMA
        r_damp = DAMP * (x[:8 * K] - m0_free.ravel())
        return np.nan_to_num(np.concatenate([r_click.ravel(), r_link.ravel(), r_size, r_damp]), nan=1e3)

    def jacobian(x, la, lb, lH, lX, lw):
        M, P, s = unpack(x)
        rows, cols, vals = [], [], []
        col_of = np.full((K, 9), -1)
        col_of[free] = np.arange(8 * K)
        nC = len(cki)
        # clicks: finite differences, each click moves only with its own keyframe and the sizes
        if nC:
            base = _click_residuals(P, s, cki, cpx_u, spec) / CLICK_PX
            for j in range(9):
                eps = 1e-6
                Mp = M.copy()
                Mp[:, j // 3, j % 3] += eps
                d = (_click_residuals(NI @ Mp @ SM, s, cki, cpx_u, spec) / CLICK_PX - base) / eps
                c = col_of[cki, j]
                for comp in (0, 1):
                    ok = c >= 0
                    rows.append(2 * np.arange(nC)[ok] + comp)
                    cols.append(c[ok])
                    vals.append(d[ok, comp])
            for j in range(nS):
                eps = 1e-5
                sp = s.copy()
                sp[j] += eps
                d = (_click_residuals(P, sp, cki, cpx_u, spec) / CLICK_PX - base) / eps
                for comp in (0, 1):
                    rows.append(2 * np.arange(nC) + comp)
                    cols.append(np.full(nC, 8 * K + j))
                    vals.append(d[:, comp])
        # links: r = proj(P_b y) - proj(H x), y = P_a^-1 x (the pixel on the pitch), P = NI M SM
        r0 = 2 * nC
        nL = len(la)
        if nL:
            inv_a = np.linalg.inv(P[la])
            y = _h(project(inv_a, lX))
            Jz = _jproj(np.einsum("nij,nj->ni", P[lb], y)) * lw[:, None, None]
            ys = y @ SM.T
            for G, kk in ((-Jz @ P[lb] @ inv_a @ NI, la), (Jz @ NI, lb)):  # d r / d (M x_s), for a then b
                d = np.einsum("nci,nj->ncij", G, ys).reshape(nL, 2, 9)
                for j in range(9):
                    c = col_of[kk, j]
                    ok = c >= 0
                    for comp in (0, 1):
                        rows.append(r0 + 2 * np.arange(nL)[ok] + comp)
                        cols.append(c[ok])
                        vals.append(d[ok, comp, j])
        r1 = r0 + 2 * nL
        rows.append(r1 + np.arange(nS))
        cols.append(8 * K + np.arange(nS))
        vals.append(1 / S_SIGMA)
        r2 = r1 + nS
        rows.append(r2 + np.arange(8 * K))
        cols.append(np.arange(8 * K))
        vals.append(np.full(8 * K, DAMP))
        J = coo_matrix((np.nan_to_num(np.concatenate(vals)), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(r2 + 8 * K, n_par))
        return J.tocsr()

    x = np.concatenate([m0_free.ravel(), S0])
    for round_ in range(2):  # pitch points each link checks come from the current estimate
        _, P, s = unpack(x)
        links = _link_samples(world, idx, P, s)
        x, c = _lm(lambda x: residuals(x, *links), lambda x: jacobian(x, *links), x, 2 * (len(cki) + len(links[0])))
        if log:
            log(f"  round {round_ + 1}: {K} keyframes, {len(links[0])} link points, cost {c:.1f}")
    _, P, s = unpack(x)
    n_key = int(max(world["key"].max(), world["links"].max(initial=0))) + 1
    P_all = np.full((n_key, 3, 3), np.nan)
    P_all[keys] = P
    resid = np.full(len(ck), np.nan)
    resid[use] = np.linalg.norm(_click_residuals(P, s, cki, cpx_u, spec), axis=1)
    return {"P": P_all, "sizes": s, "size_sd": _size_sd(jacobian(x, *links), nS), "click_key": ck, "click_px": resid,
            "root": root, "cost": c}


def _size_sd(J, nS: int) -> np.ndarray:
    """How sure the solve is of each size (one standard deviation, metres): from the normal equations,
    with residuals already in units of their noise. Near a size's prior spread = no click measured it."""
    from scipy.sparse.linalg import splu

    A = (J.T @ J).tocsc()
    n = A.shape[0]
    E = np.zeros((n, nS))
    E[n - nS + np.arange(nS), np.arange(nS)] = 1
    try:
        cov = splu(A).solve(E)[n - nS:]
    except RuntimeError:  # singular: nothing pins some keyframe down
        return S_SIGMA.copy()
    return np.sqrt(np.clip(np.diag(cov), 0, None))


# ---- placing points ----

def to_pitch(P_key: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Keyframe pixels -> pitch metres, for each point's keyframe homography P_key (n, 3, 3; NaN = none)."""
    out = np.full((len(P_key), 2), np.nan)
    ok = np.isfinite(P_key).all(axis=(1, 2))
    if ok.any():
        out[ok] = project(np.linalg.inv(P_key[ok]), np.column_stack([kx, ky])[ok])
    return out


def frame_homographies(w: dict, motion_frames: np.ndarray, motion: np.ndarray, frames: np.ndarray):
    """(keyframe, H frame pixels -> keyframe pixels (n, 3, 3)) for each frame, as world.place does it
    (keyframe -1 and NaN where the frame isn't registered)."""
    from soccer_clips.track import world

    frames = np.asarray(frames, int)
    key = np.full(len(frames), -1, np.int32)
    Hs = np.full((len(frames), 3, 3), np.nan)
    sf = w["frames"]
    steps = dict(zip(motion_frames.tolist(), motion))
    for i, f in enumerate(frames.tolist()):
        j = int(np.searchsorted(sf, f, side="right")) - 1
        if j < 0 or f - sf[j] >= world.SAMPLE * 2 or w["key"][j] < 0:
            continue
        key[i] = w["key"][j]
        Hs[i] = w["H"][j] @ np.linalg.inv(world._chain(steps, int(sf[j]), f))
    return key, Hs


def pitch_to_frame(cal_P: np.ndarray, key: np.ndarray, Hs: np.ndarray) -> np.ndarray:
    """Pitch metres -> frame pixels per frame (n, 3, 3), NaN where the frame or its keyframe isn't calibrated."""
    out = np.full((len(key), 3, 3), np.nan)
    ok = (key >= 0) & np.isfinite(Hs).all(axis=(1, 2))
    ok[ok] &= np.isfinite(cal_P[key[ok]]).all(axis=(1, 2))
    if ok.any():
        out[ok] = np.linalg.inv(Hs[ok]) @ cal_P[key[ok]]
    return out


# ---- storage ----

def points_path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "pitch_points.json"


def cal_path(store: Store, game: Game) -> Path:
    from soccer_clips.track.ball_game import tracks_dir

    return tracks_dir(store, game) / "pitch_cal.npz"


POINTS_VERSION = 2  # 1: a box's side line had the same id as its corner in the field (ga_lf), and was solved as the corner


def _upgrade(points: list[dict], version: int) -> list[dict]:
    if version < 2:  # a version 1 click on ga_lf meant the corner or anywhere on that side; the corner is on the side
        boxes = {k.removesuffix("_side") for k in LINES if k.endswith("_side")}
        points = [{**p, "target": p["target"] + "_side"} if p["target"] in boxes else p for p in points]
    return points


def load_points(store: Store, game: Game) -> list[dict]:
    p = points_path(store, game)
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return _upgrade(d["points"], d.get("version", 1))


def save_points(store: Store, game: Game, points: list[dict]) -> None:
    from soccer_clips.store import atomic_write_text

    atomic_write_text(points_path(store, game), json.dumps({"version": POINTS_VERSION, "points": points}, indent=1) + "\n")


_registration_cache: dict[str, tuple] = {}


def _registration(store: Store, game: Game):
    """(world, motion frames, motion) for the game, kept while the files don't change; None if not registered."""
    from soccer_clips.track import ball_game, world

    wp, rp = world.path(store, game), ball_game.tracks_dir(store, game) / "ball_raw.npz"
    if not wp.exists() or not rp.exists():
        return None
    stamp = (wp.stat().st_mtime_ns, rp.stat().st_mtime_ns)
    hit = _registration_cache.get(game.id)
    if hit and hit[0] == stamp:
        return hit[1]
    raw = ball_game.load_raw(store, game)
    val = (world.load(store, game), raw["motion_frames"], raw["motion"])
    _registration_cache[game.id] = (stamp, val)
    return val


def place_clicks(store: Store, game: Game, points: list[dict]):
    """(keyframe, keyframe pixels (n, 2)) for each click (keyframe -1 where the frame isn't registered)."""
    from soccer_clips.track import world

    w, mf, motion = _registration(store, game)
    f = np.array([p["frame"] for p in points], int)
    x = np.array([p["x"] for p in points], float) * game.width
    y = np.array([p["y"] for p in points], float) * game.height
    key, kx, ky = world.place(w, mf, motion, f, x, y)
    return key, np.column_stack([kx, ky])


def _sig(points: list[dict]) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(points, sort_keys=True).encode()).hexdigest()[:16]


def calibrate(store: Store, game: Game, log=None) -> dict:
    """
    Solve from the game's clicks and save pitch_cal.npz (or, when the clicks aren't enough yet,
    just why). Returns what was saved.
    """
    from soccer_clips.track import ball_game

    if _registration(store, game) is None:
        raise NotEnough("this game's frames aren't registered yet (soccer-clips ball fixed-objects)")
    points = load_points(store, game)
    sig = np.str_(_sig(points))
    key, px = place_clicks(store, game, points) if points else (np.empty(0, int), np.empty((0, 2)))
    ok = key >= 0
    w = _registration(store, game)[0]
    try:
        if not points:
            raise NotEnough("no clicks yet")
        out = solve(w, key[ok], px[ok], [p["target"] for p, o in zip(points, ok) if o], log=log)
    except NotEnough as e:
        saved = {"error": np.str_(str(e)), "sig": sig}
    else:
        resid = np.full(len(points), np.nan)
        resid[ok] = out["click_px"]
        at = np.full((len(points), 2), np.nan)
        P_click = np.full((len(points), 3, 3), np.nan)
        P_click[ok] = out["P"][key[ok]]
        at[ok] = to_pitch(P_click[ok], px[ok, 0], px[ok, 1])
        coverage = float(np.isfinite(out["P"][np.clip(w["key"], 0, None)]).all(axis=(1, 2))[w["key"] >= 0].mean())
        saved = {"P": out["P"], "sizes": out["sizes"], "size_sd": out["size_sd"], "click_px": resid,
                 "click_pitch": at, "coverage": np.float64(coverage), "sig": sig}
    ball_game.save_npz(cal_path(store, game), **saved)
    return saved


def load_cal(store: Store, game: Game) -> dict | None:
    """The saved calibration, if it was made from the clicks as they are now."""
    p = cal_path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        cal = {k: z[k] for k in z.files}
    return cal if str(cal["sig"]) == _sig(load_points(store, game)) else None


def _num(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


def state(store: Store, game: Game) -> dict:
    """Everything the tagger shows about the game's calibration (solving first if the clicks changed)."""
    registered = _registration(store, game) is not None
    points = load_points(store, game)
    cal = load_cal(store, game) if registered else None
    if registered and cal is None and points:
        cal = calibrate(store, game)
    ok = cal is not None and "P" in cal
    sizes = cal["sizes"] if ok else S0
    out_points = []
    for i, p in enumerate(points):
        px = _num(cal["click_px"][i]) if ok else None
        at = cal["click_pitch"][i] if ok else None
        out_points.append({**p, "i": i, "px": px, "bad": px is not None and px > BAD_CLICK_PX,
                           "pitch": None if at is None or not np.isfinite(at).all() else [round(float(v), 3) for v in at]})
    return {"registered": registered, "targets": targets(), "points": out_points, "calibrated": ok,
            "error": None if ok or cal is None else str(cal.get("error", "")) or None,
            "drawing": drawing(sizes), "coverage": _num(cal["coverage"]) if ok else None,
            "sizes": {k: {"m": round(float(v), 2), "sd": round(float(sd), 2), "prior_sd": round(float(ps), 2)}
                      for k, v, sd, ps in zip(SIZE_NAMES, sizes, cal["size_sd"] if ok else S_SIGMA, S_SIGMA)}}


def add_point(store: Store, game: Game, frame: int, x: float, y: float, target: str, solve: bool = True) -> dict:
    """Save a click and solve again; with solve=False, only save it (more clicks are coming) and return {count}."""
    if target not in LANDMARKS and target not in LINES and target != CIRCLE:
        raise ValueError(f"unknown target {target!r}")
    if _registration(store, game) is None:
        raise ValueError("this game's frames aren't registered yet (soccer-clips ball fixed-objects)")
    points = load_points(store, game) + [{"frame": frame, "x": round(x, 5), "y": round(y, 5), "target": target}]
    save_points(store, game, points)
    if not solve:
        return {"count": len(points)}
    calibrate(store, game)
    return state(store, game)


def remove_point(store: Store, game: Game, i: int) -> dict:
    points = load_points(store, game)
    if not 0 <= i < len(points):
        raise ValueError("no such click")
    save_points(store, game, points[:i] + points[i + 1:])
    calibrate(store, game)
    return state(store, game)


def frame_matrices(store: Store, game: Game, frames: np.ndarray) -> np.ndarray | None:
    """Pitch metres -> frame pixels (full size) at each frame (n, 3, 3), NaN where the frame isn't placed;
    None when the game isn't calibrated."""
    reg, cal = _registration(store, game), load_cal(store, game)
    if reg is None or cal is None or "P" not in cal:
        return None
    key, Hs = frame_homographies(*reg, np.asarray(frames, int))
    return pitch_to_frame(cal["P"], key, Hs)


def ground(store: Store, game: Game, frames: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Points on the ground in frame pixels (full size) at the given frames -> pitch metres (n, 2), NaN where
    the frame isn't placed; None when the game isn't calibrated."""
    out = np.full((len(frames), 2), np.nan)
    uf, inv = np.unique(np.asarray(frames, int), return_inverse=True)
    G = frame_matrices(store, game, uf)
    if G is None or not len(frames):
        return None if G is None else out
    ok = np.isfinite(G).all(axis=(1, 2))
    Gi = np.full_like(G, np.nan)
    Gi[ok] = np.linalg.inv(G[ok])
    m = ok[inv]
    out[m] = project(Gi[inv[m]], np.column_stack([x, y])[m])
    return out


def frame_rows(store: Store, game: Game, start: int, end: int) -> np.ndarray:
    """(n, 10) float32 for frames [start, end): frame, then the 3x3 pitch metres -> frame pixels (full size,
    row-major, scaled to a largest entry of 1), NaN where the frame or its view isn't calibrated."""
    frames = np.arange(start, end)
    out = np.full((len(frames), 10), np.nan, np.float32)
    out[:, 0] = frames
    reg, cal = _registration(store, game), load_cal(store, game)
    if reg is None or cal is None or "P" not in cal or not len(frames):
        return out
    key, Hs = frame_homographies(*reg, frames)
    G = pitch_to_frame(cal["P"], key, Hs)
    ok = np.isfinite(G).all(axis=(1, 2))
    G[ok] /= np.abs(G[ok]).reshape(-1, 9).max(axis=1)[:, None, None]
    out[:, 1:] = G.reshape(-1, 9)
    return out
