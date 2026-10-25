"""
Game time: when each half is played, and cutting the video down to it.

Before kickoff, at half time and after the final whistle the camera shows warm-ups, a team
talk, handshakes or the next game: nothing to tag, and garbage for tracking. A game's
`periods` are the two halves in video seconds (1st half start/end, 2nd half start/end).

`suggest` proposes them from the player tracks. In play there's a normal number of players on
the pitch (ours and opponents), they move, and the camera is zoomed in on them. At half time
there are far fewer (they go to the benches), or everyone stands about, or the camera shows
only a distant crowd (small boxes); after the game often far more (the handshake line comes
up to the camera). Each signal catches some games (2026-10-25: player height found fc-stars'
and fc-1's half time, speed found SAA's, the count PRSC's), so any one of them can rule a
stretch out. Kickoffs already set by
hand (H in the tagger) are kept. Everything is a suggestion to check in the tagger.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from soccer_clips.models import CutSegment, Game, GameCut
from soccer_clips.store import Store, atomic_write_json, atomic_write_text

WINDOW_S = 10  # activity is judged per 10-second window
PLAY_LOW, PLAY_HIGH = 0.6, 1.35  # in play: player count between these times the game's median
SMALL_HEIGHT = 0.75  # not in play if the bigger players (80th percentile height) are below this times the median
STILL_SPEED = 0.75  # not in play if the players move less than this times the game's median (camera motion
# was tried too: it hovers around any threshold at half time, while the players clearly slow down)
SMOOTH_W = 7  # windows: player counts are a running median over ~70 s, so a moment's crowd or gap doesn't count
MIN_BREAK_S = 120  # half time is at least this long
MIN_RUN_S = 90  # a stretch of play counts if at least this long
GAP_OK_S = 60  # short dips (a stoppage, the camera on the far side) don't end a half
HALF_SLACK_S = 8 * 60  # the 2nd half ends within this of the 1st half's length after its kickoff


@dataclass
class Suggestion:
    start_1h_s: float | None
    end_1h_s: float | None
    start_2h_s: float | None
    end_2h_s: float | None
    notes: list[str]

    def __post_init__(self):
        # plain floats: the window arithmetic yields numpy numbers, which don't go into JSON
        for k in ("start_1h_s", "end_1h_s", "start_2h_s", "end_2h_s"):
            v = getattr(self, k)
            setattr(self, k, None if v is None else float(v))


def activity(store: Store, game: Game) -> dict[str, np.ndarray]:
    """
    Per WINDOW_S window: "count" = mean number of on-pitch players (ours + opponents) per frame;
    "camera" = mean camera movement (pixels a frame); "speed" = mean player speed (box heights a
    second, camera motion cancelled); "height" = 80th percentile on-pitch box height (pixels);
    "tracked" = whether the window was tracked at all.
    """
    from soccer_clips.track import ball_game, people

    n = int(np.ceil(game.duration_s / WINDOW_S))
    out = {k: np.zeros(n) for k in ("count", "camera", "speed", "height")}
    out["tracked"] = np.zeros(n, bool)
    got = people.roles_for(store, game)
    t = people.tracks(store, game)
    if got is None or t is None:
        return out
    _, roles = got
    fps = game.fps

    def per_window(frames, values):
        w = np.minimum((frames / fps // WINDOW_S).astype(int), n - 1)
        return np.bincount(w, weights=values, minlength=n)[:n] / np.maximum(np.bincount(w, minlength=n)[:n], 1)

    on = (roles == 0) | (roles == 1)
    w = np.minimum((t.frame / fps // WINDOW_S).astype(int), n - 1)
    out["count"] = np.bincount(w[on], minlength=n)[:n] / (WINDOW_S * fps)
    h = (t.box[on, 3] - t.box[on, 1]) * game.height
    order = np.argsort(w[on], kind="stable")
    ws, hs = w[on][order], h[order]
    bounds = np.searchsorted(ws, np.arange(n + 1))
    out["height"] = np.array([np.percentile(hs[a:b], 80) if b > a else 0.0 for a, b in zip(bounds[:-1], bounds[1:])])
    for s, e in t.ranges:
        out["tracked"][int(s / fps // WINDOW_S):int(np.ceil(e / fps / WINDOW_S))] = True
    raw = ball_game.load_raw(store, game)
    if len(raw["motion_frames"]):
        o = np.argsort(raw["motion_frames"])
        mf, mo = raw["motion_frames"][o], raw["motion"][o]
        out["camera"] = per_window(mf, np.hypot(mo[:, 0, 2], mo[:, 1, 2]))
        order = np.lexsort((t.frame, t.track))
        tr, fr, bx, r = t.track[order], t.frame[order], t.box[order], roles[order]
        cx, cy = (bx[:, 0] + bx[:, 2]) / 2 * game.width, bx[:, 3] * game.height
        h = (bx[:, 3] - bx[:, 1]) * game.height
        step = (tr[1:] == tr[:-1]) & (fr[1:] == fr[:-1] + 1) & ((r[1:] == 0) | (r[1:] == 1))
        i = np.clip(np.searchsorted(mf, fr[1:]), 0, len(mf) - 1)
        sp = np.hypot(cx[1:] - cx[:-1] - mo[i, 0, 2], cy[1:] - cy[:-1] - mo[i, 1, 2]) / np.maximum(h[1:], 10) * fps
        out["speed"] = per_window(fr[1:][step], np.minimum(sp[step], 5))
    return out


def _runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """[start, end) runs of True, joining runs separated by at most `gap` False windows."""
    runs: list[list[int]] = []
    for i in np.flatnonzero(mask):
        if runs and i - runs[-1][1] <= gap:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return [(a, b) for a, b in runs]


def suggest(store: Store, game: Game) -> Suggestion:
    act = activity(store, game)
    count, tracked = act["count"], act["tracked"]
    notes = []
    if not tracked.any():
        return Suggestion(game.kickoff_1h_s, None, game.kickoff_2h_s, None, ["no player tracks yet"])
    med = float(np.median(count[tracked & (count > 0)]))
    def smooth(x):
        return np.median(np.lib.stride_tricks.sliding_window_view(np.pad(x, SMOOTH_W // 2, mode="edge"), SMOOTH_W),
                         axis=1)

    c = smooth(count)
    play = tracked & (c >= PLAY_LOW * med) & (c <= PLAY_HIGH * med)
    for key, low in (("speed", STILL_SPEED), ("height", SMALL_HEIGHT)):
        x = act[key]
        if x.any():
            play &= smooth(x) > low * np.median(x[tracked & (x > 0)])
    runs = [r for r in _runs(play, GAP_OK_S // WINDOW_S) if (r[1] - r[0]) * WINDOW_S >= MIN_RUN_S]
    if not runs:
        return Suggestion(game.kickoff_1h_s, None, game.kickoff_2h_s, None, ["no stretch of play found"])
    k1, k2 = game.kickoff_1h_s, game.kickoff_2h_s
    if k1 is not None and k2 is not None and k1 >= k2:
        notes.append("the 1st half kickoff is after the 2nd's; ignored it")
        k1 = None
    # half time: the longest break between runs (at least MIN_BREAK_S), or the break just before k2
    breaks = [(runs[i][1], runs[i + 1][0]) for i in range(len(runs) - 1)]
    breaks = [b for b in breaks if (b[1] - b[0]) * WINDOW_S >= MIN_BREAK_S]
    if k2 is not None:
        before = [b for b in breaks if b[1] * WINDOW_S <= k2 + 120]
        ht = max(before, key=lambda b: b[1]) if before else None
    else:
        ht = max(breaks, key=lambda b: b[1] - b[0]) if breaks else None
    if ht is None:
        notes.append("no half-time break found")
        start_1h = k1 if k1 is not None else runs[0][0] * WINDOW_S
        return Suggestion(start_1h, None, k2, runs[-1][1] * WINDOW_S, notes)
    first = [r for r in runs if r[1] <= ht[0]]
    start_1h = k1 if k1 is not None else first[0][0] * WINDOW_S
    end_1h = ht[0] * WINDOW_S
    start_2h = k2 if k2 is not None else ht[1] * WINDOW_S
    # end of the 2nd half: end of the last run after its kickoff, within the 1st half's length + slack
    limit = start_2h + (end_1h - start_1h) + HALF_SLACK_S
    after = [r for r in runs if r[1] * WINDOW_S > start_2h and r[0] * WINDOW_S < limit]
    end_2h = min(after[-1][1] * WINDOW_S, limit) if after else None
    if end_2h is not None and end_2h >= game.duration_s - WINDOW_S:
        notes.append("the video ends during the 2nd half")
        end_2h = game.duration_s
    if end_2h is None:
        notes.append("no play found after the 2nd half kickoff")
    return Suggestion(start_1h, end_1h, start_2h, end_2h, notes)


# ---- cutting the video to game time ----
# Each half (plus PAD_S either side) is copied out of the original without re-encoding, from a
# keyframe to a keyframe (Veo: one every 2.5 s, no B-frames), so frame k of a piece is exactly
# frame orig_start + k of the original; the pieces are joined into <cut folder>/<year>/<id>.mp4.
# The original stays in full/. Then every stored time and frame number of the game is moved
# onto the new timeline (TimeMap) and anything outside game time dropped, after copying all of
# it to <data root>/backup/<time>-cut-<game>/ (`restore` puts it back).

PAD_S = 10.0  # kept before each kickoff and after each final whistle


class TimeMap:
    """Original frames/seconds -> cut-video frames/seconds (None / dropped outside the kept segments)."""

    def __init__(self, segments: list[CutSegment], fps_num: int, fps_den: int):
        self.segments = segments
        self.fps = fps_num / fps_den

    def frames(self, f) -> tuple[np.ndarray, np.ndarray]:
        """(new frame numbers, kept mask) for an array of original frame numbers."""
        f = np.asarray(f, dtype=np.int64)
        new = np.full(f.shape, -1, np.int64)
        for s in self.segments:
            m = (f >= s.orig_start) & (f < s.orig_end)
            new[m] = f[m] - s.orig_start + s.new_start
        return new, new >= 0

    def frame(self, f: int) -> int | None:
        new, ok = self.frames([f])
        return int(new[0]) if ok[0] else None

    def time(self, t: float | None) -> float | None:
        """A time (seconds) on the new timeline, or None if it falls outside game time."""
        if t is None:
            return None
        for s in self.segments:
            if s.orig_start / self.fps <= t < s.orig_end / self.fps:
                return t - (s.orig_start - s.new_start) / self.fps
        return None

    def span(self, start: float, end: float) -> tuple[float, float] | None:
        """[start, end) clamped to game time and moved; None if none of it is in game time."""
        parts = [(max(start, s.orig_start / self.fps), min(end, s.orig_end / self.fps)) for s in self.segments]
        parts = [(a, b) for a, b in parts if b > a]
        if not parts:
            return None
        a, b = self.time(parts[0][0]), self.time(parts[-1][1] - 1e-6)
        return a, b + 1e-6

    def ranges(self, ranges) -> np.ndarray:
        """Frame ranges [a, b) moved onto the new timeline (split or clipped at the cuts)."""
        out = []
        for a, b in np.asarray(ranges, dtype=np.int64).reshape(-1, 2).tolist():
            for s in self.segments:
                lo, hi = max(a, s.orig_start), min(b, s.orig_end)
                if hi > lo:
                    out.append((lo - s.orig_start + s.new_start, hi - s.orig_start + s.new_start))
        out.sort()
        merged: list[list[int]] = []
        for a, b in out:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return np.array(merged, np.int64).reshape(-1, 2)


def periods(game: Game) -> list[tuple[float, float]]:
    """The two halves, checked; raises ValueError if they aren't all set and in order."""
    t = [game.kickoff_1h_s, game.end_1h_s, game.kickoff_2h_s, game.end_2h_s]
    if any(x is None for x in t):
        raise ValueError("set all four times first: 1st half start and end, 2nd half start and end")
    if not (0 <= t[0] < t[1] <= t[2] < t[3] <= game.duration_s + 1):
        raise ValueError("the times must be in order: 1st half start < end <= 2nd half start < end")
    return [(t[0], t[1]), (t[2], t[3])]


def keyframes(store: Store, path: Path, game: Game) -> np.ndarray:
    """Frame numbers of the video's keyframes."""
    out = subprocess.run([store.settings.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time,flags", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True).stdout
    ks = [float(line.split(",")[0]) for line in out.splitlines() if "K" in line.split(",", 1)[-1]]
    return np.unique(np.round(np.array(ks) * game.fps).astype(np.int64))


def plan(store: Store, game: Game) -> list[CutSegment]:
    """Keyframe-to-keyframe segments covering each half plus PAD_S either side."""
    n = game.frame_at(game.duration_s)
    keys = keyframes(store, store.full_path(game), game)
    spans = []
    for start, end in periods(game):
        a = keys[keys <= max(0, game.frame_at(start - PAD_S))]
        b = keys[keys >= game.frame_at(end + PAD_S)]
        spans.append([int(a[-1]) if len(a) else 0, int(b[0]) if len(b) else n])
    if spans[1][0] <= spans[0][1]:  # a half time shorter than the padding: one piece
        spans = [[spans[0][0], spans[1][1]]]
    out, new = [], 0
    for a, b in spans:
        out.append(CutSegment(orig_start=a, orig_end=b, new_start=new))
        new += b - a
    return out


def _count_frames(store: Store, path: Path) -> int:
    out = subprocess.run([store.settings.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().split(",")[0])


def cut_video(store: Store, game: Game, segments: list[CutSegment], out: Path, log=print) -> None:
    """Copy the segments out of the original (no re-encoding) and join them into `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = game.fps
    parts = []
    try:
        for i, s in enumerate(segments):
            part = out.with_name(f"{out.stem}.part{i}.mp4")
            subprocess.run([store.settings.ffmpeg, "-v", "error", "-y", "-ss", f"{(s.orig_start + 0.5) / fps:.6f}",
                            "-i", str(store.full_path(game)), "-t", f"{(s.orig_end - s.orig_start - 0.5) / fps:.6f}",
                            "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero",
                            str(part)], check=True)
            n = _count_frames(store, part)
            if n != s.orig_end - s.orig_start:
                raise RuntimeError(f"piece {i} has {n} frames, expected {s.orig_end - s.orig_start}")
            parts.append(part)
            log(f"{game.id}: piece {i + 1}/{len(segments)} cut ({n} frames)")
        listing = out.with_name(out.stem + ".parts.txt")
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
        tmp = out.with_name(out.stem + ".part.mp4")
        subprocess.run([store.settings.ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(tmp)], check=True)
        total = sum(s.orig_end - s.orig_start for s in segments)
        if _count_frames(store, tmp) != total:
            raise RuntimeError("the joined video doesn't have the expected number of frames")
        tmp.replace(out)
    finally:
        for p in parts:
            p.unlink(missing_ok=True)
        out.with_name(out.stem + ".parts.txt").unlink(missing_ok=True)
        out.with_name(out.stem + ".part.mp4").unlink(missing_ok=True)


def _frame(store: Store, game: Game, frame: int) -> np.ndarray:
    from soccer_clips.track import ball_game

    return next(ball_game.read_frames(store, game, frame, frame + 1)).astype(np.int16)


def verify(store: Store, game: Game, cut_game: Game, segments: list[CutSegment]) -> None:
    """Frames at the start, middle and end of each piece must match the original's exactly-ish."""
    for s in segments:
        n = s.orig_end - s.orig_start
        for k in (0, n // 2, n - 1):
            d = np.abs(_frame(store, game, s.orig_start + k) - _frame(store, cut_game, s.new_start + k)).mean()
            if d > 1.0:
                raise RuntimeError(f"cut video doesn't match the original at frame {s.orig_start + k} (diff {d:.1f})")


# ---- moving stored data onto the new timeline ----

def data_root(store: Store) -> Path:
    return store.settings.games_dir.parent


def _global_files(store: Store) -> list[Path]:
    """Files outside the game folder that hold times or frame numbers of games."""
    from soccer_clips.track.bench import bench_dir

    root = data_root(store)
    labels = root / "labels"
    out = [labels / n for n in ("ball.jsonl", "ball_tasks.json", "ball_others.json", "ball_guesses.json")]
    out += [labels / "dataset" / "val_frames.json", bench_dir(store.settings) / "clips.json",
            bench_dir(store.settings) / "clips" / ".cut.json"]
    out += sorted(store.settings.reels_dir.glob("*/*.render.json"))
    return [p for p in out if p.exists()]


def backup(store: Store, game: Game) -> Path:
    """Copy everything the cut will change to <data root>/backup/<time>-cut-<game>/ (same layout)."""
    root = data_root(store)
    dest = root / "backup" / f"{datetime.now():%Y%m%d-%H%M%S}-cut-{game.id}"
    gdir = store.game_dir(game.year, game.id)
    files = [p for p in gdir.rglob("*") if p.is_file()] + _global_files(store)
    frames = root / "labels" / "frames" / game.id
    if frames.exists():
        files += [p for p in frames.iterdir() if p.is_file()]
    for p in files:
        rel = p.relative_to(root)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / rel)
    atomic_write_json(dest / "manifest.json", {"game": game.id, "files": [str(p.relative_to(root)) for p in files]})
    return dest


def _remap_game(game: Game, tm: TimeMap, report: dict) -> None:
    kept = []
    for c in game.clips:
        span = tm.span(c.start_s, c.end_s)
        if span is None:
            report.setdefault("clips dropped (outside game time)", []).append(c.id)
            continue
        c.start_s, c.end_s = max(0.0, span[0]), span[1]
        if c.callout:
            t = tm.time(c.callout.t_s)
            if t is None or not c.start_s <= t < c.end_s:
                report.setdefault("callouts dropped", []).append(c.id)
                c.callout = None
            else:
                c.callout.t_s = t
        kept.append(c)
    game.clips = kept
    game.kickoff_1h_s, game.end_1h_s = tm.time(game.kickoff_1h_s), tm.time(game.end_1h_s - 1e-6)
    game.kickoff_2h_s, game.end_2h_s = tm.time(game.kickoff_2h_s), tm.time(game.end_2h_s - 1e-6)


def _npz(p: Path) -> dict:
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _remap_tracks(store: Store, game: Game, tm: TimeMap, report: dict) -> None:
    from soccer_clips.track.ball_game import save_npz, tracks_dir

    d = tracks_dir(store, game)
    if not d.exists():
        return
    starts = {s.new_start for s in tm.segments}
    # ball detections, camera motion, fixed objects (aligned with the detections)
    if (d / "ball_raw.npz").exists():
        r = _npz(d / "ball_raw.npz")
        nf, keep = tm.frames(r["dets"][:, 0])
        dets = r["dets"][keep].copy()
        dets[:, 0] = nf[keep]
        mf, mkeep = tm.frames(r["motion_frames"])
        motion = r["motion"][mkeep].copy()
        mf = mf[mkeep]
        motion[np.isin(mf, list(starts))] = np.eye(2, 3)  # no camera step across a cut
        save_npz(d / "ball_raw.npz", ranges=tm.ranges(r["ranges"]), dets=dets, motion_frames=mf, motion=motion)
        if (d / "fixed.npz").exists():
            fx = _npz(d / "fixed.npz")
            if int(fx["n_dets"]) == len(keep):
                save_npz(d / "fixed.npz", blocked=fx["blocked"][keep], detected=fx["detected"][keep],
                         searched=fx["searched"][keep], n_dets=np.int64(int(keep.sum())))
    if (d / "ball.npz").exists():
        b = _npz(d / "ball.npz")
        nf, keep = tm.frames(b["frame"])
        out = {k: v[keep] for k, v in b.items() if k != "ranges" and v.shape[:1] == b["frame"].shape}
        out["frame"] = nf[keep].astype(b["frame"].dtype)
        save_npz(d / "ball.npz", **out, ranges=tm.ranges(b["ranges"]).astype(b["ranges"].dtype))
    if (d / "ball_fixes.json").exists():
        fixes = json.loads((d / "ball_fixes.json").read_text())
        for key in ("balls", "suppress"):
            moved = []
            for x in fixes.get(key, []):
                f = tm.frame(x["frame"])
                if f is not None:
                    moved.append({**x, "frame": f})
            fixes[key] = moved
        atomic_write_text(d / "ball_fixes.json", json.dumps(fixes, indent=1) + "\n")
    if (d / "world.npz").exists():
        w = _npz(d / "world.npz")
        nf, keep = tm.frames(w["frames"])
        out = {k: (v[keep] if v.shape[:1] == w["frames"].shape else v) for k, v in w.items()}
        out["frames"] = nf[keep].astype(w["frames"].dtype)
        save_npz(d / "world.npz", **out)
    # players: raw boxes (and kit colours, row for row), tracklets (their det index follows the raw rows)
    for raw_p in sorted(d.glob("people_raw_*.npz")):
        model = raw_p.stem.removeprefix("people_raw_")
        r = _npz(raw_p)
        nf, keep = tm.frames(r["dets"][:, 0])
        dets = r["dets"][keep].copy()
        dets[:, 0] = nf[keep]
        save_npz(raw_p, ranges=tm.ranges(r["ranges"]), dets=dets)
        new_index = np.cumsum(keep) - 1
        kit_p = d / f"people_kit_{model}.npz"
        if kit_p.exists():
            k = _npz(kit_p)
            if len(k["feats"]) == len(keep):
                save_npz(kit_p, feats=k["feats"][keep])
            else:
                kit_p.unlink()
                report.setdefault("files dropped (out of step)", []).append(kit_p.name)
        trk_p = d / f"people_{model}.npz"
        if trk_p.exists():
            t = _npz(trk_p)
            nf2, keep2 = tm.frames(t["frame"])
            if "det" in t:
                keep2 &= keep[np.clip(t["det"], 0, len(keep) - 1)]
            out = {k: (v[keep2] if v.shape[:1] == t["frame"].shape else v) for k, v in t.items() if k != "ranges"}
            out["frame"] = nf2[keep2].astype(t["frame"].dtype)
            if "det" in t:
                out["det"] = new_index[t["det"][keep2]].astype(t["det"].dtype)
            save_npz(trk_p, **out, ranges=tm.ranges(t["ranges"]).astype(t["ranges"].dtype))
    if (d / "people_picks.json").exists():
        picks = json.loads((d / "people_picks.json").read_text())["picks"]
        moved = [{**p, "frame": tm.frame(p["frame"])} for p in picks if tm.frame(p["frame"]) is not None]
        report["player labels dropped"] = len(picks) - len(moved)
        atomic_write_text(d / "people_picks.json", json.dumps({"picks": moved}, indent=1) + "\n")


def _remap_labels(store: Store, game: Game, tm: TimeMap, report: dict) -> None:
    root = data_root(store)
    labels = root / "labels"
    gid = game.id
    p = labels / "ball.jsonl"
    if p.exists():
        lines, dropped = [], 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get("game_id") == gid:
                f = tm.frame(x["frame"])
                if f is None:
                    dropped += 1
                    continue
                x["frame"] = f
            lines.append(json.dumps(x))
        atomic_write_text(p, "\n".join(lines) + "\n")
        report["ball labels dropped"] = dropped
    p = labels / "ball_tasks.json"
    if p.exists():
        tasks = json.loads(p.read_text())
        for name, items in tasks.get("sets", {}).items():
            moved = []
            for x in items:
                if x.get("game_id") == gid:
                    f = tm.frame(x["frame"])
                    if f is None:
                        continue
                    x = {**x, "frame": f}
                moved.append(x)
            tasks["sets"][name] = moved
        atomic_write_text(p, json.dumps(tasks, indent=1) + "\n")
    for name in ("ball_others.json", "ball_guesses.json"):
        p = labels / name
        if p.exists():
            data = json.loads(p.read_text())
            frames = {}
            for k, v in data.get("frames", {}).items():
                g, _, f = k.rpartition(":")
                if g == gid:
                    nf = tm.frame(int(f))
                    if nf is None:
                        continue
                    k = f"{g}:{nf}"
                frames[k] = v
            data["frames"] = frames
            atomic_write_text(p, json.dumps(data) + "\n")
    p = labels / "dataset" / "val_frames.json"
    if p.exists():
        vals = [[g, tm.frame(f) if g == gid else f] for g, f in json.loads(p.read_text())]
        atomic_write_json(p, [v for v in vals if v[1] is not None])
    # cached frame images and the exported training set are named by frame
    frames_dir = labels / "frames" / gid
    if frames_dir.exists():
        moved = frames_dir.with_name(gid + ".moving")
        frames_dir.rename(moved)
        frames_dir.mkdir()
        for img in moved.iterdir():
            f = tm.frame(int(img.stem)) if img.stem.isdigit() else None
            if f is not None:
                img.rename(frames_dir / f"{f}{img.suffix}")
        shutil.rmtree(moved)
    for sub in ("images", "labels"):
        for split in (labels / "dataset" / sub).glob("*"):
            files = sorted(split.glob(f"{gid}_*"))
            renames = []
            for fp in files:
                parts = fp.stem.split("_")
                f = tm.frame(int(parts[-2])) if len(parts) >= 3 and parts[-2].isdigit() else None
                if f is None:
                    fp.unlink()
                else:
                    renames.append((fp, fp.with_name(f"{gid}_{f}_{parts[-1]}{fp.suffix}")))
            for fp, new in renames:  # via temporary names: new names can clash with old ones
                fp.rename(fp.with_name(fp.name + ".moving"))
            for fp, new in renames:
                fp.with_name(fp.name + ".moving").rename(new)


def _remap_bench_and_reels(store: Store, game: Game, tm: TimeMap, report: dict) -> None:
    from soccer_clips.track import bench

    bset = bench.load_set(store.settings)
    stamps_p = bench.bench_dir(store.settings) / "clips" / ".cut.json"
    stamps = json.loads(stamps_p.read_text()) if stamps_p.exists() else {}
    kept = []
    for c in bset.clips:
        if c.game_id == game.id:
            span = tm.span(c.start_s, c.end_s)
            if span is None:
                report.setdefault("bench clips dropped", []).append(c.name)
                continue
            c.start_s, c.end_s = span
            if c.callout:
                t = tm.time(c.callout.t_s)
                c.callout = c.callout.model_copy(update={"t_s": t}) if t is not None else None
            if c.name in stamps:
                stamps[c.name] = [c.game_id, c.start_s, c.end_s]
        kept.append(c)
    bset.clips = kept
    if bset.clips or bench.bench_dir(store.settings).joinpath("clips.json").exists():
        bench.save_set(store.settings, bset)
    if stamps:
        atomic_write_json(stamps_p, stamps)
    for p in sorted(store.settings.reels_dir.glob("*/*.render.json")):
        r = json.loads(p.read_text())
        changed = False
        for sg in r.get("segments", []):
            if sg.get("game_id") != game.id:
                continue
            span = tm.span(sg["source_start_s"], sg["source_end_s"])
            if span:
                sg["source_start_s"], sg["source_end_s"] = span
            if sg.get("callout") and tm.time(sg["callout"]["t_s"]) is not None:
                sg["callout"]["t_s"] = tm.time(sg["callout"]["t_s"])
            changed = True
        if changed:
            atomic_write_json(p, r)


def cut_game(store: Store, game_id: str, log=print) -> dict:
    """Cut the game's video to game time and move all its data onto the new timeline (see above).
    Returns a report of what was dropped."""
    from soccer_clips.scan import ffprobe

    game = store.load_game(game_id)
    if game.cut:
        raise ValueError("this game has already been cut (restore it first to cut again)")
    segments = plan(store, game)
    rel = f"{game.year}/{game.id}.mp4"
    out = store.settings.cut_root / rel
    cut_video(store, game, segments, out, log)
    info = ffprobe(store.settings, out)
    cut = GameCut(file=rel, segments=segments, original_duration_s=game.duration_s)
    verify(store, game, game.model_copy(update={"cut": cut}), segments)
    log(f"{game.id}: cut video checked against the original")
    dest = backup(store, game)
    cut.backup = str(dest.relative_to(data_root(store)))
    log(f"{game.id}: backed up to {dest}")
    tm = TimeMap(segments, game.fps_num, game.fps_den)
    report: dict = {}
    _remap_tracks(store, game, tm, report)
    _remap_labels(store, game, tm, report)
    _remap_bench_and_reels(store, game, tm, report)
    shutil.rmtree(data_root(store) / "ballsrc" / game.id, ignore_errors=True)  # segments named by old frames
    with store.edit_game(game.id) as g:
        _remap_game(g, tm, report)
        g.cut = cut
        g.duration_s = float(info["format"]["duration"])
        g.proxy = None
    from soccer_clips.track import people

    people._cache.clear()
    people._roles_cache.clear()
    report["kept"] = f"{sum(s.orig_end - s.orig_start for s in segments) / game.fps / 60:.1f} of {game.duration_s / 60:.1f} min"
    log(f"{game.id}: done; {report}")
    return report


def restore(store: Store, game_id: str, log=print) -> Path:
    """Put back everything as it was before the game's cut (from its backup); the cut video stays."""
    game = store.load_game(game_id)
    if not game.cut or not game.cut.backup:
        raise ValueError("this game hasn't been cut")
    root = data_root(store)
    src = root / game.cut.backup
    manifest = json.loads((src / "manifest.json").read_text())
    shutil.rmtree(root / "labels" / "frames" / game_id, ignore_errors=True)  # images cached since, named by new frames
    for rel in manifest["files"]:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dest)
    from soccer_clips.track import people

    people._cache.clear()
    people._roles_cache.clear()
    log(f"{game_id}: restored from {src}")
    return src
