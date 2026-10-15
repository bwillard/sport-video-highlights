"""
Render a reel: normalise each clip to an intermediate file (same size, frame
rate, pixel format and audio layout; cached), measure the intermediates, then
join them with hard cuts into <reels_dir>/<reel_id>/<preset>.mp4.

Next to the mp4, <preset>.render.json records where each clip landed in the
reel and where it came from in its game, so the reel page can jump back to the
full game. Segment spans come from the measured intermediates, not from the
clip ranges, so they stay right once transitions and speed changes arrive.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from soccer_clips import ffmpeg
from soccer_clips.jobs import Job
from soccer_clips.models import Clip, Game, Preset, Reel, ReelItem, Render, Segment
from soccer_clips.store import Store

# Bump to invalidate every cached intermediate (e.g. after changing the normalise filters).
CACHE_VERSION = 1
OUTPUT_FPS = Fraction(30000, 1001)  # Veo's rate, so normally no frame-rate conversion


@dataclass(frozen=True)
class PresetSpec:
    width: int
    height: int
    x264_preset: str  # for the final encode
    crf: int  # for the final encode
    # Intermediates are re-encoded once more, so they're made at higher quality.
    intermediate_preset: str
    intermediate_crf: int


PRESETS: dict[str, PresetSpec] = {
    "landscape_1080p": PresetSpec(1920, 1080, "medium", 20, "veryfast", 16),
    "preview_540p": PresetSpec(960, 540, "ultrafast", 26, "ultrafast", 23),
}


def render_job(reel_id: str, preset: Preset) -> Job:
    return Job(kind="render", reel_id=reel_id, preset=preset)


def run_render(store: Store, job: Job) -> None:
    render = render_reel(store, job.reel_id, job.preset, lambda f: setattr(job, "progress", min(f, 0.99)))
    job.output = f"{job.reel_id}/{render.file}"


def _resolve(store: Store, reel: Reel) -> tuple[list[tuple[Game, Clip]], list[ReelItem]]:
    games: dict[str, Game] = {}
    parts, skipped = [], []
    for item in reel.items:
        try:
            game = games.get(item.game_id) or store.load_game(item.game_id)
            games[game.id] = game
            parts.append((game, game.clip(item.clip_id)))
        except KeyError:
            skipped.append(item)
    return parts, skipped


def _normalise(store: Store, game: Game, seek_s: float, duration_s: float, spec: PresetSpec,
               out: Path, on_progress: Callable[[float], None]) -> None:
    s = store.settings
    w, h = spec.width, spec.height
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    if Fraction(game.fps_num, game.fps_den) != OUTPUT_FPS:
        vf += f",fps={OUTPUT_FPS}"
    vf += ",format=yuv420p"
    inputs = ["-ss", f"{seek_s:.6f}", "-i", str(store.original_path(game))]
    if game.has_audio:
        audio_map = "0:a:0"
    else:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map = "1:a:0"
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        s.ffmpeg, "-y", "-v", "error", *inputs,
        # Output-side -t trims both streams; apad makes sure audio is never shorter than video.
        "-t", f"{duration_s:.6f}", "-map", "0:v:0", "-map", audio_map,
        "-vf", vf, "-af", "aresample=48000,aformat=channel_layouts=stereo,apad",
        "-c:v", "libx264", "-preset", spec.intermediate_preset, "-crf", str(spec.intermediate_crf),
        "-c:a", "aac", "-b:a", "192k", str(tmp),
    ]
    try:
        ffmpeg.run(cmd, duration_s, on_progress)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)


def render_reel(store: Store, reel_id: str, preset: Preset,
                on_progress: Callable[[float], None] = lambda f: None) -> Render:
    s = store.settings
    reel = store.load_reel(reel_id)
    spec = PRESETS[preset]
    folder = store.reel_dir(reel_id)
    cache = folder / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    parts, skipped = _resolve(store, reel)
    if not parts:
        raise ValueError("reel has no clips to render (all missing or none added)")
    windows = [ffmpeg.cut_window(game, clip.start_s, clip.end_s) for game, clip in parts]
    total = sum(d for _, d in windows)
    # Progress: normalising is the first half of the work, the final encode the second.
    done = 0.0

    def report(fraction: float, weight: float) -> None:
        on_progress((done + fraction * weight) / (2 * total))

    intermediates = []
    for (game, clip), (seek_s, duration_s) in zip(parts, windows):
        key = hashlib.sha1(json.dumps(
            [CACHE_VERSION, preset, game.id, game.original, round(seek_s, 6), round(duration_s, 6)]
        ).encode()).hexdigest()[:16]
        path = cache / f"{preset}-{key}.mp4"
        if not path.exists():
            _normalise(store, game, seek_s, duration_s, spec, path, lambda f: report(f, duration_s))
        done += duration_s
        intermediates.append(path)

    segments, t = [], 0.0
    for path, (game, clip) in zip(intermediates, parts):
        length = max(ffmpeg.stream_durations(s, path).values())
        frame_s = game.fps_den / game.fps_num
        segments.append(Segment(
            game_id=game.id, clip_id=clip.id,
            reel_start_s=t, reel_end_s=t + length,
            source_start_s=game.frame_at(clip.start_s) * frame_s,
            source_end_s=game.frame_at(clip.end_s) * frame_s,
        ))
        t += length

    out = folder / f"{preset}.mp4"
    tmp = folder / f"{preset}.part.mp4"
    inputs = [arg for p in intermediates for arg in ("-i", str(p))]
    n = len(intermediates)
    graph = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
    cmd = [
        s.ffmpeg, "-y", "-v", "error", *inputs,
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", spec.x264_preset, "-crf", str(spec.crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp),
    ]
    try:
        ffmpeg.run(cmd, t, lambda f: report(f, total))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)

    render = Render(
        reel_id=reel.id, preset=preset, file=out.name,
        duration_s=max(ffmpeg.stream_durations(s, out).values()),
        segments=segments, skipped=skipped, reel=reel,
    )
    store.save_render(render)

    # Drop cached intermediates this preset no longer uses (clips edited or removed).
    for p in cache.glob(f"{preset}-*.mp4"):
        if p not in intermediates:
            p.unlink()
    return render


def stale_reasons(store: Store, reel: Reel, render: Render) -> list[str]:
    """Why a render no longer matches its reel and clips (empty if it's current)."""
    reasons = []
    if reel.items != render.reel.items:
        reasons.append("clips were added, removed or reordered")
    games: dict[str, Game] = {}
    changed = deleted = 0
    for seg in render.segments:
        try:
            game = games.get(seg.game_id) or store.load_game(seg.game_id)
            games[game.id] = game
            clip = game.clip(seg.clip_id)
        except KeyError:
            deleted += 1
            continue
        frame_s = game.fps_den / game.fps_num
        if (round(game.frame_at(clip.start_s) * frame_s, 6), round(game.frame_at(clip.end_s) * frame_s, 6)) != (
            round(seg.source_start_s, 6), round(seg.source_end_s, 6)
        ):
            changed += 1
    if changed:
        reasons.append(f"{changed} clip(s) had their start/end changed")
    if deleted:
        reasons.append(f"{deleted} clip(s) were deleted")
    return reasons
