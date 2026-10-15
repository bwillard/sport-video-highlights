"""
Render a reel: normalise each clip to an intermediate file (same size, frame
rate, pixel format and audio layout; cached), make the title cards (an opening
card, then one before each clip; also cached), measure everything, then join
it with hard cuts into <reels_dir>/<reel_id>/<preset>.mp4.

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
from soccer_clips.reel.callout import STYLE_VERSION, callout_frame, splice_callout
from soccer_clips.reel.cards import clip_card_lines, draw_card, font_path, start_card_lines
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


def _card(store: Store, lines: list[str], spec: PresetSpec, duration_s: float, out: Path) -> None:
    """A silent clip of a still title card, exactly round(duration_s * fps) frames long."""
    png = out.with_suffix(".png")
    draw_card(lines, spec.width, spec.height).save(png)
    frames = max(1, round(duration_s * OUTPUT_FPS))
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        store.settings.ffmpeg, "-y", "-v", "error",
        "-loop", "1", "-framerate", str(OUTPUT_FPS), "-i", str(png),
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-map", "0:v", "-map", "1:a", "-frames:v", str(frames), "-t", f"{frames / OUTPUT_FPS:.6f}",
        "-vf", "setsar=1,format=yuv420p",
        "-c:v", "libx264", "-preset", spec.intermediate_preset, "-crf", str(spec.intermediate_crf),
        "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", str(tmp),
    ]
    try:
        ffmpeg.run(cmd, frames / OUTPUT_FPS)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        png.unlink(missing_ok=True)
    os.replace(tmp, out)


def _cache_name(preset: str, *key) -> str:
    digest = hashlib.sha1(json.dumps([CACHE_VERSION, preset, *key]).encode()).hexdigest()[:16]
    return f"{preset}-{digest}.mp4"


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
    # Progress: normalising clips is the first half of the work, the final
    # encode the second. Cards are quick and aren't counted.
    done = 0.0

    def report(fraction: float, weight: float) -> None:
        on_progress((done + fraction * weight) / (2 * total))

    def card(lines: list[str]) -> Path | None:
        if not lines or reel.card_s <= 0:
            return None
        path = cache / _cache_name(preset, "card", lines, round(reel.card_s, 3), font_path())
        if not path.exists():
            _card(store, lines, spec, reel.card_s, path)
        return path

    start_lines = start_card_lines(reel, [game for game, _ in parts]) if reel.card_s > 0 else []
    start_card = card(start_lines)
    entries = []  # (game, clip, card lines, card path or None, clip path, callout frame or None)
    used = set()  # cache files to keep
    freeze_s = float(max(1, round(reel.callout_s * OUTPUT_FPS)) / OUTPUT_FPS) if reel.callout_s > 0 else 0.0
    for (game, clip), (seek_s, duration_s) in zip(parts, windows):
        lines = clip_card_lines(game, clip) if reel.card_s > 0 else []
        card_path = card(lines)
        path = cache / _cache_name(preset, game.id, game.original, round(seek_s, 6), round(duration_s, 6))
        frame_n = callout_frame(game, clip) if freeze_s else None
        spliced = None
        if frame_n is not None:
            spliced = cache / _cache_name(preset, "callout", path.name, frame_n, clip.callout.model_dump(),
                                          round(reel.callout_s, 3), STYLE_VERSION)
        needs_clip = spliced is None or not spliced.exists()  # the splice is made from the clip
        if needs_clip and not path.exists():
            _normalise(store, game, seek_s, duration_s, spec, path, lambda f: report(f, duration_s))
        if spliced and not spliced.exists():
            splice_callout(store, path, frame_n, clip.callout, game, reel.callout_s, OUTPUT_FPS,
                           spec.intermediate_preset, spec.intermediate_crf, spliced)
        done += duration_s
        if path.exists():
            used.add(path)  # kept even when spliced, so moving the callout only redoes the splice
        entries.append((game, clip, lines, card_path, spliced or path, frame_n))

    length = lambda p: max(ffmpeg.stream_durations(s, p).values())
    intermediates, segments, t = [], [], 0.0
    if start_card:
        intermediates.append(start_card)
        t += length(start_card)
    for game, clip, lines, card_path, path, frame_n in entries:
        card_start = None
        if card_path:
            intermediates.append(card_path)
            card_start = t
            t += length(card_path)
        intermediates.append(path)
        clip_len = length(path)
        frame_s = game.fps_den / game.fps_num
        segments.append(Segment(
            game_id=game.id, clip_id=clip.id,
            reel_start_s=t, reel_end_s=t + clip_len,
            source_start_s=game.frame_at(clip.start_s) * frame_s,
            source_end_s=game.frame_at(clip.end_s) * frame_s,
            card_start_s=card_start, card_lines=lines if card_path else [],
            freeze_at_s=None if frame_n is None else t + float(frame_n / OUTPUT_FPS),
            freeze_s=0.0 if frame_n is None else freeze_s,
            callout=None if frame_n is None else clip.callout,
        ))
        t += clip_len

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
        duration_s=length(out), segments=segments, skipped=skipped, reel=reel,
        start_card_lines=start_lines if start_card else [],
    )
    store.save_render(render)

    # Drop cached intermediates this preset no longer uses (clips edited or removed).
    used |= set(intermediates)
    for p in cache.glob(f"{preset}-*.mp4"):
        if p not in used:
            p.unlink()
    return render


def stale_reasons(store: Store, reel: Reel, render: Render) -> list[str]:
    """Why a render no longer matches its reel and clips (empty if it's current)."""
    reasons = []
    if reel.items != render.reel.items:
        reasons.append("clips were added, removed or reordered")
    if reel.card_s != render.reel.card_s:
        reasons.append("card length changed")
    if reel.callout_s != render.reel.callout_s:
        reasons.append("callout freeze length changed")
    elif reel.card_s > 0 and start_card_lines(reel, [g for g, _ in _resolve(store, reel)[0]]) != render.start_card_lines:
        reasons.append("title card changed")
    games: dict[str, Game] = {}
    changed = deleted = cards = callouts = 0
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
        else:
            if seg.card_lines and clip_card_lines(game, clip) != seg.card_lines:
                cards += 1
            if reel.callout_s > 0 and (clip.callout if callout_frame(game, clip) is not None else None) != seg.callout:
                callouts += 1
    if callouts:
        reasons.append(f"{callouts} callout(s) added, moved or removed")
    if cards:
        reasons.append(f"{cards} clip card(s) changed (note or kickoff time)")
    if changed:
        reasons.append(f"{changed} clip(s) had their start/end changed")
    if deleted:
        reasons.append(f"{deleted} clip(s) were deleted")
    return reasons
