"""
Render a reel: normalise each clip to an intermediate file (same size, frame
rate, pixel format and audio layout; cached), make the title cards (an opening
card, then one before each clip; also cached), measure everything, then join
it with hard cuts into <reels_dir>/<reel_id>/<name>.mp4, where the name is
the preset plus the variant (see models.Variant: a render can leave out the
clip notes, or the notes and the callout freezes).

Next to the mp4, <name>.render.json records where each clip landed in the
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
from soccer_clips.models import Clip, Game, Preset, Reel, ReelItem, Render, Segment, Variant, render_name
from soccer_clips.reel.callout import STYLE_VERSION, callout_frame, splice_callout
from soccer_clips.reel.cards import (
    card_seconds, clip_card_lines, draw_caption, draw_card, font_path, LAYOUT_VERSION, start_card_lines,
)
from soccer_clips.store import Store, atomic_write_text

# Bump to invalidate every cached intermediate (e.g. after changing the normalise filters).
CACHE_VERSION = 1
CAPTION_VERSION = 1  # bump when the caption look changes
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


def shows_notes(variant: Variant) -> bool:
    return variant == "all"


def shows_callouts(variant: Variant) -> bool:
    return variant != "plain"


def render_job(reel_id: str, preset: Preset, variant: Variant = "all") -> Job:
    return Job(kind="render", reel_id=reel_id, preset=preset, variant=variant)


def run_render(store: Store, job: Job) -> None:
    render = render_reel(store, job.reel_id, job.preset, lambda f: setattr(job, "progress", min(f, 0.99)),
                         variant=job.variant or "all")
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


def _caption(store: Store, src: Path, lines: list[str], spec: PresetSpec,
             windows: list[tuple[float, float]], out: Path) -> None:
    """Copy an intermediate with a caption overlaid during the given (start, end) times."""
    png = out.with_name(out.stem + ".caption.png")
    draw_caption(lines, spec.width, spec.height).save(png)
    enable = "+".join(f"between(t,{a:.4f},{b:.4f})" for a, b in windows)
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        store.settings.ffmpeg, "-y", "-v", "error", "-i", str(src), "-loop", "1", "-i", str(png),
        "-filter_complex", f"[0:v][1:v]overlay=0:0:shortest=1:enable='{enable}',format=yuv420p[v]",
        "-map", "[v]", "-map", "0:a", "-c:a", "copy",
        "-c:v", "libx264", "-preset", spec.intermediate_preset, "-crf", str(spec.intermediate_crf), str(tmp),
    ]
    try:
        ffmpeg.run(cmd, 0)
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
                on_progress: Callable[[float], None] = lambda f: None, variant: Variant = "all") -> Render:
    s = store.settings
    reel = store.load_reel(reel_id)
    spec = PRESETS[preset]
    name = render_name(preset, variant)
    notes = shows_notes(variant)
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
        seconds = card_seconds(lines, reel.card_s)
        path = cache / _cache_name(preset, "card", lines, round(seconds, 3), font_path(), LAYOUT_VERSION)
        if not path.exists():
            _card(store, lines, spec, seconds, path)
        return path

    start_lines = start_card_lines(reel, [game for game, _ in parts]) if reel.card_s > 0 else []
    start_card = card(start_lines)
    overlay = reel.card_style == "overlay" and reel.card_s > 0
    entries = []  # (game, clip, card lines, card path or None, clip path, callout frame or None, caption)
    used = set()  # cache files to keep
    freeze_s = float(max(1, round(reel.freeze_s * OUTPUT_FPS)) / OUTPUT_FPS) \
        if reel.freeze_s > 0 and shows_callouts(variant) else 0.0
    for (game, clip), (seek_s, duration_s) in zip(parts, windows):
        lines = clip_card_lines(game, clip, notes) if reel.card_s > 0 else []
        card_path = None if overlay else card(lines)
        path = cache / _cache_name(preset, game.id, game.original, round(seek_s, 6), round(duration_s, 6))
        frame_n = callout_frame(game, clip) if freeze_s else None
        spliced = None
        if frame_n is not None:
            spliced = cache / _cache_name(preset, "callout", path.name, frame_n, clip.callout.model_dump(),
                                          round(reel.freeze_s, 3), STYLE_VERSION)
        caption = None  # (lines, seconds shown from the clip start)
        captioned = None
        if overlay and lines:
            caption_s = float(max(1, round(card_seconds(lines, reel.card_s) * OUTPUT_FPS)) / OUTPUT_FPS)
            caption = (lines, caption_s)
            cap_windows = [(0.0, caption_s)]
            if frame_n is not None:  # also show it during the callout freeze
                at = float(frame_n / OUTPUT_FPS)
                cap_windows.append((at, at + freeze_s))
            captioned = cache / _cache_name(preset, "caption", (spliced or path).name, lines,
                                            [(round(a, 4), round(b, 4)) for a, b in cap_windows],
                                            font_path(), CAPTION_VERSION, LAYOUT_VERSION)
        # Each step is made from the one before (clip -> splice -> caption) and
        # each is cached, so an edit only redoes the steps after it.
        final = captioned or spliced or path
        need_caption = captioned is not None and not captioned.exists()
        need_splice = spliced is not None and not spliced.exists() and (need_caption or final == spliced)
        need_clip = not path.exists() and (need_splice or need_caption and spliced is None or final == path)
        if need_clip:
            _normalise(store, game, seek_s, duration_s, spec, path, lambda f: report(f, duration_s))
        if need_splice:
            splice_callout(store, path, frame_n, clip.callout, game, reel.freeze_s, OUTPUT_FPS,
                           spec.intermediate_preset, spec.intermediate_crf, spliced)
        if need_caption:
            _caption(store, spliced or path, caption[0], spec, cap_windows, captioned)
        done += duration_s
        for p in (path, spliced):  # kept so moving a callout or editing a note redoes only the later steps
            if p and p.exists():
                used.add(p)
        entries.append((game, clip, lines, card_path, final, frame_n, caption))

    length = lambda p: max(ffmpeg.stream_durations(s, p).values())
    intermediates, segments, t = [], [], 0.0
    if start_card:
        intermediates.append(start_card)
        t += length(start_card)
    for game, clip, lines, card_path, path, frame_n, caption in entries:
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
            caption_lines=caption[0] if caption else [],
            caption_s=caption[1] if caption else 0.0,
        ))
        t += clip_len

    out = folder / f"{name}.mp4"
    tmp = folder / f"{name}.part.mp4"
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
        reel_id=reel.id, preset=preset, variant=variant, file=out.name,
        duration_s=length(out), segments=segments, skipped=skipped, reel=reel,
        start_card_lines=start_lines if start_card else [],
        card_style=reel.card_style,
    )
    store.save_render(render)

    # Drop cached intermediates no render of this preset uses any more (clips
    # edited or removed). Each variant lists what it used, since they share
    # intermediates (the same clip, or a card without a note).
    used |= set(intermediates)
    atomic_write_text(cache / f"{name}.used.json", json.dumps(sorted(p.name for p in used)) + "\n")
    keep = {n for f in cache.glob(f"{preset}*.used.json") for n in json.loads(f.read_text())}
    for p in cache.glob(f"{preset}-*.mp4"):
        if p.name not in keep:
            p.unlink()
    return render


def _card_timing_changed(reel: Reel, render: Render) -> bool:
    """True if the cards in a render aren't the length card_seconds() would give them now."""
    def expected(lines: list[str]) -> float:
        return float(max(1, round(card_seconds(lines, reel.card_s) * OUTPUT_FPS)) / OUTPUT_FPS)

    shown = []  # (lines, measured length)
    if render.start_card_lines and render.segments:
        first = render.segments[0]
        shown.append((render.start_card_lines, first.card_start_s if first.card_start_s is not None else first.reel_start_s))
    shown += [(s.card_lines, s.reel_start_s - s.card_start_s) for s in render.segments if s.card_start_s is not None]
    shown += [(s.caption_lines, s.caption_s) for s in render.segments if s.caption_lines]
    # measured lengths include a few ms of audio past the last frame
    return any(abs(length - expected(lines)) > 0.05 for lines, length in shown)


def stale_reasons(store: Store, reel: Reel, render: Render) -> list[str]:
    """Why a render no longer matches its reel and clips (empty if it's current)."""
    reasons = []
    if reel.items != render.reel.items:
        reasons.append("clips were added, removed or reordered")
    if reel.card_style != render.card_style:
        reasons.append("descriptions switched between overlay and title cards")
    if reel.card_s != render.reel.card_s:
        reasons.append("minimum card length changed")
    notes, freezes = shows_notes(render.variant), shows_callouts(render.variant)
    if freezes and reel.freeze_s != render.reel.freeze_s:
        reasons.append("callout freeze length changed")
    if reel.card_s > 0 and start_card_lines(reel, [g for g, _ in _resolve(store, reel)[0]]) != render.start_card_lines:
        reasons.append("title card changed")
    if reel.card_s == render.reel.card_s and _card_timing_changed(reel, render):
        reasons.append("title card timing changed")
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
            shown = seg.card_lines or seg.caption_lines
            if shown and clip_card_lines(game, clip, notes) != shown:
                cards += 1
            if freezes and reel.freeze_s > 0 and (clip.callout if callout_frame(game, clip) is not None else None) != seg.callout:
                callouts += 1
    if callouts:
        reasons.append(f"{callouts} callout(s) added, moved or removed")
    if cards:
        reasons.append(f"{cards} clip description(s) changed (note or kickoff time)")
    if changed:
        reasons.append(f"{changed} clip(s) had their start/end changed")
    if deleted:
        reasons.append(f"{deleted} clip(s) were deleted")
    return reasons
