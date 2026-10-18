"""
Render a reel: normalise each clip to an intermediate file (same size, frame
rate, pixel format and audio layout; cached), make the title cards (an opening
card, then one before each clip; also cached), measure everything, then join
it with hard cuts into <reels_dir>/<reel_id>/<name>.mp4, where the name is
the preset plus the variant (see models.Variant: a render can leave out the
clip notes, or the notes and the callout freezes).

Every piece is encoded once, at the final quality, by the same encoder (the
GPU when there is one; see encode.py), so the join copies the video instead of
encoding it again; only the audio is re-encoded there, to smooth the joins.

Next to the mp4, <name>.render.json records where each clip landed in the
reel and where it came from in its game, so the reel page can jump back to the
full game. Segment spans come from the measured intermediates, not from the
clip ranges, so they stay right once transitions and speed changes arrive.
"""

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path

from PIL import Image

from soccer_clips import ffmpeg
from soccer_clips.encode import FROM_RGB, Encoder, choose_encoder
from soccer_clips.jobs import Job
from soccer_clips.models import Callout, Clip, Game, Preset, Reel, ReelItem, Render, Segment, Variant, render_name
from soccer_clips.reel.callout import STYLE_VERSION, callout_frame, draw_callout
from soccer_clips.reel.cards import (
    card_seconds, clip_card_lines, draw_caption, draw_card, font_path, LAYOUT_VERSION, start_card_lines,
)
from soccer_clips.store import Store, atomic_write_text

# Bump to invalidate every cached intermediate (e.g. after changing the normalise filters).
CACHE_VERSION = 4
CAPTION_VERSION = 1  # bump when the caption look changes
OUTPUT_FPS = Fraction(30000, 1001)  # Veo's rate, so normally no frame-rate conversion
# Pieces keep uncompressed audio, which (unlike AAC) can end exactly with the
# video, so pieces join without tiny gaps; the join encodes it to AAC.
PIECE = ".mov"
PIECE_AUDIO = "pcm_s16le"
PIECE_WORKERS = 2  # pieces made at once (see _make_pieces)


@dataclass(frozen=True)
class PresetSpec:
    width: int
    height: int
    preview: bool  # fixed fast settings instead of the .env encoder and quality


PRESETS: dict[str, PresetSpec] = {
    "landscape_1080p": PresetSpec(1920, 1080, False),
    "preview_540p": PresetSpec(960, 540, True),
}
# Share of the progress bar for the final join (it copies the video, so it's quick).
JOIN_SHARE = 0.05


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


def _shape(game: Game, spec: PresetSpec) -> str:
    """CPU filters that make a game's frames the preset's size and rate (letterboxed if needed)."""
    w, h = spec.width, spec.height
    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    if Fraction(game.fps_num, game.fps_den) != OUTPUT_FPS:
        vf += f",fps={OUTPUT_FPS}"
    return vf + ",format=yuv420p"


@dataclass(frozen=True)
class Span:
    """One piece of a clip: its frames [start, end), or (hold) the callout frame held still."""

    start: int = 0
    end: int = 0
    hold: bool = False
    caption: bool = False  # drawn with the caption over it


def clip_spans(frames: int, caption_frames: int, freeze_at: int | None, freeze_frames: int) -> list[Span]:
    """
    How a clip is cut into pieces. The caption shows for the first
    caption_frames frames of the clip as it plays in the reel (a freeze
    counts) and during the freeze. Only the captioned pieces need the CPU;
    the rest can stay on the GPU, and a note edit only redoes those.
    """
    c = caption_frames
    if freeze_at is None:
        spans = [Span(0, min(c, frames), caption=True), Span(min(c, frames), frames)]
    else:
        n = freeze_at
        after = max(0, min(c - n - freeze_frames, frames - n))  # caption frames left after the freeze
        spans = [Span(0, min(c, n), caption=True), Span(min(c, n), n), Span(hold=True, caption=c > 0),
                 Span(n, n + after, caption=True), Span(n + after, frames)]
    return [s for s in spans if s.hold or s.end > s.start]


def _encode(store: Store, enc: Encoder, inputs: list[str], graph: str, seconds: float, out: Path,
            on_progress: Callable[[float], None] | None = None) -> None:
    """Run a filter graph with [v] and [a] outputs into a piece of the reel."""
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    cmd = [
        store.settings.ffmpeg, "-y", "-v", "error", *enc.global_args(), *inputs,
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        *enc.video_args(), "-c:a", PIECE_AUDIO, str(tmp),
    ]
    try:
        ffmpeg.run(cmd, seconds, on_progress)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)


def _make_pieces(todo: list[tuple[float, Callable]], total_s: float, on_progress: Callable[[float], None]) -> None:
    """
    Make the pieces, PIECE_WORKERS at a time: each ffmpeg run spends a while
    starting up and seeking, and captioned pieces mostly use the CPU while
    plain ones mostly use the GPU, so two at once finish sooner.
    """
    done = [0.0] * len(todo)
    cached = total_s - sum(seconds for seconds, _ in todo)

    def report(i: int, seconds: float, fraction: float) -> None:
        done[i] = fraction * seconds
        on_progress((cached + sum(done)) / total_s)

    with ThreadPoolExecutor(PIECE_WORKERS) as pool:
        futures = [pool.submit(make, partial(report, i, seconds)) for i, (seconds, make) in enumerate(todo)]
        try:
            for f in futures:
                f.result()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def _video_piece(store: Store, game: Game, seek_s: float, span: Span, spec: PresetSpec, enc: Encoder,
                 caption: Image.Image | None, out: Path, on_progress: Callable[[float], None]) -> None:
    """Frames [span.start, span.end) of the clip that starts at seek_s, with the caption drawn over if given."""
    frames = span.end - span.start
    seconds = float(frames / OUTPUT_FPS)
    start_s = seek_s + float(span.start / OUTPUT_FPS)
    same_shape = (game.width, game.height) == (spec.width, spec.height) and \
        Fraction(game.fps_num, game.fps_den) == OUTPUT_FPS
    on_gpu = enc.gpu and same_shape and caption is None
    if on_gpu:
        # Already the right size and rate (Veo downloads are) and nothing to
        # draw: decode on the GPU and keep the frames there.
        decode = enc.gpu_decode_args()
    else:
        # The GPU still decodes when there is one; frames come back for the CPU filters.
        decode = ["-hwaccel", "vaapi", "-hwaccel_device", enc.gpu_device] if enc.gpu else []
    # -t stops reading the game soon after the piece; trim/atrim cut it exactly.
    inputs = [*decode, "-ss", f"{start_s:.6f}", "-t", f"{seconds + 1:.6f}", "-i", str(store.original_path(game))]
    audio_in = "0:a:0"
    if not game.has_audio:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_in = "1:a:0"
    graph = (f"[{audio_in}]aresample=48000,aformat=channel_layouts=stereo,"
             f"apad=whole_dur={seconds:.6f},atrim=end={seconds:.6f},asetpts=PTS-STARTPTS[a];")
    if on_gpu:
        graph += f"[0:v:0]setsar=1,scale_vaapi=format=nv12,trim=end_frame={frames}[v]"
    else:
        graph += f"[0:v:0]{_shape(game, spec)},trim=end_frame={frames},setpts=PTS-STARTPTS"
        if caption:
            caption_png = out.with_suffix(".caption.png")
            caption.save(caption_png)
            inputs += ["-loop", "1", "-i", str(caption_png)]
            graph += f"[m];[m][{inputs.count('-i') - 1}:v]overlay=0:0:shortest=1"
        graph += f",format=yuv420p{enc.upload}[v]"
    try:
        _encode(store, enc, inputs, graph, seconds, out, on_progress)
    finally:
        if caption:
            caption_png.unlink(missing_ok=True)


def _still_piece(store: Store, png: Path, frames: int, enc: Encoder, out: Path) -> None:
    """A silent piece showing one picture (a title card, or a held callout frame) for `frames` frames."""
    seconds = float(frames / OUTPUT_FPS)
    inputs = ["-loop", "1", "-framerate", str(OUTPUT_FPS), "-i", str(png),
              "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    graph = (f"[0:v]setsar=1,{FROM_RGB},trim=end_frame={frames}{enc.upload}[v];"
             f"[1:a]atrim=end={seconds:.6f}[a]")
    _encode(store, enc, inputs, graph, seconds, out)


def _card(store: Store, lines: list[str], spec: PresetSpec, enc: Encoder, duration_s: float, out: Path) -> None:
    """A silent piece of a title card, exactly round(duration_s * fps) frames long."""
    png = out.with_suffix(".png")
    draw_card(lines, spec.width, spec.height).save(png)
    try:
        _still_piece(store, png, max(1, round(duration_s * OUTPUT_FPS)), enc, out)
    finally:
        png.unlink(missing_ok=True)


def _hold(store: Store, game: Game, seek_s: float, frame_n: int, spec: PresetSpec, enc: Encoder,
          callout: Callout, frames: int, caption: Image.Image | None, out: Path,
          on_progress: Callable[[float], None] = lambda f: None) -> None:
    """The clip's frame frame_n with the callout drawn on (and the caption, if given), held for `frames` frames."""
    png = out.with_suffix(".png")
    # seek_s is half a frame before the clip's first frame, so this lands half a frame before frame_n
    at_s = seek_s + float(frame_n / OUTPUT_FPS)
    ffmpeg.run([store.settings.ffmpeg, "-y", "-v", "error", "-ss", f"{at_s:.6f}", "-i", str(store.original_path(game)),
                "-vf", _shape(game, spec), "-frames:v", "1", str(png)], 0)
    try:
        picture = draw_callout(Image.open(png).convert("RGB"), callout, game.width, game.height)
        if caption:
            picture = Image.alpha_composite(picture.convert("RGBA"), caption).convert("RGB")
        picture.save(png)
        _still_piece(store, png, frames, enc, out)
    finally:
        png.unlink(missing_ok=True)


def _join(store: Store, pieces: list[Path], duration_s: float, out: Path,
          on_progress: Callable[[float], None]) -> None:
    """
    Join the pieces end to end. The video is copied (every piece was encoded
    with the same settings); the audio is re-encoded, with aresample filling or
    trimming the few ms where one piece's audio runs past its video.
    """
    listing = out.with_name(out.stem + ".concat.txt")
    listing.write_text("".join("file '{}'\n".format(str(p).replace("'", "'\\''")) for p in pieces))
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        store.settings.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-map", "0:v", "-map", "0:a", "-c:v", "copy",
        "-af", "aresample=async=1:first_pts=0", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(tmp),
    ]
    try:
        ffmpeg.run(cmd, duration_s, on_progress)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        listing.unlink(missing_ok=True)
    os.replace(tmp, out)


def _cache_name(preset: str, *key) -> str:
    digest = hashlib.sha1(json.dumps([CACHE_VERSION, preset, *key]).encode()).hexdigest()[:16]
    return f"{preset}-{digest}{PIECE}"


def render_reel(store: Store, reel_id: str, preset: Preset,
                on_progress: Callable[[float], None] = lambda f: None, variant: Variant = "all") -> Render:
    s = store.settings
    reel = store.load_reel(reel_id)
    spec = PRESETS[preset]
    name = render_name(preset, variant)
    notes = shows_notes(variant)
    enc = choose_encoder(s, spec.width, spec.height, spec.preview)
    cache_name = lambda *key: _cache_name(preset, enc.key, *key)  # pieces from other settings can't be joined
    folder = store.reel_dir(reel_id)
    cache = folder / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    parts, skipped = _resolve(store, reel)
    if not parts:
        raise ValueError("reel has no clips to render (all missing or none added)")
    windows = [ffmpeg.cut_window(game, clip.start_s, clip.end_s) for game, clip in parts]
    total = sum(d for _, d in windows)
    def card(lines: list[str]) -> Path | None:
        if not lines or reel.card_s <= 0:
            return None
        seconds = card_seconds(lines, reel.card_s)
        path = cache / cache_name("card", lines, round(seconds, 3), font_path(), LAYOUT_VERSION)
        if not path.exists():
            _card(store, lines, spec, enc, seconds, path)
        return path

    start_lines = start_card_lines(reel, [game for game, _ in parts]) if reel.card_s > 0 else []
    start_card = card(start_lines)
    overlay = reel.card_style == "overlay" and reel.card_s > 0
    entries = []  # (game, clip, card lines, card path or None, [(span, piece)], caption)
    todo = []  # (seconds of video, make(on_progress)) for each piece that isn't cached
    freeze_frames = max(1, round(reel.freeze_s * OUTPUT_FPS)) if reel.freeze_s > 0 and shows_callouts(variant) else 0
    freeze_s = float(freeze_frames / OUTPUT_FPS)
    for (game, clip), (seek_s, duration_s) in zip(parts, windows):
        lines = clip_card_lines(game, clip, notes) if reel.card_s > 0 else []
        card_path = None if overlay else card(lines)
        frame_n = callout_frame(game, clip) if freeze_frames else None
        caption = None  # (lines, seconds shown from the clip start)
        caption_frames = 0
        if overlay and lines:
            caption_frames = max(1, round(card_seconds(lines, reel.card_s) * OUTPUT_FPS))
            caption = (lines, float(caption_frames / OUTPUT_FPS))
        source = [game.id, game.original, round(seek_s, 6)]
        caption_key = [lines, font_path(), CAPTION_VERSION, LAYOUT_VERSION] if caption else None
        caption_img = None  # drawn when a piece needs it
        pieces = []
        for span in clip_spans(round(duration_s * OUTPUT_FPS), caption_frames, frame_n, freeze_frames):
            if span.hold:
                path = cache / cache_name("hold", *source, frame_n, clip.callout.model_dump(), STYLE_VERSION,
                                          freeze_frames, span.caption and caption_key)
            else:
                path = cache / cache_name("clip", *source, span.start, span.end, span.caption and caption_key)
            if not path.exists():
                if span.caption and caption_img is None:
                    caption_img = draw_caption(lines, spec.width, spec.height)
                over = caption_img if span.caption else None
                if span.hold:
                    todo.append((0.0, partial(_hold, store, game, seek_s, frame_n, spec, enc, clip.callout,
                                              freeze_frames, over, path)))
                else:
                    todo.append((float((span.end - span.start) / OUTPUT_FPS),
                                 partial(_video_piece, store, game, seek_s, span, spec, enc, over, path)))
            pieces.append((span, path))
        entries.append((game, clip, lines, card_path, pieces, caption))
    # Progress: making the pieces is nearly all of the work, then the join
    # (JOIN_SHARE). Cards and callout holds are quick and aren't counted.
    _make_pieces(todo, total, lambda f: on_progress(f * (1 - JOIN_SHARE)))

    length = lambda p: max(ffmpeg.stream_durations(s, p).values())
    intermediates, segments, t = [], [], 0.0
    if start_card:
        intermediates.append(start_card)
        t += length(start_card)
    for game, clip, lines, card_path, pieces, caption in entries:
        card_start = None
        if card_path:
            intermediates.append(card_path)
            card_start = t
            t += length(card_path)
        start, freeze_at = t, None
        for span, path in pieces:
            intermediates.append(path)
            if span.hold:
                freeze_at = t
            t += length(path)
        frame_s = game.fps_den / game.fps_num
        segments.append(Segment(
            game_id=game.id, clip_id=clip.id,
            reel_start_s=start, reel_end_s=t,
            source_start_s=game.frame_at(clip.start_s) * frame_s,
            source_end_s=game.frame_at(clip.end_s) * frame_s,
            card_start_s=card_start, card_lines=lines if card_path else [],
            freeze_at_s=freeze_at,
            freeze_s=0.0 if freeze_at is None else freeze_s,
            callout=None if freeze_at is None else clip.callout,
            caption_lines=caption[0] if caption else [],
            caption_s=caption[1] if caption else 0.0,
        ))

    out = folder / f"{name}.mp4"
    _join(store, intermediates, t, out, lambda f: on_progress(1 - JOIN_SHARE + f * JOIN_SHARE))

    render = Render(
        reel_id=reel.id, preset=preset, variant=variant, file=out.name,
        duration_s=length(out), segments=segments, skipped=skipped, reel=reel,
        start_card_lines=start_lines if start_card else [],
        card_style=reel.card_style, encoder=enc.label,
    )
    store.save_render(render)

    # Drop cached intermediates no render of this preset uses any more (clips
    # edited or removed). Each variant lists what it used, since they share
    # intermediates (the same clip, or a card without a note).
    atomic_write_text(cache / f"{name}.used.json", json.dumps(sorted(p.name for p in intermediates)) + "\n")
    keep = {n for f in cache.glob(f"{preset}*.used.json") for n in json.loads(f.read_text())}
    for p in [*cache.glob(f"{preset}-*{PIECE}"), *cache.glob(f"{preset}-*.mp4")]:  # .mp4: before CACHE_VERSION 4
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
