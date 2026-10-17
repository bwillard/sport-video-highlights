"""
Freeze-frame callout (Phase 3.1): the reel stops on one frame of a clip, dims
everything except a spotlight on the player, draws a ring and an arrow, holds for a
moment, then plays on.

The overlay is drawn with Pillow onto the frozen frame; the renderer splices
it into the clip's intermediate with ffmpeg.
"""

import os
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from soccer_clips import ffmpeg
from soccer_clips.models import Callout, Clip, Game
from soccer_clips.store import Store

DIM = 0.45  # brightness outside the spotlight
RING = (255, 255, 255)
STYLE_VERSION = 1  # bump when the look changes, to re-render cached freezes


def callout_frame(game: Game, clip: Clip) -> int | None:
    """Index of the callout frame within the clip's frames, or None if there's no usable callout."""
    c = clip.callout
    if c is None or not clip.start_s <= c.t_s < clip.end_s:
        return None
    first, last = game.frame_at(clip.start_s), game.frame_at(clip.end_s)
    return min(max(game.frame_at(c.t_s) - first, 0), last - first - 1)


def draw_callout(frame: Image.Image, callout: Callout, src_w: int, src_h: int) -> Image.Image:
    """
    Spotlight + ring + arrow on a frame of the rendered size. Callout x/y/r are
    relative to the source frame, which may be letterboxed inside this one.
    """
    w, h = frame.size
    scale = min(w / src_w, h / src_h)
    ox, oy = (w - src_w * scale) / 2, (h - src_h * scale) / 2
    cx, cy = ox + callout.x * src_w * scale, oy + callout.y * src_h * scale
    r = max(callout.r * src_h * scale, h * 0.02)

    # dim everything, then paste the original back through a soft-edged circle
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((cx - r * 1.15, cy - r * 1.15, cx + r * 1.15, cy + r * 1.15), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(r * 0.15))
    out = Image.composite(frame, Image.eval(frame, lambda v: int(v * DIM)), mask)

    draw = ImageDraw.Draw(out)
    width = max(3, round(h * 0.006))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=RING, width=width)
    # arrow pointing down at the ring; flipped below it if she's near the top
    size = max(h * 0.03, r * 0.5)
    gap = r + width + size * 0.4
    direction = -1 if cy - gap - size * 1.6 > 0 else 1  # -1 = arrow above the ring
    tip_y = cy + direction * gap
    base_y = tip_y + direction * size
    draw.polygon([(cx, tip_y), (cx - size * 0.6, base_y), (cx + size * 0.6, base_y)], fill=RING)
    draw.rectangle((cx - size * 0.18, min(base_y, base_y + direction * size * 0.6),
                    cx + size * 0.18, max(base_y, base_y + direction * size * 0.6)), fill=RING)
    return out


def splice_callout(store: Store, clip_path: Path, frame_n: int, callout: Callout, game: Game,
                   freeze_s: float, fps: Fraction, x264_preset: str, crf: int, out: Path) -> float:
    """
    Copy a normalised clip intermediate to `out` with a freeze inserted at
    frame `frame_n`: frames [0, n) play, frame n is held (with the overlay)
    for `freeze_s` seconds with silence, then frames [n, end) play. Returns
    the actual freeze length (a whole number of frames).
    """
    s = store.settings
    png = out.with_name(out.stem + ".callout.png")
    raw = out.with_name(out.stem + ".frame.png")
    ffmpeg.run([s.ffmpeg, "-y", "-v", "error", "-i", str(clip_path),
                "-vf", f"select=eq(n\\,{frame_n})", "-frames:v", "1", "-fps_mode", "vfr", str(raw)], 0)
    frame = Image.open(raw).convert("RGB")
    draw_callout(frame, callout, game.width, game.height).save(png)
    raw.unlink(missing_ok=True)

    freeze_frames = max(1, round(freeze_s * fps))
    freeze = freeze_frames / fps
    split_t = frame_n / fps
    graph = (
        f"[0:v]trim=end_frame={frame_n},setpts=PTS-STARTPTS[va];"
        f"[0:v]trim=start_frame={frame_n},setpts=PTS-STARTPTS[vb];"
        f"[0:a]atrim=end={float(split_t):.6f},asetpts=PTS-STARTPTS[aa];"
        f"[0:a]atrim=start={float(split_t):.6f},asetpts=PTS-STARTPTS[ab];"
        f"[1:v]setsar=1,format=yuv420p[vf];"
        f"[va][aa][vf][2:a][vb][ab]concat=n=3:v=1:a=1[v][a]"
    )
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [
        s.ffmpeg, "-y", "-v", "error", "-i", str(clip_path),
        "-loop", "1", "-framerate", str(fps), "-t", f"{float(freeze):.6f}", "-i", str(png),
        "-f", "lavfi", "-t", f"{float(freeze):.6f}", "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-r", str(fps),
        "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "192k", str(tmp),
    ]
    try:
        ffmpeg.run(cmd, 0)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        png.unlink(missing_ok=True)
    os.replace(tmp, out)
    return float(freeze)
